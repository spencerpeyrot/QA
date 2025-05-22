"""
call_summary_pipeline.py  –  Audit-grade evaluator & fixer for earnings-call summaries
"""

# --------------------------------------------------------------------- #
# Imports & logging setup
# --------------------------------------------------------------------- #
import os, re, json, random, asyncio, logging, textwrap
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

# Import helpers using relative path
from ..helpers import get_mongo_client, get_openai_client, call_llm_with_function_call, split_text_into_items

env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ---------- logging ---------- #
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%(H:%M:%S)",
    level=logging.INFO
)
log = logging.getLogger("CALL-SUM")
log.setLevel(logging.DEBUG)
for v in ("motor", "pymongo", "httpcore", "httpx", "openai"):
    logging.getLogger(v).setLevel(logging.WARNING)

# --------------------------------------------------------------------- #
# ENV / CONFIG
# --------------------------------------------------------------------- #
# OPENAI_API_KEYS will be removed as helpers.get_openai_client handles key selection.
# OPENAI_API_KEYS = [k for k in (
#     os.getenv("OPENAI_API_KEY"),
#     os.getenv("OPENAI_API_KEY_BACKUP1"),
#     os.getenv("OPENAI_API_KEY_BACKUP2")) if k]

MONGO_URI         = os.getenv("EVAL_MONGO_URI", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "o4-mini")
DAILY_SAMPLE_SIZE = int(os.getenv("DAILY_SAMPLE_SIZE", 1))

EVAL_COLL_NAME    = "evaluations"
PIPELINE_TAG      = "CALL-SUM"

TRANSCRIPT_CAP_CHARS = 70000

# --------------------------------------------------------------------- #
# Function-call schema
# --------------------------------------------------------------------- #
evaluation_schema = {
    "name": "evaluate_call_summary",
    "description": "Fact-check each bullet in an earnings-call summary against the official transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "factual_criteria": {
                "type": "object",
                "properties": {
                    "numbers_match_transcript": {
                        "type": "boolean",
                        "description": "True if every numerical figure in the summary bullets exactly matches the numbers spoken in the transcript."
                    },
                    "statements_supported": {
                        "type": "boolean",
                        "description": "True if all non-numeric statements *in the summary* are directly supported by or are reasonable inferences from the transcript content. Minor paraphrasing is acceptable."
                    }
                },
                "required": ["numbers_match_transcript", "statements_supported"]
            },
            "completeness_criteria": {
                "type": "object",
                "properties": {
                    "covers_key_points": {
                        "type": "boolean",
                        "description": "True if the summary covers all major points discussed in the earnings call relevant to the summary's scope."
                    },
                    "includes_context": {
                        "type": "boolean",
                        "description": "True if sufficient context is provided *within the summary* for each bullet point to understand its significance based on transcript information."
                    }
                },
                "required": ["covers_key_points", "includes_context"]
            },
            "quality_criteria": {
                "type": "object",
                "properties": {
                    "clear_structure": {
                        "type": "boolean",
                        "description": "True if the summary bullets are logically organized, clear, and concise."
                    },
                    "professional_tone": {
                        "type": "boolean",
                        "description": "True if the language used is professional and appropriate for financial reporting."
                    }
                },
                "required": ["clear_structure", "professional_tone"]
            },
            "hallucination_free": {
                "type": "boolean",
                "description": "True if the summary does not introduce substantive information, details, or assertions not found in or reasonably inferable from the transcript. Minor paraphrasing or summarization of transcript content, including slight variations in phrasing or context for otherwise supported numbers, is allowed and does not count as hallucination. False if the summary invents new facts or makes claims with no basis in the transcript."
            },
            "quality_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall quality rating (0–100), balanced across factual accuracy, completeness, structure, and absence of hallucinations."
            },
            "criteria_explanations": {
                "type": "object",
                "description": "Explanations for each criterion evaluation",
                "properties": {
                    "numbers_match_transcript": {"type": "string"},
                    "statements_supported": {"type": "string"},
                    "covers_key_points": {"type": "string"},
                    "includes_context": {"type": "string"},
                    "clear_structure": {"type": "string"},
                    "professional_tone": {"type": "string"},
                    "hallucination_free": {"type": "string"}
                }
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List specific claims or figures *present in the summary* that are substantively contradicted by the transcript or for which no clear supporting evidence can be found in the transcript. Do not list items here if they are minor paraphrases or slight numerical variations that are generally consistent with the transcript (these might affect 'numbers_match_transcript' but not necessarily 'unsupported_claims' or 'hallucination_free'). This field is for information *in the summary* that is demonstrably false or absent from the transcript."
            },
            "score_explanation": {
                "type": "string",
                "description": "Narrative justification for the assigned quality_score and criteria decisions."
            }
        },
        "required": [
            "factual_criteria",
            "completeness_criteria",
            "quality_criteria",
            "hallucination_free",
            "quality_score",
            "criteria_explanations",
            "unsupported_claims",
            "score_explanation"
        ]
    }
}

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
para_re = re.compile(r"^- ", re.M)

# split_bullets function will be removed and replaced by helpers.split_text_into_items
# def split_bullets(text: str) -> List[str]:
#     """Return individual bullet blocks (keeps heading lines)."""
#     if not text:
#         return []
#     return [b.strip() for b in text.split("\n\n") if para_re.match(b.strip())]

# --------------------------------------------------------------------- #
class CallSummaryEvaluator:
    def __init__(self, mongo_uri: str):
        if not mongo_uri:
            raise ValueError("EVAL_MONGO_URI env var not set")
        cli = get_mongo_client(mongo_uri) # Use helper directly
        db  = cli["asc-fin-data"]
        self.src  = db["user_activities"]
        self.eval = db[EVAL_COLL_NAME]

    # --------------------------- prompt ---------------------------- #
    def build_prompt(self, summary: str, transcript: str) -> str:
        if not summary:
            log.error("Empty summary text received")
            return ""
        
        bullet_blocks = "\\n".join(
            f"BULLET:\\n{b}" for b in split_text_into_items(summary, item_separator="\\n\\n", item_start_regex=para_re) # Use helper directly
        ) or summary   # fallback if no bullets

        prompt = (
            "You are an audit-grade fact checker specializing in earnings-call summaries. "
            "Compare each bullet of the analyst-written summary with the official transcript excerpt, "
            "and evaluate numerical accuracy, statement support, and overall clarity. "
            "Note: it's acceptable for summaries to omit details; missing content should not be marked as an error. "
            "Only unsupported or hallucinated statements should count against the summary."
            "\\n\\nSUMMARY:\\n" +
            bullet_blocks +
            "\\n\\nCALL TRANSCRIPT:\\n" + # Changed from CALL TRANSCRIPT (excerpt, may be truncated)
            transcript + # Use full transcript
            "\\n\\nRespond using the evaluate_call_summary function."
        )
        log.debug("Prompt size for evaluation ≈ %d chars", len(prompt))
        return prompt

    # --------------------------- evaluate -------------------------- #
    async def evaluate(self, doc: Dict[str, Any], corrected=False) -> Dict[str, Any]:
        """Evaluate a single earnings call summary against its transcript."""
        summary = doc.get("agent_response_corrected") if corrected else doc.get("agent_response")
        transcript = doc.get("agent_sources", {}).get("transcript")
        
        if not summary or not transcript:
            log.error("Missing content in document %s - summary: %s, transcript: %s", 
                     doc.get("_id"), bool(summary), bool(transcript))
            return {
                "document_id": str(doc["_id"]),
                "ticker": doc["user_question"],
                "timestamp": doc["timestamp"],
                "pipeline": PIPELINE_TAG,
                "is_correction": corrected,
                "evaluation": {"error": "Missing required content"},
                "evaluated_at": datetime.now(timezone.utc)
            }

        system_prompt = (
            "You are an audit-grade earnings-call fact checker.\n"
            "Your task is to evaluate the provided \"SUMMARY\" against the \"CALL TRANSCRIPT (excerpt)\".\n"
            "Focus SOLELY on the claims and statements made *within the SUMMARY*.\n\n"
            "Key Evaluation Points:\n"
            "1.  **Factual Accuracy**: Verify if numbers and statements *in the SUMMARY* match the transcript.\n"
            "2.  **Support**: Determine if claims *in the SUMMARY* are supported by the transcript.\n"
            "3.  **Omissions are Allowed**: The SUMMARY is not expected to cover everything in the transcript. "
            "Do NOT penalize the summary or list claims as \"unsupported\" if the summary simply omits information "
            "present in the transcript. An \"unsupported claim\" is a piece of information *present in the summary* "
            "that cannot be verified by the transcript.\n"
            "4.  **Hallucinations**: A hallucination occurs if the SUMMARY introduces information or details that are NOT "
            "found in the transcript. Minor paraphrasing of transcript content is acceptable and not a hallucination.\n"
            "5.  **`unsupported_claims` Field**: Use this field ONLY to list specific sentences or data points "
            "*from the SUMMARY* that you find are not supported by the transcript. Do not use this field to list items "
            "*missing* from the summary that are present in the transcript.\n\n"
            "Return your evaluation using the `evaluate_call_summary` function.\n"
            "If all boolean criteria within `factual_criteria`, `completeness_criteria`, and `quality_criteria` are true, "
            "AND `hallucination_free` is true, then `quality_score` MUST be 100."
        )
        user_prompt = self.build_prompt(summary, transcript)

        # Use helpers.call_llm_with_function_call
        ev_result = await call_llm_with_function_call( # Use helper directly
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            function_schema=evaluation_schema,
            openai_model=OPENAI_MODEL,
            tool_choice={"type": "function", "function": {"name": "evaluate_call_summary"}},
            temperature=1.0, # Set to 1.0 as required by o4-mini model
            logger=log
        )

        if ev_result is None or "error" in ev_result:
            error_detail = "Unknown LLM error" # Default if ev_result is None
            if isinstance(ev_result, dict): # Check if ev_result is a dict before accessing keys
                if "details" in ev_result:
                    error_detail = ev_result["details"]
                elif "function_name" in ev_result: 
                    error_detail = f"Unexpected function called: {ev_result['function_name']}"
                elif "error" in ev_result: 
                    error_detail = str(ev_result.get("error", "Unknown LLM error")) # Use str representation of error value
            
            log.error("LLM eval failed for %s: %s", doc["user_question"], error_detail)
            ev = {"error": f"eval_error: {error_detail}"}
        else:
            ev = ev_result
        
        # Force score to 100 if all criteria are true - only if ev is not an error dict
        if "error" not in ev:
            if (ev.get("hallucination_free", False) and
                all(ev.get("factual_criteria", {}).values()) and
                all(ev.get("completeness_criteria", {}).values()) and
                all(ev.get("quality_criteria", {}).values())):
                ev["quality_score"] = 100
                if "score_explanation" in ev:
                    ev["score_explanation"] = "Perfect score of 100 awarded as all evaluation criteria were met."
                    
            log.debug("Eval result for %s: %s", doc["user_question"], textwrap.shorten(json.dumps(ev), 200))
        # Removed old try-except for LLM call as it's handled by the helper and above logic

        return {
            "document_id": str(doc["_id"]),
            "ticker": doc["user_question"],
            "timestamp": doc["timestamp"],
            "pipeline": PIPELINE_TAG,
            "is_correction": corrected,
            "evaluation": ev,
            "evaluated_at": datetime.now(timezone.utc)
        }

    # -------------------------- sampling -------------------------- #
    async def sample_docs(self, start: datetime, end: datetime):
        """Sample documents from the specified date range."""
        pipeline = [
            {
                "$match": {
                    "agent": "earnings",
                    "mode": "earnings_transcript_summary",
                    "timestamp": {"$gte": start, "$lt": end},
                    "agent_response": {"$exists": True},
                    "agent_sources.transcript": {"$exists": True}
                }
            }
        ]
        
        docs = await self.src.aggregate(pipeline).to_list(length=None)
        if not docs:
            log.info("No documents found in date range %s to %s", start.date(), end.date())
            return []
            
        sample = random.sample(docs, min(len(docs), DAILY_SAMPLE_SIZE))
        log.info("Sampled %d docs: %s", len(sample), ", ".join(d["user_question"] for d in sample))
        return sample

    # ---------------------- range processing ---------------------- #
    async def process_range(self, start: datetime, end: datetime):
        """Process documents in the date range."""
        cur = start
        evaluated_docs = []
        
        while cur < end:
            nxt = cur + timedelta(days=1)
            log.info("Evaluating %s", cur.date())
            docs = await self.sample_docs(cur, nxt)
            
            if not docs:
                cur = nxt
                continue

            for doc in docs:
                try:
                    # Log the initial agent response for every processed document
                    initial_response_full = doc.get("agent_response", "N/A")
                    log.info("Initial agent_response for %s (doc_id: %s, full):\\n%s",
                             doc["user_question"], str(doc["_id"]), initial_response_full)

                    ev = await self.evaluate(doc)
                    await self.eval.insert_one(ev)
                    evaluated_docs.append({
                        "ticker": doc["user_question"],
                        "doc_id": str(doc["_id"]),
                        "needs_correction": await self.needs_fix(ev),
                        "eval_id": ev["_id"]
                    })
                    log.info("Evaluated %s: %s", doc["user_question"], 
                             "passed" if not await self.needs_fix(ev) else "needs correction")
                except Exception as e:
                    log.error("Failed to evaluate %s: %s", doc["user_question"], e)
                await asyncio.sleep(1)
            cur = nxt
            
        return evaluated_docs

    # ----------------------- correction logic --------------------- #
    async def needs_fix(self, ev: Dict[str, Any]) -> bool:
        """Only flag for fix when hallucinations or factual inaccuracies occur."""
        if "error" in ev["evaluation"]:
            return True
        e = ev["evaluation"]
        return not (
            e.get("hallucination_free") and
            e.get("factual_criteria", {}).get("numbers_match_transcript") and
            e.get("factual_criteria", {}).get("statements_supported")
        )

    async def correct(self, ev_doc):
        base = await self.src.find_one({"_id": ObjectId(ev_doc["document_id"])})
        if not base:
            return {"error": "orig_missing"}

        e = ev_doc["evaluation"]
        instr = []

        factual_criteria = e.get("factual_criteria", {})
        if not factual_criteria.get("numbers_match_transcript"):
            instr.append("- Fix numbers that don't match the transcript.")
        if not factual_criteria.get("statements_supported"):
            instr.append("- Remove or cite unsupported statements.")

        completeness_criteria = e.get("completeness_criteria", {})
        if not completeness_criteria.get("covers_key_points"):
            instr.append("- Add missing key points from the transcript.")
        if not completeness_criteria.get("includes_context"):
            instr.append("- Add necessary context for better understanding.")

        quality_criteria = e.get("quality_criteria", {})
        if not quality_criteria.get("clear_structure"):
            instr.append("- Improve organization and clarity of bullet points.")
        if not quality_criteria.get("professional_tone"):
            instr.append("- Adjust language to be more professional.")

        if not e.get("hallucination_free"):
            instr.append("- Remove or correct unsupported claims:")
            for claim in e.get("unsupported_claims", []):
                instr.append(f"  • {claim}")

        transcript = base["agent_sources"].get("transcript", "")
        system_corr = (
            "You are an expert financial editor and fact checker. Your task is to meticulously correct the ORIGINAL SUMMARY.\n"
            "**Critical Instructions - Adhere Strictly:**\n"
            "1.  **Header Preservation**: The first two lines of the summary, specifically `### Earnings Call Summary ###` and the `Date: YYYY-MM-DD` line immediately following it, MUST be preserved VERBATIM from the ORIGINAL SUMMARY. These lines are NEVER to be considered for correction, alteration, or removal.\n"
            "2.  **Minimal Changes Only**: Beyond the preserved header, make ONLY the changes necessary to address the specific issues listed in the \"TASK\". Do NOT rephrase, restructure, or alter any part of the report that is already correct or not mentioned in the \"TASK\".\n"
            "3.  **Preserve Correct Content Verbatim**: If a statement, bullet point, sentence, or any part of the ORIGINAL SUMMARY (after the initial two header lines) is factually accurate, correctly cited, and does not pertain to a listed issue, it MUST be preserved EXACTLY AS IS in the corrected version. Do NOT omit, add, or reword it.\n"
            "4.  **Maintain Original Formatting and Structure**: The CORRECTED SUMMARY MUST retain the exact same markdown formatting (e.g., `**Header**`, bullet points `- `), paragraphing, line breaks, and overall structure as the ORIGINAL SUMMARY for all content following the initial two header lines. Correct only the textual content of problematic elements as identified in the \"TASK\".\n"
            "5.  **Focus Solely on the \"TASK\"**: Your corrections should exclusively target the problems detailed in the \"TASK\" section of the user prompt. Do not introduce new information or re-evaluate claims not listed in the \"TASK\".\n"
            "6.  **Return Only Corrected Summary**: Provide only the full text of the corrected summary, ensuring it reflects all original formatting and structure, with targeted fixes applied according to these instructions."
        )
        user_corr = (
            "ORIGINAL SUMMARY:\n" + base["agent_response"] +
            "\\n\\nTRANSCRIPT:\n" + transcript +
            "\\n\\nTASK:\nBased on a previous evaluation, the following issues were identified in the ORIGINAL SUMMARY. Correct them precisely:\n" + "\\n".join(instr) +
            "\\n\\nINSTRUCTIONS FOR CORRECTION:\n"
            "1.  Review each issue listed in the \"TASK\". Correct ONLY these specific issues within the ORIGINAL SUMMARY.\n"
            "2.  **For any part of the ORIGINAL SUMMARY NOT listed as an issue: copy it VERBATIM to the CORRECTED SUMMARY, including all original markdown (bolding, headers, bullet styles, etc.) and line breaks.**\n"
            "3.  If an issue involves an \"unsupported claim\", either correct the claim to match the TRANSCRIPT or remove the specific unsupported part of the claim. Ensure the correction is minimal.\n"
            "4.  If an issue involves numerical inaccuracies, correct the numbers to match the TRANSCRIPT precisely.\n"
            "5.  The final CORRECTED SUMMARY must mirror the ORIGINAL SUMMARY's formatting (headings, markdown, bullet points, paragraph structure) perfectly, with only the necessary textual corrections applied to address the \"TASK\".\n"
            "6.  Return only the full text of the corrected summary."
        )
        log.debug("Prompt size for correction ≈ %d chars", len(system_corr) + len(user_corr))

        client = await get_openai_client() # Use helper directly
        try:
            rsp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_corr},
                    {"role": "user",   "content": user_corr}
                ]
            )
            corrected = rsp.choices[0].message.content.strip()
        except Exception as e:
            log.error("LLM correction failed: %s", e)
            return {"error": f"corr_error: {e}"}

        log.info("Doc %s corrected (dev-mode, not patched)", base["_id"])

        # TODO: Uncomment the following block when ready to enable automatic corrections in production
        # This will update the original document in user_activities with the corrected content
        # await self.src.update_one(
        #     {"_id": base["_id"]},
        #     {"$set": {
        #         "agent_response_corrected": corrected,
        #         "corrected_at": datetime.now(timezone.utc)
        #     }}
        # )

        return {"corrected": True, "corrected_text": corrected}

    async def process_corrections(self, start: datetime, end: datetime):
        """Process corrections for documents that failed evaluation."""
        evaluated_docs = await self.process_range(start, end)
        docs_needing_correction = [doc for doc in evaluated_docs if doc["needs_correction"]]

        if not docs_needing_correction:
            log.info("No documents need correction")
            return

        log.info("%d documents need correction: %s", 
                 len(docs_needing_correction), 
                 ", ".join(d["ticker"] for d in docs_needing_correction))

        for doc in docs_needing_correction:
            try:
                original_doc = await self.src.find_one({"_id": ObjectId(doc["doc_id"])})
                if not original_doc:
                    log.error("Could not find original document for %s", doc["ticker"])
                    continue

                eval_doc = await self.eval.find_one({"_id": doc["eval_id"]})
                if not eval_doc:
                    log.error("Could not find evaluation for %s", doc["ticker"])
                    continue

                # Log what was identified as wrong using the evaluation details
                evaluation_details = eval_doc.get("evaluation", {})
                if "error" in evaluation_details:
                    log.info("Document %s (%s) flagged for correction due to evaluation error: %s",
                             doc["ticker"], doc["doc_id"], evaluation_details["error"])
                else:
                    wrong_items = []
                    # Factual Criteria
                    fc = evaluation_details.get("factual_criteria", {})
                    if not fc.get("numbers_match_transcript", True): wrong_items.append("Numbers don't match transcript")
                    if not fc.get("statements_supported", True): wrong_items.append("Statements not supported")
                    
                    # Completeness Criteria
                    cc = evaluation_details.get("completeness_criteria", {})
                    if not cc.get("covers_key_points", True): wrong_items.append("Doesn't cover key points")
                    if not cc.get("includes_context", True): wrong_items.append("Lacks context")

                    # Quality Criteria
                    qc = evaluation_details.get("quality_criteria", {})
                    if not qc.get("clear_structure", True): wrong_items.append("Unclear structure")
                    if not qc.get("professional_tone", True): wrong_items.append("Unprofessional tone")

                    # Hallucinations
                    if not evaluation_details.get("hallucination_free", True):
                        wrong_items.append("Contains hallucinations")
                        unsupported_claims = evaluation_details.get("unsupported_claims", [])
                        if unsupported_claims:
                            wrong_items.append(f"Unsupported claims: {', '.join(unsupported_claims)}")
                    
                    if wrong_items:
                         log.info("Document %s (%s) identified issues for correction: %s", 
                                  doc["ticker"], doc["doc_id"], "; ".join(wrong_items))
                    elif not doc["needs_correction"]: # Should not happen if logic is sound, but as a fallback
                         log.info("Document %s (%s) needs correction but no specific issues logged.", doc["ticker"], doc["doc_id"])


                res = await self.correct(eval_doc)
                if res.get("corrected"):
                    corrected_text_full = res.get("corrected_text", "N/A")
                    log.info("Correction attempt for %s generated text (full):\\n%s", 
                             doc["ticker"], corrected_text_full)
                    
                    # Create a temporary document for re-evaluation that includes the corrected text
                    doc_for_re_eval = {
                        **original_doc, # Carry over _id, user_question, timestamp, agent_sources etc.
                        "agent_response_corrected": res.get("corrected_text") # Add the new corrected text
                    }
                    reval = await self.evaluate(doc_for_re_eval, corrected=True) # Pass the temporary doc
                    
                    # Log the updated analysis
                    reval_eval = reval.get("evaluation", {})
                    if "error" in reval_eval:
                        log.info("Re-evaluation for %s after correction resulted in an error: %s", 
                                 doc["ticker"], reval_eval["error"])
                    else:
                        reval_score = reval_eval.get('quality_score', 'N/A')
                        reval_hallu = reval_eval.get('hallucination_free', 'N/A')
                        reval_fact_nums = reval_eval.get('factual_criteria', {}).get('numbers_match_transcript', 'N/A')
                        reval_fact_stmts = reval_eval.get('factual_criteria', {}).get('statements_supported', 'N/A')
                        log.info("Re-evaluation for %s after correction: Score: %s, Hallucination Free: %s, Numbers Match: %s, Statements Supported: %s",
                                 doc["ticker"], reval_score, reval_hallu, reval_fact_nums, reval_fact_stmts)


                    # TODO: Uncomment the following block when ready to enable automatic corrections in production
                    # This will update the evaluation document with correction metadata
                    # await self.eval.update_one(
                    #     {"_id": doc["eval_id"]},
                    #     {"$set": {
                    #         "with_corrections": True,
                    #         "corrected_evaluation": reval["evaluation"],
                    #         "corrected_evaluated_at": reval["evaluated_at"],
                    #         "corrected_text": res["corrected_text"]
                    #     }}
                    # )

                    log.info("Would have updated evaluation for %s with correction", doc["ticker"])
            except Exception as e:
                log.error("Error processing correction for %s: %s", doc["ticker"], str(e))
            await asyncio.sleep(1)

# --------------------------------------------------------------------- #
async def main():
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=1)

    ev = CallSummaryEvaluator(MONGO_URI)
    log.info("=== Evaluation pass ===")
    await ev.process_corrections(start, end)

if __name__ == "__main__":
    asyncio.run(main())

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
OPENAI_API_KEYS = [k for k in (
    os.getenv("OPENAI_API_KEY"),
    os.getenv("OPENAI_API_KEY_BACKUP1"),
    os.getenv("OPENAI_API_KEY_BACKUP2")) if k]

MONGO_URI         = os.getenv("EVAL_MONGO_URI", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "o4-mini")
DAILY_SAMPLE_SIZE = int(os.getenv("DAILY_SAMPLE_SIZE", 1))

EVAL_COLL_NAME    = "evaluations"
PIPELINE_TAG      = "CALL-SUM"

TRANSCRIPT_CAP_CHARS = 51000

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
                        "description": "True if all non-numeric statements in the summary are directly supported by the transcript content."
                    }
                },
                "required": ["numbers_match_transcript", "statements_supported"]
            },
            "completeness_criteria": {
                "type": "object",
                "properties": {
                    "covers_key_points": {
                        "type": "boolean",
                        "description": "True if the summary covers all major points discussed in the earnings call."
                    },
                    "includes_context": {
                        "type": "boolean",
                        "description": "True if sufficient context is provided for each bullet point to understand its significance."
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
                "description": "False if any bullet contains details or assertions not found in the transcript. Minor paraphrasing is allowed and does not count as hallucination."
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
                "description": "List of any specific claims or figures in the summary that are not present in the transcript."
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

def split_bullets(text: str) -> List[str]:
    """Return individual bullet blocks (keeps heading lines)."""
    if not text:
        return []
    return [b.strip() for b in text.split("\n\n") if para_re.match(b.strip())]

# --------------------------------------------------------------------- #
class CallSummaryEvaluator:
    def __init__(self, mongo_uri: str):
        if not mongo_uri:
            raise ValueError("EVAL_MONGO_URI env var not set")
        cli = AsyncIOMotorClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db  = cli["asc-fin-data"]
        self.src  = db["user_activities"]
        self.eval = db[EVAL_COLL_NAME]

    # --------------------------- prompt ---------------------------- #
    def build_prompt(self, summary: str, transcript: str) -> str:
        if not summary:
            log.error("Empty summary text received")
            return ""
        
        bullet_blocks = "\n\n".join(
            f"BULLET:\n{b}" for b in split_bullets(summary)
        ) or summary   # fallback if no bullets

        prompt = (
            "You are an audit-grade fact checker specializing in earnings-call summaries. "
            "Compare each bullet of the analyst-written summary with the official transcript excerpt, "
            "and evaluate numerical accuracy, statement support, and overall clarity. "
            "Note: it's acceptable for summaries to omit details; missing content should not be marked as an error. "
            "Only unsupported or hallucinated statements should count against the summary."
            "\n\nSUMMARY:\n" +
            bullet_blocks +
            "\n\nCALL TRANSCRIPT (excerpt, may be truncated):\n" +
            transcript[:TRANSCRIPT_CAP_CHARS] +
            "\n\nRespond using the evaluate_call_summary function."
        )
        log.debug("Prompt size ≈ %d chars", len(prompt))
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
            "You are an audit-grade earnings-call fact checker. "
            "Compare the summary against the official transcript excerpt. "
            "Allow summaries to omit content; missing details should not lower the score. "
            "Only unsupported statements should be flagged as hallucinations. "
            "Return results using the evaluation schema. "
            "If all criteria (factual_criteria, completeness_criteria, quality_criteria) "
            "are true and there are no hallucinations, the quality_score must be 100."
        )
        user_prompt = self.build_prompt(summary, transcript)

        client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
        try:
            llm = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={"type": "function", "function": {"name": "evaluate_call_summary"}}
            )
            ev = json.loads(llm.choices[0].message.tool_calls[0].function.arguments)
            
            # Force score to 100 if all criteria are true
            if (ev.get("hallucination_free", False) and
                all(ev.get("factual_criteria", {}).values()) and
                all(ev.get("completeness_criteria", {}).values()) and
                all(ev.get("quality_criteria", {}).values())):
                ev["quality_score"] = 100
                if "score_explanation" in ev:
                    ev["score_explanation"] = "Perfect score of 100 awarded as all evaluation criteria were met."
                    
            log.debug("Eval result for %s: %s", doc["user_question"], textwrap.shorten(json.dumps(ev), 200))
        except Exception as e:
            log.error("LLM eval failed for %s: %s", doc["user_question"], e)
            ev = {"error": f"eval_error: {e}"}

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
                    initial_response_snippet = (doc.get("agent_response", "")[:150] + "...") if doc.get("agent_response") else "N/A"
                    log.info("Initial agent_response for %s (doc_id: %s, first 150 chars): %s",
                             doc["user_question"], str(doc["_id"]), initial_response_snippet)

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
        system_corr = "You are an expert financial editor and fact checker."
        user_corr = (
            "ORIGINAL SUMMARY:\n" + base["agent_response"] +
            "\n\nTRANSCRIPT EXCERPT:\n" + transcript[:TRANSCRIPT_CAP_CHARS] +
            "\n\nTASK:\n" + "\n".join(instr) +
            "\n\nINSTRUCTIONS:\n"
            "1. Correct only the problematic bullets identified above.\n"
            "2. Preserve the overall bullet structure and order.\n"
            "3. Return only the corrected summary text."
        )

        client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
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
                    corrected_text_snippet = (res.get("corrected_text", "")[:150] + "...") if res.get("corrected_text") else "N/A"
                    log.info("Correction attempt for %s generated text (first 150 chars): %s", 
                             doc["ticker"], corrected_text_snippet)
                    
                    reval = await self.evaluate(original_doc, corrected=True)
                    
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

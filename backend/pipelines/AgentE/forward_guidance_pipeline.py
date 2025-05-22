"""
forward_guidance_pipeline.py  –  Agent-E Forward-Guidance Evaluation + Correction
"""

# ------------------------------------------------------------------ #
# Imports & logging setup
# ------------------------------------------------------------------ #
import os, re, json, random, asyncio, logging, textwrap
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pandas_market_calendars as mcal

# ---------- logging ---------- #
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
log = logging.getLogger("FWD-GUID")
log.setLevel(logging.DEBUG)
for noisy in ("motor", "pymongo", "httpcore", "httpx", "openai", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ------------------------------------------------------------------ #
# ENV / CONFIG
# ------------------------------------------------------------------ #
OPENAI_API_KEYS   = [k for k in (
    os.getenv("OPENAI_API_KEY"),
    os.getenv("OPENAI_API_KEY_BACKUP1"),
    os.getenv("OPENAI_API_KEY_BACKUP2")) if k]
MONGO_URI         = os.getenv("EVAL_MONGO_URI", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "o4-mini")
DAILY_SAMPLE_SIZE = int(os.getenv("DAILY_SAMPLE_SIZE", 1))
EVAL_COLL_NAME    = "evaluations"
PIPELINE_TAG      = "FWD-GUID"

# ------------------------------------------------------------------ #
# Function-calling schema
# ------------------------------------------------------------------ #
evaluation_schema = {
    "name": "evaluate_forward_guidance",
    "description": "Paragraph-level fact check of a forward-guidance analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "factual_criteria": {
                "type": "object",
                "description": "Checks on raw data accuracy and citation correctness",
                "properties": {
                    "accurate_numbers": {
                        "type": "boolean",
                        "description": "True if all numerical figures (e.g., growth rates, forecasted metrics) in each paragraph precisely match the content of the cited source documents."
                    },
                    "correct_citations": {
                        "type": "boolean",
                        "description": "True if every bracketed reference [1], [2], etc. correctly points to the matching SOURCE DOCUMENT entry and supporting text."
                    }
                },
                "required": ["accurate_numbers", "correct_citations"]
            },
            "completeness_criteria": {
                "type": "object",
                "description": "Coverage checks for analysis and guidance sections",
                "properties": {
                    "paragraphs_all_supported": {
                        "type": "boolean",
                        "description": "True if each analysis paragraph is backed by at least one source citation; no paragraph is left unverified."
                    },
                    "guidance_section_present": {
                        "type": "boolean",
                        "description": "True if a dedicated 'GUIDANCE SECTION' (i.e. the forward-looking 'Predictive Estimates') is included at the end."
                    }
                },
                "required": ["paragraphs_all_supported", "guidance_section_present"]
            },
            "quality_criteria": {
                "type": "object",
                "description": "Stylistic and structural consistency checks",
                "properties": {
                    "clear_structure": {
                        "type": "boolean",
                        "description": "True if the analysis is logically organized, with clear headings, bullet points, and consistent formatting."
                    },
                    "guidance_section_consistent": {
                        "type": "boolean",
                        "description": "True if the guidance section's tone and data align with the preceding analysis paragraphs (no contradictions or new unsupported claims)."
                    }
                },
                "required": ["clear_structure", "guidance_section_consistent"]
            },
            "hallucination_free": {
                "type": "boolean",
                "description": "False if there are any details not supported by the SOURCE DOCUMENTS. Common market knowledge, straightforward inferences, and minor editorial phrasing are allowed and do not count as hallucinations."
            },
            "quality_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall quality rating (0–100), balancing factual accuracy, completeness, structure, and absence of hallucinations."
            },
            "criteria_explanations": {
                "type": "object",
                "description": "Detailed rationale for each criterion judgment",
                "properties": {
                    "accurate_numbers":             {"type": "string"},
                    "correct_citations":            {"type": "string"},
                    "paragraphs_all_supported":     {"type": "string"},
                    "guidance_section_present":     {"type": "string"},
                    "clear_structure":              {"type": "string"},
                    "guidance_section_consistent":  {"type": "string"},
                    "hallucination_free":           {"type": "string"}
                }
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of any statements or forecasts not directly supported by the source documents."
            },
            "score_explanation": {
                "type": "string",
                "description": "Overall narrative explaining how the quality_score was determined."
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

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
# para_re = re.compile(r"^- +(?:\*\*)?", re.M)
# cid_re  = re.compile(r"\[(\d+)\]")

# def split_paragraphs(text: str) -> List[str]:
#     return [p.strip() for p in text.split("\n\n") if para_re.match(p.strip())]

# def extract_ids(para: str) -> List[str]:
#     return cid_re.findall(para)

# ------------------------------------------------------------------ #
class ForwardGuidanceEvaluator:
    def __init__(self, mongo_uri: str):
        if not mongo_uri:
            raise ValueError("EVAL_MONGO_URI env var not set")
        cli = AsyncIOMotorClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db  = cli["asc-fin-data"]
        self.src  = db["user_activities"]
        self.eval = db[EVAL_COLL_NAME]

    # --------------------------- sampling ------------------------- #
    async def sample_docs(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        docs = await self.src.find({
            "agent": "earnings",
            "mode":  "forward_guidance",
            "timestamp": {"$gte": start, "$lt": end},
            "agent_sources": {"$exists": True}
        }).to_list(length=None)
        
        sample = []
        if docs:
            sample = random.sample(docs, min(len(docs), DAILY_SAMPLE_SIZE))
        
        log.info("Sampled %d docs: %s", len(sample),
                 ", ".join(d.get("user_question", "N/A") for d in sample))
        return sample

    # --------------------------- prompt --------------------------- #
    def build_prompt(self, analysis: str, sources: Dict[str, Any]) -> str:
        blocks, all_ids = [], set()

        # 1) Each analysis paragraph
        for para in split_paragraphs(analysis):
            ids = extract_ids(para)
            all_ids.update(ids)
            blocks.append(f"PARAGRAPH:\n{para}\n")

        # 2) Guidance section
        guidance = analysis.split("**Predictive Estimates**:")[-1].strip()
        gids     = extract_ids(guidance)
        all_ids.update(gids)
        blocks.append(f"GUIDANCE SECTION:\n{guidance}\n")

        # 3) SOURCE DOCUMENTS
        srcs = []
        for sid in sorted(all_ids, key=int):
            if sid in sources:
                doc  = sources[sid]
                meta = doc.get("metadata", {})
                body = doc.get("text", "")
                srcs.append(
                    f"[{sid}] {meta.get('title','')}  \n"
                    f"Source: {meta.get('source','unknown')}  \n"
                    f"URL: {meta.get('url','n/a')}\n\n"
                    f"{body}\n"
                )

        source_section = "\n\n".join(srcs) or "NO SOURCE DOCUMENTS FOUND"

        return (
            "\n\n".join(blocks)
            + "\n\nSOURCE DOCUMENTS:\n" + source_section
            + "\n\nPlease respond via the `evaluate_forward_guidance` function."
        )

    # --------------------------- evaluate ------------------------- #
    async def evaluate_document(self, doc: Dict[str, Any], corrected: bool = False) -> Dict[str, Any]:
        try:
            response_text = doc.get("agent_response_corrected") if corrected else doc.get("agent_response")
            if not response_text:
                log.error(f"No response text found for doc_id {doc.get('_id')} (corrected={corrected})")
                return {
                    "document_id": str(doc.get("_id")), "ticker": doc.get("user_question"), "timestamp": doc.get("timestamp"),
                    "pipeline": PIPELINE_TAG, "is_correction": corrected,
                    "evaluation": {"error": "Missing agent_response or agent_response_corrected"},
                    "evaluated_at": datetime.now(timezone.utc)
                }

            if not corrected:
                log.info("Original report text for doc_id %s (%s):\n%s", doc.get("_id"), doc.get("user_question", "N/A"), response_text)

            source_docs_data = doc.get("agent_sources", {})
            source_texts = []
            if isinstance(source_docs_data, dict):
                for sid, source_content in source_docs_data.items():
                    if isinstance(source_content, dict):
                        meta = source_content.get("metadata", {})
                        body = source_content.get("text", "")
                        source_texts.append(
                            f"[{sid}] {meta.get('title','')}\n"
                            f"Source: {meta.get('source','unknown')}\n"
                            f"URL: {meta.get('url','n/a')}\n\n"
                            f"{body}\n"
                        )
                    else:
                        log.warning(f"Source content for sid {sid} in doc {doc.get('_id')} is not a dict, skipping.")
            else:
                log.warning(f"agent_sources in doc {doc.get('_id')} is not a dict, skipping source formatting.")

            source_docs_formatted = "\n\n".join(source_texts) if source_texts else "NO SOURCE DOCUMENTS PROVIDED."

            user_prompt = f"""
Evaluate this Forward Guidance analysis report against the source documents:

ANALYSIS REPORT:
{response_text}

SOURCE DOCUMENTS:
{source_docs_formatted}

Evaluate each criterion as TRUE or FALSE with brief explanations.
Calculate a quality score from 0-100 based on these criteria.
List any statements not supported by the sources.
"""
            system_prompt = """You are an expert financial evaluator specializing in Forward Guidance analysis.
Assess the ANALYSIS REPORT against all provided SOURCE DOCUMENTS.
For 'accurate_numbers', a claim is true if supported by *any* provided source, even if its explicit citation is incorrect (note incorrect citations under 'correct_citations').
For 'correct_citations', verify that the cited source [N] matches the claim; if a claim is supported by source [M] but cited as [N], 'correct_citations' for that instance is False.
The "GUIDANCE SECTION" refers to the forward-looking "Predictive Estimates".
Be objective and thorough."""

            log.info("Evaluation prompt length for %s: %d chars", doc.get("user_question", "N/A"), len(user_prompt))

            client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
            response_llm = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={"type": "function", "function": {"name": "evaluate_forward_guidance"}}
            )

            tool_calls = response_llm.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                eval_result = json.loads(tool_calls[0].function.arguments)
                
                if (eval_result.get("hallucination_free") and
                    all(eval_result.get("factual_criteria", {}).values()) and
                    all(eval_result.get("completeness_criteria", {}).values()) and
                    all(eval_result.get("quality_criteria", {}).values()) and
                    not eval_result.get("unsupported_claims")):
                    log.info("All criteria passed in evaluation for %s. Setting quality_score to 100.", doc.get("user_question", "N/A"))
                    eval_result["quality_score"] = 100
                
                log.debug("Eval %s → %s", doc.get("_id"), textwrap.shorten(json.dumps(eval_result), 200))
            else:
                log.error("No function response received from LLM for doc_id %s", doc.get("_id"))
                eval_result = {"error": "No function response received"}

        except Exception as e:
            log.error(f"LLM eval failed for doc_id {doc.get('_id')}: {e}", exc_info=True)
            eval_result = {"error": f"eval_error: {str(e)}"}

        return {
            "document_id": str(doc.get("_id")),
            "ticker": doc.get("user_question"),
            "timestamp": doc.get("timestamp"),
            "pipeline": PIPELINE_TAG,
            "is_correction": corrected,
            "evaluation": eval_result,
            "evaluated_at": datetime.now(timezone.utc)
        }

    # ----------------------- range processing --------------------- #
    async def process_range(self, start: datetime, end: datetime) -> Dict[str, Any]:
        results = {
            "documents_evaluated": 0,
            "documents_failed": 0,
        }
        nyse = mcal.get_calendar("NYSE")
        days = set(d.date() for d in nyse.valid_days(start, end))
        cur  = start
        while cur < end:
            nxt = cur + timedelta(days=1)
            if cur.date() not in days:
                cur = nxt
                continue
            log.info("Evaluating %s", cur.date())
            docs_to_eval = await self.sample_docs(cur, nxt)

            async def _eval_and_store(d_eval):
                eval_output = await self.evaluate_document(d_eval)
                if "error" in eval_output.get("evaluation", {}):
                    results["documents_failed"] += 1
                results["documents_evaluated"] += 1
                await self.eval.insert_one(eval_output)

            for batch in [docs_to_eval[i:i+10] for i in range(0, len(docs_to_eval), 10)]:
                await tqdm_asyncio.gather(*[_eval_and_store(d_batch) for d_batch in batch])
            cur = nxt
        return results

    # ------------------- correction helpers ---------------------- #
    async def needs_fix(self, ev: Dict[str, Any]) -> bool:
        if "error" in ev.get("evaluation", {}):
            return True
        e = ev["evaluation"]
        return not (
            e.get("hallucination_free", False) and
            all(e.get("factual_criteria", {}).values()) and
            all(e.get("completeness_criteria", {}).values()) and
            all(e.get("quality_criteria", {}).values())
        )

    async def find_failing_evals(self, start: datetime, end: datetime, run_start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        match_criteria = {
            "pipeline": PIPELINE_TAG,
            "timestamp": {"$gte": start, "$lt": end},
            "with_corrections": {"$exists": False}
        }
        if run_start_time:
            match_criteria["evaluated_at"] = {"$gte": run_start_time}

        rows = await self.eval.find(match_criteria).to_list(length=None)
        return [ev_row for ev_row in rows if await self.needs_fix(ev_row)]

    async def correct_document(self, ev_doc: Dict[str, Any]) -> Dict[str, Any]:
        base_doc_id_obj = ObjectId(ev_doc["document_id"])
        base = await self.src.find_one({"_id": base_doc_id_obj})
        if not base:
            log.error(f"Original document not found for correction: {ev_doc['document_id']}")
            return {"error": "Original document not found"}

        original_response_text = base.get("agent_response")
        if not original_response_text:
            log.error(f"Original agent_response missing in doc_id {base_doc_id_obj}")
            return {"document_id": str(base_doc_id_obj), "ticker": base.get("user_question"), "error": "Original agent_response missing"}

        e = ev_doc["evaluation"]
        crit_expl = e.get("criteria_explanations", {})
        issues = []

        if "unsupported_claims" in e and e["unsupported_claims"]:
            issues.append("UNSUPPORTED CLAIMS:")
            for claim in e["unsupported_claims"]:
                issues.append(f"- {claim}")
        
        if not e.get("hallucination_free", True):
            issues.append(f"HALLUCINATION ISSUE: {crit_expl.get('hallucination_free', 'Content may not be fully supported by sources or contains invented details.')}")

        fc = e.get("factual_criteria", {})
        if not fc.get("accurate_numbers", True):
            issues.append(f"ACCURATE NUMBERS ISSUE: {crit_expl.get('accurate_numbers', 'Numerical data may not accurately match cited sources.')}")
        if not fc.get("correct_citations", True):
            issues.append(f"CORRECT CITATIONS ISSUE: {crit_expl.get('correct_citations', 'Bracketed citations may not correctly correspond to the content/location in SOURCE DOCUMENTS.')}")

        cc = e.get("completeness_criteria", {})
        if not cc.get("paragraphs_all_supported", True):
            issues.append(f"PARAGRAPHS SUPPORT ISSUE: {crit_expl.get('paragraphs_all_supported', 'Not every analysis paragraph is supported by at least one cited source.')}")
        if not cc.get("guidance_section_present", True):
            default_msg = 'The analysis may be missing a dedicated "GUIDANCE SECTION" (i.e., "Predictive Estimates").'
            issues.append(f"GUIDANCE SECTION ISSUE: {crit_expl.get('guidance_section_present', default_msg)}")

        qc = e.get("quality_criteria", {})
        if not qc.get("clear_structure", True):
            issues.append(f"CLEAR STRUCTURE ISSUE: {crit_expl.get('clear_structure', 'The report may not be logically organized with clear headings/bullets.')}")
        if not qc.get("guidance_section_consistent", True):
            issues.append(f"GUIDANCE CONSISTENCY ISSUE: {crit_expl.get('guidance_section_consistent', 'The guidance section may not align with the main analysis in data or tone.')}")
        
        if e.get("score_explanation"):
            issues.append(f"OVERALL SCORE RATIONALE FROM INITIAL EVALUATION: {e['score_explanation']}")
        
        if not issues:
            issues.append("General review needed for accuracy, completeness, and quality based on initial evaluation flagging.")
        
        source_docs_data_corr = base.get("agent_sources", {})
        source_texts_corr = []
        if isinstance(source_docs_data_corr, dict):
            for sid, source_content in source_docs_data_corr.items():
                if isinstance(source_content, dict):
                    meta = source_content.get("metadata", {})
                    body = source_content.get("text", "")
                    source_texts_corr.append(
                        f"[{sid}] {meta.get('title','')}\n"
                        f"Source: {meta.get('source','unknown')}\n"
                        f"URL: {meta.get('url','n/a')}\n\n"
                        f"{body}\n"
                    )
        source_docs_formatted_corr = "\n\n".join(source_texts_corr) if source_texts_corr else "NO SOURCE DOCUMENTS PROVIDED."
        
        system_prompt_corr = """You are an expert financial editor. Your primary goal is to meticulously correct factual errors, citation issues, and completeness gaps in the provided Forward Guidance analysis report, based *only* on the "ISSUES TO FIX" list.

Key Principles for Correction:
1.  **Minimal Changes**: Make ONLY the changes necessary to address the specific issues listed. Do NOT rephrase, restructure, or alter any part of the report that is already correct and does not pertain to a listed issue.
2.  **Preserve Correct Content**: If a statement or section is factually accurate, correctly cited (if applicable), and complete according to the original evaluation, it MUST be preserved exactly as is.
3.  **Maintain Original Formatting**: The corrected report MUST retain the exact same bullet-point structure, paragraphing, and overall formatting as the ORIGINAL REPORT. Only the textual content of problematic bullets should change. The "GUIDANCE SECTION" (Predictive Estimates) should remain at the end.
4.  **Focus on "ISSUES TO FIX"**: Your corrections should solely target the problems detailed in the "ISSUES TO FIX" section of the user prompt.

Your task is to return a corrected version of the report that is factually sound, properly cited, complete, and maintains the integrity and style of the original where it was already correct."""

        user_prompt_corr = f"""
Please correct the following Forward Guidance analysis report that contains issues:

ORIGINAL REPORT:
{original_response_text}

SOURCE DOCUMENTS:
{source_docs_formatted_corr}

ISSUES TO FIX:
{chr(10).join(f"- {issue}" for issue in issues)}

INSTRUCTIONS:
1. Fix all identified issues while preserving accurate information from the ORIGINAL REPORT.
2. Ensure all numbers precisely match supporting data within the SOURCE DOCUMENTS.
3. Review all claims and their citations:
    - If a claim is cited but the citation is incorrect (i.e., the cited source [N] does not support the claim):
        - First, check if another document within the provided SOURCE DOCUMENTS *does* support the claim.
        - If yes, update the claim to cite the correct source [M] and ensure the claim accurately reflects source [M].
        - If no other provided source supports the claim, the claim must be revised to be supportable by one of the provided sources, or removed if unsupportable.
    - If a claim is made without a citation but requires one (e.g., specific data):
        - Find support for it in the SOURCE DOCUMENTS and add the correct citation.
        - If no support is found, revise the claim to be supportable or remove it.
    - Ensure all final citations accurately point to the content in the SOURCE DOCUMENTS that supports them.
4. Maintain clear bullet-point structure and overall report formatting.
5. Ensure the "GUIDANCE SECTION" (Predictive Estimates) is present, well-supported, and consistent.
6. Keep the corrected report's length and tone similar to the original.
7. Return ONLY the corrected report text.

CORRECTED REPORT:"""

        log.info("Correction prompt length for %s: %d chars", base.get("user_question", "N/A"), len(user_prompt_corr))
        client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
        try:
            rsp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt_corr},
                    {"role": "user",   "content": user_prompt_corr}
                ]
            )
            corrected_text_val = rsp.choices[0].message.content.strip() if rsp.choices[0].message.content else ""
            log.info("Doc %s corrected by LLM. Length: %d", base.get("_id"), len(corrected_text_val))
            return {"document_id": str(base.get("_id")), "ticker": base.get("user_question"), "corrected": True, "corrected_text": corrected_text_val}
        except Exception as err:
            log.error(f"LLM correction failed for {base.get('_id')}: {err}", exc_info=True)
            return {"document_id": str(base.get("_id")), "ticker": base.get("user_question"), "error": f"LLM correction failed: {str(err)}"}

    async def process_corrections(self, start: datetime, end: datetime, run_start_time: Optional[datetime] = None):
        fails = await self.find_failing_evals(start, end, run_start_time=run_start_time)
        log.info("%d docs need correction in fwd-guid pipeline", len(fails))
        if not fails:
            return {"corrections_processed": 0, "corrections_succeeded": 0, "corrections_failed": 0, "improvements_summary": {}}

        async def _fix_and_re_evaluate(eval_doc: Dict[str, Any]):
            try:
                original_eval = eval_doc.get("evaluation", {})
                ticker = eval_doc.get("ticker", "UnknownTicker")
                log.info("\nProcessing correction for %s (fwd-guid):", ticker)
                log.info("Original scores (fwd-guid):")
                log.info("- Quality score: %d", original_eval.get("quality_score", 0))

                crit_expl_orig = original_eval.get("criteria_explanations", {})
                log.info("Detailed issues from original evaluation (fwd-guid):")
                if not original_eval.get("hallucination_free", True): log.info("  - Hallucination: %s", crit_expl_orig.get("hallucination_free", "N/A"))
                fc_orig = original_eval.get("factual_criteria", {})
                if not fc_orig.get("accurate_numbers", True): log.info("  - Inaccurate Numbers: %s", crit_expl_orig.get("accurate_numbers", "N/A"))
                if not fc_orig.get("correct_citations", True): log.info("  - Incorrect Citations: %s", crit_expl_orig.get("correct_citations", "N/A"))
                cc_orig = original_eval.get("completeness_criteria", {})
                if not cc_orig.get("paragraphs_all_supported", True): log.info("  - Paragraphs Not Supported: %s", crit_expl_orig.get("paragraphs_all_supported", "N/A"))
                if not cc_orig.get("guidance_section_present", True): log.info("  - Guidance Section Missing: %s", crit_expl_orig.get("guidance_section_present", "N/A"))
                qc_orig = original_eval.get("quality_criteria", {})
                if not qc_orig.get("clear_structure", True): log.info("  - Unclear Structure: %s", crit_expl_orig.get("clear_structure", "N/A"))
                if not qc_orig.get("guidance_section_consistent", True): log.info("  - Guidance Inconsistent: %s", crit_expl_orig.get("guidance_section_consistent", "N/A"))
                unsupported_claims_list_orig = original_eval.get("unsupported_claims", [])
                if unsupported_claims_list_orig:
                    log.info("  - Unsupported Claims:")
                    for claim in unsupported_claims_list_orig: log.info("    * %s", claim)
                log.info("  - Original Score Explanation: %s", original_eval.get("score_explanation", "N/A"))
                log.info("- Factual accuracy (orig): %s", all(fc_orig.values()))
                log.info("- Completeness (orig): %s", all(cc_orig.values()))
                log.info("- Quality criteria (orig): %s", all(qc_orig.values()))
                log.info("- Hallucination free (orig): %s", original_eval.get("hallucination_free", False))

                correction_output = await self.correct_document(eval_doc)
                if "error" in correction_output or not correction_output.get("corrected"):
                    log.error(f"Correction generation failed for {ticker}: {correction_output.get('error', 'Not corrected')}")
                    return {"status": "failed", "reason": "correction_generation_failed", "document_id": eval_doc.get("document_id")}

                original_doc_for_re_eval = await self.src.find_one({"_id": ObjectId(correction_output["document_id"])})
                if not original_doc_for_re_eval:
                     log.error(f"Original document not found for re-evaluation: {correction_output['document_id']}")
                     return {"status": "failed", "reason": "original_doc_missing_for_re_eval", "document_id": correction_output.get("document_id")}

                temp_doc_for_re_eval = {**original_doc_for_re_eval, "agent_response_corrected": correction_output["corrected_text"]}
                re_evaluation_result = await self.evaluate_document(temp_doc_for_re_eval, corrected=True)
                new_eval_data = re_evaluation_result.get("evaluation", {})

                if "error" in new_eval_data:
                    log.error(f"Re-evaluation failed for {ticker}: {new_eval_data['error']}")
                    return {"status": "failed", "reason": f"re_evaluation_error: {new_eval_data['error']}", "document_id": eval_doc.get("document_id")}

                if (new_eval_data.get("hallucination_free", False) and
                    all(new_eval_data.get("factual_criteria", {}).values()) and
                    all(new_eval_data.get("completeness_criteria", {}).values()) and
                    all(new_eval_data.get("quality_criteria", {}).values())):
                    log.info("All criteria passed in corrected version for %s. Setting quality_score to 100.", ticker)
                    new_eval_data["quality_score"] = 100
                
                log.info("\nCorrected scores for %s (fwd-guid):", ticker)
                log.info("- Quality score: %d", new_eval_data.get("quality_score", 0))
                log.info("- Factual accuracy: %s", all(new_eval_data.get("factual_criteria",{}).values()))
                log.info("- Completeness: %s", all(new_eval_data.get("completeness_criteria",{}).values()))
                log.info("- Quality criteria: %s", all(new_eval_data.get("quality_criteria",{}).values()))
                log.info("- Hallucination free: %s", new_eval_data.get("hallucination_free", False))

                original_quality_score = original_eval.get("quality_score", 0)
                new_quality_score = new_eval_data.get("quality_score", 0)
                score_improved_flag = new_quality_score > original_quality_score
                
                factual_improved_flag = (all(new_eval_data.get("factual_criteria", {}).values()) and 
                                    not all(original_eval.get("factual_criteria", {}).values()))
                completeness_improved_flag = (all(new_eval_data.get("completeness_criteria", {}).values()) and
                                        not all(original_eval.get("completeness_criteria", {}).values()))
                quality_improved_flag = (all(new_eval_data.get("quality_criteria", {}).values()) and
                                    not all(original_eval.get("quality_criteria", {}).values()))
                hallucination_improved_flag = (new_eval_data.get("hallucination_free", False) and
                                        not original_eval.get("hallucination_free", False))
                any_improvement_flag = (score_improved_flag or factual_improved_flag or completeness_improved_flag or quality_improved_flag or hallucination_improved_flag)

                if any_improvement_flag:
                    log.info("\nImprovements detected for %s (fwd-guid):", ticker)
                    if score_improved_flag: log.info("- Quality score improved: %d → %d", original_quality_score, new_quality_score)
                    if factual_improved_flag: log.info("- Factual accuracy fixed")
                    if completeness_improved_flag: log.info("- Completeness improved")
                    if quality_improved_flag: log.info("- Quality criteria improved")
                    if hallucination_improved_flag: log.info("- Hallucinations removed")

                    await self.eval.update_one(
                        {"_id": ObjectId(eval_doc["_id"])},
                        {"$set": {
                            "with_corrections": True,
                            "corrected_evaluation": new_eval_data,
                            "corrected_evaluated_at": re_evaluation_result["evaluated_at"],
                            "corrected_text": correction_output["corrected_text"],
                            "improvement_metrics": {
                                "quality_score_delta": new_quality_score - original_quality_score,
                                "factual_improved": factual_improved_flag,
                                "completeness_improved": completeness_improved_flag,
                                "quality_improved": quality_improved_flag,
                                "hallucination_improved": hallucination_improved_flag
                            }
                        }}
                    )
                    log.info("Stored verified correction for %s (fwd-guid) in evaluations collection.", ticker)
                    return {"status": "success", "document_id": correction_output.get("document_id"), "improvements_made": True}
                else:
                    log.warning("No improvements detected in correction for %s (fwd-guid), skipping update.", ticker)
                    return {"status": "no_improvement", "document_id": correction_output.get("document_id"), "improvements_made": False}
            except Exception as e_outer_fix:
                log.error(f"Outer error in _fix_and_re_evaluate for {eval_doc.get('ticker', 'UnknownTickerOnError')} (fwd-guid): {e_outer_fix}", exc_info=True)
                return {"status": "exception", "reason": str(e_outer_fix), "document_id": eval_doc.get("document_id")}

        results_list = []
        for i in range(0, len(fails), 6):
            batch_items = fails[i:i+6]
            log.info("Processing fwd-guid correction batch %d/%d (%d documents)", i//6 + 1, (len(fails)+5)//6, len(batch_items))
            batch_processing_results = await tqdm_asyncio.gather(*[_fix_and_re_evaluate(ev_item) for ev_item in batch_items], desc=f"Fwd-Guid Correction Batch {i//6 + 1}")
            results_list.extend(batch_processing_results)
            if i + 6 < len(fails):
                log.info("Pausing between fwd-guid correction batches...")
                await asyncio.sleep(2)
        
        succeeded_count = sum(1 for r in results_list if r and r.get("status") == "success")
        score_improved_count = sum(1 for r in results_list if r and r.get("status") == "success" and r.get("improvement_metrics", {}).get("quality_score_delta", 0) > 0)
        factual_fixed_count = sum(1 for r in results_list if r and r.get("status") == "success" and r.get("improvement_metrics", {}).get("factual_improved"))

        return {
            "corrections_processed": len(fails),
            "corrections_succeeded": succeeded_count,
            "corrections_failed": len(fails) - succeeded_count,
            "improvements_summary": {
                "score_improved_count": score_improved_count,
                "factual_fixed_count": factual_fixed_count,
            },
            "note": "Forward Guidance corrections stored in evaluations collection."
        }

# ------------------------------------------------------------------ #
async def main():
    run_start_time = datetime.now(timezone.utc)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=1)

    evaluator = ForwardGuidanceEvaluator(MONGO_URI)
    log.info("=== Evaluation pass (Forward Guidance) ===")
    eval_results = await evaluator.process_range(start, end)
    log.info("Evaluation results (Forward Guidance): %s", eval_results)
    
    log.info("=== Correction pass (Forward Guidance) ===")
    correction_summary = await evaluator.process_corrections(start, end, run_start_time=run_start_time)
    log.info("Correction results (Forward Guidance): %s", correction_summary)

if __name__ == "__main__":
    asyncio.run(main())

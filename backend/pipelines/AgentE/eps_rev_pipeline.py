"""
epsrev_pipeline.py  –  Agent-E EPS/Revenue Analysis Evaluation + Correction
"""

# --------------------------------------------------------------------- #
# Imports & logging setup
# --------------------------------------------------------------------- #
import os
import re
import json
import random
import asyncio
import logging
import textwrap
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pathlib import Path

from ..helpers import (
    get_mongo_client,
    get_openai_client,
    setup_logger,
    aggregate_evaluation_stats,
    calculate_final_pass_rates
)

# Load .env file
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pandas_market_calendars as mcal

# ----------  logging  ---------- #
log = setup_logger(
    "EPSREV",
    level=logging.DEBUG,
    fmt="%(asctime)s [%(levelname)s] %(message)s", # Match original format
    datefmt="%H:%M:%S", # Match original datefmt
    noisy_loggers_to_warn=["motor", "pymongo", "httpcore", "httpx", "openai", "asyncio", "anyio"]
)

# --------------------------------------------------------------------- #
# ENV / CONFIG
# --------------------------------------------------------------------- #
# OPENAI_API_KEYS = [
#     k for k in (
#         os.getenv("OPENAI_API_KEY"),
#         os.getenv("OPENAI_API_KEY_BACKUP1"),
#         os.getenv("OPENAI_API_KEY_BACKUP2"),
#     ) if k
# ]
MONGO_URI      = os.getenv("EVAL_MONGO_URI", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "o4-mini")
DAILY_SAMPLE_SIZE = int(os.getenv("DAILY_SAMPLE_SIZE", 1))

EVAL_COLL_NAME = "evaluations"
PIPELINE_TAG   = "EPS-REV"

# --------------------------------------------------------------------- #
# Function-calling schema
# --------------------------------------------------------------------- #
evaluation_schema = {
    "name": "evaluate_eps_rev",
    "description": "Evaluate an EPS/Revenue analysis report for accuracy, completeness and quality",
    "parameters": {
        "type": "object",
        "properties": {
            "factual_criteria": {
                "type": "object",
                "properties": {
                    "accurate_numbers": {
                        "type": "boolean",
                        "description": "True if all numerical data points (EPS, revenue figures, percentages) in each paragraph accurately match the cited source documents."
                    },
                    "correct_citations": {
                        "type": "boolean",
                        "description": "True if each bracketed citation [1], [2], etc. correctly corresponds to the content and location in the SOURCE DOCUMENTS section."
                    }
                },
                "required": ["accurate_numbers", "correct_citations"]
            },
            "completeness_criteria": {
                "type": "object",
                "properties": {
                    "paragraphs_all_supported": {
                        "type": "boolean",
                        "description": "True if every analysis paragraph is supported by at least one cited source; no paragraph is left unverified."
                    },
                    "predictive_section_present": {
                        "type": "boolean",
                        "description": "True if the analysis includes a dedicated '**Predictive Estimates**' section for forward-looking metrics."
                    }
                },
                "required": ["paragraphs_all_supported", "predictive_section_present"]
            },
            "quality_criteria": {
                "type": "object",
                "properties": {
                    "clear_structure": {
                        "type": "boolean",
                        "description": "True if the report is logically organized with clear headings and paragraph structure."
                    },
                    "predictive_section_consistent": {
                        "type": "boolean",
                        "description": "True if the predictive estimates section aligns with the main analysis in both data and tone."
                    }
                },
                "required": ["clear_structure", "predictive_section_consistent"]
            },
            "hallucination_free": {
                "type": "boolean",
                "description": "False if there are any details unsupported by the SOURCE DOCUMENTS. Note that common market knowledge, straightforward inferences, and minor editorial liberties are allowed and thus do not count as hallucination. Mark True if the response is a natural extension of the sources."
            },
            "quality_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall quality score (0–100), reflecting factual accuracy, completeness, structure, and absence of hallucinations."
            },
            "criteria_explanations": {
                "type": "object",
                "description": "Detailed explanations for each criterion judgment.",
                "properties": {
                    "accurate_numbers": {"type": "string"},
                    "correct_citations": {"type": "string"},
                    "paragraphs_all_supported": {"type": "string"},
                    "predictive_section_present": {"type": "string"},
                    "clear_structure": {"type": "string"},
                    "predictive_section_consistent": {"type": "string"},
                    "hallucination_free": {"type": "string"}
                }
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of any statements or numbers not supported by the source documents."
            },
            "score_explanation": {
                "type": "string",
                "description": "Overall rationale for the assigned quality score."
            }
        },
        "required": [
            "factual_criteria", "completeness_criteria", "quality_criteria",
            "hallucination_free", "quality_score", "criteria_explanations",
            "unsupported_claims", "score_explanation"
        ]
    }
}

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
para_re = re.compile(r"^- +(?:\*\*)?", re.M)
cid_re  = re.compile(r"\[(\d+)\]")

def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if para_re.match(p.strip())]

def extract_ids(para: str) -> List[str]:
    return cid_re.findall(para)

# --------------------------------------------------------------------- #
class EPSRevEvaluator:
    def __init__(self, mongo_uri: str):
        if not mongo_uri:
            raise ValueError("EVAL_MONGO_URI env var not set")
        # cli = AsyncIOMotorClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        # db  = cli["asc-fin-data"]
        client = get_mongo_client(mongo_uri)
        db = client["asc-fin-data"]
        self.src  = db["user_activities"]
        self.eval = db[EVAL_COLL_NAME]

    async def sample_docs(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Sample documents from a specific date range."""
        pipeline = [
            {
                "$match": {
                    "agent": "earnings",
                    "mode": "eps_revenue",
                    "timestamp": {"$gte": start, "$lt": end},
                    "agent_sources": {"$exists": True}
                }
            }
        ]
        
        all_docs = await self.src.aggregate(pipeline).to_list(length=None)
        sample = random.sample(all_docs, min(len(all_docs), DAILY_SAMPLE_SIZE))
        log.info("Sampled %d docs: %s",
                 len(sample),
                 ", ".join(d.get('user_question', '') for d in sample))
        return sample

    async def evaluate_document(self, doc: Dict[str, Any], corrected: bool = False) -> Dict[str, Any]:
        """Evaluate a document using the evaluation schema."""
        try:
            # Get the appropriate response text based on whether we're evaluating original or corrected
            response_text = doc.get("agent_response_corrected") if corrected else doc["agent_response"]
            
            if not corrected:
                log.info("Original report text for doc_id %s (%s):\n%s", doc.get("_id"), doc.get("user_question", "N/A"), response_text)
            
            # Extract all citation numbers from the response
            citations = set(re.findall(r'\[(\d+)\]', response_text))
            log.info("Citations found in response: %s", citations)
            
            # Format source documents and verify citations
            source_docs = doc.get("agent_sources", {})
            source_texts = []
            available_sources = set(source_docs.keys())
            log.info("Available sources: %s", available_sources)
            
            # Check for missing citations
            missing_citations = citations - available_sources
            if missing_citations:
                log.warning("Citations referenced but not found in sources: %s", missing_citations)
            
            # Check for unused sources
            unused_sources = available_sources - citations
            if unused_sources:
                log.info("Sources available but not cited: %s", unused_sources)
            
            # Log character counts for cited sources
            log.info("Character counts for cited sources:")
            for citation in sorted(citations):
                if citation in source_docs:
                    source = source_docs[citation]
                    text = source.get("text", "")
                    char_count = len(text)
                    title = source.get("metadata", {}).get("title", "No title")
                    log.info("Source [%s] - %s chars - Title: %s", citation, char_count, title)
                    if char_count < 10:  # Arbitrary small number to catch obviously truncated sources
                        log.warning("Source [%s] appears truncated with only %d characters", citation, char_count)
            
            # Format sources and verify text field
            for sid, source in source_docs.items():
                if "text" not in source:
                    log.warning("Source [%s] missing 'text' field", sid)
                    continue
                    
                meta = source.get("metadata", {})
                body = source.get("text", "")
                source_texts.append(
                    f"[{sid}] {meta.get('title','')}\n"
                    f"Source: {meta.get('source','unknown')}\n"
                    f"URL: {meta.get('url','n/a')}\n\n"
                    f"{body}\n"
                )
            
            source_docs_formatted = "\n\n".join(source_texts)
            
            # Create evaluation prompt
            user_prompt = f"""
Evaluate this EPS/Revenue analysis report against the source documents:

ANALYSIS REPORT:
{response_text}

SOURCE DOCUMENTS:
{source_docs_formatted}

Evaluate each criterion as TRUE or FALSE with brief explanations.
Calculate a quality score from 0-100 based on these criteria.
List any statements not supported by the sources.
"""

            system_prompt = """You are an expert financial evaluator specializing in EPS/Revenue analysis. 
Assess the ANALYSIS REPORT against all provided SOURCE DOCUMENTS. 
For 'accurate_numbers', a claim is true if supported by *any* provided source, even if its explicit citation is incorrect (note incorrect citations under 'correct_citations'). 
For 'correct_citations', verify that the cited source [N] matches the claim; if a claim is supported by source [M] but cited as [N], 'correct_citations' for that instance is False. 
Be objective and thorough."""

            log.info("Evaluation prompt length for %s: %d chars", doc.get("user_question", "N/A"), len(user_prompt))
            # Make LLM call
            # client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
            client = await get_openai_client()
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={"type": "function", "function": {"name": "evaluate_eps_rev"}}
            )
            
            # Extract function response
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                function_response = json.loads(tool_calls[0].function.arguments)
                
                return {
                    "document_id": str(doc["_id"]),
                    "ticker": doc["user_question"],
                    "timestamp": doc["timestamp"],
                    "pipeline": PIPELINE_TAG,
                    "evaluation": function_response,
                    "evaluated_at": datetime.now(timezone.utc)
                }
            else:
                return {
                    "document_id": str(doc["_id"]),
                    "ticker": doc["user_question"],
                    "timestamp": doc["timestamp"],
                    "pipeline": PIPELINE_TAG,
                    "evaluation": {"error": "No function response received"},
                    "evaluated_at": datetime.now(timezone.utc)
                }
                
        except Exception as e:
            log.error("Evaluation failed: %s", e)
            return {
                "document_id": str(doc["_id"]),
                "ticker": doc["user_question"],
                "timestamp": doc["timestamp"],
                "pipeline": PIPELINE_TAG,
                "evaluation": {"error": f"Evaluation failed: {str(e)}"},
                "evaluated_at": datetime.now(timezone.utc)
            }

    async def process_range(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Process all documents within a date range."""
        results = {
            # "factual_accuracy_pass_rate": 0, # Will be added by helper as factual_pass_rate
            # "completeness_pass_rate": 0, # Will be added by helper
            # "quality_usefulness_pass_rate": 0, # Will be added by helper as quality_pass_rate
            # "hallucination_free_rate": 0, # Will be added by helper
            "quality_scores": [],
            "documents_evaluated": 0,
            "documents_failed": 0,
            "days_skipped": 0,
            
            "factual_true_count": 0,
            "factual_total_count": 0,
            "completeness_true_count": 0,
            "completeness_total_count": 0,
            "quality_true_count": 0,
            "quality_total_count": 0,
            "hallucination_free_count": 0,
            "hallucination_total_count": 0
            # "avg_quality_score": 0 # Will be added by helper
        }
        
        criteria_mapping = {
            "factual": ["factual_criteria.accurate_numbers", "factual_criteria.correct_citations"],
            "completeness": ["completeness_criteria.paragraphs_all_supported", "completeness_criteria.predictive_section_present"],
            "quality": ["quality_criteria.clear_structure", "quality_criteria.predictive_section_consistent"]
        }

        # Get NYSE calendar for market day checks
        nyse = mcal.get_calendar('NYSE')
        days = set(d.date() for d in nyse.valid_days(start, end))
        cur  = start
        
        while cur < end:
            nxt = cur + timedelta(days=1)
            if cur.date() not in days:
                log.info("Skipping %s - not a market day", cur.date())
                results["days_skipped"] += 1
                cur = nxt
                continue
            
            log.info("Processing market day: %s", cur.date())
            docs = await self.sample_docs(cur, nxt)
            log.info("Sampled %d documents", len(docs))
            
            async def process_doc(doc):
                try:
                    evaluation = await self.evaluate_document(doc)
                    await self.eval.insert_one(evaluation)
                    return evaluation
                except Exception as e:
                    log.error("Error processing document %s: %s", doc.get("_id", "unknown"), e)
                    return {"error": str(e)}
            
            # Process documents in batches of 10
            all_evaluations = []
            for i in range(0, len(docs), 10):
                batch = docs[i:i+10]
                log.info("Processing batch %d/%d (%d documents)",
                         i//10 + 1, (len(docs)+9)//10, len(batch))
                
                batch_evaluations = await tqdm_asyncio.gather(
                    *[process_doc(doc) for doc in batch],
                    desc=f"Batch {i//10 + 1}"
                )
                
                all_evaluations.extend(batch_evaluations)
                
                if i + 10 < len(docs):
                    log.info("Pausing between batches...")
                    await asyncio.sleep(2)
            
            # Process results
            for evaluation in all_evaluations:
                if "error" in evaluation or not evaluation.get("evaluation"):
                    results["documents_failed"] = results.get("documents_failed", 0) + 1
                    continue
                    
                eval_data = evaluation["evaluation"] # evaluation.get("evaluation", {})
                # if not eval_data: # Handled by check above and by helper
                #     results["documents_failed"] += 1
                #     continue
                
                aggregate_evaluation_stats(eval_data, results, criteria_mapping)

                # # Count factual criteria
                # factual_criteria = eval_data.get("factual_criteria", {})
                # factual_true = sum(1 for _, v in factual_criteria.items() if v)
                # factual_total = len(factual_criteria)
                # 
                # # Count completeness criteria
                # completeness_criteria = eval_data.get("completeness_criteria", {})
                # completeness_true = sum(1 for _, v in completeness_criteria.items() if v)
                # completeness_total = len(completeness_criteria)
                # 
                # # Count quality criteria
                # quality_criteria = eval_data.get("quality_criteria", {})
                # quality_true = sum(1 for _, v in quality_criteria.items() if v)
                # quality_total = len(quality_criteria)
                # 
                # # Track quality score
                # if "quality_score" in eval_data:
                #     results["quality_scores"].append(eval_data["quality_score"])
                # 
                # # Track hallucination
                # if eval_data.get("hallucination_free", False):
                #     results["hallucination_free_count"] += 1
                # 
                # # Add to totals
                # results["factual_true_count"] += factual_true
                # results["factual_total_count"] += factual_total
                # results["completeness_true_count"] += completeness_true
                # results["completeness_total_count"] += completeness_total
                # results["quality_true_count"] += quality_true
                # results["quality_total_count"] += quality_total
                # results["hallucination_total_count"] += 1
                # results["documents_evaluated"] += 1 # Handled by aggregate_evaluation_stats
            
            cur = nxt
        
        # Calculate final rates
        # if results["documents_evaluated"] > 0:
        #     results["factual_accuracy_pass_rate"] = (results["factual_true_count"] / results["factual_total_count"]) * 100 if results["factual_total_count"] > 0 else 0
        #     results["completeness_pass_rate"] = (results["completeness_true_count"] / results["completeness_total_count"]) * 100 if results["completeness_total_count"] > 0 else 0
        #     results["quality_usefulness_pass_rate"] = (results["quality_true_count"] / results["quality_total_count"]) * 100 if results["quality_total_count"] > 0 else 0
        #     results["hallucination_free_rate"] = (results["hallucination_free_count"] / results["hallucination_total_count"]) * 100 if results["hallucination_total_count"] > 0 else 0
        #     results["avg_quality_score"] = sum(results["quality_scores"]) / len(results["quality_scores"]) if results["quality_scores"] else 0
        categories_for_pass_rates = ["factual", "completeness", "quality"]
        calculate_final_pass_rates(results, categories_for_pass_rates)
        
        return results

    # --------------------------------------------------------------------- #
    # Correction Methods
    # --------------------------------------------------------------------- #
    
    async def needs_fix(self, ev: Dict[str, Any]) -> bool:
        """Determine if a document needs correction based on evaluation results."""
        if "error" in ev["evaluation"]:
            return True
        e = ev["evaluation"]
        return not (
            e.get("hallucination_free", False)
            and all(e["factual_criteria"].values())
            and all(e["completeness_criteria"].values())
            and all(e["quality_criteria"].values())
        )

    async def find_failing_evals(self, start: datetime, end: datetime, run_start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Find evaluations that need correction in the specified date range."""
        match_criteria = {
            "pipeline": PIPELINE_TAG,
            "timestamp": {"$gte": start, "$lt": end}, # This is the original document's timestamp
            "with_corrections": {"$exists": False}
        }
        if run_start_time:
            match_criteria["evaluated_at"] = {"$gte": run_start_time}

        pipeline = [
            {
                "$match": match_criteria
            }
        ]
        
        rows = await self.eval.aggregate(pipeline).to_list(length=None)
        return [ev for ev in rows if await self.needs_fix(ev)]

    async def correct_document(self, ev_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a corrected version of the document based on evaluation feedback."""
        # Get original document
        base = await self.src.find_one({"_id": ObjectId(ev_doc["document_id"])})
        if not base:
            return {"error": "Original document not found"}

        # Collect issues from evaluation
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
            issues.append(f"ACCURATE NUMBERS ISSUE: {crit_expl.get('accurate_numbers', 'Numerical data (EPS, revenue, percentages) may not accurately match cited sources.')}")
        if not fc.get("correct_citations", True):
            issues.append(f"CORRECT CITATIONS ISSUE: {crit_expl.get('correct_citations', 'Bracketed citations [1], [2], etc. may not correctly correspond to the content/location in SOURCE DOCUMENTS.')}")

        cc = e.get("completeness_criteria", {})
        if not cc.get("paragraphs_all_supported", True):
            issues.append(f"PARAGRAPHS SUPPORT ISSUE: {crit_expl.get('paragraphs_all_supported', 'Not every analysis paragraph is supported by at least one cited source.')}")
        if not cc.get("predictive_section_present", True):
            default_message = 'The analysis may be missing a dedicated "**Predictive Estimates**" section for forward-looking metrics.'
            issues.append(f"PREDICTIVE SECTION ISSUE: {crit_expl.get('predictive_section_present', default_message)}")

        qc = e.get("quality_criteria", {})
        if not qc.get("clear_structure", True):
            issues.append(f"CLEAR STRUCTURE ISSUE: {crit_expl.get('clear_structure', 'The report may not be logically organized with clear headings and paragraph structure.')}")
        if not qc.get("predictive_section_consistent", True):
            issues.append(f"PREDICTIVE CONSISTENCY ISSUE: {crit_expl.get('predictive_section_consistent', 'The predictive estimates section may not align with the main analysis in data or tone.')}")
        
        if e.get("score_explanation"):
            issues.append(f"OVERALL SCORE RATIONALE FROM INITIAL EVALUATION: {e['score_explanation']}")
        
        if not issues: # Fallback if no specific issues were parsed but correction is triggered
            issues.append("General review needed for accuracy, completeness, and quality based on initial evaluation flagging.")

        # Format source documents
        source_docs = base.get("agent_sources", {})
        source_texts = []
        for sid, source in source_docs.items():
            meta = source.get("metadata", {})
            body = source.get("text", "")
            source_texts.append(
                f"[{sid}] {meta.get('title','')}\n"
                f"Source: {meta.get('source','unknown')}\n"
                f"URL: {meta.get('url','n/a')}\n\n"
                f"{body}\n"
            )
        
        source_docs_formatted = "\n\n".join(source_texts)

        # Create correction prompt
        system_prompt = """You are an expert financial editor. Your primary goal is to meticulously correct factual errors, citation issues, and completeness gaps in the provided EPS/Revenue analysis report, based *only* on the "ISSUES TO FIX" list.

Key Principles for Correction:
1.  **Minimal Changes**: Make ONLY the changes necessary to address the specific issues listed. Do NOT rephrase, restructure, or alter any part of the report that is already correct and does not pertain to a listed issue.
2.  **Preserve Correct Content**: If a statement or section is factually accurate, correctly cited (if applicable), and complete according to the original evaluation, it MUST be preserved exactly as is.
3.  **Maintain Original Formatting**: The corrected report MUST retain the exact same bullet-point structure, paragraphing, and overall formatting as the ORIGINAL REPORT. Only the textual content of problematic bullets should change. Do not add, remove, or merge bullet points unless an issue explicitly requires it (e.g., a missing predictive section).
4.  **Focus on "ISSUES TO FIX"**: Your corrections should solely target the problems detailed in the "ISSUES TO FIX" section of the user prompt.

Your task is to return a corrected version of the report that is factually sound, properly cited, complete, and maintains the integrity and style of the original where it was already correct."""

        user_prompt = f"""
Please correct the following EPS/Revenue analysis report that contains issues:

ORIGINAL REPORT:
{base["agent_response"]}

SOURCE DOCUMENTS:
{source_docs_formatted}

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
4. Maintain clear bullet-point structure.
5. Ensure the '**Predictive Estimates**' section is present, well-supported by the overall analysis, and consistent in tone.
6. Keep the corrected report's length and tone similar to the original.
7. Return ONLY the corrected report text.

CORRECTED REPORT:"""

        try:
            log.info("Correction prompt length for %s: %d chars", base["user_question"], len(user_prompt))
            # Make LLM call for correction
            # client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
            client = await get_openai_client()
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            corrected_text = response.choices[0].message.content.strip()
            
            return {
                "document_id": str(base["_id"]),
                "ticker": base["user_question"],
                "corrected": True,
                "corrected_text": corrected_text
            }
            
        except Exception as e:
            log.error("Correction failed: %s", e)
            return {"error": f"Correction failed: {str(e)}"}

    async def process_corrections(self, start: datetime, end: datetime, run_start_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Process corrections for documents that failed evaluation."""
        # Find documents needing correction
        fails = await self.find_failing_evals(start, end, run_start_time=run_start_time)
        log.info("%d documents need correction", len(fails))
        
        if not fails:
            return {"corrections_processed": 0}
        
        async def process_correction(eval_doc):
            try:
                original_eval = eval_doc["evaluation"]
                log.info("\nProcessing correction for %s:", eval_doc["ticker"])
                log.info("Original scores:")
                log.info("- Quality score: %d", original_eval.get("quality_score", 0))

                # Log detailed reasons for failure from original evaluation
                log.info("Detailed issues from original evaluation:")
                crit_expl = original_eval.get("criteria_explanations", {})
                if not original_eval.get("hallucination_free", True):
                    log.info("  - Hallucination: %s", crit_expl.get("hallucination_free", "No specific explanation."))
                if not original_eval.get("factual_criteria", {}).get("accurate_numbers", True):
                    log.info("  - Inaccurate Numbers: %s", crit_expl.get("accurate_numbers", "No specific explanation."))
                if not original_eval.get("factual_criteria", {}).get("correct_citations", True):
                    log.info("  - Incorrect Citations: %s", crit_expl.get("correct_citations", "No specific explanation."))
                if not original_eval.get("completeness_criteria", {}).get("paragraphs_all_supported", True):
                    log.info("  - Paragraphs Not Supported: %s", crit_expl.get("paragraphs_all_supported", "No specific explanation."))
                if not original_eval.get("completeness_criteria", {}).get("predictive_section_present", True):
                    log.info("  - Predictive Section Missing: %s", crit_expl.get("predictive_section_present", "No specific explanation."))
                if not original_eval.get("quality_criteria", {}).get("clear_structure", True):
                    log.info("  - Unclear Structure: %s", crit_expl.get("clear_structure", "No specific explanation."))
                if not original_eval.get("quality_criteria", {}).get("predictive_section_consistent", True):
                    log.info("  - Predictive Section Inconsistent: %s", crit_expl.get("predictive_section_consistent", "No specific explanation."))
                
                unsupported_claims_list = original_eval.get("unsupported_claims", [])
                if unsupported_claims_list:
                    log.info("  - Unsupported Claims:")
                    for claim in unsupported_claims_list:
                        log.info("    * %s", claim)
                log.info("  - Original Score Explanation: %s", original_eval.get("score_explanation", "N/A"))
                
                log.info("- Factual accuracy: %s", all(original_eval["factual_criteria"].values()))
                log.info("- Completeness: %s", all(original_eval["completeness_criteria"].values()))
                log.info("- Quality criteria: %s", all(original_eval["quality_criteria"].values()))
                log.info("- Hallucination free: %s", original_eval.get("hallucination_free", False))
                
                # Generate correction
                correction = await self.correct_document(eval_doc)
                
                if "error" in correction:
                    log.error("Correction failed: %s", correction["error"])
                    return {"status": "failed", "reason": "correction_failed"}
                
                # Create a temporary document for evaluation
                original_doc = await self.src.find_one({"_id": ObjectId(correction["document_id"])})
                if not original_doc:
                    return {"status": "failed", "reason": "original_doc_not_found"}
                    
                eval_doc = {
                    **original_doc,
                    "agent_response_corrected": correction["corrected_text"]
                }
                
                # Re-evaluate the correction
                re_evaluation = await self.evaluate_document(eval_doc, corrected=True)
                new_eval = re_evaluation["evaluation"]

                # If all criteria pass in the new evaluation, set score to 100
                if (new_eval.get("hallucination_free", False) and
                        all(new_eval.get("factual_criteria", {}).values()) and
                        all(new_eval.get("completeness_criteria", {}).values()) and
                        all(new_eval.get("quality_criteria", {}).values())):
                    log.info("All criteria passed in corrected version. Setting quality_score to 100.")
                    new_eval["quality_score"] = 100
                
                log.info("\nCorrected scores:")
                log.info("- Quality score: %d", new_eval.get("quality_score", 0))
                log.info("- Factual accuracy: %s", all(new_eval["factual_criteria"].values()))
                log.info("- Completeness: %s", all(new_eval["completeness_criteria"].values()))
                log.info("- Quality criteria: %s", all(new_eval["quality_criteria"].values()))
                log.info("- Hallucination free: %s", new_eval.get("hallucination_free", False))
                
                # Verify improvement
                original_score = original_eval.get("quality_score", 0)
                new_score = new_eval.get("quality_score", 0)
                score_improved = new_score > original_score
                
                factual_improved = (
                    all(new_eval["factual_criteria"].values()) and 
                    not all(original_eval["factual_criteria"].values())
                )
                
                completeness_improved = (
                    all(new_eval["completeness_criteria"].values()) and 
                    not all(original_eval["completeness_criteria"].values())
                )
                
                quality_improved = (
                    all(new_eval["quality_criteria"].values()) and 
                    not all(original_eval["quality_criteria"].values())
                )
                
                hallucination_improved = (
                    new_eval.get("hallucination_free", False) and 
                    not original_eval.get("hallucination_free", False)
                )
                
                any_improvement = (
                    score_improved or 
                    factual_improved or 
                    completeness_improved or 
                    quality_improved or 
                    hallucination_improved
                )
                
                if any_improvement:
                    log.info("\nImprovements detected:")
                    if score_improved:
                        log.info("- Quality score improved: %d → %d", original_score, new_score)
                    if factual_improved:
                        log.info("- Factual accuracy fixed")
                    if completeness_improved:
                        log.info("- Completeness improved")
                    if quality_improved:
                        log.info("- Quality criteria improved")
                    if hallucination_improved:
                        log.info("- Hallucinations removed")
                    
                    log.info("Final corrected report text:\n%s", correction["corrected_text"])

                    # Update evaluation document with verified improvement
                    await self.eval.update_one(
                        {"_id": eval_doc["_id"]},
                        {"$set": {
                            "with_corrections": True,
                            "corrected_evaluation": new_eval,
                            "corrected_evaluated_at": re_evaluation["evaluated_at"],
                            "corrected_text": correction["corrected_text"],
                            "improvement_metrics": {
                                "quality_score_delta": new_score - original_score,
                                "factual_improved": factual_improved,
                                "completeness_improved": completeness_improved,
                                "quality_improved": quality_improved,
                                "hallucination_improved": hallucination_improved
                            }
                        }}
                    )
                    
                    log.info("\nStored verified correction in evaluations collection")
                    return {
                        "status": "success",
                        "document_id": correction["document_id"],
                        "improvements": {
                            "score": score_improved,
                            "factual": factual_improved,
                            "completeness": completeness_improved,
                            "quality": quality_improved,
                            "hallucination": hallucination_improved
                        }
                    }
                else:
                    log.warning("\nNo improvements detected in correction, skipping update")
                    return {"status": "failed", "reason": "no_improvement"}
                
            except Exception as e:
                log.error("Error in correction process: %s", e)
                return {"status": "failed", "reason": str(e)}
        
        # Process in batches of 6
        results = []
        for i in range(0, len(fails), 6):
            batch = fails[i:i+6]
            log.info("Processing correction batch %d/%d (%d documents)",
                     i//6 + 1, (len(fails)+5)//6, len(batch))
            
            batch_results = await tqdm_asyncio.gather(
                *[process_correction(eval_doc) for eval_doc in batch],
                desc=f"Correction Batch {i//6 + 1}"
            )
            
            results.extend(batch_results)
            
            if i + 6 < len(fails):
                log.info("Pausing between correction batches...")
                await asyncio.sleep(2)
        
        # Summarize results
        successes = sum(1 for r in results if r["status"] == "success")
        improvements = {
            "score": sum(1 for r in results if r["status"] == "success" and r["improvements"]["score"]),
            "factual": sum(1 for r in results if r["status"] == "success" and r["improvements"]["factual"]),
            "completeness": sum(1 for r in results if r["status"] == "success" and r["improvements"]["completeness"]),
            "quality": sum(1 for r in results if r["status"] == "success" and r["improvements"]["quality"]),
            "hallucination": sum(1 for r in results if r["status"] == "success" and r["improvements"]["hallucination"])
        }
        
        return {
            "corrections_processed": len(fails),
            "corrections_succeeded": successes,
            "corrections_failed": len(fails) - successes,
            "improvements": improvements,
            "note": "Running in dev-mode: corrections stored in evaluations collection only"
        }

# --------------------------------------------------------------------- #
async def main():
    run_start_time = datetime.now(timezone.utc)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=1) # Respecting user's change

    ev = EPSRevEvaluator(MONGO_URI)
    log.info("=== Evaluation pass ===")
    eval_results = await ev.process_range(start, end)
    log.info("Evaluation results: %s", eval_results)
    
    log.info("=== Correction pass ===")
    corr_results = await ev.process_corrections(start, end, run_start_time=run_start_time)
    log.info("Correction results: %s", corr_results)

if __name__ == "__main__":
    asyncio.run(main())

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
from typing import Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pandas_market_calendars as mcal

# ----------  logging  ---------- #
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
log = logging.getLogger("EPSREV")
log.setLevel(logging.DEBUG)

for noisy in ("motor", "pymongo", "httpcore", "httpx", "openai", "asyncio", "anyio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --------------------------------------------------------------------- #
# ENV / CONFIG
# --------------------------------------------------------------------- #
OPENAI_API_KEYS = [
    k for k in (
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_API_KEY_BACKUP1"),
        os.getenv("OPENAI_API_KEY_BACKUP2"),
    ) if k
]
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
    "description": "Paragraph-level fact check of an EPS/Revenue analysis.",
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
        cli = AsyncIOMotorClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db  = cli["asc-fin-data"]
        self.src  = db["user_activities"]
        self.eval = db[EVAL_COLL_NAME]

    # ------------------------------- sampling ------------------------ #
    async def sample_docs(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        docs = await self.src.find({
            "agent":    "earnings",
            "mode":     "eps_revenue",
            "timestamp":{"$gte": start, "$lt": end}
        }).to_list(length=None)
        sample = random.sample(docs, min(len(docs), DAILY_SAMPLE_SIZE))
        log.info("Sampled %d docs: %s",
                 len(sample),
                 ", ".join(d['user_question'] for d in sample))
        return sample

    # ---------------------------- prompting ------------------------- #
    def build_prompt(self, analysis: str, sources: Dict[str, Any]) -> str:
        blocks, all_ids = [], set()

        # 1) Analysis paragraphs
        for para in split_paragraphs(analysis):
            ids = extract_ids(para)
            all_ids.update(ids)
            blocks.append(f"PARAGRAPH:\n{para}\n")

        # 2) Predictive Estimates section
        predictive = analysis.split("**Predictive Estimates**:")[-1].strip()
        ids = extract_ids(predictive)
        all_ids.update(ids)
        blocks.append(f"PREDICTIVE ESTIMATES SECTION:\n{predictive}\n")

        # 3) SOURCE DOCUMENTS section
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
            + "\n\nPlease respond via the `evaluate_eps_rev` function."
        )

    # --------------------------- evaluation ------------------------- #
    async def evaluate(self, doc: Dict[str, Any], corrected: bool = False) -> Dict[str, Any]:
        # pick corrected or original
        text = doc.get("agent_response_corrected") if corrected else doc["agent_response"]
        prompt = self.build_prompt(text, doc["agent_sources"])
        log.debug("Prompt preview for %s\n%s",
                  doc["_id"], textwrap.shorten(prompt, 800, placeholder=" …truncated…"))

        system_prompt = (
            "You are an audit-grade EPS/Revenue fact-checker. "
            "Evaluate each paragraph against the SOURCE DOCUMENTS, then the predictive estimates section. "
            "Use the provided schema to return TRUE/FALSE for each criterion, list any unsupported claims, "
            "and assign a quality_score between 0 and 100."
        )

        client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
        try:
            llm = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt}
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={"type": "function", "function": {"name": "evaluate_eps_rev"}}
            )

            ev = json.loads(llm.choices[0].message.tool_calls[0].function.arguments)

            # auto-boost to 100 if truly perfect
            if (
                ev.get("hallucination_free")
                and all(ev["factual_criteria"].values())
                and all(ev["completeness_criteria"].values())
                and all(ev["quality_criteria"].values())
                and not ev["unsupported_claims"]
            ):
                ev["quality_score"] = 100

            log.debug("Eval for %s → %s",
                      doc["_id"], textwrap.shorten(json.dumps(ev), 200))
        except Exception as e:
            log.error("LLM eval failed: %s", e)
            ev = {"error": f"eval_error: {e}"}

        return {
            "document_id":   str(doc["_id"]),
            "ticker":        doc["user_question"],
            "timestamp":     doc["timestamp"],
            "pipeline":      PIPELINE_TAG,
            "is_correction": corrected,
            "evaluation":    ev,
            "evaluated_at":  datetime.now(timezone.utc)
        }

    # ---------------------- range processing ------------------------ #
    async def process_range(self, start: datetime, end: datetime):
        nyse = mcal.get_calendar("NYSE")
        days = set(d.date() for d in nyse.valid_days(start, end))
        cur  = start
        while cur < end:
            nxt = cur + timedelta(days=1)
            if cur.date() not in days:
                cur = nxt
                continue
            log.info("Evaluating %s", cur.date())
            docs = await self.sample_docs(cur, nxt)

            async def _eval(d):
                ev = await self.evaluate(d)
                await self.eval.insert_one(ev)

            for batch in [docs[i:i+10] for i in range(0, len(docs), 10)]:
                await tqdm_asyncio.gather(*[_eval(d) for d in batch])

            cur = nxt

    # ----------------------- correction utils ----------------------- #
    async def needs_fix(self, ev: Dict[str, Any]) -> bool:
        if "error" in ev["evaluation"]:
            return True
        e = ev["evaluation"]
        return not (
            e.get("hallucination_free", False)
            and all(e["factual_criteria"].values())
            and all(e["completeness_criteria"].values())
            and all(e["quality_criteria"].values())
        )

    async def find_failing_evals(self, start: datetime, end: datetime):
        rows = await self.eval.find({
            "pipeline": PIPELINE_TAG,
            "timestamp": {"$gte": start, "$lt": end},
            "with_corrections": {"$exists": False}
        }).to_list(length=None)
        return [ev for ev in rows if await self.needs_fix(ev)]

    async def correct(self, ev_doc: Dict[str, Any]) -> Dict[str, Any]:
        base = await self.src.find_one({"_id": ObjectId(ev_doc["document_id"])})
        if not base:
            return {"error": "orig_missing"}

        e      = ev_doc["evaluation"]
        issues = e.get("unsupported_claims", [])[:]
        if not e["factual_criteria"]["accurate_numbers"]:
            issues.append("Some numbers do not match their sources.")
        if not e["factual_criteria"]["correct_citations"]:
            issues.append("Some citations do not align with the referenced sources.")

        full_prompt     = self.build_prompt(base["agent_response"], base["agent_sources"])
        source_section  = full_prompt.split("SOURCE DOCUMENTS:")[-1]

        system_corr     = "You are an expert financial editor."
        user_corr = (
            "ORIGINAL ANALYSIS:\n" + base["agent_response"] +
            "\n\nSOURCE DOCUMENTS:\n" + source_section +
            "\n\nISSUES IDENTIFIED:\n  " + "\n  ".join(issues) +
            "\n\nINSTRUCTIONS:\n"
            "1. Correct any inaccurate numbers.\n"
            "2. Fix any bad citations so that every [n] maps back to the right source.\n"
            "3. Revise or clarify any unsupported statements—**do NOT remove** or collapse any of the original paragraphs.\n"
            "4. Preserve the **entire structure and bullet formatting**, including all analysis bullets AND the Predictive Estimates section.\n\n"
            "Return ONLY the **full** corrected EPS/Revenue analysis (all bullets + predictive estimates)."
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
        except Exception as err:
            log.error("LLM correction failed: %s", err)
            return {"error": f"corr_error: {err}"}

        log.info("Doc %s corrected (dev-mode, not patched)", base["_id"])
        # when ready to write back, uncomment:
        # await self.src.update_one({ "_id": base["_id"] },
        #     { "$set": { "agent_response_corrected": corrected, "corrected_at": datetime.now(timezone.utc) } }
        # )
        return {"corrected": True, "corrected_text": corrected}

    async def process_corrections(self, start: datetime, end: datetime):
        fails = await self.find_failing_evals(start, end)
        log.info("%d docs need correction", len(fails))

        async def _fix(ev):
            res = await self.correct(ev)
            if res.get("corrected"):
                base = await self.src.find_one({"_id": ObjectId(ev["document_id"])})
                doc  = {**base, "agent_response": res["corrected_text"]}
                reval = await self.evaluate(doc, corrected=False)
                await self.eval.update_one(
                    { "_id": ev["_id"] },
                    { "$set": {
                        "with_corrections": True,
                        "corrected_evaluation": reval["evaluation"],
                        "corrected_evaluated_at": reval["evaluated_at"]
                    }}
                )

        for batch in [fails[i:i+6] for i in range(0, len(fails), 6)]:
            await tqdm_asyncio.gather(*[_fix(ev) for ev in batch])

# --------------------------------------------------------------------- #
async def main():
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=3)

    ev = EPSRevEvaluator(MONGO_URI)
    log.info("=== Evaluation pass ===")
    await ev.process_range(start, end)
    log.info("=== Correction pass ===")
    await ev.process_corrections(start, end)

if __name__ == "__main__":
    asyncio.run(main())

"""
forward_guidance_pipeline.py  –  Agent-E Forward-Guidance Evaluation + Correction
"""

# ------------------------------------------------------------------ #
# Imports & logging setup
# ------------------------------------------------------------------ #
import os, re, json, random, asyncio, logging, textwrap
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
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
for noisy in ("motor", "pymongo", "httpcore", "httpx", "openai"):
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
                        "description": "True if a dedicated “GUIDANCE SECTION” (i.e. the forward-looking “Predictive Estimates”) is included at the end."
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
                        "description": "True if the guidance section’s tone and data align with the preceding analysis paragraphs (no contradictions or new unsupported claims)."
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
para_re = re.compile(r"^- +(?:\*\*)?", re.M)
cid_re  = re.compile(r"\[(\d+)\]")

def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if para_re.match(p.strip())]

def extract_ids(para: str) -> List[str]:
    return cid_re.findall(para)

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
    async def sample_docs(self, start: datetime, end: datetime):
        docs = await self.src.find({
            "agent": "earnings",
            "mode":  "forward_guidance",
            "timestamp": {"$gte": start, "$lt": end}
        }).to_list(length=None)
        sample = random.sample(docs, min(len(docs), DAILY_SAMPLE_SIZE))
        log.info("Sampled %d docs: %s", len(sample),
                 ", ".join(d["user_question"] for d in sample))
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
    async def evaluate(self, doc: Dict[str, Any], corrected: bool=False) -> Dict[str, Any]:
        text   = doc.get("agent_response_corrected") if corrected else doc["agent_response"]
        prompt = self.build_prompt(text, doc["agent_sources"])
        log.debug("Prompt preview %s\n%s",
                  doc["_id"], textwrap.shorten(prompt, 800, placeholder=" …truncated…"))

        system_prompt = (
            "You are an audit-grade forward-guidance fact checker. "
            "Evaluate each analysis paragraph and the final guidance section against the SOURCE DOCUMENTS. "
            "Use the schema to return TRUE/FALSE for each criterion, list unsupported claims, "
            "and assign a 0–100 quality_score."
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
                tool_choice={"type": "function", "function": {"name": "evaluate_forward_guidance"}}
            )

            ev = json.loads(llm.choices[0].message.tool_calls[0].function.arguments)
            # auto-boost to 100 if perfect
            if (
                ev.get("hallucination_free")
                and all(ev["factual_criteria"].values())
                and all(ev["completeness_criteria"].values())
                and all(ev["quality_criteria"].values())
                and not ev["unsupported_claims"]
            ):
                ev["quality_score"] = 100

            log.debug("Eval %s → %s",
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

    # ----------------------- range processing --------------------- #
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

    # ------------------- correction helpers ---------------------- #
    async def needs_fix(self, ev: Dict[str, Any]) -> bool:
        if "error" in ev["evaluation"]:
            return True
        e = ev["evaluation"]
        return not (
            e.get("hallucination_free") and
            all(e["factual_criteria"].values()) and
            all(e["completeness_criteria"].values()) and
            all(e["quality_criteria"].values())
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

        context       = self.build_prompt(base["agent_response"], base["agent_sources"])
        system_corr   = "You are an expert financial editor and fact checker."
        user_corr     = (
            "ORIGINAL FORWARD-GUIDANCE ANALYSIS:\n\n" + context +
            "\n\nISSUES IDENTIFIED:\n  " + "\n  ".join(issues) +
            "\n\nINSTRUCTIONS:\n"
            "1. Fix inaccurate numbers.\n"
            "2. Update every in-line citation so each [n] matches the correct source below.\n"
            "3. Clarify unsupported statements without deleting any paragraph.\n"
            "4. Preserve bullet structure and order.\n\n"
            "Return ONLY the corrected analysis with updated citations."
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
        # when ready, uncomment to write back:
        # await self.src.update_one(
        #     {"_id": base["_id"]},
        #     {"$set": {"agent_response_corrected": corrected, "corrected_at": datetime.now(timezone.utc)}}
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
                    {"_id": ev["_id"]},
                    {"$set": {
                        "with_corrections":       True,
                        "corrected_evaluation":   reval["evaluation"],
                        "corrected_evaluated_at": reval["evaluated_at"]
                    }}
                )

        for batch in [fails[i:i+6] for i in range(0, len(fails), 6)]:
            await tqdm_asyncio.gather(*[_fix(ev) for ev in batch])

# ------------------------------------------------------------------ #
async def main():
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=2)

    ev = ForwardGuidanceEvaluator(MONGO_URI)
    log.info("=== Evaluation pass ===")
    await ev.process_range(start, end)

    log.info("=== Correction pass ===")
    await ev.process_corrections(start, end)

if __name__ == "__main__":
    asyncio.run(main())

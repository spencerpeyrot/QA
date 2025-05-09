"""
call_summary_pipeline.py  –  Audit‑grade evaluator & fixer for earnings‑call summaries
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
# Function‑call schema
# --------------------------------------------------------------------- #
evaluation_schema = {
    "name": "evaluate_call_summary",
    "description": "Fact-check each bullet in an earnings-call summary against the official transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "numbers_match_transcript": {
                "type": "boolean",
                "description": "True if every numerical figure in the summary bullets exactly matches the numbers spoken in the transcript."
            },
            "statements_supported": {
                "type": "boolean",
                "description": "True if all non-numeric statements in the summary are directly supported by the transcript content."
            },
            "hallucination_free": {
                "type": "boolean",
                "description": "False if any bullet contains details or assertions not found in the transcript. Minor paraphrasing is allowed and does not count as hallucination."
            },
            "clear_structure": {
                "type": "boolean",
                "description": "True if the summary bullets are logically organized, clear, and concise."
            },
            "quality_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall quality rating (0–100), balanced across factual accuracy, completeness, structure, and absence of hallucinations."
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
            "numbers_match_transcript",
            "statements_supported",
            "hallucination_free",
            "clear_structure",
            "quality_score",
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
        bullet_blocks = "\n\n".join(
            f"BULLET:\n{b}" for b in split_bullets(summary)
        ) or summary   # fallback if no bullets

        prompt = (
            "You are an audit-grade fact checker specializing in earnings-call summaries. "
            "Compare each bullet of the analyst-written summary with the official transcript excerpt, "
            "and evaluate numerical accuracy, statement support, and overall clarity."
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
        text = doc.get("agent_response_corrected") if corrected else doc["agent_response"]
        transcript = doc["agent_sources"].get("transcript", "")
        system_prompt = (
            "You are an audit-grade earnings-call fact checker. "
            "Use the evaluation schema to return true/false for each criterion, list any unsupported claims, "
            "and assign a 0–100 quality_score."
        )
        user_prompt = self.build_prompt(text, transcript)

        client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
        try:
            llm = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_prompt}
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={"type": "function", "function": {"name": "evaluate_call_summary"}}
            )
            ev = json.loads(llm.choices[0].message.tool_calls[0].function.arguments)
            log.debug("Eval result %s: %s", doc["_id"], textwrap.shorten(json.dumps(ev), 200))
        except Exception as e:
            log.error("LLM eval failed: %s", e)
            ev = {"error": f"eval_error: {e}"}

        return {
            "document_id": str(doc["_id"]),
            "ticker":       doc["user_question"],
            "timestamp":    doc["timestamp"],
            "pipeline":     PIPELINE_TAG,
            "is_correction":corrected,
            "evaluation":   ev,
            "evaluated_at": datetime.now(timezone.utc)
        }

    # -------------------------- sampling -------------------------- #
    async def sample_docs(self, start: datetime, end: datetime):
        docs = await self.src.find({
            "agent": "earnings",
            "mode":  "earnings_transcript_summary",
            "timestamp": {"$gte": start, "$lt": end}
        }).to_list(length=None)
        sample = random.sample(docs, min(len(docs), DAILY_SAMPLE_SIZE))
        log.info("Sampled %d docs: %s", len(sample), ", ".join(d["user_question"] for d in sample))
        return sample

    # ---------------------- range processing ---------------------- #
    async def process_range(self, start: datetime, end: datetime):
        cur = start
        while cur < end:
            nxt = cur + timedelta(days=1)
            log.info("Evaluating %s", cur.date())
            docs = await self.sample_docs(cur, nxt)

            async def _eval(d):
                ev = await self.evaluate(d)
                await self.eval.insert_one(ev)

            for batch in [docs[i:i+10] for i in range(0, len(docs), 10)]:
                await tqdm_asyncio.gather(*[_eval(d) for d in batch])
            cur = nxt

    # ----------------------- correction logic --------------------- #
    async def needs_fix(self, ev: Dict[str, Any]) -> bool:
        if "error" in ev["evaluation"]:
            return True
        e = ev["evaluation"]
        return not (
            e.get("hallucination_free") and
            e.get("numbers_match_transcript") and
            e.get("statements_supported") and
            e.get("clear_structure")
        )

    async def find_failing_evals(self, start: datetime, end: datetime):
        evals = await self.eval.find({
            "pipeline": PIPELINE_TAG,
            "timestamp": {"$gte": start, "$lt": end},
            "with_corrections": {"$exists": False}
        }).to_list(length=None)
        return [ev for ev in evals if await self.needs_fix(ev)]

    async def correct(self, ev_doc):
        base = await self.src.find_one({"_id": ObjectId(ev_doc["document_id"])})
        if not base:
            return {"error": "orig_missing"}

        e = ev_doc["evaluation"]
        instr = []
        if not e.get("numbers_match_transcript"):
            instr.append("- Fix numbers that don’t match the transcript.")
        if not e.get("statements_supported"):
            instr.append("- Remove or cite unsupported statements.")
        if e.get("unsupported_claims"):
            instr.append("- Correct unsupported claims:" + "".join(f"\n  • {c}" for c in e["unsupported_claims"]))

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
        # -- when ready to write back the corrected summary, uncomment below --
        # await self.src.update_one(
        #     {"_id": base["_id"]},
        #     {"$set": {
        #         "agent_response_corrected": corrected,
        #         "corrected_at": datetime.now(timezone.utc)
        #     }}
        # )
        return {"corrected": True, "corrected_text": corrected}
    
    async def process_corrections(self, start: datetime, end: datetime):
        fails = await self.find_failing_evals(start, end)
        log.info("%d docs need correction", len(fails))

        async def _fix(ev):
            res = await self.correct(ev)
            if res.get("corrected"):
                reval = await self.evaluate(
                    await self.src.find_one({"_id": ObjectId(ev["document_id"]) }),
                    corrected=True
                )
                await self.eval.update_one(
                    {"_id": ev["_id"]},
                    {"$set": {
                        "with_corrections": True,
                        "corrected_evaluation": reval["evaluation"],
                        "corrected_evaluated_at": reval["evaluated_at"],
                        "corrected_text": res["corrected_text"]
                    }}
                )

        for batch in [fails[i:i+6] for i in range(0, len(fails), 6)]:
            await tqdm_asyncio.gather(*[_fix(ev) for ev in batch])

# --------------------------------------------------------------------- #
async def main():
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=2)

    ev = CallSummaryEvaluator(MONGO_URI)
    log.info("=== Evaluation pass ===")
    await ev.process_range(start, end)

    log.info("=== Correction pass ===")
    await ev.process_corrections(start, end)

if __name__ == "__main__":
    asyncio.run(main())

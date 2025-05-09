"""
slvb_pipeline.py  ·  SLVB Evaluation & Correction

Pipeline goals
--------------
• Sample daily "keypoints" ticker write‑ups.
• Verify numbers vs. ground‑truth market data (dummy fetch for now).
• Scrape live hyperlinks + treat "[Newswire]" bullets as relevance‑only.
• Grade on factual / completeness / quality / hallucination‑free axes.
• Auto‑correct failing docs, update ASC‑FIN‑DATA.user_activities,
  then re‑evaluate and store results in ASC‑FIN‑DATA.evaluations.

Environment variables
---------------------
OPENAI_API_KEY, OPENAI_API_KEY_BACKUP1, OPENAI_API_KEY_BACKUP2
EVAL_MONGO_URI
DAILY_SAMPLE_SIZE   (optional, default 25)
MIN_CITATIONS       (optional, default 2)
OPENAI_MODEL        (optional, default "o4-mini")
"""

# ---------------------------------------------------------------------
# Imports & basic setup
# ---------------------------------------------------------------------
import os, json, random, asyncio, re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from itertools import islice

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pandas_market_calendars as mcal

# ---------- Third‑party fetch helpers (aiohttp / bs4) ----------
import aiohttp
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
OPENAI_API_KEYS = [k for k in (
    os.getenv("OPENAI_API_KEY"),
    os.getenv("OPENAI_API_KEY_BACKUP1"),
    os.getenv("OPENAI_API_KEY_BACKUP2")) if k]

MONGO_URI           = os.getenv("EVAL_MONGO_URI")
# PRODUCTION: DAILY_SAMPLE_SIZE = int(os.getenv("DAILY_SAMPLE_SIZE", 25))
DAILY_SAMPLE_SIZE   = int(os.getenv("DAILY_SAMPLE_SIZE", 1))  # TESTING: Sample 1 report per day
MIN_CITATIONS       = int(os.getenv("MIN_CITATIONS", 2))
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "o4-mini") 

EVAL_COLL_NAME      = "evaluations"
PIPELINE_TAG        = "SLVB"

# ---------------------------------------------------------------------
# Evaluation JSON schema for OpenAI function calling
# ---------------------------------------------------------------------
evaluation_schema = {
    "name": "evaluate_response",
    "description": "Evaluate a newsletter ticker write‑up vs. sources & market data",
    "parameters": {
        "type": "object",
        "properties": {
            "factual_criteria": {
                "type": "object",
                "properties": {
                    "accurate_numbers": {"type": "boolean"},
                    "correct_citations": {"type": "boolean"}
                },
                "required": ["accurate_numbers", "correct_citations"]
            },
            "completeness_criteria": {
                "type": "object",
                "properties": {
                    "time_relevant": {"type": "boolean"},
                    "includes_context": {"type": "boolean"}
                },
                "required": ["time_relevant", "includes_context"]
            },
            "quality_criteria": {
                "type": "object",
                "properties": {
                    "clear_presentation": {"type": "boolean"},
                    "explains_causes":   {"type": "boolean"}
                },
                "required": ["clear_presentation", "explains_causes"]
            },
            "hallucination_free": {"type": "boolean"},
            "quality_score": {
                "type": "integer",
                "minimum": 0, "maximum": 100
            },
            "criteria_explanations": {
                "type": "object",
                "properties": {
                    "accurate_numbers":  {"type": "string"},
                    "correct_citations": {"type": "string"},
                    "time_relevant":     {"type": "string"},
                    "includes_context":  {"type": "string"},
                    "clear_presentation":{"type": "string"},
                    "explains_causes":   {"type": "string"},
                    "hallucination_free":{"type": "string"}
                }
            },
            "unsupported_claims": {
                "type": "array", "items": {"type": "string"}
            },
            "score_explanation": {"type": "string"}
        },
        "required": ["factual_criteria", "completeness_criteria",
                     "quality_criteria", "hallucination_free",
                     "quality_score", "criteria_explanations",
                     "unsupported_claims", "score_explanation"]
    }
}

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def is_market_day(date_: datetime) -> bool:
    nyse = mcal.get_calendar("NYSE")
    return bool(nyse.valid_days(date_, date_))

def count_citations(bullets: List[str]) -> int:
    """Return total #citations (hyperlink OR `[Newswire]`) across bullets."""
    link_pattern   = re.compile(r'https?://')
    news_pattern   = re.compile(r'\[Newswire\]', re.IGNORECASE)
    cnt = 0
    for b in bullets:
        if link_pattern.search(b):
            cnt += 1
        elif news_pattern.search(b):
            cnt += 1
    return cnt

# ----------------------------- Market stats --------------------------
async def fetch_market_stats(symbol: str, ts: datetime) -> Dict[str, Any]:
    """
    Placeholder market‑data fetch.
    TODO → replace URLs with real endpoints once confirmed.

    Return keys: last_close, chg_pct, chg_pct_vol, market_cap, iv_value,
                 loan_fee_value, current_q_eps, next_q_eps,
                 current_q_rev, next_q_rev
    """
    # Example dummy payload
    return {
        "last_close":        None,  # e.g., 597.02
        "chg_pct":           None,  # +4.3%
        "chg_pct_vol":       None,
        "market_cap":        None,
        "iv_value":          None,
        "loan_fee_value":    None,
        "current_q_eps":     None,
        "next_q_eps":        None,
        "current_q_rev":     None,
        "next_q_rev":        None
    }

# ----------------------------- URL scrape ----------------------------
async def fetch_source_snippet(session: aiohttp.ClientSession, url: str) -> str:
    """Download URL & return title + first 2 paragraphs (plain text)."""
    try:
        async with session.get(url, timeout=10) as resp:
            html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.text.strip() if soup.title else ""
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")[:2]]
        snippet = f"{title}\n" + "\n".join(paras)
        return snippet[:1200]  # truncate for token safety
    except Exception as e:
        return f"[ERROR scraping {url}: {e}]"

# ---------------------------------------------------------------------
# Core evaluator class
# ---------------------------------------------------------------------
class SLVBTickerEvaluator:
    def __init__(self, mongo_uri: str,
                 daily_sample_size: int = DAILY_SAMPLE_SIZE):
        self.client = AsyncIOMotorClient(
            mongo_uri, tls=True, tlsAllowInvalidCertificates=True,
            minPoolSize=2, connectTimeoutMS=0)
        db                       = self.client["asc-fin-data"]
        self.src_coll            = db["user_activities"]
        self.eval_coll           = db[EVAL_COLL_NAME]
        self.daily_sample_size   = daily_sample_size

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------
    async def sample_documents(self, start: datetime,
                               end: datetime) -> List[Dict[str, Any]]:
        """Random sample of keypoints docs having ≥ MIN_CITATIONS."""
        pipeline = [
            {"$match": {
                "agent": "dpr",
                "mode": {"$in": ["keypoints", "keypoints_noquant_Direxion"]},
                "timestamp": {"$gte": start, "$lt": end}
            }},
            {"$project": {
                "agent_response": 1,
                "user_question": 1,
                "timestamp": 1,
                "cit_count": {
                    "$sum": {
                        "$map": {
                            "input": "$agent_response.bullet_points",
                            "as": "bp",
                            "in": {
                                "$cond": [
                                    {"$or": [
                                        {"$regexMatch": {"input": "$$bp", "regex": "https?://"}},
                                        {"$regexMatch": {"input": "$$bp", "regex": "\\[Newswire\\]", "options": "i"}}
                                    ]},
                                    1, 0]
                            }
                        }
                    }
                }
            }},
            {"$match": {"cit_count": {"$gte": MIN_CITATIONS}}}
        ]
        docs = await self.src_coll.aggregate(pipeline).to_list(length=None)
        return random.sample(docs, min(len(docs), self.daily_sample_size))

    # -----------------------------------------------------------------
    # Document‑level evaluation
    # -----------------------------------------------------------------
    async def evaluate_document(self, doc: Dict[str, Any],
                                corrected: bool = False) -> Dict[str, Any]:
        ar  = doc["agent_response"]
        sym = doc["user_question"]
        ts  = doc["timestamp"]

        bullets = (ar.get("corrected_bullet_points")
                   if (corrected and ar.get("corrected_bullet_points"))
                   else ar.get("bullet_points", []))

        # ---- ground‑truth fetches ----
        stats_market = await fetch_market_stats(sym, ts)

        # scrape hyperlinks
        link_pattern = re.compile(r'(https?://\S+)')
        hyperlinks = [m.group(1) for bp in bullets
                      for m in link_pattern.finditer(bp)]
        snippets = {}
        async with aiohttp.ClientSession() as sess:
            for url in hyperlinks:
                snippets[url] = await fetch_source_snippet(sess, url)

        # -------------- build prompts --------------
        system_prompt = "You are an audit‑grade equity‑research fact checker."
        user_prompt   = f"""
TICKER: {sym}
REPORT TIMESTAMP (UTC): {ts.isoformat()}

BULLET POINTS:
{json.dumps(bullets, indent=2)}

RELEVANT_STATS (model‑generated):
{json.dumps(ar.get("relevant_stats", {}), indent=2)}

MARKET_STATS (ground truth):
{json.dumps(stats_market, indent=2)}

LIVE SOURCES:
{json.dumps(snippets, indent=2)}

EVALUATE using the JSON schema supplied.
"""

        try:
            client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={"type": "function",
                             "function": {"name": "evaluate_response"}}
            )
            fn_args = json.loads(resp.choices[0].message.tool_calls[0]
                                 .function.arguments)
        except Exception as e:
            fn_args = {"error": f"LLM eval failed: {e}"}

        return {
            "document_id": str(doc["_id"]),
            "ticker":      sym,
            "timestamp":   ts,
            "evaluation":  fn_args,
            "evaluated_at": datetime.now(timezone.utc)
        }

    # -----------------------------------------------------------------
    # End‑to‑end day‑range processing (mirrors other pipelines)
    # -----------------------------------------------------------------
    async def process_date_range(self, start: datetime,
                                 end: datetime) -> Dict[str, Any]:
        nyse = mcal.get_calendar("NYSE")
        mkt_days = set(d.date() for d in nyse.valid_days(start, end))

        results = {
            "documents_evaluated": 0,
            "documents_failed": 0,
            "factual_true": 0, "factual_tot": 0,
            "complete_true": 0, "complete_tot": 0,
            "quality_true": 0, "quality_tot": 0,
            "hallucination_true": 0, "hallucination_tot": 0,
            "quality_scores": []
        }

        cur = start
        while cur < end:
            nxt = cur + timedelta(days=1)
            if cur.date() not in mkt_days:
                cur = nxt
                continue

            docs = await self.sample_documents(cur, nxt)
            async def _proc(d):
                ev = await self.evaluate_document(d)
                ev["pipeline"] = PIPELINE_TAG
                try:
                    await self.eval_coll.insert_one(ev)
                except Exception as mongo_err:
                    print("Mongo insert error:", mongo_err)
                return ev

            batches = [docs[i:i+10] for i in range(0, len(docs), 10)]
            for b in batches:
                evals = await tqdm_asyncio.gather(*[_proc(d) for d in b])
                # accumulate metrics …
                for ev in evals:
                    e = ev["evaluation"]
                    if "error" in e:
                        results["documents_failed"] += 1
                        continue
                    results["documents_evaluated"] += 1
                    fc = e["factual_criteria"];    cc = e["completeness_criteria"]
                    qc = e["quality_criteria"];    hf = e["hallucination_free"]
                    results["factual_true"]    += sum(fc.values())
                    results["factual_tot"]     += len(fc)
                    results["complete_true"]   += sum(cc.values())
                    results["complete_tot"]    += len(cc)
                    results["quality_true"]    += sum(qc.values())
                    results["quality_tot"]     += len(qc)
                    results["hallucination_true"] += int(hf)
                    results["hallucination_tot"]  += 1
                    results["quality_scores"].append(e["quality_score"])
            cur = nxt

        # compute pass‑rates
        if results["factual_tot"]:
            results["factual_pass_rate"] = 100*results["factual_true"]/results["factual_tot"]
            results["complete_pass_rate"]= 100*results["complete_true"]/results["complete_tot"]
            results["quality_pass_rate"] = 100*results["quality_true"]/results["quality_tot"]
            results["hallucination_free_rate"] = \
                100*results["hallucination_true"]/results["hallucination_tot"]
            if results["quality_scores"]:
                results["avg_quality_score"] = sum(results["quality_scores"]) / len(results["quality_scores"])
        return results

    # -----------------------------------------------------------------
    # === CORRECTION WORKFLOW (condensed) ===
    # -----------------------------------------------------------------
    async def find_failed_evals(self, start: datetime, end: datetime):
        """Find eval docs needing correction."""
        q = {
            "pipeline": PIPELINE_TAG,
            "timestamp": {"$gte": start, "$lt": end},
            "$or": [
                {"evaluation.hallucination_free": False},
                {"evaluation.factual_criteria.accurate_numbers": False},
                {"evaluation.factual_criteria.correct_citations": False},
                {"evaluation.completeness_criteria.time_relevant": False},
                {"evaluation.completeness_criteria.includes_context": False},
                {"evaluation.quality_criteria.clear_presentation": False},
                {"evaluation.quality_criteria.explains_causes": False}
            ],
            "with_corrections": {"$exists": False}
        }
        return await self.eval_coll.find(q).to_list(length=None)

    # ---------- Correction helpers ----------
    async def correct_document(self, eval_doc: Dict[str, Any]) -> Dict[str, Any]:
        doc = await self.src_coll.find_one({"_id": ObjectId(eval_doc["document_id"])})
        if not doc:
            return {"error": "Original doc not found"}

        ar   = doc["agent_response"]
        sym  = doc["user_question"]
        ts   = doc["timestamp"]
        bullets = ar["bullet_points"]

        # Build instruction text from failed criteria & unsupported claims
        e     = eval_doc["evaluation"]
        instr = [f"Bullet(s) need correction due to: {k}"
                 for k, v in {
                    "inaccurate numbers":  not e["factual_criteria"]["accurate_numbers"],
                    "bad citations":       not e["factual_criteria"]["correct_citations"],
                    "stale info":          not e["completeness_criteria"]["time_relevant"],
                    "missing context":     not e["completeness_criteria"]["includes_context"],
                    "presentation issues": not e["quality_criteria"]["clear_presentation"],
                    "no causal expl.":     not e["quality_criteria"]["explains_causes"],
                    "hallucinations":      not e["hallucination_free"]
                 }.items() if v]

        if e["unsupported_claims"]:
            instr.append("Remove or fix unsupported claims:")
            instr.extend([f"- {c}" for c in e["unsupported_claims"]])

        user_prompt = f"""
TICKER  : {sym}
TS      : {ts.isoformat()}

ORIGINAL BULLETS:
{json.dumps(bullets, indent=2)}

INSTRUCTIONS:
{chr(10).join(instr)}

RULES:
1. Make the minimum edits to satisfy the instructions.
2. Preserve order unless bullet logically belongs elsewhere.
3. Do not add new citations except those present in live links.
4. Return *only* the corrected bullets (JSON array).
"""
        sys_prompt = "You are an expert editor of equity notes."

        try:
            client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_prompt}
                ]
            )
            corrected = json.loads(resp.choices[0].message.content)
        except Exception as e:
            return {"error": f"Correction LLM failed: {e}"}

        # -------- update original doc --------
        update = await self.src_coll.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "agent_response.corrected_bullet_points": corrected,
                "corrected_timestamp": datetime.now(timezone.utc),
                "with_corrections": True
            }}
        )
        if update.modified_count != 1:
            return {"error": "Mongo update failed"}

        return {"corrected": True, "ticker": sym, "document_id": str(doc["_id"])}

    # -----------------------------------------------------------------
    async def process_corrections(self, start: datetime, end: datetime):
        fails = await self.find_failed_evals(start, end)
        print(f"{len(fails)} ticker docs need correction")
        async def _fix(ev):
            corr = await self.correct_document(ev)
            if corr.get("corrected"):
                reval = await self.evaluate_document(
                    await self.src_coll.find_one({"_id": ObjectId(ev["document_id"])}),
                    corrected=True)
                await self.eval_coll.update_one(
                    {"_id": ev["_id"]},
                    {"$set": {
                        "with_corrections": True,
                        "corrected_evaluation": reval["evaluation"],
                        "corrected_evaluated_at": reval["evaluated_at"]
                    }}
                )
            return corr
        batches = [fails[i:i+6] for i in range(0, len(fails), 6)]
        for b in batches:
            await tqdm_asyncio.gather(*[_fix(ev) for ev in b])

# ---------------------------------------------------------------------
# Quick test‑runner
# ---------------------------------------------------------------------
async def main():
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=3)

    ev = SLVBTickerEvaluator(MONGO_URI)
    print("=== EVALUATION PASS ===")
    summary = await ev.process_date_range(start, end)
    print("Summary:", summary)

    print("\n=== CORRECTION PASS ===")
    await ev.process_corrections(start, end)

if __name__ == "__main__":
    asyncio.run(main())

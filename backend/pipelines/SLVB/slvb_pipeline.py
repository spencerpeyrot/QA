"""
slvb_ticker_pipeline.py  –  QA + auto-correction for DPR "keypoints" ticker blurbs
------------------------------------------------------------------
• Source docs live in ticker_dashboard/all_agents_daily (DAILY & WEEKLY blobs)
• Each bullet must cite - or be exempt (price/options chatter) - exactly 1 URL
• URLs are matched against metadata.url in source_docs_ticker
• Bullets with two bracketed URLs are supported
------------------------------------------------------------------
"""

# -------------------------------------------------- imports & setup
import os, re, json, random, asyncio, logging, textwrap
from datetime import datetime, timezone, timedelta, time
from typing import Dict, Any, List, Tuple, Set
from dotenv import load_dotenv
from pathlib import Path

# Set up logging first
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("SLVB-QA")
log.setLevel(logging.DEBUG)

for noisy in ("motor", "pymongo", "httpcore", "httpx", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Load .env file
env_path = Path(__file__).resolve().parents[1] / '.env'
log.info("Looking for .env file at: %s", env_path)
log.info(".env file exists: %s", env_path.exists())

load_dotenv(dotenv_path=env_path)

# Initialize API keys list (silently)
OPENAI_API_KEYS = [
    k for k in (
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_API_KEY_BACKUP1"),
        os.getenv("OPENAI_API_KEY_BACKUP2"),
    ) if k
]

if not OPENAI_API_KEYS:
    raise ValueError("No OpenAI API keys found in environment")

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pandas_market_calendars as mcal

# -------------------------------------------------- env / config
MONGO_URI = os.getenv("EVAL_MONGO_URI","mongodb+srv://overlord-one:sbNciWt8sf5KUkmU@asc-fin-data.oxz1gjj.mongodb.net/?retryWrites=true&w=majority")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

DAILY_SAMPLE_SIZE = int(os.getenv("DAILY_SAMPLE_SIZE", "1"))  # Number of documents to sample per day
EVAL_COLL_NAME = "evaluations"
PIPELINE_TAG = "SLVB-TICKER"
# -------------------------------------------------- regex helpers
url_re = re.compile(r"\[((?:https?://[a-zA-Z0-9_\-./?=&%:,]+)|(?:Newswire(?:_\d+)?))]")
stat_re = re.compile(r"\d+(\.\d+)?%\b|\$?\d[\d,.]+")  # crude numeric detector

# Define keys for new context
CONTEXT_TA_VOLUME_KEY = "CONTEXT_TA_VOLUME"
CONTEXT_EARNINGS_KEY = "CONTEXT_EARNINGS"
CONTEXT_OPTIONS_KEY = "CONTEXT_OPTIONS"
CONTEXT_MACRO_OUTLOOK_KEY = "CONTEXT_MACRO_OUTLOOK"
CONTEXT_CATALYST_KEY = "CONTEXT_CATALYST" # New key for Catalyst context

# -------------------------------------------------- evaluation-schema
evaluation_schema = {
    "name": "evaluate_slvb_ticker",
    "description": "Checks each bullet-point claim in a DPR-SLVB keypoints report against its cited source article(s).",
    "parameters": {
        "type": "object",
        "properties": {
            "factual_criteria": {
                "type": "object",
                "description": "Per-bullet fact-checking results",
                "properties": {
                    "statements_supported": {
                        "type": "boolean",
                        "description": "True if claims in each bullet are supported by source documents. Standard financial inferences (e.g. dividend payments indicating stability, strong metrics suggesting performance) are considered supported if the base facts are present."
                    },
                    "correct_citations": {
                        "type": "boolean",
                        "description": "True if every URL in brackets exists in the dashboard sources and the bullet content actually comes from that article."
                    },
                    "no_missing_required_cites": {
                        "type": "boolean",
                        "description": "True iff bullets that contain numeric/statistical assertions ALSO contain a bracketed URL (price/flow chatter may omit)."
                    },
                },
                "required": [
                    "statements_supported",
                    "correct_citations",
                    "no_missing_required_cites",
                ],
            },
            "quality_criteria": {
                "type": "object",
                "description": "Stylistic / structural checks",
                "properties": {
                    "clear_structure": {
                        "type": "boolean",
                        "description": "True if bullets are concise and clearly written."
                    }
                },
                "required": ["clear_structure"],
            },
            "hallucination_free": {
                "type": "boolean",
                "description": "False if any bullet contains information that contradicts or goes far beyond reasonable inference from its matched source(s). Standard financial analysis and logical inferences are not considered hallucinations."
            },
            "quality_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "0–100 overall score balancing factual, citation, and clarity dimensions."
            },
            "criteria_explanations": {
                "type": "object",
                "description": "Detailed reasons for each T/F decision",
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List specific bullet fragments that directly contradict sources or make claims far beyond reasonable financial inference. Standard financial analysis and logical market interpretations are not considered unsupported."
            },
            "score_explanation": {"type": "string"},
        },
        "required": [
            "factual_criteria",
            "quality_criteria",
            "hallucination_free",
            "quality_score",
            "criteria_explanations",
            "unsupported_claims",
            "score_explanation",
        ],
    },
}

# -------------------------------------------------- market calendar helpers
def get_previous_market_days(current_date: datetime, num_days: int = 7) -> List[datetime]:
    """Get a list of previous market days from the given date."""
    nyse = mcal.get_calendar("NYSE")
    # Look back 30 days to ensure we find enough market days
    start_date = current_date - timedelta(days=30)
    # Ensure dates for nyse.valid_days are strings or pandas Timestamps without tz for broad compatibility
    # The output of nyse.valid_days are pandas Timestamps, which are timezone-aware by default (UTC)
    # if the calendar is timezone-aware, or naive if not. NYSE is UTC.
    market_days_pd = nyse.valid_days(start_date=start_date.strftime('%Y-%m-%d'), 
                                     end_date=current_date.strftime('%Y-%m-%d'))
    
    # Convert pandas Timestamps to python datetimes and ensure they are UTC for comparison
    # current_date is already UTC-aware
    market_days = [
        pd_ts.to_pydatetime().replace(tzinfo=timezone.utc) 
        for pd_ts in market_days_pd
    ]
    
    # Filter out any days that are not strictly before current_date (especially current_date itself if it's a market day)
    # And ensure current_date is also timezone.utc for comparison if it wasn't already.
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc) # Should already be by convention
        
    market_days = [d for d in market_days if d < current_date]
    market_days = sorted(market_days, reverse=True)[:num_days]
    return market_days

async def fetch_dashboard_sources_for_date(
    coll: AsyncIOMotorClient,
    ticker: str,
    target_date: datetime,
    missing_urls: Set[str] | None = None
) -> Dict[str, Dict[str, Any]]:
    """
    Pull the all_agents_daily doc for `ticker` on a specific date.
    If missing_urls is provided, only fetches those specific URLs.
    Return a dict keyed by URL (or context key) with the full source-doc payload.
    Also extracts contextual information for TA/volume, earnings, options, and macro outlook.
    """
    # Ensure target_date has timezone info
    if target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=timezone.utc)

    target_start = datetime.combine(target_date.date(), time.min, tzinfo=timezone.utc)
    target_end = datetime.combine(target_date.date(), time.max, tzinfo=timezone.utc)

    if missing_urls:
        log.info(f"\nSearching on {target_date.date()} for missing URLs: {missing_urls}")
    else:
        log.info(f"\nFetching all dashboard sources and context for {ticker} on {target_date.date()}")

    dash = await coll.find_one(
        {
            "agent": "ticker_dashboard",
            "mode": "all_agents_daily",
            "user_question": ticker,
            "timestamp": {"$gte": target_start, "$lt": target_end}
        },
        sort=[("timestamp", -1)]
    )

    sources: Dict[str, Dict[str, Any]] = {} 

    if not dash:
        log.info(f"No dashboard document found for {ticker} on {target_date.date()}")
        # Attempt to find historical options data even if current day dash is missing
        # This is a rare case, usually handled if dash exists but options are insufficient
        # This options backfill should only run if this is the primary call for the document's date.
        if not missing_urls:
            previous_days_for_options = get_previous_market_days(target_date, num_days=7)
            found_historical_options = False
            for past_date_options_only in previous_days_for_options:
                past_dash_options = await coll.find_one(
                    {
                        "agent": "ticker_dashboard",
                        "mode": "all_agents_daily",
                        "user_question": ticker,
                        "timestamp": {
                            "$gte": datetime.combine(past_date_options_only.date(), time.min, tzinfo=timezone.utc),
                            "$lt": datetime.combine(past_date_options_only.date(), time.max, tzinfo=timezone.utc)
                        }
                    },
                    projection={"agent_response.unusual_options.agent_response": 1},
                    sort=[("timestamp", -1)]
                )
                if past_dash_options and past_dash_options.get("agent_response", {}).get("unusual_options", {}).get("agent_response"):
                    historical_options_text = past_dash_options["agent_response"]["unusual_options"]["agent_response"]
                    if "Insufficient options data" not in historical_options_text:
                        sources[CONTEXT_OPTIONS_KEY] = {
                            "metadata": {"url": CONTEXT_OPTIONS_KEY, "title": f"Options Context for {ticker} (from {past_date_options_only.date()})"},
                            "text": historical_options_text
                        }
                        log.info(f"Found sufficient historical Options context for {ticker} from {past_date_options_only.date()} (Length: {len(historical_options_text)})")
                        found_historical_options = True
                        break
            if not found_historical_options:
                log.info(f"No dashboard document and no sufficient historical options found for {ticker} on {target_date.date()}")
        return sources # Return sources which might only contain historical options if found

    agent_response_from_dash = dash.get("agent_response", {})

    # Only extract general global contexts if this is the primary call (missing_urls is None)
    # The options context has its own specific backfilling logic which is self-contained and runs
    # based on the sufficiency of options data for the 'target_date' of this function call.
    if not missing_urls:
        if isinstance(agent_response_from_dash, dict):
            # Extract new contextual information from the dashboard's agent_response
            consolidated_data = agent_response_from_dash.get("consolidated", {})
            if isinstance(consolidated_data, dict) and consolidated_data.get("agent_response"):
                text_content = consolidated_data["agent_response"]
                sources[CONTEXT_TA_VOLUME_KEY] = {
                    "metadata": {"url": CONTEXT_TA_VOLUME_KEY, "title": f"TA/Volume Context for {ticker}"},
                    "text": text_content
                }
                log.info(f"Extracted TA/Volume context for {ticker} (Length: {len(text_content)})")

            # Earnings Context (from .catalyst blob as per current structure)
            earnings_data_blob = agent_response_from_dash.get("catalyst", {}) # Path: agent_response.catalyst
            if isinstance(earnings_data_blob, dict) and earnings_data_blob.get("agent_response"):
                text_content = earnings_data_blob["agent_response"]
                sources[CONTEXT_EARNINGS_KEY] = {
                    "metadata": {"url": CONTEXT_EARNINGS_KEY, "title": f"Earnings Context for {ticker}"},
                    "text": text_content
                }
                log.info(f"Extracted Earnings context for {ticker} (Length: {len(text_content)})")

            # Catalyst Context (from .catalysts blob)
            true_catalyst_data_blob = agent_response_from_dash.get("catalysts", {}) # Path: agent_response.catalysts
            if isinstance(true_catalyst_data_blob, dict):
                catalyst_agent_response = true_catalyst_data_blob.get("agent_response")
                if isinstance(catalyst_agent_response, str) and catalyst_agent_response.strip():
                    sources[CONTEXT_CATALYST_KEY] = {
                        "metadata": {"url": CONTEXT_CATALYST_KEY, "title": f"Catalyst Context for {ticker}"},
                        "text": catalyst_agent_response
                    }
                    log.info(f"Extracted Catalyst context for {ticker} (Length: {len(catalyst_agent_response)})")
                elif isinstance(catalyst_agent_response, dict) and catalyst_agent_response.get("agent_response"):
                    nested_catalyst_text = catalyst_agent_response.get("agent_response")
                    if isinstance(nested_catalyst_text, str) and nested_catalyst_text.strip():
                        sources[CONTEXT_CATALYST_KEY] = {
                            "metadata": {"url": CONTEXT_CATALYST_KEY, "title": f"Catalyst Context for {ticker}"},
                            "text": nested_catalyst_text
                        }
                        log.info(f"Extracted Catalyst context (from nested structure) for {ticker} (Length: {len(nested_catalyst_text)})")
                    else:
                        log.info(f"Catalyst context (nested) found but content is empty for {ticker}.")
                else:
                    log.info(f"No Catalyst context content found at 'agent_response.catalysts.agent_response' for {ticker}.")
            else:
                log.info(f"No 'catalysts' object found in agent_response for {ticker} to extract Catalyst context.")

            # Options Context - This entire block handles options for the current target_date,
            # including its specific historical backfill if data is insufficient.
            # This should run when missing_urls is None.
            unusual_options_data = agent_response_from_dash.get("unusual_options", {})
            options_text_content = None
            if isinstance(unusual_options_data, dict) and unusual_options_data.get("agent_response"):
                options_text_content = unusual_options_data["agent_response"]
            
            if options_text_content and "Insufficient options data" in options_text_content:
                log.info(f"Insufficient options data for {ticker} on {target_date.date()}. Attempting historical lookup.")
                previous_days_for_options = get_previous_market_days(target_date, num_days=7)
                found_historical_options = False
                for past_date_options_only in previous_days_for_options:
                    log.info(f"Checking for options data on {past_date_options_only.date()} for {ticker}")
                    past_dash_options = await coll.find_one(
                        {
                            "agent": "ticker_dashboard",
                            "mode": "all_agents_daily",
                            "user_question": ticker,
                            "timestamp": {
                                "$gte": datetime.combine(past_date_options_only.date(), time.min, tzinfo=timezone.utc),
                                "$lt": datetime.combine(past_date_options_only.date(), time.max, tzinfo=timezone.utc)
                            }
                        },
                        projection={"agent_response.unusual_options.agent_response": 1},
                        sort=[("timestamp", -1)]
                    )
                    if past_dash_options and past_dash_options.get("agent_response", {}).get("unusual_options", {}).get("agent_response"):
                        historical_options_text = past_dash_options["agent_response"]["unusual_options"]["agent_response"]
                        if "Insufficient options data" not in historical_options_text:
                            options_text_content = historical_options_text
                            sources[CONTEXT_OPTIONS_KEY] = {
                                "metadata": {"url": CONTEXT_OPTIONS_KEY, "title": f"Options Context for {ticker} (from {past_date_options_only.date()})"},
                                "text": options_text_content
                            }
                            log.info(f"Found sufficient historical Options context for {ticker} from {past_date_options_only.date()} (Length: {len(options_text_content)})")
                            found_historical_options = True
                            break
                        else:
                            log.info(f"Historical options data on {past_date_options_only.date()} for {ticker} also insufficient.")
                    else:
                        log.info(f"No options data found on {past_date_options_only.date()} for {ticker}")
                
                if not found_historical_options:
                    log.info(f"No sufficient historical options data found for {ticker} after checking previous 7 market days. Using original data from {target_date.date()}.")
                    if options_text_content:
                        sources[CONTEXT_OPTIONS_KEY] = {
                            "metadata": {"url": CONTEXT_OPTIONS_KEY, "title": f"Options Context for {ticker}"},
                            "text": options_text_content
                        }
                        log.info(f"Using original (insufficient) Options context for {ticker} (Length: {len(options_text_content)})")
            
            elif options_text_content: # Options data was sufficient on target_date
                sources[CONTEXT_OPTIONS_KEY] = {
                    "metadata": {"url": CONTEXT_OPTIONS_KEY, "title": f"Options Context for {ticker}"},
                    "text": options_text_content
                }
                log.info(f"Extracted Options context for {ticker} (Length: {len(options_text_content)})")
            else:
                log.info(f"No options data found in agent_response for {ticker} on {target_date.date()}")

            # Macro Outlook Context
            macro_outlook_content = None
            macro_outlook_blob = agent_response_from_dash.get("macro_outlook", {})
            if isinstance(macro_outlook_blob, dict):
                macro_outlook_content_candidate = macro_outlook_blob.get("agent_response")
                if isinstance(macro_outlook_content_candidate, str) and macro_outlook_content_candidate.strip():
                    macro_outlook_content = macro_outlook_content_candidate
            
            if macro_outlook_content: # Check if content was found and is not empty
                sources[CONTEXT_MACRO_OUTLOOK_KEY] = {
                    "metadata": {"url": CONTEXT_MACRO_OUTLOOK_KEY, "title": f"General Macro Outlook Context for {ticker}"},
                    "text": macro_outlook_content
                }
                log.info(f"Extracted General Macro Outlook context for {ticker} (Length: {len(macro_outlook_content)})")
            else:
                log.info(f"No distinct Macro Outlook context found via 'agent_response.macro_outlook.agent_response' path for {ticker}.")
        else: # agent_response_from_dash is not a dict
            log.info(f"Agent_response in dashboard for {ticker} on {target_date.date()} is not a dictionary. Skipping all context extraction.")

    # Original URL source harvesting logic - this runs regardless of missing_urls,
    # as harvest_from_blob itself filters based on missing_urls if provided.
    seen_urls_on_this_date = set()
    newswire_idx_on_this_date = 0

    def _add_found_source(key_url: str, data: Dict[str, Any], original_url: str, type_info: str):
        text_content = data.get("text", "")
        log.info(f"Found source on {target_date.date()} ({type_info}): {key_url} (Original: {original_url}, Chars: {len(text_content)})")
        sources[key_url] = {
            "metadata": data.get("metadata", {}),
            "text": text_content
        }

    def harvest_from_blob(blob_data: Any, type_label: str):
        nonlocal newswire_idx_on_this_date
        if not blob_data: return

        agent_sources_list = blob_data.get("agent_sources", [])
        if not isinstance(agent_sources_list, list): 
            if isinstance(agent_sources_list, dict) and type_label=="WEEKLY": 
                 agent_sources_list = list(agent_sources_list.values())
            else:
                log.warning(f"Unexpected agent_sources format in {type_label} blob for {ticker} on {target_date.date()}. Skipping.")
                return

        for item_data in agent_sources_list:
            if not isinstance(item_data, dict): continue
            
            docs_container = item_data.get("source_docs_ticker") or \
                             item_data.get("source_docs_sector")
            
            actual_docs_to_process: List[Dict[str, Any]] = []
            if isinstance(docs_container, dict): 
                actual_docs_to_process.extend(list(docs_container.values()))
            elif isinstance(docs_container, list): 
                actual_docs_to_process.extend(docs_container)
            elif "metadata" in item_data and item_data.get("metadata", {}).get("url"): 
                actual_docs_to_process.append(item_data)

            for doc_content in actual_docs_to_process:
                if not isinstance(doc_content, dict): continue
                
                url_from_meta = doc_content.get("metadata", {}).get("url")
                if not url_from_meta: continue

                if missing_urls and url_from_meta not in missing_urls:
                    continue
                
                is_newswire = url_from_meta.startswith("Newswire") or doc_content.get("text", "").startswith("Newswire")

                if not missing_urls and is_newswire: 
                    newswire_idx_on_this_date += 1
                    current_key = f"Newswire_{newswire_idx_on_this_date}"
                    if current_key in seen_urls_on_this_date: continue 
                    seen_urls_on_this_date.add(current_key)
                    _add_found_source(current_key, doc_content, url_from_meta, f"{type_label}/Newswire")
                else: 
                    if url_from_meta in seen_urls_on_this_date: continue
                    seen_urls_on_this_date.add(url_from_meta)
                    _add_found_source(url_from_meta, doc_content, url_from_meta, type_label)
    
    # Ensure this uses agent_response_from_dash which is already fetched
    # The agent_response_from_dash is the dictionary from which DAILY/WEEKLY blobs are taken.
    if isinstance(agent_response_from_dash, dict):
        for blob_type in ["DAILY", "WEEKLY"]: 
            data_blob = agent_response_from_dash.get(blob_type)
            if data_blob:
                harvest_from_blob(data_blob, blob_type)
            
    if not sources and missing_urls:
        log.info(f"No matching missing URLs found for {ticker} on {target_date.date()}")
    elif not any(k not in [CONTEXT_TA_VOLUME_KEY, CONTEXT_EARNINGS_KEY, CONTEXT_OPTIONS_KEY, CONTEXT_MACRO_OUTLOOK_KEY, CONTEXT_CATALYST_KEY] for k in sources.keys()):
        # This condition checks if ONLY context keys were added and no URL sources
        log.info(f"Only contextual data found, no URL-based sources for {ticker} on {target_date.date()}")
        
    return sources

# -------------------------------------------------- pipeline class
class SLVBEvaluator:
    def __init__(self, mongo_uri: str):
        if not mongo_uri:
            raise ValueError("EVAL_MONGO_URI env var not set")
        self.cli = AsyncIOMotorClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db = self.cli["asc-fin-data"]
        self.src: AsyncIOMotorClient = db["user_activities"]
        self.eval: AsyncIOMotorClient = db[EVAL_COLL_NAME]

    # ---------- sampling dpr/keypoints
    async def sample_docs(self, start: datetime, end: datetime) -> List[Dict[str,Any]]:
        """Sample documents from a specific date range."""
        docs = await self.src.find(
            {
                "agent": "dpr",
                "mode": "keypoints",
                "timestamp": {"$gte": start, "$lt": end},
            }
        ).to_list(length=None)
        
        if not docs:
            log.info("No documents found in range")
            return []
            
        sample_size = DAILY_SAMPLE_SIZE
        sample = random.sample(docs, min(len(docs), sample_size))
        log.info(
            "Sampled %d/%d docs for %s: %s",
            len(sample),
            sample_size,
            start.date(),
            ", ".join(d.get("user_question", "Unknown") for d in sample)
        )
        return sample

    async def gather_all_sources(self, ticker: str, doc_timestamp: datetime, bullets: List[str]) -> Dict[str, Dict[str, Any]]:
        current_day_sources_and_contexts = await fetch_dashboard_sources_for_date(self.src, ticker, doc_timestamp, missing_urls=None)
        log.info(f"Gathered {len(current_day_sources_and_contexts)} sources and contexts from current day ({doc_timestamp.date()}) for {ticker}")

        all_sources_accumulated = current_day_sources_and_contexts.copy()

        all_cited_non_newswire_urls = set()
        for bullet_text in bullets:
            matches = url_re.findall(bullet_text)
            for m_url in matches:
                if not m_url.startswith("Newswire"):
                    all_cited_non_newswire_urls.add(m_url)
        
        if not all_cited_non_newswire_urls:
            log.info(f"No non-Newswire URLs cited in bullets for {ticker}.")
        else:
            log.info(f"Found {len(all_cited_non_newswire_urls)} unique non-Newswire URLs cited in bullets for {ticker}: {all_cited_non_newswire_urls}")

        urls_to_find_historically = {
            cited_url for cited_url in all_cited_non_newswire_urls 
            if cited_url not in all_sources_accumulated
        }
        
        if urls_to_find_historically:
            log.info(f"Attempting historical lookup for {len(urls_to_find_historically)} missing URLs for {ticker}: {urls_to_find_historically}")
            previous_days_to_check = get_previous_market_days(doc_timestamp, num_days=7)
            for past_date in previous_days_to_check:
                if not urls_to_find_historically:  
                    log.info(f"All missing URLs found for {ticker} before checking all historical dates.")
                    break
                log.info(f"Searching on {past_date.date()} for remaining {len(urls_to_find_historically)} URLs for {ticker}: {urls_to_find_historically}")
                sources_from_past_date = await fetch_dashboard_sources_for_date(
                    self.src, ticker, past_date, missing_urls=urls_to_find_historically
                )
                if sources_from_past_date:
                    log.info(f"Found {len(sources_from_past_date)} sources on {past_date.date()} for {ticker}.")
                    for found_url, source_data in sources_from_past_date.items():
                        if found_url in urls_to_find_historically:
                            all_sources_accumulated[found_url] = source_data
                            urls_to_find_historically.remove(found_url)
                            log.info(f"Successfully added missing URL {found_url} for {ticker} from {past_date.date()}. Remaining: {len(urls_to_find_historically)}")
                else:
                    log.info(f"No additional missing URLs found for {ticker} on {past_date.date()}.")
            if urls_to_find_historically:
                log.warning(f"After historical search, {len(urls_to_find_historically)} URLs remain unfound for {ticker}: {urls_to_find_historically}")
        else:
            log.info(f"All cited non-Newswire URLs for {ticker} were found in current day's sources or no non-Newswire URLs were cited.")

        compiled_newswire_text = ""
        for url_key, source_content in all_sources_accumulated.items():
            if url_key.startswith("Newswire") and not url_key.startswith("CONTEXT_"): # ensure not a context key
                compiled_newswire_text += f"\n{source_content.get('text', '')}"

        if compiled_newswire_text:
            all_sources_accumulated["COMPILED_NEWSWIRE"] = {
                "metadata": {"url": "COMPILED_NEWSWIRE", "title": "Compiled Newswire Sources"},
                "text": compiled_newswire_text.strip()
            }
            all_sources_accumulated["Newswire"] = all_sources_accumulated["COMPILED_NEWSWIRE"]
            log.info(f"Compiled Newswire content for {ticker} (Length: {len(compiled_newswire_text)}).")
        else:
            log.info(f"No Newswire sources found to compile for {ticker}.")
            all_sources_accumulated.pop("COMPILED_NEWSWIRE", None)
            all_sources_accumulated.pop("Newswire", None)

        log.info(f"Final source and context count for {ticker} after all gathering: {len(all_sources_accumulated)}")
        return all_sources_accumulated

    # ---------- prompt construction
    def build_prompt(
        self,
        ticker: str,
        bullets: List[str],
        bullet_sources: List[List[Dict[str, Any]]], # URL-specific sources for each bullet
        missing_flags: List[str],
        global_contexts: Dict[str, str] # New: TA, Earnings, Options, Macro context texts
    ) -> str:
        """
        bullets[i] <- raw bullet text
        bullet_sources[i] <- list of doc dictionaries (may be empty) for cited URLs
        missing_flags[i] <- "" or explanation of missing/needed citation or context check
        global_contexts <- dictionary of context sections like TA, Earnings, etc.
        """
        prompt_segments = []
        current_prompt_char_count = 0 # For debugging length

        # Add Global Contexts to the top of the prompt
        if global_contexts:
            segment_header = "ADDITIONAL CONTEXTUAL INFORMATION:"
            prompt_segments.append(segment_header)
            current_prompt_char_count += len(segment_header)
            
            context_order = [CONTEXT_MACRO_OUTLOOK_KEY, CONTEXT_EARNINGS_KEY, CONTEXT_CATALYST_KEY, CONTEXT_TA_VOLUME_KEY, CONTEXT_OPTIONS_KEY, "COMPILED_NEWSWIRE"]
            
            for ctx_key in context_order:
                if ctx_key in global_contexts and global_contexts[ctx_key]:
                    title = ctx_key # Default title
                    if ctx_key == CONTEXT_MACRO_OUTLOOK_KEY: title = "MACRO OUTLOOK CONTEXT"
                    elif ctx_key == CONTEXT_EARNINGS_KEY: title = "EARNINGS CONTEXT"
                    elif ctx_key == CONTEXT_CATALYST_KEY: title = "CATALYST CONTEXT"
                    elif ctx_key == CONTEXT_TA_VOLUME_KEY: title = "TA/VOLUME CONTEXT"
                    elif ctx_key == CONTEXT_OPTIONS_KEY: title = "OPTIONS CONTEXT"
                    elif ctx_key == "COMPILED_NEWSWIRE": title = "COMPILED NEWSWIRE (Full Text)"
                        
                    text_content = global_contexts[ctx_key]
                    segment = f"  {title}:\n{text_content}"
                    prompt_segments.append(segment)
                    current_prompt_char_count += len(segment)
                    log.info(f"[build_prompt] Added global context: {title}, Length: {len(text_content)}")
            
            separator = "-" * 30
            prompt_segments.append(separator)
            current_prompt_char_count += len(separator)

        for idx, (bt, sources, flag) in enumerate(zip(bullets, bullet_sources, missing_flags)):
            bullet_header = f"BULLET {idx+1}:\n{bt}\n"
            seg = bullet_header
            current_prompt_char_count += len(bullet_header)
            
            if sources: # These are URL-specific sources
                for j, doc in enumerate(sources, 1):
                    meta = doc.get("metadata", {})
                    source_text = doc.get('text','') # Get full text
                    source_url = meta.get('url','n/a')
                    source_title = meta.get('title','')
                    source_provider = meta.get('source','')
                    
                    cited_source_segment = (
                        f"  CITED SOURCE {j} (url={source_url}): "
                        f"{source_title} – {source_provider}\n"
                        f"{source_text}\n" # Use full source_text here
                    )
                    seg += cited_source_segment
                    current_prompt_char_count += len(cited_source_segment)
                    log.info(f"[build_prompt] Added cited source for bullet {idx+1}: {source_url}, Length: {len(source_text)}")
            
            if flag: 
                note_segment = f"  NOTE: {flag}\n"
                seg += note_segment
                current_prompt_char_count += len(note_segment)
            else: 
                note_segment = "  NOTE: All cited URLs found. Check against these and relevant global context.\n"
                seg += note_segment
                current_prompt_char_count += len(note_segment)
                
            prompt_segments.append(seg)

        base_prompt_text = f"You are an audit-grade fact checker. Ticker: {ticker}\n\n"
        current_prompt_char_count += len(base_prompt_text)
        
        final_instructions = "\n\nReview each bullet. For claims not supported by explicitly CITED URLs, check if they are supported by the ADDITIONAL CONTEXTUAL INFORMATION sections (TA/Volume, Earnings, Options, Macro Outlook). Respond via the `evaluate_slvb_ticker` function."
        current_prompt_char_count += len(final_instructions)

        prompt = (
            base_prompt_text
            + "\n\n".join(prompt_segments)
            + final_instructions
        )
        log.info(f"[build_prompt] Total calculated prompt length for {ticker}: {current_prompt_char_count} (actual via len(prompt)): {len(prompt)}")
        # For extreme debugging:
        # if len(prompt) < 20000 and ticker == "IEF": # Condition to avoid flooding logs
        #    log.debug(f"Full prompt for {ticker}:\n{prompt}")
        return prompt

    # ---------- evaluate single doc
    async def evaluate(self, doc: Dict[str, Any], corrected: bool = False, dash_sources_override: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        agent_resp_key = "agent_response_corrected" if corrected else "agent_response"
        response_data = doc.get(agent_resp_key, {})
        
        if isinstance(response_data, str): 
            bullets_text_list = [b.strip() for b in response_data.split('\n') if b.strip()]
        elif isinstance(response_data.get("bullet_points"), list): 
            bullets_text_list = response_data["bullet_points"]
        elif isinstance(response_data.get("bullet_points"), dict): 
            bullets_dict = response_data["bullet_points"]
            bullets_text_list = [v for _, v in sorted(bullets_dict.items(), key=lambda kv: int(kv[0]))]
        else: 
            bullets_text_list = []
            log.warning(f"Bullets not found or in unexpected format in {agent_resp_key} for doc_id {doc.get('_id')}. Proceeding with empty bullets.")

        ticker = doc["user_question"]
        doc_timestamp = doc["timestamp"]
        
        log.info(f"Evaluating document for {ticker}, timestamp: {doc_timestamp}, corrected: {corrected}")
        
        all_sources_and_contexts: Dict[str, Any]
        if dash_sources_override is not None:
            log.info(f"Using pre-fetched dash_sources_override for {ticker}.")
            all_sources_and_contexts = dash_sources_override
        else:
            log.info(f"No dash_sources_override for {ticker}. Calling gather_all_sources.")
            all_sources_and_contexts = await self.gather_all_sources(ticker, doc_timestamp, bullets_text_list)

        # Separate global contexts from URL-specific sources
        global_contexts_data: Dict[str, str] = {}
        url_specific_sources: Dict[str, Any] = {}
        context_keys = [CONTEXT_TA_VOLUME_KEY, CONTEXT_EARNINGS_KEY, CONTEXT_OPTIONS_KEY, CONTEXT_MACRO_OUTLOOK_KEY, CONTEXT_CATALYST_KEY, "COMPILED_NEWSWIRE", "Newswire"]

        for key, value in all_sources_and_contexts.items():
            if key in context_keys:
                if isinstance(value, dict) and "text" in value:
                    global_contexts_data[key] = value["text"]
                elif isinstance(value, str): # Should not happen based on current structure, but good fallback
                     global_contexts_data[key] = value
            else: # URL-specific source
                url_specific_sources[key] = value
        
        # COMPILED_NEWSWIRE and Newswire are also treated as global context for matching purposes later
        if "COMPILED_NEWSWIRE" in all_sources_and_contexts:
            global_contexts_data["COMPILED_NEWSWIRE"] = all_sources_and_contexts["COMPILED_NEWSWIRE"].get("text","")
        if "Newswire" in all_sources_and_contexts: # This key points to the same as COMPILED_NEWSWIRE
             global_contexts_data["Newswire"] = all_sources_and_contexts["Newswire"].get("text","")

        log.info(f"Proceeding with {len(bullets_text_list)} bullet points for {ticker}. Using {len(url_specific_sources)} URL-specific sources and {len(global_contexts_data)} global context sections.")
        
        bullet_src_lists: List[List[Dict[str, Any]]] = [] 
        missing_flags: List[str] = [] 
        
        for idx, bullet_content in enumerate(bullets_text_list, 1):
            log.info(f"\n--- Processing Bullet {idx}/{len(bullets_text_list)} for {ticker} ---")
            log.info(f"Full Bullet Content: {bullet_content}")
            
            current_bullet_url_sources: List[Dict[str, Any]] = []
            cited_entities = url_re.findall(bullet_content) 
            log.info(f"Cited Entities in Bullet: {cited_entities or 'None'}")
            
            if cited_entities:
                log.info("Attempting to match cited entities to URL-specific sources or Newswire context:")
                all_citations_resolved_for_bullet = True
                for cited_entity_key in cited_entities:
                    if cited_entity_key.startswith("Newswire") and "COMPILED_NEWSWIRE" in all_sources_and_contexts:
                        # For Newswire, we pass the actual source doc dict for consistency in build_prompt
                        current_bullet_url_sources.append(all_sources_and_contexts["COMPILED_NEWSWIRE"])
                        log.info(f"  [MATCH] '{cited_entity_key}' -> Mapped to COMPILED_NEWSWIRE context.")
                    elif cited_entity_key in url_specific_sources:
                        current_bullet_url_sources.append(url_specific_sources[cited_entity_key])
                        log.info(f"  [MATCH] '{cited_entity_key}' -> Found in URL-specific sources.")
                    else:
                        log.warning(f"  [MISSING CITED URL] '{cited_entity_key}' -> Not found in URL-specific sources or Newswire context.")
                        all_citations_resolved_for_bullet = False
                
                bullet_src_lists.append(current_bullet_url_sources)
                if current_bullet_url_sources:
                    missing_flags.append("") 
                    if all_citations_resolved_for_bullet:
                        log.info("Outcome: All cited URLs successfully matched to sources.")
                    else:
                        log.warning("Outcome: Some cited URLs were NOT found in sources (see [MISSING CITED URL] logs). LLM will need to verify if content is supported by global context.")
                        # Even if some URLs are missing, the flag is empty because we want the LLM to check global context.
                        # The build_prompt will indicate to the LLM if URLs were missing for this bullet.
                else: 
                    missing_flags.append("(Cited URL(s) not found in dashboard sources or Newswire context - check global context)")
                    log.warning("Outcome: NO sources found for ANY cited entities in this bullet. LLM must check global context.")

            else: # No URL citations in bullet
                bullet_src_lists.append([]) 
                log.info("No URLs cited in this bullet.")
                if stat_re.search(bullet_content.lower()): # Numeric/statistical claim
                    missing_flags.append("(Numeric/statistical claim without URL citation - check global context for support)")
                    log.warning("Outcome: Bullet contains numeric/statistical claim(s) but has NO URL citation. Needs check against global context.")
                else: # Non-numeric, no URL cite, might be qualitative or general
                    missing_flags.append("(No URL citation - check global context for support, especially for opinions/analysis)")
                    log.info("Outcome: Bullet is non-numeric and has no URL citations. Needs check against global context for support.")
        
        prompt_text = self.build_prompt(ticker, bullets_text_list, bullet_src_lists, missing_flags, global_contexts_data)

        system_prompt = (
            "You are an audit-grade ticker-bullet fact checker. "
            "For each bullet, assess the criteria. "
            "A claim is considered 'statements_supported' if its factual content is backed by EITHER its CITED SOURCE URLs OR the relevant ADDITIONAL CONTEXTUAL INFORMATION sections (TA/Volume, Earnings, Options, Macro Outlook). "
            "'correct_citations' applies to bracketed URLs; if no URLs are cited, this can be true if context supports the claim. "
            "'no_missing_required_cites' is true if numeric/statistical claims are either cited via URL OR clearly supported by the ADDITIONAL CONTEXTUAL INFORMATION. Qualitative statements not citing URLs should also be checked against global context. "
            "For bullets with [Newswire] or [Newswire_X] citations, check against the compiled Newswire content. "
            "Populate the JSON schema, explain decisions, and give a 0-100 score."
        )

        client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS)) # type: ignore
        try:
            log.info(f"\nSending to LLM for evaluation for {ticker}... Prompt length: {len(prompt_text)} chars.")
            # For debugging, print first/last N chars of prompt
            # log.debug(f"Prompt Start: {prompt_text[:500]}")
            # log.debug(f"Prompt End: {prompt_text[-500:]}")
            llm_response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={
                    "type": "function",
                    "function": {"name": "evaluate_slvb_ticker"},
                },
            )
            eval_result_json = json.loads(llm_response.choices[0].message.tool_calls[0].function.arguments) # type: ignore
            
            log.info(f"\nEvaluation Results for {ticker}:")
            log.info(f"Hallucination Free: {eval_result_json.get('hallucination_free', False)}")
            log.info(f"Factual Criteria: {eval_result_json.get('factual_criteria')}")
            log.info(f"Quality Criteria: {eval_result_json.get('quality_criteria')}")
            
            if eval_result_json.get("unsupported_claims"):
                log.warning(f"Unsupported Claims Found for {ticker}:")
                for claim in eval_result_json["unsupported_claims"]:
                    log.warning(f"- {claim}")

            fc = eval_result_json.get("factual_criteria", {})
            qc = eval_result_json.get("quality_criteria", {})
            if (
                eval_result_json.get("hallucination_free")
                and fc.get("statements_supported")
                and fc.get("correct_citations")
                and fc.get("no_missing_required_cites")
                and qc.get("clear_structure")
                and not eval_result_json.get("unsupported_claims")
            ):
                eval_result_json["quality_score"] = 100
                log.info(f"Perfect score (100) auto-assigned for {ticker} based on criteria.")
            else:
                log.info(f"Quality Score for {ticker}: {eval_result_json.get('quality_score', 0)}")

        except Exception as e:
            log.error(f"LLM eval failed for {ticker}: {e}", exc_info=True)
            eval_result_json = {"error": f"eval_error: {str(e)}"}
        
        final_eval_doc = {
            "document_id": str(doc["_id"]),
            "ticker": ticker,
            "timestamp": doc["timestamp"],
            "pipeline": PIPELINE_TAG,
            "is_correction": corrected,
            "evaluation": eval_result_json,
            "evaluated_at": datetime.now(timezone.utc),
            "original_bullet_details": [
                {"text": bullets_text_list[i], "flag": missing_flags[i]} 
                for i in range(len(bullets_text_list))
            ]
        }
        return final_eval_doc, all_sources_and_contexts # Return all sources and contexts for potential reuse

    # ---------- range processing
    async def process_range(self, start: datetime, end: datetime):
        nyse = mcal.get_calendar("NYSE")
        days = set(d.date() for d in nyse.valid_days(start, end))
        cur = start
        while cur < end:
            nxt = cur + timedelta(days=1)
            if cur.date() not in days:
                cur = nxt
                continue
            log.info("Evaluating %s", cur.date())
            docs = await self.sample_docs(cur, nxt)

            async def _eval(d):
                ev, _ = await self.evaluate(d)
                await self.eval.insert_one(ev)

            for batch in [docs[i : i + 10] for i in range(0, len(docs), 10)]:
                await tqdm_asyncio.gather(*[_eval(d) for d in batch])
            cur = nxt

    # ---------- correction helpers
    async def needs_fix(self, ev_doc: Dict[str, Any]) -> bool:
        if "error" in ev_doc.get("evaluation", {}): # Check if evaluation key exists
            return True
        e = ev_doc["evaluation"]
        return not (
            e.get("hallucination_free")
            and e["factual_criteria"]["statements_supported"]
            and e["factual_criteria"]["correct_citations"]
            and e["factual_criteria"]["no_missing_required_cites"]
            and e["quality_criteria"]["clear_structure"]
        )

    async def find_failing_evals(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Find eval docs needing correction."""
        # First get the document IDs we evaluated in this run
        rows = await self.eval.find(
            {
                "pipeline": PIPELINE_TAG,
                "timestamp": {"$gte": start, "$lt": end},
                "with_corrections": {"$exists": False},
                "evaluated_at": {"$gte": start}  # Only get evaluations from this run
            }
        ).to_list(length=None)
        
        # Filter for docs that need fixing
        failing_evals = []
        for ev in rows:
            if "error" in ev["evaluation"]:
                failing_evals.append(ev)
                continue
            
            e = ev["evaluation"]
            # Don't flag TA/Options bullets as unsupported
            unsupported_claims = []
            for claim in e.get("unsupported_claims", []):
                # Skip if claim is about TA, price, or options
                if any(term in claim.lower() for term in ["price", "ema", "ma", "support", "resistance", "trend", "technical", "option", "volume", "trading volume", "uptick", "downtick"]):
                    continue
                unsupported_claims.append(claim)
            
            # Only consider non-TA/options claims for support check
            needs_fix = (
                not e.get("hallucination_free", False) or
                not e["factual_criteria"]["statements_supported"] or
                not e["factual_criteria"]["correct_citations"] or
                not e["factual_criteria"]["no_missing_required_cites"] or
                not e["quality_criteria"]["clear_structure"] or
                (unsupported_claims and not any(term in claim.lower() for claim in unsupported_claims 
                                              for term in ["price", "ema", "ma", "support", "resistance", "trend", "technical", "option", "volume", "trading volume"]))
            )
            
            if needs_fix:
                # Update unsupported claims to exclude TA/options
                e["unsupported_claims"] = unsupported_claims
                failing_evals.append(ev)
        
        return failing_evals

    def _extract_core_claim_text(self, eval_claim_full_text: str) -> str:
        """
        Extracts the core assertion from an LLM's "unsupported_claim" string.
        Example: 'Bullet 1: "This part is wrong" – because of XYZ.' -> "This part is wrong"
        """
        # Attempt to find text within double quotes, which often signifies the direct claim
        match_quoted = re.search(r'(?:Bullet \\d+:\\s*)?"(.*?)"', eval_claim_full_text)
        if match_quoted:
            return match_quoted.group(1).strip()

        # If no quotes, fall back to stripping common prefixes/suffixes
        text = re.sub(r'^Bullet \\d+:\\s*', '', eval_claim_full_text) # Remove "Bullet X: "
        text = re.sub(r'\\s*–.*$', '', text) # Remove " – explanation"
        text = re.sub(r'\\s*--.*$', '', text) # Remove " -- explanation"
        return text.strip()

    async def correct(self, ev_doc: Dict[str, Any], dash_sources_override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate a corrected version of the document based on evaluation feedback."""
        base_doc_id_str = ev_doc.get("document_id")
        if not base_doc_id_str:
            log.error("Evaluation document missing 'document_id'. Cannot correct.")
            return {"error": "eval_missing_doc_id"}
        
        base_doc = await self.src.find_one({"_id": ObjectId(base_doc_id_str)})
        if not base_doc:
            log.error(f"Original document {base_doc_id_str} not found for correction.")
            return {"error": "orig_missing"}

        evaluation_details = ev_doc.get("evaluation", {})
        
        original_agent_response = base_doc.get("agent_response", {})
        if isinstance(original_agent_response.get("bullet_points"), list):
            original_bullets_text = original_agent_response["bullet_points"]
        elif isinstance(original_agent_response.get("bullet_points"), dict):
            original_bullets_text = [v for _, v in sorted(original_agent_response["bullet_points"].items(), key=lambda kv: int(kv[0]))]
        else:
            log.warning(f"Original bullets not found or in unexpected format for doc {base_doc_id_str}. Proceeding with empty list.")
            original_bullets_text = []

        ticker = base_doc["user_question"]
        doc_timestamp = base_doc["timestamp"]
        
        original_bullet_details_from_eval = ev_doc.get("original_bullet_details") 
        if not original_bullet_details_from_eval:
            log.warning(f"'original_bullet_details' not found in eval_doc for {ticker}. Constructing from original_bullets_text.")
            # This fallback uses the raw original bullets and assumes no specific flags from initial eval were available
            original_bullet_details_from_eval = [{"text": b, "flag": "(No URL citation - check global context for support, especially for opinions/analysis)" if not url_re.findall(b) else ""} for b in original_bullets_text]
        
        all_sources_and_contexts: Dict[str, Any]
        if dash_sources_override is not None:
            log.info(f"Using pre-fetched dash_sources_override (all sources and contexts) for correction of {ticker}.")
            all_sources_and_contexts = dash_sources_override
        else:
            log.warning(f"No dash_sources_override for correction of {ticker}. Calling gather_all_sources. This might be inefficient.")
            # Pass original_bullets_text for context if re-fetching sources
            all_sources_and_contexts = await self.gather_all_sources(ticker, doc_timestamp, original_bullets_text)

        # Separate global contexts from URL-specific sources for the correction prompt
        global_contexts_data_for_correction: Dict[str, str] = {}
        # url_specific_sources_for_correction: Dict[str, Any] = {} # Not directly used in prompt structure, but sources are in all_sources_and_contexts
        context_keys_list = [CONTEXT_TA_VOLUME_KEY, CONTEXT_EARNINGS_KEY, CONTEXT_OPTIONS_KEY, CONTEXT_MACRO_OUTLOOK_KEY, CONTEXT_CATALYST_KEY, "COMPILED_NEWSWIRE", "Newswire"]

        for key, value in all_sources_and_contexts.items():
            if key in context_keys_list and isinstance(value, dict) and "text" in value:
                global_contexts_data_for_correction[key] = value["text"]
            # else: Not collecting URL specific sources separately here as the full source details are not individually listed in correction prompt for brevity
            # The LLM is expected to use the "AVAILABLE SOURCES" block which summarizes all available data.

        # Prepare general issues from evaluation
        issues_to_address_parts = []
        if "unsupported_claims" in evaluation_details and evaluation_details["unsupported_claims"]:
            issues_to_address_parts.append("UNSUPPORTED CLAIMS IDENTIFIED (These need to be addressed based on available CITED URLs or GLOBAL CONTEXTS):")
            for claim in evaluation_details["unsupported_claims"]:
                issues_to_address_parts.append(f"- {claim}")
        
        criteria_explanations = evaluation_details.get("criteria_explanations", {})
        factual_criteria = evaluation_details.get("factual_criteria", {})
        quality_criteria = evaluation_details.get("quality_criteria", {})

        if not evaluation_details.get("hallucination_free", True):
             issues_to_address_parts.append(f"POTENTIAL HALLUCINATIONS: {criteria_explanations.get('hallucination_free', 'No specific explanation.')}")
        if not factual_criteria.get("statements_supported", True):
            issues_to_address_parts.append(f"STATEMENTS NOT SUPPORTED (by URLs or Global CONTEXTS): {criteria_explanations.get('statements_supported', 'No specific explanation.')}")
        if not factual_criteria.get("correct_citations", True):
            issues_to_address_parts.append(f"CITATION ERRORS (for URL citations): {criteria_explanations.get('correct_citations', 'No specific explanation.')}")
        if not factual_criteria.get("no_missing_required_cites", True):
            issues_to_address_parts.append(f"MISSING REQUIRED CITATIONS (URL or Global Context support needed for numeric/stats): {criteria_explanations.get('no_missing_required_cites', 'No specific explanation.')}")
        if not quality_criteria.get("clear_structure", True):
            issues_to_address_parts.append(f"STRUCTURE/CLARITY ISSUES: {criteria_explanations.get('clear_structure', 'No specific explanation.')}")
        
        if not issues_to_address_parts and not evaluation_details.get("error") and not await self.needs_fix(ev_doc): 
            log.info(f"Initial evaluation for {ticker} (doc: {base_doc_id_str}) was good. No correction needed.")
            return {"document_id": str(base_doc["_id"]), "ticker": ticker, "corrected": False, "corrected_text": "\n".join(original_bullets_text), "note": "Initial evaluation was good."}

        if not issues_to_address_parts and evaluation_details.get("error"):
             log.warning(f"Correction attempt for {ticker} where initial eval had an error: {evaluation_details.get('error')}. No specific claims to guide LLM. Returning original.")
             return {"document_id": str(base_doc["_id"]), "ticker": ticker, "corrected": False, "corrected_text": "\n".join(original_bullets_text), "note": "Initial evaluation had an error, no specific correction guidance."}
        
        if not original_bullet_details_from_eval: # Handles if original_bullets_text was also empty
            log.warning(f"No original bullets found for {ticker} (Doc ID: {base_doc_id_str}) to correct. Returning empty.")
            return {"document_id": str(base_doc["_id"]), "ticker": ticker, "corrected": False, "corrected_text": "", "note": "No original bullets to correct."}

        sys_prompt_correction = """You are an expert financial content corrector. Your task is to meticulously correct issues in financial ticker reports. 
Ensure all information is accurate and well-supported by EITHER the provided CITED SOURCE URLs (if any per bullet) OR the relevant GLOBAL CONTEXT sections (TA/Volume, Earnings, Options, Macro Outlook, Catalyst). Clearly present all information.

CORRECTION GUIDELINES:
1.  **Ticker Relevance is Key**: ALL bullets MUST be directly relevant to the specified TICKER. 
    *   For ETFs or indices (e.g., SPY, QQQ, MDY), focus on the ETF/index itself: its price action, technicals, options activity, sector trends it represents, significant news impacting its major holdings *as they relate to the ETF*, or overall market sentiment affecting it. 
    *   Avoid generic news about individual companies unless it's a top holding AND the information is framed in the context of its impact on the TICKER ETF/index.
2.  **Analyze Bullet Status & Identified Issues**: For each bullet, review its STATUS. For bullets marked for REVIEW or CORRECTION, carefully consider the general "ISSUES IDENTIFIED".
3.  **Verify with All Available Information**: Cross-reference claims against any CITED URLs for that bullet AND all GLOBAL CONTEXTS. GLOBAL CONTEXTS can also be used to generate new, relevant claims if needed.
4.  **Correct Inaccuracies & Cite Appropriately**: If claims are inaccurate or unsupported, rewrite them to be factual, ticker-relevant, and fully backed. 
    *   If support comes from a CITED URL, ensure the URL is present `[URL_of_source]` and the content matches.
    *   **If primary support for a claim comes from the 'TA/VOLUME CONTEXT' or 'OPTIONS CONTEXT', cite this as `[Newswire]` in the bullet.**
    *   For claims primarily supported by other global contexts (e.g., Macro Outlook, Earnings, Catalyst), explicit bracketed citation in the bullet is not required if the statement is clearly supported by that context and is directly about the TICKER.
    *   If a claim is supported by an actual news article from the `COMPILED_NEWSWIRE` global context, cite it as `[Newswire]` (or `[Newswire_X]` if distinguishable and appropriate from original citations).
5.  **Address Citation Problems**: If URL citations (for web articles) are missing where required, or are incorrect, add or fix them. Ensure content matches its cited source if a URL is used.
6.  **Handle Claims Needing Contextual Support**: For bullets flagged for contextual check (e.g., no URL, numeric/statistical), verify them against the GLOBAL CONTEXTS. If supported and ticker-relevant, the bullet can remain or be refined (citing as per Guideline 4 if TA/Options based). If not, it must be corrected to be supportable and relevant, or removed.
7.  **Improve Clarity & Structure**: Refine all bullets for conciseness and clear writing.
8.  **Handle Unsupportable Content**: If a claim cannot be made supportable and relevant to the TICKER using any CITED URL or GLOBAL CONTEXT, it MUST be removed. Do not invent information or include unrelated content to meet bullet counts.
9.  **Maintain Integrity**: Preserve accurate information. Maintain a professional financial tone.
10. **Output Format**:
    *   Return a MINIMUM of 4 bullet points if possible, ensuring all are relevant and supported. 
    *   If the original report had fewer than 4 bullets, try to generate additional, relevant, and supported bullets using the GLOBAL CONTEXTS to meet this minimum. If the original had 0 bullets, try to generate 1-4. Prioritize quality and relevance over mere quantity.
    *   You MUST return the COMPLETE LIST of ALL bullet points (original kept/polished + corrected + newly generated if any), maintaining their original relative order where applicable.
    *   If a bullet point was marked for CORRECTION or REVIEW and changes were made, it MUST appear in its corrected form.
    *   If an original bullet point was marked to KEEP AS IS, or if after REVIEW it was found to be accurate, well-supported, and ticker-relevant, it MUST be returned in its original (or minimally polished) form and position.
    *   The final output must be a single block of text, with each bullet on a new line. No preamble or commentary.
"""
        
        # Prepare global context strings for the prompt
        global_context_prompt_parts = ["GLOBAL CONTEXTS (Use these to verify claims not supported by URLs):"]
        for key, text_content in global_contexts_data_for_correction.items():
            title = key # Default to key if no specific title logic needed here
            if key == CONTEXT_TA_VOLUME_KEY: title = "TA/VOLUME CONTEXT"
            elif key == CONTEXT_EARNINGS_KEY: title = "EARNINGS CONTEXT"
            elif key == CONTEXT_CATALYST_KEY: title = "CATALYST CONTEXT"
            elif key == CONTEXT_OPTIONS_KEY: title = "OPTIONS CONTEXT"
            elif key == CONTEXT_MACRO_OUTLOOK_KEY: title = "MACRO OUTLOOK CONTEXT"
            elif key == "COMPILED_NEWSWIRE": title = "COMPILED NEWSWIRE"
            # 'Newswire' key also points to compiled, handled by COMPILED_NEWSWIRE title
            if key != "Newswire": # Avoid duplicate Newswire if COMPILED_NEWSWIRE is present
                global_context_prompt_parts.append(f"  {title}:\n{textwrap.shorten(text_content, width=300, placeholder='...')}")
        global_context_for_prompt = "\n".join(global_context_prompt_parts) if len(global_context_prompt_parts) > 1 else "No global contexts provided."

        # Prepare CITED URL sources summary for the prompt (optional, for general reference)
        # url_sources_summary_parts = ["AVAILABLE CITED URL SOURCES (Referenced in original bullets if any):"]
        # for key, doc_content_dict in all_sources_and_contexts.items():
        #     if key not in context_keys_list and isinstance(doc_content_dict, dict):
        #         meta = doc_content_dict.get("metadata", {})
        #         text_snippet = textwrap.shorten(doc_content_dict.get("text", "No text content"), width=100, placeholder="...")
        #         url_sources_summary_parts.append(f"  URL: {meta.get('url','n/a')} Title: '{meta.get('title','N/A')}' Snippet: {text_snippet}")
        # url_sources_summary_for_prompt = "\n".join(url_sources_summary_parts) if len(url_sources_summary_parts) > 1 else "No specific URL sources were cited or found separate from global contexts."
        # Decided against detailed URL list here to keep prompt focused; LLM sees cited sources per bullet if any.

        issues_for_prompt = "\n".join(issues_to_address_parts) if issues_to_address_parts else "No specific major issues flagged from overall evaluation, but review each bullet's STATUS and ensure all claims are accurate and well-supported by CITED URLs or GLOBAL CONTEXTS."

        bullets_for_llm_prompt_parts = ["ORIGINAL BULLET POINTS AND CORRECTION STATUS:"]
        any_bullet_needs_substantive_review = False
        
        for i, bullet_detail in enumerate(original_bullet_details_from_eval):
            bullet_text = bullet_detail.get("text", "")
            bullet_flag = bullet_detail.get("flag", "") # Flag from initial evaluation
            bullets_for_llm_prompt_parts.append(f"\nBULLET {i+1}: {bullet_text}")

            # Determine status based on initial eval flag and overall evaluation quality
            # This logic refines the status determination based on the new context-aware flags.
            is_generally_problematic_eval = await self.needs_fix(ev_doc) and not evaluation_details.get("error")

            if bullet_flag == "(price/options chatter – cite not required)": # This flag is from old logic, should be phased out by new flags
                 bullets_for_llm_prompt_parts.append("  STATUS: This was previously identified as chatter. Verify against relevant GLOBAL CONTEXT (TA/Volume, Options). KEEP AS IS if supported by context or general market understanding, or refine for clarity. Correct if it contradicts context.")
                 any_bullet_needs_substantive_review = True # Chatter still needs context check
            elif not bullet_flag and not is_generally_problematic_eval: # No specific flag for this bullet, and overall eval was good
                bullets_for_llm_prompt_parts.append("  STATUS: This bullet had no specific issues and overall evaluation was good. KEEP AS IS.")
            elif "check global context for support" in bullet_flag or "Citation(s) not found" in bullet_flag: 
                bullets_for_llm_prompt_parts.append(f"  STATUS: REVIEW & POTENTIALLY CORRECT. Original note: '{bullet_flag}'. Verify against GLOBAL CONTEXTS. If supported, keep or refine. If not, correct or remove. Also address any general ISSUES IDENTIFIED.")
                any_bullet_needs_substantive_review = True
            elif not bullet_flag and is_generally_problematic_eval: # No specific flag, but overall eval had issues
                bullets_for_llm_prompt_parts.append("  STATUS: REVIEW & POTENTIALLY CORRECT. This bullet had no specific citation flag, but the overall evaluation indicated issues. Verify against CITED URLs (if any) and GLOBAL CONTEXTS. Address any relevant general ISSUES IDENTIFIED.")
                any_bullet_needs_substantive_review = True
            else: # Other specific flags from initial eval, or default if logic is complex
                bullets_for_llm_prompt_parts.append(f"  STATUS: REVIEW & POTENTIALLY CORRECT. Original note: '{bullet_flag if bullet_flag else 'General review due to overall evaluation.'}'. Verify against CITED URLs (if any) and GLOBAL CONTEXTS. Address any relevant general ISSUES IDENTIFIED.")
                any_bullet_needs_substantive_review = True
        
        # Fallback if no original_bullet_details_from_eval but original_bullets_text exists
        if not original_bullet_details_from_eval and original_bullets_text:
            bullets_for_llm_prompt_parts = ["ORIGINAL BULLET POINTS (Review all based on general issues):"]
            for i, bullet_text in enumerate(original_bullets_text):
                bullets_for_llm_prompt_parts.append(f"\nBULLET {i+1}: {bullet_text}")
                bullets_for_llm_prompt_parts.append("  STATUS: REVIEW & POTENTIALLY CORRECT based on general ISSUES IDENTIFIED and available CITED URLs/GLOBAL CONTEXTS.")
            any_bullet_needs_substantive_review = True
            
        if not any_bullet_needs_substantive_review and not (issues_to_address_parts and any(p not in ["UNSUPPORTED CLAIMS IDENTIFIED:"] for p in issues_to_address_parts)):
            # This condition is tricky: if no bullet needs review AND general issues are empty or only the header for unsupported_claims exists
            log.info(f"No specific bullets or significant general issues flagged for correction for {ticker}. Returning original bullets.")
            return {"document_id": str(base_doc["_id"]), "ticker": ticker, "corrected": False, "corrected_text": "\n".join(original_bullets_text), "note": "Initial evaluation found no correctable substantive issues."}

        user_prompt_correction = (
            f"TICKER: {ticker}\n\n"
            f"{chr(10).join(bullets_for_llm_prompt_parts)}\n\n"
            f"{global_context_for_prompt}\n\n"
            # f"{url_sources_summary_for_prompt}\n\n" # Decided against including this detailed list here
            f"ISSUES IDENTIFIED IN OVERALL EVALUATION (Apply these to bullets marked for REVIEW/CORRECTION):\n"
            f"{issues_for_prompt}\n\n"
            f"INSTRUCTIONS: Based on the CORRECTION GUIDELINES and the STATUS of each bullet, please revise the bullets. Return the complete, final list of all bullet points as a single block of text, with each bullet on a new line.\n\n"
            f"CORRECTED BULLET POINTS BLOCK:"
        )
        
        try:
            client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS)) # type: ignore
            log.info(f"Sending to LLM for correction for {ticker}. Prompt length: {len(user_prompt_correction)} chars.")
            # log.debug(f"CORRECTION PROMPT for {ticker}:\n{user_prompt_correction}") # For intense debugging
            llm_correction_response = await client.chat.completions.create(
                model=OPENAI_MODEL, 
                messages=[
                    {"role": "system", "content": sys_prompt_correction},
                    {"role": "user", "content": user_prompt_correction}
                ]
            ) # type: ignore
            
            corrected_text_str = llm_correction_response.choices[0].message.content.strip() if llm_correction_response.choices[0].message.content else ""
            log.info(f"Correction LLM call successful for {ticker}. Corrected text length: {len(corrected_text_str)}")
            
        except Exception as e:
            log.error(f"Correction LLM call failed for {ticker}: {e}", exc_info=True)
            return {"document_id": str(base_doc["_id"]), "ticker": ticker, "corrected": False, "corrected_text": "\n".join(original_bullets_text), "error": f"Correction LLM call failed: {str(e)}"}

        return {
            "document_id": str(base_doc["_id"]),
            "ticker": ticker,
            "corrected": True, 
            "corrected_text": corrected_text_str
        }

    async def process_corrections(self, start: datetime, end: datetime):
        """Process corrections for documents that failed evaluation."""
        # Find documents needing correction
        failing_eval_docs = await self.find_failing_evals(start, end)
        log.info(f"{len(failing_eval_docs)} docs need correction based on find_failing_evals.")
        
        if not failing_eval_docs:
            return {"corrections_processed": 0}
        
        async def _fix_and_reevaluate_one_doc(eval_doc_from_db: Dict[str, Any]):
            try:
                original_eval_content = eval_doc_from_db.get("evaluation", {})
                ticker_for_log = eval_doc_from_db.get("ticker", "UnknownTicker")
                log.info(f"Processing correction for {ticker_for_log} (Eval ID: {eval_doc_from_db.get('_id')})")

                # Step 1: Get original doc and its bullets for source gathering context if needed by `correct`
                # This is implicitly handled if `correct` calls `gather_all_sources`
                # For `correct` to receive dash_sources_override, they must have been stored or passed from initial eval.
                # Assuming `correct` will fetch its own sources if override is None.
                # For efficiency, it would be best if initial eval's sources were available.
                # For now, `correct` can re-fetch if not provided.

                # Generate correction. `dash_sources_override` is None here, so `correct` will call `gather_all_sources`.
                # This might be inefficient if sources were *just* gathered for the initial evaluation.
                # A more advanced flow might persist/pass the initially gathered sources.
                correction_result = await self.correct(eval_doc_from_db, dash_sources_override=None) 
                
                if "error" in correction_result or not correction_result.get("corrected"):
                    log.error(f"Correction failed or no changes made for {ticker_for_log}: {correction_result.get('error') or correction_result.get('note')}")
                    return {"status": "failed", "reason": f"correction_gen_failed: {correction_result.get('error') or correction_result.get('note')}"}
                
                original_doc_for_reeval = await self.src.find_one({"_id": ObjectId(correction_result["document_id"])})
                if not original_doc_for_reeval:
                    log.error(f"Original document {correction_result['document_id']} not found for re-evaluation of {ticker_for_log}.")
                    return {"status": "failed", "reason": "original_doc_not_found_for_reeval"}
                    
                # Create a temporary document structure for re-evaluation
                # The corrected text is a flat string of bullets separated by newline.
                # `evaluate` handles splitting this.
                temp_eval_doc_struct = {
                    **original_doc_for_reeval,
                    "agent_response_corrected": correction_result["corrected_text"] 
                }
                
                # Re-evaluate the correction.
                # For re-evaluation, we need the sources again.
                # If `correct` fetched sources, they are not returned by it.
                # So, re-evaluation `evaluate` call will also call `gather_all_sources`.
                # This is correct to ensure sources match the (potentially new) bullets in corrected_text.
                re_evaluation_output, final_sources_used_for_reeval = await self.evaluate(
                    temp_eval_doc_struct, 
                    corrected=True, 
                    dash_sources_override=None # Force re-gather based on new corrected bullets
                )
                new_eval_content = re_evaluation_output.get("evaluation", {})
                
                # ... (rest of score comparison and DB update logic as before) ...
                # Ensure original_eval_content and new_eval_content are valid before accessing.
                original_score = original_eval_content.get("quality_score", 0) if isinstance(original_eval_content, dict) else 0
                new_score = new_eval_content.get("quality_score", 0) if isinstance(new_eval_content, dict) else 0
                
                # Simplified improvement check
                improvement_detected = new_score > original_score # Add more criteria if needed

                if improvement_detected:
                    log.info(f"Improvement detected for {ticker_for_log}. Original: {original_score}, New: {new_score}")
                    await self.eval.update_one(
                        {"_id": eval_doc_from_db["_id"]},
                        {"$set": {
                            "with_corrections": True,
                            "corrected_evaluation": new_eval_content["evaluation"],
                            "corrected_evaluated_at": re_evaluation_output["evaluated_at"],
                            "corrected_text": correction_result["corrected_text"]
                        }}
                    )
                    log.info(f"Stored verified correction for {ticker_for_log} in evaluations collection.")
                    return {"status": "success", "document_id": correction_result["document_id"], "improvements_made": True}
                else:
                    log.warning(f"No improvement detected for {ticker_for_log} after correction. Original: {original_score}, New: {new_score}. Skipping update.")
                    return {"status": "no_improvement", "document_id": correction_result["document_id"], "improvements_made": False}
                
            except Exception as e_outer:
                ticker_for_exc_log = eval_doc_from_db.get("ticker", "UnknownTickerOnError")
                log.error(f"Outer error in _fix_and_reevaluate_one_doc for {ticker_for_exc_log}: {e_outer}", exc_info=True)
                return {"status": "exception", "reason": str(e_outer)}

        # ... (rest of batch processing for _fix_and_reevaluate_one_doc) ...

# -------------------------------------------------- main
async def main():
    # Calculate date range
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)  # Start of today
    days_to_process = int(os.getenv("DAYS_TO_PROCESS", "2"))  # Default to 2 days if not set
    
    # Get NYSE calendar and find the last N market days
    nyse = mcal.get_calendar("NYSE")
    market_days = nyse.valid_days(end - timedelta(days=30), end)  # Look back up to 30 days to find enough market days
    market_days = [d.date() for d in market_days if d.date() < end.date()]  # Exclude today and future dates
    market_days = sorted(market_days, reverse=True)[:days_to_process]  # Get the last N market days
    
    if not market_days:
        log.error("No market days found in the specified range")
        return
        
    start = datetime.combine(market_days[-1], time.min, tzinfo=timezone.utc)
    end = datetime.combine(market_days[0], time.max, tzinfo=timezone.utc)
    
    log.info("Processing range: %s to %s", start.date(), end.date())
    log.info("Daily sample size: %d", DAILY_SAMPLE_SIZE)
    log.info("Days to process: %d", days_to_process)
    log.info("Market days to process: %s", [d.strftime("%Y-%m-%d") for d in market_days])

    qa_evaluator = SLVBEvaluator(MONGO_URI) # type: ignore
    
    for market_day_date_obj in market_days:
        current_processing_day_start = datetime.combine(market_day_date_obj, time.min, tzinfo=timezone.utc)
        current_processing_day_end = datetime.combine(market_day_date_obj, time.max, tzinfo=timezone.utc) # Use EOD for range
        
        log.info(f"Processing documents for market day: {market_day_date_obj.strftime('%Y-%m-%d')}")
        
        # Sample docs for this specific market day
        docs_for_day = await qa_evaluator.sample_docs(current_processing_day_start, current_processing_day_end)
        
        if not docs_for_day:
            log.info(f"No documents sampled for {market_day_date_obj.strftime('%Y-%m-%d')}. Moving to next day.")
            continue

        for doc_to_process in docs_for_day:
            ticker_symbol = doc_to_process.get("user_question", "UnknownTicker")
            doc_id_for_log = doc_to_process.get("_id", "UnknownDocID")
            log.info(f"Beginning processing for ticker: {ticker_symbol}, Doc ID: {doc_id_for_log}")

            # Step 1: Initial evaluation. 
            # `evaluate` will now call `gather_all_sources` internally because dash_sources_override is None.
            log.info(f"Step 1: Initial evaluation for {ticker_symbol} (Doc ID: {doc_id_for_log})")
            initial_eval_output, sources_used_in_initial_eval = await qa_evaluator.evaluate(doc_to_process, corrected=False, dash_sources_override=None)
            await qa_evaluator.eval.insert_one(initial_eval_output) # type: ignore
            log.info(f"Initial evaluation stored for {ticker_symbol} (Eval ID: {initial_eval_output.get('_id')})")
            
            # Step 2: Check if correction is needed
            if await qa_evaluator.needs_fix(initial_eval_output):
                log.info("Step 2: Correction needed, generating correction")
                # Pass sources_used_in_initial_eval so `correct` can use them for LLM context
                correction_result_dict = await qa_evaluator.correct(initial_eval_output, dash_sources_override=sources_used_in_initial_eval)
                
                if "error" not in correction_result_dict and correction_result_dict.get("corrected_text") is not None:
                    corrected_text_str = correction_result_dict["corrected_text"]
                    # Parse the corrected text string into a list of bullet strings
                    corrected_text_bullets = [b.strip() for b in corrected_text_str.split('\\n') if b.strip()]

                    log.info(f"--- Final Corrected Bullet Points for {ticker_symbol} (Doc ID: {doc_id_for_log}) ---")
                    log.info(f"\n{corrected_text_str}\n--------------------------------------------------")

                    log.info(f"Gathering sources for re-evaluation of corrected text for {ticker_symbol}")
                    sources_for_corrected_text = await qa_evaluator.gather_all_sources(
                        ticker_symbol,
                        doc_to_process["timestamp"], # Timestamp of the original document
                        corrected_text_bullets # Use the newly corrected bullets to find relevant sources
                    )
                    
                    # Create temporary document for re-evaluation with proper structure
                    temp_eval_doc_struct = {
                        **doc_to_process, # Contains original _id, user_question, timestamp etc.
                        "agent_response_corrected": corrected_text_str 
                    }
                    
                    # Step 3: Re-evaluate correction
                    log.info("Step 3: Re-evaluating correction, passing newly gathered sources for corrected text")
                    # Pass sources_for_corrected_text as the override
                    corrected_evaluation_output, _ = await qa_evaluator.evaluate(
                        temp_eval_doc_struct, 
                        corrected=True, 
                        dash_sources_override=sources_for_corrected_text
                    )
                    
                    # Step 4: Store results if improvement found
                    # Ensure 'evaluation' key exists and has 'quality_score'
                    initial_score = initial_eval_output.get("evaluation", {}).get("quality_score", 0)
                    corrected_score = corrected_evaluation_output.get("evaluation", {}).get("quality_score", 0)

                    if corrected_score > initial_score:
                        log.info("Improvement found, storing correction")
                        await qa_evaluator.eval.update_one(
                            {"_id": initial_eval_output["_id"]},
                            {"$set": {
                                "with_corrections": True,
                                "corrected_evaluation": corrected_evaluation_output["evaluation"],
                                "corrected_evaluated_at": corrected_evaluation_output["evaluated_at"],
                                "corrected_text": correction_result_dict["corrected_text"]
                            }}
                        )
                    else:
                        log.warning(f"No improvement found in correction for {initial_eval_output.get('ticker', 'UnknownTicker')}. Original score: {initial_score}, Corrected score: {corrected_score}")
            else:
                log.info(f"No correction needed for {ticker_symbol} (Doc ID: {doc_id_for_log})")

if __name__ == "__main__":
    asyncio.run(main())

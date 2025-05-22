import os
import re
import json
import random
import asyncio
import logging
from datetime import datetime, timezone, timedelta, time
from typing import Dict, Any, List, Optional, Set, Pattern

from dotenv import load_dotenv # Typically used by calling scripts
from pathlib import Path      # Typically used by calling scripts

from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
import pandas_market_calendars as mcal

# --- Environment & API Keys ---

def get_openai_api_keys() -> List[str]:
    """Retrieves OpenAI API keys from environment variables."""
    keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_API_KEY_BACKUP1"),
        os.getenv("OPENAI_API_KEY_BACKUP2"),
    ]
    return [k for k in keys if k]

def get_random_openai_api_key() -> str | None:
    """
    Selects a random OpenAI API key from the ones loaded from environment variables.
    Returns None if no keys are available.
    """
    keys = get_openai_api_keys()
    if not keys:
        return None
    return random.choice(keys)

# --- MongoDB ---

def get_mongo_client(
    mongo_uri: str,
    tls_allow_invalid_certs: bool = True,
    min_pool_size: Optional[int] = None,
    connect_timeout_ms: Optional[int] = None
) -> AsyncIOMotorClient:
    """
    Initializes and returns an AsyncIOMotorClient.
    
    Args:
        mongo_uri: The MongoDB connection string.
        tls_allow_invalid_certs: Whether to allow invalid TLS certificates.
        min_pool_size: Optional minimum connection pool size.
        connect_timeout_ms: Optional connection timeout in milliseconds.
        
    Raises:
        ValueError: If mongo_uri is not provided.
    """
    if not mongo_uri:
        raise ValueError("MONGO_URI must be provided to initialize MongoDB client.")
    
    client_options: Dict[str, Any] = {
        "tls": True,
        "tlsAllowInvalidCertificates": tls_allow_invalid_certs
    }
    if min_pool_size is not None:
        client_options["minPoolSize"] = min_pool_size
    if connect_timeout_ms is not None:
        client_options["connectTimeoutMS"] = connect_timeout_ms
        
    return AsyncIOMotorClient(mongo_uri, **client_options)

# --- Logging ---

def setup_logger(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt: str = "%H:%M:%S",
    noisy_loggers_to_warn: Optional[List[str]] = None
) -> logging.Logger:
    """
    Configures basic logging and returns a logger instance for the application.
    Sets the root logger level to WARNING by default and then configures
    the specified application logger and other common noisy loggers.
    """
    logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.WARNING)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if noisy_loggers_to_warn is None:
        noisy_loggers_to_warn = [
            "motor", "pymongo", "httpcore", "httpx",
            "openai", "asyncio", "anyio", "urllib3", "aiohttp.client"
        ]

    for noisy_logger_name in noisy_loggers_to_warn:
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)
    
    return logger

# --- OpenAI Client ---

async def get_openai_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """
    Initializes and returns an AsyncOpenAI client.
    If api_key is not provided, it tries to select one randomly from environment variables.
    
    Raises:
        ValueError: If no API key can be found/selected.
    """
    if api_key is None:
        selected_key = get_random_openai_api_key()
        if selected_key is None:
            raise ValueError("No OpenAI API key provided or found in environment variables.")
        api_key = selected_key
    return AsyncOpenAI(api_key=api_key)

# --- Text Processing Regexes (Commonly used patterns) ---

# Matches paragraphs/bullets starting with "- " optionally followed by markdown bold (e.g., "- **Headline**" or "- Topic")
RE_MD_BULLET_WITH_OPTIONAL_BOLD: Pattern[str] = re.compile(r"^- +(?:\\*\\*)?", re.M)

# Extracts bracketed numeric IDs like [1], [23] (common for citations)
RE_CITE_BRACKETED_NUMERIC_ID: Pattern[str] = re.compile(r"\[(\d+)\]")

# Extracts URLs or "Newswire" variants from brackets (SLVB pipeline style)
RE_CITE_SLVB_URL_NEWSWIRE: Pattern[str] = re.compile(r"\[((?:https?://[a-zA-Z0-9_\-./?=&%:,]+)|(?:Newswire(?:_\d+)?))]")

# Detects numeric values, percentages, or currency amounts (SLVB pipeline style)
RE_STAT_SLVB_NUMERIC_DETECT: Pattern[str] = re.compile(r"\d+(?:\.\d+)?%?\b|\$?\d[\d,.]*\d\b")


# --- Text Processing Functions ---

def split_text_into_items(
    text: str,
    item_separator: str = "\\n\\n",
    item_start_regex: Optional[Pattern[str]] = None
) -> List[str]:
    """
    Splits a block of text into items (e.g., paragraphs, bullets).
    
    Args:
        text: The input string.
        item_separator: The string that separates items (e.g., "\\n\\n" for paragraphs).
        item_start_regex: An optional compiled regex pattern. If provided,
                               only items matching this pattern at their start
                               will be included after splitting.
    
    Returns:
        A list of item strings, stripped of leading/trailing whitespace.
    """
    if not text:
        return []
    
    potential_items = [item.strip() for item in text.split(item_separator) if item.strip()]
    
    if item_start_regex:
        return [item for item in potential_items if item_start_regex.match(item)]
    else:
        return potential_items

def extract_citations_from_text(text: str, citation_regex: Pattern[str]) -> List[str]:
    """
    Extracts citation identifiers from text using the provided regex.
    The regex should ideally have one capturing group for the identifier.
    """
    if not text:
        return []
    return citation_regex.findall(text)

# --- Market Calendar Utilities ---

def get_previous_market_days(
    current_date: datetime,
    num_days: int = 7,
    calendar_name: str = 'NYSE'
) -> List[datetime]:
    """
    Get a list of previous market days from the given date for a specific calendar.
    Ensures dates are timezone-aware (UTC) and are strictly before current_date.

    Args:
        current_date: The reference date (timezone-aware or naive; will be handled as UTC).
        num_days: The number of previous market days to retrieve.
        calendar_name: The name of the market calendar (e.g., 'NYSE').

    Returns:
        A list of datetime objects representing the previous market days, sorted most recent first.
    """
    calendar = mcal.get_calendar(calendar_name)
    
    if current_date.tzinfo is None or current_date.tzinfo.utcoffset(current_date) is None:
        current_date_aware = current_date.replace(tzinfo=timezone.utc)
    else:
        current_date_aware = current_date.astimezone(timezone.utc)

    # Generous lookback period to find enough market days, accounting for holidays/weekends
    lookback_days_heuristic = max(30, num_days * 2 + 10) 
    start_date_for_mcal = current_date_aware - timedelta(days=lookback_days_heuristic)
    
    # mcal 'end_date' is inclusive. We want days *before* current_date_aware.
    # So, the query range should end the day before current_date_aware.
    end_date_for_mcal_query = current_date_aware - timedelta(days=1)

    # Ensure start_date is not after end_date for mcal query
    if start_date_for_mcal > end_date_for_mcal_query:
        return [] # Not possible to find previous market days in this scenario

    market_days_pd = calendar.valid_days(
        start_date=start_date_for_mcal.strftime('%Y-%m-%d'),
        end_date=end_date_for_mcal_query.strftime('%Y-%m-%d')
    )

    market_days_dt: List[datetime] = []
    for pd_ts in market_days_pd:
        dt_val = pd_ts.to_pydatetime() # Converts to pandas Timestamp then to python datetime
        # Ensure UTC timezone for consistency
        if dt_val.tzinfo is None or dt_val.tzinfo.utcoffset(dt_val) is None:
            dt_val_aware = dt_val.replace(tzinfo=timezone.utc)
        else:
            dt_val_aware = dt_val.astimezone(timezone.utc)
        market_days_dt.append(dt_val_aware)

    return sorted(market_days_dt, reverse=True)[:num_days]

# --- LLM Interaction (Focus on Function Calling) ---

async def call_llm_with_function_call(
    system_prompt: str,
    user_prompt: str,
    function_schema: Dict[str, Any],
    openai_model: str,
    tool_choice: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1, # Default to a low temperature for factual tasks
    max_tokens: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any] | None:
    """
    Makes a call to an OpenAI LLM expecting a function call in the response.

    Args:
        system_prompt: The system message for the LLM.
        user_prompt: The user message for the LLM.
        function_schema: The schema of the function the LLM is expected to call.
        openai_model: The OpenAI model to use (e.g., "o4-mini").
        tool_choice: Specific tool choice for the LLM (e.g., {"type": "function", "function": {"name": "my_func"}}).
                     If None, defaults to choosing the provided function_schema.
        temperature: Sampling temperature for the LLM.
        max_tokens: Optional maximum tokens for the LLM response.
        logger: Optional logger instance for debugging/errors.

    Returns:
        A dictionary parsed from the LLM's function call arguments, or None if an error occurs
        or no valid function call is made.
    """
    try:
        client = await get_openai_client() # Uses random key selection internally
        
        if tool_choice is None:
            tool_choice = {"type": "function", "function": {"name": function_schema.get("name", "evaluate_response")}}

        completion_params: Dict[str, Any] = {
            "model": openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "tools": [{"type": "function", "function": function_schema}],
            "tool_choice": tool_choice,
            "temperature": temperature,
        }
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens

        if logger:
            logger.debug(f"LLM Call Params: model={openai_model}, tool_choice={tool_choice}, temp={temperature}")
            # logger.debug(f"System Prompt: {system_prompt[:500]}...") # Be cautious logging full prompts
            # logger.debug(f"User Prompt: {user_prompt[:1000]}...")

        llm_response = await client.chat.completions.create(**completion_params)
        
        response_message = llm_response.choices[0].message
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            if tool_call.function.name == function_schema.get("name"):
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    if logger:
                        logger.debug(f"LLM function call successful: {tool_call.function.name}")
                    return arguments
                except json.JSONDecodeError as e:
                    if logger:
                        logger.error(f"JSONDecodeError parsing LLM function arguments: {e}\\nArguments: {tool_call.function.arguments}")
                    return {"error": "llm_json_decode_error", "details": str(e)}
            else:
                if logger:
                    logger.warning(f"LLM called unexpected function: {tool_call.function.name}")
                return {"error": "llm_unexpected_function_call", "function_name": tool_call.function.name}
        else:
            if logger:
                logger.warning("LLM response did not contain tool calls.")
            return {"error": "llm_no_tool_calls"}

    except Exception as e:
        if logger:
            logger.error(f"Exception during LLM call: {e}", exc_info=True)
        return {"error": "llm_api_call_exception", "details": str(e)}


# --- Source Document Formatting for Prompts ---

def format_source_documents_for_prompt(
    agent_sources: Dict[str, Any],
    include_keys: Optional[List[str]] = None, # e.g., ["1", "2", "CONTEXT_KEY"]
    max_text_length: Optional[int] = None, # Optional: truncate individual source texts
    global_source_doc_id_prefix: str = "Source", # e.g. "Source [1]", "Source [transcript]"
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Formats agent_sources into a string suitable for LLM prompts.
    Handles nested structures (like source_docs_ticker/sector in SLVB)
    and different metadata fields.

    Args:
        agent_sources: The raw agent_sources dictionary from the database.
                       Can be a flat dict of {id: {metadata:..., text:...}}
                       or nested like in SLVB: { "DAILY": { "agent_sources": [ { "source_docs_ticker": {id: doc} } ] } }
                       or a list of such source doc dicts directly under a key like "source_docs_ticker".
        include_keys: Optional list of source keys/IDs to include. If None, all are included.
        max_text_length: Optional. If provided, truncates the 'text' of each source.
        global_source_doc_id_prefix: Prefix for the source identifier in the prompt.
        logger: Optional logger for warnings.

    Returns:
        A formatted string of source documents.
    """
    source_texts_parts: List[str] = []
    
    processed_sources_count = 0

    def process_single_source_doc(key: str, doc_content: Dict[str, Any]):
        nonlocal processed_sources_count
        if not isinstance(doc_content, dict):
            if logger:
                logger.warning(f"Source content for key '{key}' is not a dictionary. Skipping.")
            return

        metadata = doc_content.get("metadata", {})
        text = str(doc_content.get("text", "")) # Ensure text is a string

        if max_text_length is not None and len(text) > max_text_length:
            text = text[:max_text_length] + "... (truncated)"

        meta_parts = [f"{global_source_doc_id_prefix} [{key}]"]
        if metadata.get("title"): meta_parts.append(f"Title: {metadata['title']}")
        if metadata.get("source"): meta_parts.append(f"Source Provider: {metadata['source']}")
        if metadata.get("url"): meta_parts.append(f"URL: {metadata['url']}")
        # Add other common metadata fields if needed, e.g., description, summary

        source_texts_parts.append("\\n".join(meta_parts) + f"\\n\\n{text}\\n")
        processed_sources_count +=1

    # --- Iteration Logic ---
    # This needs to be flexible to handle various structures of agent_sources
    
    sources_to_iterate: Dict[str, Any] = {}

    # Heuristic 1: Flat dictionary of sources (common case, e.g., EPS/REV, FWD-GUID)
    if all(isinstance(v, dict) and ("metadata" in v or "text" in v) for v in agent_sources.values()):
        sources_to_iterate = agent_sources
    # Heuristic 2: SLVB-like structure (DAILY/WEEKLY blobs)
    elif any(key in agent_sources for key in ["DAILY", "WEEKLY"]):
        temp_slvb_sources: Dict[str, Any] = {}
        newswire_idx = 1
        for blob_type in ["DAILY", "WEEKLY"]:
            blob_data = agent_sources.get(blob_type)
            if not blob_data or not isinstance(blob_data.get("agent_sources"), list):
                continue
            for item_data in blob_data["agent_sources"]:
                if not isinstance(item_data, dict): continue
                docs_container = item_data.get("source_docs_ticker") or item_data.get("source_docs_sector")
                
                actual_docs_to_process: List[Dict[str, Any]] = []
                if isinstance(docs_container, dict):
                    actual_docs_to_process.extend(docs_container.values())
                elif isinstance(docs_container, list):
                    actual_docs_to_process.extend(docs_container)
                
                for doc_c in actual_docs_to_process:
                    if not isinstance(doc_c, dict): continue
                    url_from_meta = doc_c.get("metadata", {}).get("url")
                    if not url_from_meta: continue
                    
                    if url_from_meta.startswith("Newswire"):
                        key_to_use = f"Newswire_{newswire_idx}"
                        newswire_idx +=1
                    else:
                        key_to_use = url_from_meta
                    temp_slvb_sources[key_to_use] = doc_c
        sources_to_iterate = temp_slvb_sources
    # Heuristic 3: LTV-like structure (keys are numbers, values are source docs)
    elif all(k.isdigit() for k in agent_sources.keys()) and \
         all(isinstance(v, dict) for v in agent_sources.values()):
        sources_to_iterate = agent_sources
    # Heuristic 4: Context keys (e.g., from SLVB, like CONTEXT_TA_VOLUME_KEY)
    elif any(k.startswith("CONTEXT_") for k in agent_sources.keys()):
        # This might be mixed with other types; handle context keys if present
        for key, value_dict in agent_sources.items():
            if key.startswith("CONTEXT_") and isinstance(value_dict, dict) and "text" in value_dict:
                if (include_keys is None or key in include_keys):
                     process_single_source_doc(key, value_dict)
    # Fallback or for agent_sources that are simple dicts of text by key (like SLVB global_contexts)
    else:
        # This might catch cases where agent_sources is already pre-formatted for context
        # or if it's a direct { "some_id": {metadata: ..., text:...} } structure not caught by heuristic 1
        # If values are just strings (text), wrap them for process_single_source_doc
        temp_wrapped_sources: Dict[str, Any] = {}
        can_be_wrapped = True
        for key, value in agent_sources.items():
            if isinstance(value, str): # If value is just text
                temp_wrapped_sources[key] = {"text": value, "metadata": {"title": key}}
            elif isinstance(value, dict) and ("metadata" in value or "text" in value): # Already a source doc
                temp_wrapped_sources[key] = value
            else:
                can_be_wrapped = False; break # Unrecognized structure
        if can_be_wrapped:
            sources_to_iterate = temp_wrapped_sources


    for key_id, source_doc in sources_to_iterate.items():
        if include_keys is None or key_id in include_keys:
            process_single_source_doc(key_id, source_doc)
            
    if not source_texts_parts:
        if logger:
            logger.info(f"No source documents formatted. Input agent_sources keys: {list(agent_sources.keys()) if isinstance(agent_sources, dict) else 'Not a dict'}")
        return "NO SOURCE DOCUMENTS PROVIDED OR MATCHED FILTERS."
        
    if logger:
        logger.debug(f"Formatted {processed_sources_count} source documents for prompt.")

    return "\\n\\n".join(source_texts_parts)


# --- Generic `needs_fix` Checker ---

def check_needs_fix(
    evaluation_result: Dict[str, Any],
    hallucination_key: str = "hallucination_free",
    criteria_groups: Optional[Dict[str, List[str]]] = None, # e.g., {"factual_criteria": ["accurate_numbers", "correct_citations"]}
    unsupported_claims_key: str = "unsupported_claims", # Key for list of unsupported claims
    allow_empty_unsupported_if_perfect: bool = True # If all criteria pass, empty unsupported_claims is ok
) -> bool:
    """
    Generic check to determine if an evaluation result indicates a need for correction.

    Args:
        evaluation_result: The evaluation dictionary from the LLM.
        hallucination_key: Key for the hallucination boolean flag.
        criteria_groups: A dictionary where keys are main criteria group names (e.g., "factual_criteria")
                         and values are lists of sub-criteria keys within that group that must be true.
        unsupported_claims_key: Key for the list of unsupported claims. An empty list is good.
        allow_empty_unsupported_if_perfect: If True and all other criteria pass, an empty 'unsupported_claims'
                                            list doesn't trigger 'needs_fix'. If False, any non-empty list
                                            would trigger it if the key exists.

    Returns:
        True if correction is needed, False otherwise.
    """
    if not evaluation_result or "error" in evaluation_result:
        return True # Errors always need fixing

    if not evaluation_result.get(hallucination_key, False): # False for hallucination_free means problem
        return True

    if criteria_groups:
        for group_key, sub_keys in criteria_groups.items():
            group_data = evaluation_result.get(group_key, {})
            if not isinstance(group_data, dict): return True # malformed evaluation
            for sub_key in sub_keys:
                if not group_data.get(sub_key, False): # False for any sub-criterion means problem
                    return True
    
    # Check unsupported claims:
    # If the key exists and the list is not empty, it generally means a fix is needed,
    # UNLESS allow_empty_unsupported_if_perfect is true AND all other criteria have passed so far.
    unsupported = evaluation_result.get(unsupported_claims_key)
    if isinstance(unsupported, list) and unsupported: # If list exists and is not empty
        if allow_empty_unsupported_if_perfect:
            # At this point, if we are here, it means hallucination_free was True and all criteria passed.
            # So, if unsupported_claims are present, it's a problem.
            return True 
        else: # If any unsupported claim exists, it's a problem regardless of other flags
             return True
    # If unsupported_claims_key is not present or is empty, it's fine from this check's POV.

    return False # All checks passed

# --- Evaluation Result Aggregation for process_range ---

def aggregate_evaluation_stats(
    evaluation_data: Dict[str, Any], # The direct "evaluation" sub-dictionary
    stats_accumulator: Dict[str, Any],
    criteria_mapping: Dict[str, List[str]] = None # e.g. {"factual": ["factual_criteria.accurate_numbers", ...]}
):
    """
    Updates an accumulator dictionary with statistics from a single evaluation.
    Modifies stats_accumulator in-place.

    Args:
        evaluation_data: The evaluation data (e.g., result from LLM function call).
        stats_accumulator: A dictionary to store aggregated stats (e.g., counts, scores).
                           Expected to have keys like:
                           'quality_scores', 'documents_evaluated', 'documents_failed',
                           '{category}_true_count', '{category}_total_count',
                           'hallucination_free_count', 'hallucination_total_count'.
        criteria_mapping: Maps friendly category names (like "factual") to lists of dot-separated paths
                          to boolean criteria within evaluation_data.
                          Example: {"factual": ["factual_criteria.accurate_numbers", "factual_criteria.correct_citations"]}
    """
    if not evaluation_data or "error" in evaluation_data:
        stats_accumulator["documents_failed"] = stats_accumulator.get("documents_failed", 0) + 1
        return

    stats_accumulator["documents_evaluated"] = stats_accumulator.get("documents_evaluated", 0) + 1

    if "quality_score" in evaluation_data:
        if not isinstance(stats_accumulator.get("quality_scores"), list):
            stats_accumulator["quality_scores"] = []
        stats_accumulator["quality_scores"].append(evaluation_data["quality_score"])

    if evaluation_data.get("hallucination_free", False):
        stats_accumulator["hallucination_free_count"] = stats_accumulator.get("hallucination_free_count", 0) + 1
    stats_accumulator["hallucination_total_count"] = stats_accumulator.get("hallucination_total_count", 0) + 1
    
    if criteria_mapping:
        for category, paths in criteria_mapping.items():
            cat_true_key = f"{category}_true_count"
            cat_total_key = f"{category}_total_count"
            
            stats_accumulator.setdefault(cat_true_key, 0)
            stats_accumulator.setdefault(cat_total_key, 0)

            for path_str in paths:
                path_parts = path_str.split('.')
                current_val = evaluation_data
                valid_path = True
                for part in path_parts:
                    if isinstance(current_val, dict) and part in current_val:
                        current_val = current_val[part]
                    else:
                        valid_path = False
                        break
                
                if valid_path and isinstance(current_val, bool):
                    stats_accumulator[cat_total_key] += 1
                    if current_val:
                        stats_accumulator[cat_true_key] += 1
                # else: path not found or not a boolean, so we don't count it.
                # A logger warning could be added here if strictness is required.

def calculate_final_pass_rates(stats_accumulator: Dict[str, Any], categories: List[str]):
    """
    Calculates pass rates for given categories and average quality score.
    Modifies stats_accumulator in-place by adding keys like:
    '{category}_pass_rate', 'avg_quality_score'.

    Args:
        stats_accumulator: The dictionary containing aggregated counts and scores.
        categories: A list of category names (e.g., ["factual", "completeness", "quality"])
                    for which pass rates should be calculated.
    """
    if stats_accumulator.get("documents_evaluated", 0) > 0:
        for category in categories:
            true_count = stats_accumulator.get(f"{category}_true_count", 0)
            total_count = stats_accumulator.get(f"{category}_total_count", 0)
            stats_accumulator[f"{category}_pass_rate"] = (true_count / total_count * 100) if total_count > 0 else 0
        
        if stats_accumulator.get("hallucination_total_count", 0) > 0:
            stats_accumulator["hallucination_free_rate"] = \
                (stats_accumulator.get("hallucination_free_count", 0) / \
                 stats_accumulator["hallucination_total_count"] * 100)
        else:
            stats_accumulator["hallucination_free_rate"] = 0
            
        quality_scores = stats_accumulator.get("quality_scores", [])
        if isinstance(quality_scores, list) and quality_scores:
            stats_accumulator["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
        else:
            stats_accumulator["avg_quality_score"] = 0
    else: # No documents evaluated
        for category in categories:
            stats_accumulator[f"{category}_pass_rate"] = 0
        stats_accumulator["hallucination_free_rate"] = 0
        stats_accumulator["avg_quality_score"] = 0

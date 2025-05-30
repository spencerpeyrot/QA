from motor.motor_asyncio import AsyncIOMotorClient
import json
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import pandas_market_calendars as mcal
import asyncio
from itertools import islice
from bson import ObjectId
import os

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY_BACKUP1=os.getenv("OPENAI_API_KEY_BACKUP1")
OPENAI_API_KEY_BACKUP2=os.getenv("OPENAI_API_KEY_BACKUP2")
MONGO_URI=os.getenv("EVAL_MONGO_URI")

OPENAI_API_KEYS = [k for k in (OPENAI_API_KEY, OPENAI_API_KEY_BACKUP1, OPENAI_API_KEY_BACKUP2) if k]

ltv_keys_macro_context = [
    "US_Growth",
    "US_Consumer Defensive",
    "US_Healthcare",
    "US_Utilities",
    "US_Geopolitics",
    "US_Real Estate",
    "US_Inflation",
    "US_Policy",
    "US_Jobs",
    "US_Technology", 
    "US_Energy",
    "US_Industrials",
    "US_Consumer Sentiment",
    "US_Basic Materials",
    "US_Communication Services",
    "US_Consumer Cyclical",
    "US_Financial Services"
]

evaluation_schema = {
    "name": "evaluate_response",
    "description": "Evaluate a financial report response against source documents and for overall quality",
    "parameters": {
        "type": "object",
        "properties": {
            "factual_criteria": {
                "type": "object",
                "properties": {
                    "accurate_numbers": {
                        "type": "boolean",
                        "description": "True if all data points such as percentages, trends, and numbers in the response are accurate and match the content either the MACRO RELEASES or the SOURCE DOCUMENTS, depending on where it was sourced from. Return True if not applicable."
                    },
                    "correct_citations": {
                        "type": "boolean",
                        "description": "True if references to sources [1], [2], etc. correctly match the content from the SOURCE DOCUMENTS dictionary citation number (keys) and the content (values) from the SOURCE DOCUMENTS dictionary. Feel free to ignore references to the MACRO RELEASES and [Background Information]: they are not part of this evaluation."
                    },
                },
                "required": ["accurate_numbers", "correct_citations"]
            },
            "completeness_criteria": {
                "type": "object",
                "properties": {
                    "covers_macro_context": {
                        "type": "boolean",
                        "description": "True if the report captures key macro-economic or sector-wide trends, challenges, and opportunities from the SOURCE DOCUMENTS, if provided. It is not required to cover the MACRO RELEASES section, only if relevant."
                    },
                    "includes_context": {
                        "type": "boolean",
                        "description": "True if sufficient context is provided to relevant information and trends about the macro-economic or sector-wide trends, in about 3-5 bullet points from diverse sources if provided in the SOURCE DOCUMENTS. No need to cover all the sources, just a small selection that provide relevant context. It is not required to cover the MACRO RELEASES section, only if relevant."
                    }
                },
                "required": ["covers_macro_context", "includes_context"]
            },
            "quality_criteria": {
                "type": "object",
                "properties": {
                    "clear_presentation": {
                        "type": "boolean",
                        "description": "True if information is logically organized and clear"
                    },
                    "explains_causes": {
                        "type": "boolean",
                        "description": "True if it explains drivers behind stock movements"
                    }
                },
                "required": ["clear_presentation", "explains_causes"]
            },
            "hallucination_free": {
                "type": "boolean",
                "description": "False if there are any particular details in this report unsupported by the the material in the MACRO RELEASES or the SOURCE DOCUMENTS. Note that common market knowledge, straightforward inferences, and minor editorial liberties are allowed and thus does not count as hallucination. Thus, mark True if the response is a natural extension of the sources, or if it is a common-sense interpretation of the sources."
            },
            "quality_score": {
                "type": "integer",
                "description": "Overall quality score from 0-100. Assume that you're a finance professional, and evaluate the response based on the factual accuracy, completeness, and quality criteria, and the overall quality and content.",
                "minimum": 0,
                "maximum": 100
            },
            "criteria_explanations": {
                "type": "object",
                "description": "Explanations for each criterion evaluation",
                "properties": {
                    "accurate_numbers": {"type": "string"},
                    "correct_citations": {"type": "string"},
                    "covers_macro_context": {"type": "string"},
                    "includes_context": {"type": "string"},
                    "clear_presentation": {"type": "string"},
                    "explains_causes": {"type": "string"},
                    "hallucination_free": {"type": "string"},
                }
            },
            "unsupported_claims": {
                "type": "array",
                "description": "List of statements not supported by source documents",
                "items": {"type": "string"}
            },
            "score_explanation": {
                "type": "string",
                "description": "Explanation for the overall quality score"
            }
        },
        "required": ["factual_criteria", "completeness_criteria", "quality_criteria", 
                    "hallucination_free", "quality_score", "criteria_explanations", 
                    "unsupported_claims", "score_explanation"]
    }
}

def get_async_mongo_client():
    uri = MONGO_URI 
    client = AsyncIOMotorClient(
        uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        minPoolSize=2,
        connectTimeoutMS=0
    )
    print("New connection to Eval MongoDB established!")
    return client


# Configuration
EVAL_SAMPLE_SIZE = 25  # Number of documents to sample per day
OPENAI_MODEL = "o4-mini"  # Or "claude-3-opus" if using Anthropic
MONGODB_EVAL_COLLECTION = "evaluations"  # Where to store evaluation results

class MacroLTVEvaluator:
    def __init__(self, mongo_uri: str, daily_sample_size: int = EVAL_SAMPLE_SIZE):
        self.client = AsyncIOMotorClient(
            mongo_uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            minPoolSize=2,
            connectTimeoutMS=0
        )
        self.source_db = self.client['asc-fin-data']
        self.source_collection = self.source_db['user_activities']
        self.eval_collection = self.source_db[MONGODB_EVAL_COLLECTION]
        self.daily_sample_size = daily_sample_size
        
    async def sample_documents(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Sample random documents from a specific date range."""
        # Using aggregation pipeline to find documents with 3+ source_docs
        pipeline = [
            {
                "$match": {
                    "agent": "neo",
                    "mode": "LTV",
                    "timestamp": {"$gte": start_date, "$lt": end_date},
                    "agent_sources": {"$exists": True},
                    "user_question": {"$in": ltv_keys_macro_context}
                }
            },
            {
                "$addFields": {
                    "source_docs_count": {"$size": {"$ifNull": [{"$objectToArray": "$agent_sources"}, []]}}
                }
            },
            {
                "$match": {
                    "source_docs_count": {"$gte": 3}
                }
            }
        ]

        all_docs = await self.source_collection.aggregate(pipeline).to_list(length=None)
        
        # Get all document IDs
        
        # Random sampling
        if len(all_docs) <= self.daily_sample_size:
            return all_docs
        
        sampled_docs = random.sample(all_docs, self.daily_sample_size)
        return sampled_docs

    async def evaluate_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Submit document to LLM for evaluation using function calling."""
        # Define the function schema

        # Create user prompt with context

        agent_response = doc.get("agent_response", "")
        user_question = doc.get("user_question", "")
        agent_sources = doc.get("agent_sources", {})
        print(f'Processing ltv output for {user_question}')
        
        doc_timestamp = doc.get("timestamp")

        catalyst_doc = await self.get_nearest_catalyst_document(doc_timestamp)
        catalyst_content = ""
        if catalyst_doc:
            catalyst_response = catalyst_doc.get("agent_response", "")
            # catalyst_timestamp = catalyst_doc.get("timestamp")
            # time_diff = doc_timestamp - catalyst_timestamp if doc_timestamp and catalyst_timestamp else None
            # time_diff_str = f"{time_diff.total_seconds() / 3600:.1f} hours" if time_diff else "unknown time"
            
            catalyst_content = f"""
    RELATED CATALYST DOCUMENT:
    {catalyst_response}
    """

        # Extract and format source documents
        source_texts = []
        for key, value in agent_sources.items():
            metadata_text = ""
            
            # Add metadata if available
            if "metadata" in value:
                metadata = value["metadata"]
                
                if "title" in metadata:
                    metadata_text += f"Title: {metadata['title']}\n"
                if "source" in metadata:
                    metadata_text += f"Source: {metadata['source']}\n"
                if "url" in metadata:
                    metadata_text += f"URL: {metadata['url']}\n"
                if "description" in metadata and metadata["description"]:
                    metadata_text += f"Description: {metadata['description']}\n"
                if "summary" in metadata and metadata["summary"]:
                    metadata_text += f"Summary: {metadata['summary']}\n"
            
            # Add the source text with metadata
            if "text" in value:
                source_text = f"Source [{key}]:\n{metadata_text}\n{value['text']}"
                source_texts.append(source_text)

        source_docs_formatted = "\n\n" + "\n\n".join(source_texts)
        
        user_prompt = f"""
    Evaluate this financial resport on a macro-theme or sector against the source documents:

    MACRO-THEME OR SECTOR:
    {user_question}

    MACRO-THEME OR SECTOR RESPONSE:
    {agent_response}

    MACRO RELEASES:
    {catalyst_content}

    SOURCE DOCUMENTS:
    {source_docs_formatted}

    Evaluate each criterion as TRUE or FALSE with brief explanations.
    Calculate a quality score from 0-100 based on these criteria.
    List any statements not supported by the sources.
    """

        system_prompt = """You are an expert financial evaluator. Assess financial macro-theme or sector responses against source documents using TRUE/FALSE criteria. Be objective and thorough in your evaluation."""
        
        # print(user_prompt)
        # print(system_prompt)

        try:
            # Initialize client
            client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
            
            # Make the function call
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": evaluation_schema}],
                tool_choice={"type": "function", "function": {"name": "evaluate_response"}},
                # temperature=0.1,
            )
            
            # Extract function call response
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                function_response = json.loads(tool_calls[0].function.arguments)
                # print(f'Eval for doc {doc.get("_id", "")} is {function_response}')

                # Return the evaluation result
                return {
                    "document_id": str(doc.get("_id", "")),
                    "user_question": user_question,
                    "timestamp": doc.get("timestamp"),
                    "evaluation": function_response,
                    "evaluated_at": datetime.now(timezone.utc)
                }
            else:
                return {
                    "document_id": str(doc.get("_id", "")),
                    "user_question": user_question,
                    "timestamp": doc.get("timestamp"),
                    "evaluation": {"error": "No function response received"},
                    "evaluated_at": datetime.now(timezone.utc)
                }
                
        except Exception as e:
            # Handle errors
            return {
                "document_id": str(doc.get("_id", "")),
                "user_question": user_question,
                "timestamp": doc.get("timestamp"),
                "evaluation": {"error": f"Evaluation failed: {str(e)}"},
                "evaluated_at": datetime.now(timezone.utc)
            }



    async def process_date_range(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Process all documents within a date range."""
        results = {
            # Category pass rates
            "factual_accuracy_pass_rate": 0,
            "completeness_pass_rate": 0, 
            "quality_usefulness_pass_rate": 0,
            "hallucination_free_rate": 0,
            
            # Overall scores
            "quality_scores": [],
            "documents_evaluated": 0,
            "documents_failed": 0,
            "days_skipped": 0,
            
            # Counters
            "factual_true_count": 0,
            "factual_total_count": 0,
            "completeness_true_count": 0,
            "completeness_total_count": 0,
            "quality_true_count": 0,
            "quality_total_count": 0,
            "hallucination_free_count": 0,
            "hallucination_total_count": 0
        }
        
        # Get NYSE calendar for market day checks
        nyse = mcal.get_calendar('NYSE')
        market_days = nyse.valid_days(start_date, end_date)
        market_days_set = set(d.date() for d in market_days)
        
        current_date = start_date
        while current_date < end_date:
            next_date = current_date + timedelta(days=1)
            
            # Check if it's a market day
            if current_date.date() not in market_days_set:
                print(f"Skipping {current_date.date()} - not a market day (weekend or holiday)")
                results["days_skipped"] += 1
                current_date = next_date
                continue
            
            print(f"Processing market day: {current_date.date()}")
            sampled_docs = await self.sample_documents(current_date, next_date)
            print(f"Sampled {len(sampled_docs)} documents")
            
            # Define an async function to process each document
            async def process_document(doc):
                try:
                    # Evaluate document
                    evaluation = await self.evaluate_document(doc)
                    
                    # Add pipeline name
                    evaluation["pipeline"] = "LTV"

                    # if score is based on 0-5, multiply by 20 to convert to 0-100
                    if "quality_score" in evaluation["evaluation"] and evaluation["evaluation"]["quality_score"] <= 5:
                        evaluation["evaluation"]["quality_score"] = evaluation["evaluation"]["quality_score"] * 20
                    
                    # Store in MongoDB
                    try:
                        await self.eval_collection.insert_one(evaluation)
                    except Exception as mongo_err:
                        print(f"MongoDB insertion error: {mongo_err}")
                    
                    return evaluation
                except Exception as e:
                    print(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                    return {"error": str(e)}
            
            # Process documents in batches of 10
            all_evaluations = []
            for i in range(0, len(sampled_docs), 10):
                batch = sampled_docs[i:i+10]
                print(f"Processing batch {i//10 + 1}/{(len(sampled_docs)+9)//10} ({len(batch)} documents)")
                
                # Process batch with progress bar
                batch_evaluations = await tqdm_asyncio.gather(
                    *[process_document(doc) for doc in batch],
                    desc=f"Batch {i//10 + 1}"
                )
                
                all_evaluations.extend(batch_evaluations)
                
                # Add a delay between batches to avoid rate limits
                if i + 10 < len(sampled_docs):
                    print("Pausing between batches to respect rate limits...")
                    await asyncio.sleep(2)  # 2 second pause between batches
            
            # Process results
            for evaluation in all_evaluations:
                if "error" in evaluation:
                    results["documents_failed"] += 1
                    continue
                    
                eval_data = evaluation.get("evaluation", {})
                if not eval_data:
                    results["documents_failed"] += 1
                    continue
                
                # Count factual criteria
                factual_criteria = eval_data.get("factual_criteria", {})
                factual_true = sum(1 for _, v in factual_criteria.items() if v)
                factual_total = len(factual_criteria)
                
                # Count completeness criteria
                completeness_criteria = eval_data.get("completeness_criteria", {})
                completeness_true = sum(1 for _, v in completeness_criteria.items() if v)
                completeness_total = len(completeness_criteria)
                
                # Count quality criteria
                quality_criteria = eval_data.get("quality_criteria", {})
                quality_true = sum(1 for _, v in quality_criteria.items() if v)
                quality_total = len(quality_criteria)
                
                # Get quality score
                if "quality_score" in eval_data:
                    results["quality_scores"].append(eval_data["quality_score"])
                
                # Track hallucination
                hallucination_free = eval_data.get("hallucination_free", False)
                if hallucination_free:
                    results["hallucination_free_count"] += 1
                
                # Add to totals
                results["factual_true_count"] += factual_true
                results["factual_total_count"] += factual_total
                
                results["completeness_true_count"] += completeness_true
                results["completeness_total_count"] += completeness_total
                
                results["quality_true_count"] += quality_true
                results["quality_total_count"] += quality_total
                
                results["hallucination_total_count"] += 1
                
                results["documents_evaluated"] += 1
            
            current_date = next_date
        
        # Calculate final rates
        if results["documents_evaluated"] > 0:
            results["factual_accuracy_pass_rate"] = (results["factual_true_count"] / results["factual_total_count"]) * 100 if results["factual_total_count"] > 0 else 0
            results["completeness_pass_rate"] = (results["completeness_true_count"] / results["completeness_total_count"]) * 100 if results["completeness_total_count"] > 0 else 0
            results["quality_usefulness_pass_rate"] = (results["quality_true_count"] / results["quality_total_count"]) * 100 if results["quality_total_count"] > 0 else 0
            results["hallucination_free_rate"] = (results["hallucination_free_count"] / results["hallucination_total_count"]) * 100 if results["hallucination_total_count"] > 0 else 0
            results["avg_quality_score"] = sum(results["quality_scores"]) / len(results["quality_scores"]) if results["quality_scores"] else 0
        
        # Add market day statistics
        results["market_days_processed"] = results.get("days_processed", 0)
        results["market_days_skipped"] = results.get("days_skipped", 0)
        
        return results

    async def get_nearest_catalyst_document(self, doc_timestamp: datetime) -> Dict[str, Any]:
        """Find the most recent catalysts_macro_US_LTV document with timestamp not exceeding doc_timestamp."""
        try:
            one_day_prior = doc_timestamp - timedelta(days=1)

            pipeline = [
                {
                    "$match": {
                        "mode": "catalysts_macro_US_LTV",
                        "timestamp": {
                            "$lte": doc_timestamp,
                            "$gte": one_day_prior
                        },
                        "agent_response": {"$exists": True}
                    }
                },
                {
                    "$sort": {
                        "timestamp": -1  # Sort by timestamp descending (most recent first)
                    }
                },
                {
                    "$limit": 1  # Only get the most recent document
                }
            ]
            
            # Execute the aggregation pipeline
            cursor = self.source_collection.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            # Return the document if found, otherwise None
            return results[0] if results else None
        except Exception as e:
            print(f"Error fetching catalyst document: {e}")
            return None

    async def find_failed_hallucination_checks(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Find evaluations that failed the hallucination check."""
        pipeline = [
            {
                "$match": {
                    "pipeline": "LTV",
                    "timestamp": {"$gte": start_date, "$lt": end_date},
                    "evaluation.hallucination_free": False,
                    "corrected": {"$exists": False}  # Only get uncorrected documents
                }
            }
        ]
        
        return await self.eval_collection.aggregate(pipeline).to_list(length=None)
    
    async def correct_document(self, evaluation_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a corrected version of the document based on evaluation feedback."""
        # Get the original document from user_activities collection
        doc_id = evaluation_doc.get("document_id")
        original_doc = await self.source_collection.find_one({"_id": ObjectId(doc_id)})
        
        if not original_doc:
            return {"error": f"Original document not found: {doc_id}"}
        
        # Format based on document mode
        return await self._correct_doc(original_doc, evaluation_doc)

    async def _correct_doc(self, original_doc: Dict[str, Any], evaluation_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Correct an LTV document."""
        agent_response = original_doc.get("agent_response", "")
        user_question = original_doc.get("user_question", "")
        agent_sources = original_doc.get("agent_sources", {})
        doc_timestamp = original_doc.get("timestamp")
        
        # Get the catalyst document as in the original evaluation
        catalyst_doc = await self.get_nearest_catalyst_document(doc_timestamp)
        catalyst_content = ""
        if catalyst_doc:
            catalyst_response = catalyst_doc.get("agent_response", "")
            catalyst_content = f"""
    RELATED CATALYST DOCUMENT:
    {catalyst_response}
    """
        
        # Extract evaluation feedback
        evaluation = evaluation_doc.get("evaluation", {})
        unsupported_claims = evaluation.get("unsupported_claims", [])
        
        # Build specific correction instructions based on failing criteria
        correction_instructions = []
        
        # Check hallucinations
        if not evaluation.get("hallucination_free", True):
            hallucination_explanation = evaluation.get("criteria_explanations", {}).get("hallucination_free", "")
            correction_instructions.append(f"HALLUCINATION ISSUES: {hallucination_explanation}")
            if unsupported_claims:
                correction_instructions.append("Unsupported claims to remove or correct:")
                for claim in unsupported_claims:
                    correction_instructions.append(f"- {claim}")
        
        # Check factual accuracy issues
        factual_criteria = evaluation.get("factual_criteria", {})
        if not factual_criteria.get("accurate_numbers", True):
            accuracy_explanation = evaluation.get("criteria_explanations", {}).get("accurate_numbers", "")
            correction_instructions.append(f"NUMBER ACCURACY ISSUE: {accuracy_explanation}")
            correction_instructions.append("- Ensure all data points, percentages, trends, and numbers are accurate and match the content in the sources")
        
        if not factual_criteria.get("correct_citations", True):
            citation_explanation = evaluation.get("criteria_explanations", {}).get("correct_citations", "")
            correction_instructions.append(f"CITATION ISSUE: {citation_explanation}")
            correction_instructions.append("- Ensure references to sources [1], [2], etc. correctly match the source documents")
        
        # Check completeness issues
        completeness_criteria = evaluation.get("completeness_criteria", {})
        if not completeness_criteria.get("covers_macro_context", True):
            context_explanation = evaluation.get("criteria_explanations", {}).get("covers_macro_context", "")
            correction_instructions.append(f"MACRO CONTEXT COVERAGE ISSUE: {context_explanation}")
            correction_instructions.append("- Ensure the report captures key macro-economic or sector-wide trends, challenges, and opportunities")
        
        if not completeness_criteria.get("includes_context", True):
            context_explanation = evaluation.get("criteria_explanations", {}).get("includes_context", "")
            correction_instructions.append(f"CONTEXT ISSUE: {context_explanation}")
            correction_instructions.append("- Add sufficient context about macro-economic or sector-wide trends from diverse sources")
        
        # Format source documents
        source_texts = []
        for key, value in agent_sources.items():
            metadata_text = ""
            
            # Add metadata if available
            if "metadata" in value:
                metadata = value["metadata"]
                
                if "title" in metadata:
                    metadata_text += f"Title: {metadata['title']}\n"
                if "source" in metadata:
                    metadata_text += f"Source: {metadata['source']}\n"
                if "url" in metadata:
                    metadata_text += f"URL: {metadata['url']}\n"
                if "description" in metadata and metadata["description"]:
                    metadata_text += f"Description: {metadata['description']}\n"
                if "summary" in metadata and metadata["summary"]:
                    metadata_text += f"Summary: {metadata['summary']}\n"
            
            # Add the source text with metadata
            if "text" in value:
                source_text = f"Source [{key}]:\n{metadata_text}\n{value['text']}"
                source_texts.append(source_text)

        source_docs_formatted = "\n\n" + "\n\n".join(source_texts)
        
        # Join all correction instructions
        correction_instructions_text = "\n".join(correction_instructions)
        
        # Create correction prompt
        system_prompt = """You are an expert financial content corrector. Your task is to correct hallucinations and other issues in financial macro-theme or sector reports while preserving accurate information. Make minimal changes necessary to fix the identified issues. Keep the tone, structure, and length similar to the original."""
        
        user_prompt = f"""
    Please correct the following financial report on a macro-theme or sector that contains specific issues identified by our evaluation system.

    MACRO-THEME OR SECTOR:
    {user_question}

    ORIGINAL REPORT:
    {agent_response}

    MACRO RELEASES:
    {catalyst_content}

    SOURCE DOCUMENTS:
    {source_docs_formatted}

    EXCEPTIONS RAISED BY THE EVALUATION:
    {correction_instructions_text}

    INSTRUCTIONS:
    1. Correct only the problematic content highlighted by EXCEPTIONS RAISED BY THE EVALUATION while preserving accurate information from the original report
    2. For hallucinations and accuracy issues: remove or correct unsupported claims and add the missing information from RELEVANT STATS or SOURCE DOCUMENTS
    3. For incorrect citations or misrepresentations of source documents: add the correct citations from SOURCE DOCUMENTS and make sure the bullet points are faithful to the cited source
    4. For completeness issues: add the missing information from RELEVANT STATS or context from SOURCE DOCUMENTS
    5. Maintain the same overall structure, tone, and length where corrections are not needed.
    6. Return ONLY the corrected report, with no explanations or commentary

    CORRECTED REPORT:
    """

        # print(user_prompt)

        try:
            # Initialize client
            client = AsyncOpenAI(api_key=random.choice(OPENAI_API_KEYS))
            
            # Make the API call
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            corrected_content = response.choices[0].message.content
            
            # Return the corrected document
            return {
                "document_id": str(original_doc.get("_id", "")),
                "user_question": user_question,
                "original_response": agent_response,
                "corrected_response": corrected_content,
                "corrected_at": datetime.now(timezone.utc)
            }
                
        except Exception as e:
            # Handle errors
            return {
                "document_id": str(original_doc.get("_id", "")),
                "user_question": user_question,
                "error": f"Correction failed: {str(e)}"
            }

    async def update_original_document(self, correction_result: Dict[str, Any]) -> bool:
        """Update the original document with the corrected response."""
        if "error" in correction_result:
            return False
            
        doc_id = correction_result.get("document_id")
        corrected_response = correction_result.get("corrected_response")
        
        try:
            update_result = await self.source_collection.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {
                    "corrected_agent_response": corrected_response,
                    "corrected_timestamp": datetime.now(timezone.utc),
                    "corrected": True,
                }}
            )

            return update_result.modified_count > 0
            
        except Exception as e:
            print(f"Error updating original document: {e}")
            return False

    async def re_evaluate_corrected_document(self, correction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Re-evaluate the corrected document."""
        if "error" in correction_result:
            return {"error": correction_result["error"]}
            
        doc_id = correction_result.get("document_id")
        
        try:
            original_doc = await self.source_collection.find_one({"_id": ObjectId(doc_id)})
            
            if not original_doc:
                return {"error": f"Document not found after correction: {doc_id}"}
            
            # Create a copy of the document for re-evaluation with the corrected content
            eval_doc = original_doc.copy()
            
            # For LTV, use the corrected_agent_response field instead of original agent_response
            if "corrected_agent_response" in eval_doc:
                eval_doc["agent_response"] = eval_doc["corrected_agent_response"]
            
            # Re-evaluate the corrected document
            evaluation = await self.evaluate_document(eval_doc)
            
            # Mark as corrected evaluation
            evaluation["is_correction"] = True
            evaluation["original_document_id"] = str(doc_id)
            
            return evaluation
            
        except Exception as e:
            return {"error": f"Re-evaluation failed: {str(e)}"}

    async def update_evaluation_document(self, eval_id: str, 
                                    re_evaluation: Dict[str, Any]) -> bool:
        """Update the evaluation document with the corrected evaluation."""
        try:
            update_result = await self.eval_collection.update_one(
                {"_id": ObjectId(eval_id)},
                {"$set": {
                    "corrected": True,
                    "corrected_evaluation": re_evaluation.get("evaluation"),
                    "corrected_evaluated_at": re_evaluation.get("evaluated_at")
                }}
            )
            
            return update_result.modified_count > 0
            
        except Exception as e:
            print(f"Error updating evaluation document: {e}")
            return False

    async def process_corrections(self, start_date: datetime, end_date: datetime):
        """Process corrections for documents that failed hallucination checks in parallel batches."""
        # Get failed LTV evaluations
        failures = await self.find_failed_hallucination_checks(start_date, end_date)
        print(f"Found {len(failures)} LTV documents requiring correction")
        
        if not failures:
            return {"corrections_processed": 0}
        
        # Define an async function to process each document
        async def process_correction(eval_doc):
            try:
                # Correct document
                correction = await self.correct_document(eval_doc)
                
                if "error" in correction:
                    print(f"Correction failed: {correction['error']}")
                    return {"status": "failed", "reason": "correction_failed", "document_id": eval_doc.get("document_id")}
                    
                # Update original document
                update_success = await self.update_original_document(correction)
                if not update_success:
                    print(f"Failed to update original document: {correction['document_id']}")
                    return {"status": "failed", "reason": "update_failed", "document_id": correction['document_id']}
                
                # Re-evaluate
                re_evaluation = await self.re_evaluate_corrected_document(correction)
                
                if "error" in re_evaluation:
                    print(f"Re-evaluation failed: {re_evaluation['error']}")
                    return {"status": "failed", "reason": "re_evaluation_failed", "document_id": correction['document_id']}
                
                # Update evaluation
                eval_update_success = await self.update_evaluation_document(
                    eval_doc["_id"], re_evaluation)
                
                if eval_update_success:
                    print(f"Successfully corrected and re-evaluated document: {correction['document_id']}")
                    return {"status": "success", "document_id": correction['document_id']}
                else:
                    return {"status": "failed", "reason": "eval_update_failed", "document_id": correction['document_id']}
                    
            except Exception as e:
                print(f"Error processing correction for document {eval_doc.get('document_id', 'unknown')}: {e}")
                return {"status": "failed", "reason": str(e), "document_id": eval_doc.get("document_id")}
        
        # Process documents in batches of 6
        all_results = []
        for i in range(0, len(failures), 6):
            batch = failures[i:i+6]
            print(f"Processing correction batch {i//6 + 1}/{(len(failures)+5)//6} ({len(batch)} documents)")
            
            # Process batch with progress bar
            batch_results = await tqdm_asyncio.gather(
                *[process_correction(eval_doc) for eval_doc in batch],
                desc=f"Correction Batch {i//6 + 1}"
            )
            
            all_results.extend(batch_results)
            
            # Add a delay between batches to avoid rate limits
            if i + 6 < len(failures):
                print("Pausing between correction batches to respect rate limits...")
                await asyncio.sleep(2)  # 2 second pause between batches
        
        # Summarize results
        successes = sum(1 for r in all_results if r.get("status") == "success")
        
        print(f"Correction batch completed. {len(failures)} documents processed. {successes} corrections succeeded. {len(failures) - successes} corrections failed.")
        print(f"Updated doc Ids: {', '.join([r.get('document_id') for r in all_results if r.get('status') == 'success'])}")
        return {
            "corrections_processed": len(failures),
            "corrections_succeeded": successes,
            "corrections_failed": len(failures) - successes
        }


async def main():
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=3)

    evaluator = MacroLTVEvaluator(MONGO_URI, daily_sample_size=EVAL_SAMPLE_SIZE)
    summary   = await evaluator.process_date_range(start_dt, end_dt)
    print("Evaluation summary:", summary)

if __name__ == "__main__":
    asyncio.run(main())
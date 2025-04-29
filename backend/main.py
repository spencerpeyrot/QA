from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
from datetime import datetime, timedelta, timezone
import json
from prompt_manager import PromptManager
from bson import ObjectId
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (you'll need to create a .env file)
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="QA Platform API")

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB clients
EVAL_MONGO_URI = os.getenv('EVAL_MONGO_URI')
if not EVAL_MONGO_URI:
    raise ValueError("EVAL_MONGO_URI environment variable is not set")

def get_async_mongo_client():
    client = AsyncIOMotorClient(EVAL_MONGO_URI, tls=True, tlsAllowInvalidCertificates=True, minPoolSize=2, connectTimeoutMS=0)
    return client

# Initialize database connections
qa_mongo_client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
automation_mongo_client = get_async_mongo_client()

qa_db = qa_mongo_client.qa_platform
qa_evaluations = qa_db.qa_evaluations

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize PromptManager
prompt_manager = PromptManager()

# Pydantic models
class QARequest(BaseModel):
    agent: str
    sub_component: Optional[str] = None
    variables: Dict[str, Any]

class QAPassUpdate(BaseModel):
    qa_rating: bool

class ReportPassUpdate(BaseModel):
    report_rating: int

# New Pydantic models for evaluation stats
class EvaluationMetrics(BaseModel):
    factualAccuracyRate: float
    completenessRate: float
    qualityUsefulnessRate: float
    hallucinationFreeRate: float
    averageQualityScore: float
    totalDocumentsEvaluated: int
    documentsRequiringCorrection: int

class AgentMetrics(BaseModel):
    successRate: float
    averageRunTime: float
    totalRuns: int
    failureRate: float
    lastWeekTrend: str
    commonErrors: List[str]
    ltvMetrics: Optional[EvaluationMetrics] = None
    tickerPulseMetrics: Optional[EvaluationMetrics] = None

@app.get("/")
async def root():
    return {"message": "QA Platform API is running"}

@app.post("/qa")
async def create_qa_evaluation(request: QARequest):
    try:
        # Add current date
        request.variables["current_date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Validate variables
        missing_vars = prompt_manager.validate_variables(
            request.agent,
            request.sub_component,
            request.variables
        )
        if missing_vars:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required variables: {', '.join(missing_vars)}"
            )
        
        # Load and format prompt template
        template = prompt_manager.load_prompt(request.agent, request.sub_component)
        formatted_prompt = prompt_manager.format_prompt(template, request.variables)
        
        # Call OpenAI
        response = await openai_client.chat.completions.create(
            model=os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            messages=[
                {"role": "system", "content": "You are a financial analysis QA expert. Your task is to evaluate the quality and accuracy of financial analysis reports."},
                {"role": "user", "content": formatted_prompt}
            ],
            stream=True
        )
        
        # Stream and collect response
        collected_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                collected_response += chunk.choices[0].delta.content
        
        # Create document with timestamp logging
        created_at = datetime.utcnow()
        current_utc = datetime.utcnow()
        logger.info(f"Current UTC time: {current_utc.isoformat()}")
        logger.info(f"Creating QA evaluation with UTC timestamp: {created_at.isoformat()}")
        logger.info(f"Timestamp components - Year: {created_at.year}, Month: {created_at.month}, Day: {created_at.day}, Hour: {created_at.hour}, Minute: {created_at.minute}")
        
        doc = {
            "agent": request.agent,
            "sub_component": request.sub_component,
            "variables": request.variables,
            "injected_date": request.variables["current_date"],
            "openai_model": os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            "response_markdown": collected_response,
            "qa_pass": None,
            "report_pass": None,
            "created_at": created_at
        }
        
        result = await qa_evaluations.insert_one(doc)
        stored_doc = await qa_evaluations.find_one({"_id": result.inserted_id})
        logger.info(f"Stored document timestamp: {stored_doc['created_at'].isoformat() if stored_doc else 'Not found'}")
        
        return {
            "id": str(result.inserted_id),
            "markdown": collected_response
        }
        
    except Exception as e:
        logger.error(f"Error creating QA evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/qa/{qa_id}/qa_pass")
async def update_qa_pass(qa_id: str, update: QAPassUpdate):
    try:
        # Convert string ID to MongoDB ObjectId
        object_id = ObjectId(qa_id)
        result = await qa_evaluations.update_one(
            {"_id": object_id},
            {"$set": {
                "qa_rating": update.qa_rating,
                "qa_pass": update.qa_rating
            }}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/qa/{qa_id}/report_pass")
async def update_report_pass(qa_id: str, update: ReportPassUpdate):
    try:
        # Convert string ID to MongoDB ObjectId
        object_id = ObjectId(qa_id)
        # Ensure rating is between 1 and 5
        rating = max(1, min(5, update.report_rating))
        result = await qa_evaluations.update_one(
            {"_id": object_id},
            {"$set": {
                "report_rating": rating,
                "report_pass": rating >= 3
            }}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa/{qa_id}")
async def get_qa_evaluation(qa_id: str):
    try:
        # Convert string ID to MongoDB ObjectId
        object_id = ObjectId(qa_id)
        doc = await qa_evaluations.find_one({"_id": object_id})
        if not doc:
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        logger.info(f"Retrieved QA evaluation {qa_id} with timestamp: {doc['created_at'].isoformat()}")
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        return doc
    except Exception as e:
        logger.error(f"Error retrieving QA evaluation {qa_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa")
async def list_qa_evaluations(limit: int = 20):
    try:
        cursor = qa_evaluations.find().sort("created_at", -1).limit(limit)
        evaluations = []
        async for doc in cursor:
            logger.info(f"Listed QA evaluation {doc['_id']} with timestamp: {doc['created_at'].isoformat()}")
            doc["_id"] = str(doc["_id"])
            evaluations.append(doc)
        return evaluations
    except Exception as e:
        logger.error(f"Error listing QA evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/qa/{qa_id}")
async def delete_qa_evaluation(qa_id: str):
    try:
        object_id = ObjectId(qa_id)
        result = await qa_evaluations.delete_one({"_id": object_id})
        if result.deleted_count == 0:
            logger.warning(f"Attempted to delete non-existent QA evaluation: {qa_id}")
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        logger.info(f"Deleted QA evaluation: {qa_id}")
        return {"status": "deleted", "deleted_id": qa_id}
    except Exception as e:
        logger.error(f"Error deleting QA evaluation {qa_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluations/ltv")
async def get_ltv_evaluations():
    try:
        # Use the automation MongoDB client
        eval_db = automation_mongo_client['asc-fin-data']
        eval_collection = eval_db['evaluations']
        
        logger.info("Connected to asc-fin-data database")
        
        # Find all LTV evaluations, sorted by timestamp
        cursor = eval_collection.find(
            {"pipeline": "LTV"},
            {
                "document_id": 1,
                "user_question": 1,
                "timestamp": 1,
                "evaluated_at": 1,
                "evaluation": {
                    "factual_criteria": 1,
                    "completeness_criteria": 1,
                    "quality_criteria": 1,
                    "hallucination_free": 1,
                    "quality_score": 1,
                    "criteria_explanations": 1,
                    "unsupported_claims": 1,
                    "score_explanation": 1
                }
            }
        ).sort("timestamp", -1)
        
        # Convert cursor to list and format for JSON response
        evaluations = []
        async for doc in cursor:
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            evaluations.append(doc)
            
        logger.info(f"Found {len(evaluations)} LTV evaluations")
        if len(evaluations) > 0:
            logger.info(f"First evaluation: {evaluations[0]}")
        
        return evaluations
        
    except Exception as e:
        logger.error(f"Error fetching LTV evaluations: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluations/ltv/test")
async def create_test_ltv_evaluation():
    try:
        # Connect to the asc-fin-data database for evaluations
        eval_db = automation_mongo_client['asc-fin-data']
        eval_collection = eval_db['evaluations']
        
        # Create a test document based on the example
        test_doc = {
            "document_id": "680a7182a26c8e99cf8566e5",
            "user_question": "US_Policy",
            "timestamp": datetime.now(timezone.utc),
            "evaluation": {
                "factual_criteria": {
                    "accurate_numbers": True,
                    "correct_citations": True
                },
                "completeness_criteria": {
                    "covers_macro_context": True,
                    "includes_context": True
                },
                "quality_criteria": {
                    "clear_presentation": True,
                    "explains_causes": True
                },
                "hallucination_free": True,
                "quality_score": 95,
                "criteria_explanations": {
                    "accurate_numbers": "All numerical values (inflation at 2.3%, unemployment at 4.2%, rates at 4.25%-4.50%, yields >4.9%, deficit $1.3T, tax plan $4.5T) match source figures.",
                    "correct_citations": "Citations [1], [2], [10], [11], [6], [7], [4], [5] align correctly with the referenced source content.",
                    "covers_macro_context": "Key themes (Fed dilemma, tariffs, global divergence, political pressure, market and fiscal impacts, global spillovers) reflect the major topics in the source documents.",
                    "includes_context": "Provides cohesive context in 6 bullet points drawn from multiple sources, covering main trends and drivers.",
                    "clear_presentation": "Information is structured in clear bullet points with headings and logical flow.",
                    "explains_causes": "Each market movement is linked to drivers such as tariffs, policy divergence, political attacks and fiscal deficits.",
                    "hallucination_free": "All statements are supported by the provided sources, with no extraneous claims."
                },
                "unsupported_claims": [],
                "score_explanation": "The response accurately and comprehensively synthesizes information from multiple source documents, uses correct data and citations, and presents clear, logical analysis of drivers behind policy and market trends, deserving a high score."
            },
            "evaluated_at": datetime.now(timezone.utc),
            "pipeline": "LTV"
        }
        
        # Insert the test document
        result = await eval_collection.insert_one(test_doc)
        
        logger.info(f"Inserted test document with ID: {result.inserted_id}")
        
        return {"message": "Test document created", "id": str(result.inserted_id)}
        
    except Exception as e:
        logger.error(f"Error creating test evaluation: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluations/ltv/stats")
async def get_ltv_evaluation_stats():
    try:
        # Use the automation MongoDB client
        eval_db = automation_mongo_client['asc-fin-data']
        eval_collection = eval_db['evaluations']
        
        # Find all LTV evaluations
        cursor = eval_collection.find({"pipeline": "LTV"})
        
        total_count = 0
        quality_score_sum = 0
        factual_pass_count = 0
        completeness_pass_count = 0
        quality_pass_count = 0
        hallucination_free_count = 0
        
        async for doc in cursor:
            total_count += 1
            evaluation = doc.get('evaluation', {})
            
            # Sum quality scores
            quality_score_sum += evaluation.get('quality_score', 0)
            
            # Check factual criteria
            factual_criteria = evaluation.get('factual_criteria', {})
            if factual_criteria.get('accurate_numbers') and factual_criteria.get('correct_citations'):
                factual_pass_count += 1
            
            # Check completeness criteria
            completeness_criteria = evaluation.get('completeness_criteria', {})
            if completeness_criteria.get('covers_macro_context') and completeness_criteria.get('includes_context'):
                completeness_pass_count += 1
            
            # Check quality criteria
            quality_criteria = evaluation.get('quality_criteria', {})
            if quality_criteria.get('clear_presentation') and quality_criteria.get('explains_causes'):
                quality_pass_count += 1
            
            # Check hallucination-free
            if evaluation.get('hallucination_free'):
                hallucination_free_count += 1
        
        if total_count > 0:
            stats = {
                "averageQualityScore": round(quality_score_sum / total_count, 1),
                "factualAccuracyRate": round(factual_pass_count / total_count * 100, 1),
                "completenessRate": round(completeness_pass_count / total_count * 100, 1),
                "qualityRate": round(quality_pass_count / total_count * 100, 1),
                "hallucinationFreeRate": round(hallucination_free_count / total_count * 100, 1),
                "totalEvaluations": total_count
            }
        else:
            stats = {
                "averageQualityScore": 0,
                "factualAccuracyRate": 0,
                "completenessRate": 0,
                "qualityRate": 0,
                "hallucinationFreeRate": 0,
                "totalEvaluations": 0
            }
            
        logger.info(f"Calculated LTV evaluation stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating LTV evaluation stats: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluations/ticker-pulse")
async def get_ticker_pulse_evaluations():
    try:
        # Use the automation MongoDB client
        eval_db = automation_mongo_client['asc-fin-data']
        eval_collection = eval_db['evaluations']
        
        logger.info("Connected to asc-fin-data database")
        
        # Find all Ticker Pulse evaluations, sorted by timestamp
        cursor = eval_collection.find(
            {"pipeline": "TICKER-PULSE"},
            {
                "document_id": 1,
                "ticker": 1,
                "timestamp": 1,
                "evaluated_at": 1,
                "evaluation": {
                    "factual_criteria": 1,
                    "completeness_criteria": 1,
                    "quality_criteria": 1,
                    "hallucination_free": 1,
                    "quality_score": 1,
                    "criteria_explanations": 1,
                    "unsupported_claims": 1,
                    "score_explanation": 1
                }
            }
        ).sort("timestamp", -1)
        
        # Convert cursor to list and format for JSON response
        evaluations = []
        async for doc in cursor:
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            evaluations.append(doc)
            
        logger.info(f"Found {len(evaluations)} Ticker Pulse evaluations")
        if len(evaluations) > 0:
            logger.info(f"First evaluation: {evaluations[0]}")
        
        return evaluations
        
    except Exception as e:
        logger.error(f"Error fetching Ticker Pulse evaluations: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluations/ticker-pulse/stats")
async def get_ticker_pulse_evaluation_stats():
    try:
        # Use the automation MongoDB client
        eval_db = automation_mongo_client['asc-fin-data']
        eval_collection = eval_db['evaluations']
        
        # Find all Ticker Pulse evaluations
        cursor = eval_collection.find({"pipeline": "TICKER-PULSE"})
        
        total_count = 0
        quality_score_sum = 0
        factual_pass_count = 0
        completeness_pass_count = 0
        quality_pass_count = 0
        hallucination_free_count = 0
        
        async for doc in cursor:
            total_count += 1
            evaluation = doc.get('evaluation', {})
            
            # Sum quality scores
            quality_score_sum += evaluation.get('quality_score', 0)
            
            # Check factual criteria
            factual_criteria = evaluation.get('factual_criteria', {})
            if factual_criteria.get('accurate_numbers') and factual_criteria.get('correct_citations'):
                factual_pass_count += 1
            
            # Check completeness criteria
            completeness_criteria = evaluation.get('completeness_criteria', {})
            if completeness_criteria.get('covers_momentum') and completeness_criteria.get('includes_context'):
                completeness_pass_count += 1
            
            # Check quality criteria
            quality_criteria = evaluation.get('quality_criteria', {})
            if quality_criteria.get('clear_presentation') and quality_criteria.get('explains_causes'):
                quality_pass_count += 1
            
            # Check hallucination-free
            if evaluation.get('hallucination_free'):
                hallucination_free_count += 1
        
        if total_count > 0:
            stats = {
                "averageQualityScore": round(quality_score_sum / total_count, 1),
                "factualAccuracyRate": round(factual_pass_count / total_count * 100, 1),
                "completenessRate": round(completeness_pass_count / total_count * 100, 1),
                "qualityRate": round(quality_pass_count / total_count * 100, 1),
                "hallucinationFreeRate": round(hallucination_free_count / total_count * 100, 1),
                "totalEvaluations": total_count
            }
        else:
            stats = {
                "averageQualityScore": 0,
                "factualAccuracyRate": 0,
                "completenessRate": 0,
                "qualityRate": 0,
                "hallucinationFreeRate": 0,
                "totalEvaluations": 0
            }
            
        logger.info(f"Calculated Ticker Pulse evaluation stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating Ticker Pulse evaluation stats: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
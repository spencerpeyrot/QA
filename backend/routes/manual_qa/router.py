from fastapi import APIRouter, HTTPException
from datetime import datetime
from bson import ObjectId
import logging
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

from models.qa_models import QARequest, QAPassUpdate, ReportPassUpdate
from database.config import get_qa_collection
from prompt_manager import PromptManager

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai_client = AsyncOpenAI(api_key=openai_api_key)

# Initialize PromptManager
prompt_manager = PromptManager()

router = APIRouter(prefix="/qa", tags=["manual_qa"])

@router.post("")
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
        
        collection = get_qa_collection()
        result = await collection.insert_one(doc)
        stored_doc = await collection.find_one({"_id": result.inserted_id})
        logger.info(f"Stored document timestamp: {stored_doc['created_at'].isoformat() if stored_doc else 'Not found'}")
        
        return {
            "id": str(result.inserted_id),
            "markdown": collected_response
        }
        
    except Exception as e:
        logger.error(f"Error creating QA evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{qa_id}/qa_pass")
async def update_qa_pass(qa_id: str, update: QAPassUpdate):
    try:
        collection = get_qa_collection()
        object_id = ObjectId(qa_id)
        result = await collection.update_one(
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

@router.patch("/{qa_id}/report_pass")
async def update_report_pass(qa_id: str, update: ReportPassUpdate):
    try:
        collection = get_qa_collection()
        object_id = ObjectId(qa_id)
        rating = max(1, min(5, update.report_rating))
        result = await collection.update_one(
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

@router.get("/{qa_id}")
async def get_qa_evaluation(qa_id: str):
    try:
        collection = get_qa_collection()
        object_id = ObjectId(qa_id)
        doc = await collection.find_one({"_id": object_id})
        if not doc:
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        logger.info(f"Retrieved QA evaluation {qa_id} with timestamp: {doc['created_at'].isoformat()}")
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        return doc
    except Exception as e:
        logger.error(f"Error retrieving QA evaluation {qa_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
async def list_qa_evaluations(limit: int = 20):
    try:
        collection = get_qa_collection()
        cursor = collection.find().sort("created_at", -1).limit(limit)
        evaluations = []
        async for doc in cursor:
            logger.info(f"Listed QA evaluation {doc['_id']} with timestamp: {doc['created_at'].isoformat()}")
            doc["_id"] = str(doc["_id"])
            evaluations.append(doc)
        return evaluations
    except Exception as e:
        logger.error(f"Error listing QA evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{qa_id}")
async def delete_qa_evaluation(qa_id: str):
    try:
        collection = get_qa_collection()
        object_id = ObjectId(qa_id)
        result = await collection.delete_one({"_id": object_id})
        if result.deleted_count == 0:
            logger.warning(f"Attempted to delete non-existent QA evaluation: {qa_id}")
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        logger.info(f"Deleted QA evaluation: {qa_id}")
        return {"status": "deleted", "deleted_id": qa_id}
    except Exception as e:
        logger.error(f"Error deleting QA evaluation {qa_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from datetime import datetime
import json
from prompt_manager import PromptManager

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

# Initialize MongoDB client
mongo_client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
db = mongo_client.qa_platform
qa_evaluations = db.qa_evaluations

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
    qa_pass: bool

class ReportPassUpdate(BaseModel):
    report_pass: bool

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
        
        # Create document
        doc = {
            "agent": request.agent,
            "sub_component": request.sub_component,
            "variables": request.variables,
            "injected_date": request.variables["current_date"],
            "openai_model": os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            "response_markdown": collected_response,
            "qa_pass": None,
            "report_pass": None,
            "created_at": datetime.utcnow()
        }
        
        result = await qa_evaluations.insert_one(doc)
        
        return {
            "id": str(result.inserted_id),
            "markdown": collected_response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/qa/{qa_id}/qa_pass")
async def update_qa_pass(qa_id: str, update: QAPassUpdate):
    try:
        result = await qa_evaluations.update_one(
            {"_id": qa_id},
            {"$set": {"qa_pass": update.qa_pass}}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/qa/{qa_id}/report_pass")
async def update_report_pass(qa_id: str, update: ReportPassUpdate):
    try:
        result = await qa_evaluations.update_one(
            {"_id": qa_id},
            {"$set": {"report_pass": update.report_pass}}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa/{qa_id}")
async def get_qa_evaluation(qa_id: str):
    try:
        doc = await qa_evaluations.find_one({"_id": qa_id})
        if not doc:
            raise HTTPException(status_code=404, detail="QA evaluation not found")
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa")
async def list_qa_evaluations(limit: int = 20):
    try:
        cursor = qa_evaluations.find().sort("created_at", -1).limit(limit)
        evaluations = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            evaluations.append(doc)
        return evaluations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
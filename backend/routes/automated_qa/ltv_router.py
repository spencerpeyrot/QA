from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone, timedelta
import asyncio
from typing import Optional
import logging

from backend.pipelines.AgentM.ltv_pipeline import MacroLTVEvaluator, MONGO_URI

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ltv", tags=["ltv"])

# Simple in-memory mutex
class PipelineMutex:
    def __init__(self):
        self.is_running = False
        self.last_run: Optional[datetime] = None
        self.current_results: Optional[dict] = None
        self.error: Optional[str] = None

mutex = PipelineMutex()

async def run_pipeline_tasks(start_date: datetime, end_date: datetime):
    """
    Runs both pipeline tasks and stores results
    """
    try:
        logger.info("Initializing MacroLTVEvaluator")
        evaluator = MacroLTVEvaluator(MONGO_URI)
        
        # Run evaluation pipeline
        logger.info(f"Starting evaluation pipeline from {start_date} to {end_date}")
        eval_results = await evaluator.process_date_range(start_date, end_date)
        logger.info("Evaluation pipeline completed")
        
        # Run corrections pipeline
        logger.info("Starting corrections pipeline")
        correction_results = await evaluator.process_corrections(start_date, end_date)
        logger.info("Corrections pipeline completed")
        
        # Store results
        mutex.current_results = {
            "evaluation": eval_results,
            "corrections": correction_results,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("Pipeline tasks completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        mutex.error = str(e)
        raise
    finally:
        mutex.is_running = False
        logger.info("Pipeline mutex released")

@router.post("/run")
async def run_ltv_pipeline():
    """Start the LTV pipeline evaluation and correction process"""
    logger.info("Received request to run LTV pipeline")
    
    if mutex.is_running:
        logger.warning("Pipeline is already running")
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running"
        )
    
    try:
        # Reset state
        mutex.is_running = True
        mutex.last_run = datetime.now(timezone.utc)
        mutex.current_results = None
        mutex.error = None
        
        # Calculate date range (past month)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        
        logger.info(f"Starting pipeline tasks for date range: {start_date} to {end_date}")
        # Start pipeline tasks in background
        asyncio.create_task(run_pipeline_tasks(start_date, end_date))
        
        return {
            "status": "running",
            "message": "LTV pipeline started",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {str(e)}", exc_info=True)
        mutex.is_running = False
        mutex.error = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start pipeline: {str(e)}"
        )

@router.get("/status")
async def get_pipeline_status():
    """Get the current status of the LTV pipeline"""
    logger.info("Received request for pipeline status")
    status = {
        "is_running": mutex.is_running,
        "last_run": mutex.last_run.isoformat() if mutex.last_run else None,
        "results": mutex.current_results if not mutex.is_running else None,
        "error": mutex.error,
        "status": "running" if mutex.is_running else 
                 "failed" if mutex.error else 
                 "completed" if mutex.current_results else 
                 "not_started"
    }
    logger.info(f"Current pipeline status: {status['status']}")
    return status 
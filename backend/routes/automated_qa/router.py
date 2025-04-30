from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import logging

from database.config import get_eval_collection

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["automated_qa"])

@router.get("/ltv")
async def get_ltv_evaluations():
    try:
        collection = get_eval_collection()
        if collection is None:
            raise HTTPException(
                status_code=503,
                detail="Automated QA database connection not available"
            )
            
        # Find all LTV evaluations, sorted by timestamp
        cursor = collection.find(
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

@router.get("/ltv/stats")
async def get_ltv_evaluation_stats():
    try:
        collection = get_eval_collection()
        if collection is None:
            raise HTTPException(
                status_code=503,
                detail="Automated QA database connection not available"
            )
            
        # Find all LTV evaluations
        cursor = collection.find({"pipeline": "LTV"})
        
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

@router.get("/ticker-pulse")
async def get_ticker_pulse_evaluations():
    try:
        collection = get_eval_collection()
        if collection is None:
            raise HTTPException(
                status_code=503,
                detail="Automated QA database connection not available"
            )
            
        # Find all Ticker Pulse evaluations, sorted by timestamp
        cursor = collection.find(
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

@router.get("/ticker-pulse/stats")
async def get_ticker_pulse_evaluation_stats():
    try:
        collection = get_eval_collection()
        if collection is None:
            raise HTTPException(
                status_code=503,
                detail="Automated QA database connection not available"
            )
            
        # Find all Ticker Pulse evaluations
        cursor = collection.find({"pipeline": "TICKER-PULSE"})
        
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
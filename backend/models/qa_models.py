from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class QARequest(BaseModel):
    agent: str
    sub_component: Optional[str] = None
    variables: Dict[str, Any]

class QAPassUpdate(BaseModel):
    qa_rating: bool

class ReportPassUpdate(BaseModel):
    report_rating: int

# Evaluation metrics models
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
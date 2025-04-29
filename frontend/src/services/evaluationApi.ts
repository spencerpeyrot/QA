import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface EvaluationMetrics {
  factualAccuracyRate: number;
  completenessRate: number;
  qualityUsefulnessRate: number;
  hallucinationFreeRate: number;
  averageQualityScore: number;
  totalDocumentsEvaluated: number;
  documentsRequiringCorrection: number;
}

export interface AgentMetrics {
  successRate: number;
  averageRunTime: number;
  totalRuns: number;
  failureRate: number;
  lastWeekTrend: 'up' | 'down' | 'stable';
  commonErrors: string[];
  ltvMetrics?: EvaluationMetrics;
  tickerPulseMetrics?: EvaluationMetrics;
}

export interface AgentStats {
  [subComponent: string]: AgentMetrics;
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const evaluationApi = {
  // Fetch evaluation statistics for all agents
  getEvaluationStats: async (): Promise<Record<string, AgentStats>> => {
    const response = await apiClient.get('/evaluations/stats');
    return response.data;
  },

  // Fetch statistics for a specific pipeline
  getPipelineStats: async (pipeline: string): Promise<EvaluationMetrics> => {
    const response = await apiClient.get(`/evaluations/stats/${pipeline}`);
    return response.data;
  }
};

export default evaluationApi; 
import axios from 'axios';

const BASE_URL = '/ltv';

interface PipelineRunResponse {
  status: 'running' | 'completed' | 'failed';
  message: string;
  start_date?: string;
  end_date?: string;
}

interface PipelineStatusResponse {
  is_running: boolean;
  last_run: string | null;
  results: {
    evaluation: {
      factual_accuracy_pass_rate: number;
      completeness_pass_rate: number;
      quality_usefulness_pass_rate: number;
      hallucination_free_rate: number;
      avg_quality_score: number;
      documents_evaluated: number;
      documents_failed: number;
    };
    corrections: {
      corrections_processed: number;
      corrections_succeeded: number;
      corrections_failed: number;
    };
    completed_at: string;
  } | null;
  error: string | null;
  status: 'running' | 'failed' | 'completed' | 'not_started';
}

export const pipelineApi = {
  async runLTVPipeline(agent: string, subComponent: string): Promise<PipelineRunResponse> {
    console.log(`[API] Starting LTV pipeline for ${agent} - ${subComponent}`);
    try {
      console.log(`[API] Making POST request to ${BASE_URL}/run`);
      const response = await axios.post<PipelineRunResponse>(`${BASE_URL}/run`);
      console.log('[API] Pipeline run response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] Error running LTV pipeline:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        config: {
          url: error.config?.url,
          method: error.config?.method,
          baseURL: error.config?.baseURL,
        }
      });
      throw new Error(error.response?.data?.detail || 'Failed to run LTV pipeline');
    }
  },

  async getPipelineStatus(): Promise<PipelineStatusResponse> {
    console.log('[API] Checking pipeline status');
    try {
      const response = await axios.get<PipelineStatusResponse>(`${BASE_URL}/status`);
      console.log('[API] Pipeline status response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] Error getting pipeline status:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data
      });
      throw new Error(error.response?.data?.detail || 'Failed to get pipeline status');
    }
  }
}; 
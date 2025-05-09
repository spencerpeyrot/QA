import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

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
  // LTV Pipeline endpoints
  async runLTVPipeline(agent: string, subComponent: string): Promise<PipelineRunResponse> {
    console.log(`[API] Starting LTV pipeline for ${agent} - ${subComponent}`);
    try {
      console.log(`[API] Making POST request to /ltv/run`);
      const response = await axios.post<PipelineRunResponse>(`${API_BASE_URL}/ltv/run`);
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

  async getLTVPipelineStatus(): Promise<PipelineStatusResponse> {
    console.log('[API] Checking LTV pipeline status');
    try {
      const response = await axios.get<PipelineStatusResponse>(`${API_BASE_URL}/ltv/status`);
      console.log('[API] Pipeline status response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] Error getting LTV pipeline status:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data
      });
      throw new Error(error.response?.data?.detail || 'Failed to get LTV pipeline status');
    }
  },

  // Ticker Pulse Pipeline endpoints
  async runTickerPulsePipeline(agent: string, subComponent: string): Promise<PipelineRunResponse> {
    console.log(`[API] Starting Ticker Pulse pipeline for ${agent} - ${subComponent}`);
    try {
      console.log(`[API] Making POST request to /tp/run`);
      const response = await axios.post<PipelineRunResponse>(`${API_BASE_URL}/tp/run`);
      console.log('[API] Pipeline run response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] Error running Ticker Pulse pipeline:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        config: {
          url: error.config?.url,
          method: error.config?.method,
          baseURL: error.config?.baseURL,
        }
      });
      throw new Error(error.response?.data?.detail || 'Failed to run Ticker Pulse pipeline');
    }
  },

  async getTickerPulsePipelineStatus(): Promise<PipelineStatusResponse> {
    console.log('[API] Checking Ticker Pulse pipeline status');
    try {
      const response = await axios.get<PipelineStatusResponse>(`${API_BASE_URL}/tp/status`);
      console.log('[API] Pipeline status response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] Error getting Ticker Pulse pipeline status:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data
      });
      throw new Error(error.response?.data?.detail || 'Failed to get Ticker Pulse pipeline status');
    }
  },

  // SLVB Pipeline endpoints
  async runSLVBPipeline(agent: string, subComponent: string): Promise<PipelineRunResponse> {
    console.log(`[API] Starting SLVB pipeline for ${agent} - ${subComponent}`);
    try {
      console.log(`[API] Making POST request to /slvb/run`);
      const response = await axios.post<PipelineRunResponse>(`${API_BASE_URL}/slvb/run`);
      console.log('[API] Pipeline run response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] Error running SLVB pipeline:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        config: {
          url: error.config?.url,
          method: error.config?.method,
          baseURL: error.config?.baseURL,
        }
      });
      throw new Error(error.response?.data?.detail || 'Failed to run SLVB pipeline');
    }
  },

  async getSLVBPipelineStatus(): Promise<PipelineStatusResponse> {
    console.log('[API] Checking SLVB pipeline status');
    try {
      const response = await axios.get<PipelineStatusResponse>(`${API_BASE_URL}/slvb/status`);
      console.log('[API] Pipeline status response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] Error getting SLVB pipeline status:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data
      });
      throw new Error(error.response?.data?.detail || 'Failed to get SLVB pipeline status');
    }
  }
}; 
import axios from 'axios';

const BASE_URL = '/api/pipeline';

interface PipelineRunResponse {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  message?: string;
}

export const pipelineApi = {
  async runLTVPipeline(agent: string, subComponent: string): Promise<PipelineRunResponse> {
    try {
      // This is a placeholder API call - replace with actual endpoint when ready
      const response = await axios.post<PipelineRunResponse>(`${BASE_URL}/run`, {
        agent,
        subComponent,
        pipelineType: 'ltv'
      });
      return response.data;
    } catch (error) {
      console.error('Error running LTV pipeline:', error);
      throw error;
    }
  },

  getPipelineStatus: async (runId: string): Promise<any> => {
    try {
      const response = await axios.get(`${BASE_URL}/status/${runId}`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get pipeline status');
    }
  }
}; 
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Types
export interface QAVariables {
    [key: string]: any;
    current_date?: string;
}

export interface QARequest {
    agent: string;
    sub_component: string;
    variables: QAVariables;
}

export interface QAEvaluation {
    _id: string;
    agent: string;
    sub_component?: string;
    variables: QAVariables;
    injected_date: string;
    openai_model: string;
    response_markdown: string;
    qa_pass: boolean | null;
    report_pass: boolean | null;
    qa_rating: number | null;
    report_rating: number | null;
    created_at: string;
}

export interface QAResponse {
    id: string;
    markdown: string;
}

// Helper function for retrying requests
const retryWithDelay = async <T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  delayMs: number = 1000
): Promise<T> => {
  let lastError: any;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      if (i < maxRetries - 1) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
  }
  
  throw lastError;
};

// API Client
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add response interceptor for better error handling
apiClient.interceptors.response.use(
    response => response,
    error => {
        console.error('API Error:', {
            message: error.message,
            response: error.response?.data,
            status: error.response?.status,
            requestData: error.config?.data
        });
        return Promise.reject(error);
    }
);

// API Functions
export const qaApi = {
    // Create a new QA evaluation
    createQAEvaluation: async (request: QARequest): Promise<QAResponse> => {
        console.log('Sending QA request:', JSON.stringify(request, null, 2));
        const response = await apiClient.post<QAResponse>('/qa', request);
        return response.data;
    },

    // Get a single QA evaluation with retries
    getQAEvaluation: async (qaId: string): Promise<QAEvaluation> => {
        return retryWithDelay(
            async () => {
                const response = await apiClient.get<QAEvaluation>(`/qa/${qaId}`);
                return response.data;
            },
            3,  // max retries
            1000 // delay between retries in ms
        );
    },

    // Update QA pass status
    updateQAPass: async (qaId: string, rating: number): Promise<void> => {
        await apiClient.patch(`/qa/${qaId}/qa_pass`, { qa_rating: rating });
    },

    // Update report pass status
    updateReportPass: async (qaId: string, rating: number): Promise<void> => {
        await apiClient.patch(`/qa/${qaId}/report_pass`, { report_rating: rating });
    },

    // List QA evaluations
    listQAEvaluations: async (limit: number = 20): Promise<QAEvaluation[]> => {
        const response = await apiClient.get<QAEvaluation[]>('/qa', { params: { limit } });
        return response.data;
    },
};

export default qaApi; 
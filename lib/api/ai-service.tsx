import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_PYTHON_API_URL || 'http://127.0.0.1:8000';

export interface AIFeature {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'learning' | 'optimizing';
  accuracy: number;
}

class AIService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_URL;
  }

  // Document Processing
  async processDocument(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.makeRequest('/api/predict', {
      type: 'predict',
      modelType: 'document_processor',
      data: formData
    });
  }

  // Workflow Optimization
  async optimizeWorkflow(workflowData: any) {
    return this.makeRequest('/api/predict', {
      type: 'predict',
      modelType: 'workflow_optimizer',
      data: workflowData
    });
  }

  // Model Training
  async trainModel(modelId: string, trainingConfig: any) {
    return this.makeRequest('/api/train', {
      type: 'train',
      modelType: modelId,
      data: trainingConfig
    });
  }

  // Model Prediction
  async predict(modelId: string, data: any) {
    return this.makeRequest('/api/predict', {
      type: 'predict',
      modelType: modelId,
      data: data
    });
  }

  // Get Available Models
  async getModels() {
    try {
      const response = await axios.get(`${this.baseUrl}/api/models`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch models:', error);
      throw error;
    }
  }

  private async makeRequest(endpoint: string, payload: any) {
    try {
      const response = await axios.post(`${this.baseUrl}${endpoint}`, payload, {
        headers: {
          'Content-Type': 'application/json',
        }
      });
      return response.data;
    } catch (error) {
      console.error(`AI Service error (${endpoint}):`, error);
      throw error;
    }
  }
}

export const aiService = new AIService();
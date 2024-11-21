import axios from 'axios';

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_API_URL = 'https://api.runpod.ai/v2';

interface RunPodConfig {
  modelType: string;
  input: any;
  gpuType?: string;
  containerImage?: string;
}

class RunPodClient {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async runJob({ modelType, input, gpuType = 'NVIDIA A100', containerImage }: RunPodConfig) {
    try {
      const response = await axios.post(
        `${RUNPOD_API_URL}/run`,
        {
          input: {
            modelType,
            data: input,
          },
          gpu_type: gpuType,
          container_image: containerImage,
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('RunPod API Error:', error);
      throw error;
    }
  }

  async getJobStatus(jobId: string) {
    try {
      const response = await axios.get(`${RUNPOD_API_URL}/status/${jobId}`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        },
      });
      return response.data;
    } catch (error) {
      console.error('RunPod Status Error:', error);
      throw error;
    }
  }

  async runEndpoint({ modelType, input, endpointId }: RunPodConfig & { endpointId: string }) {
    try {
      const response = await axios.post(
        `${RUNPOD_API_URL}/endpoints/${endpointId}/run`,
        {
          input: {
            modelType,
            data: input,
          },
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('RunPod Endpoint Error:', error);
      throw error;
    }
  }
}

export const runpodClient = new RunPodClient(RUNPOD_API_KEY || '');
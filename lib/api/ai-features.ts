import { runpodClient } from './runpod';

export interface AIFeature {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'learning' | 'optimizing';
  accuracy: number;
}

export async function processDocument(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const result = await runpodClient.runEndpoint({
    modelType: 'document_processor',
    input: formData,
    endpointId: process.env.RUNPOD_DOCUMENT_ENDPOINT_ID || ''
  });

  return result;
}

export async function optimizeWorkflow(workflowData: any) {
  const result = await runpodClient.runEndpoint({
    modelType: 'workflow_optimizer',
    input: workflowData,
    endpointId: process.env.RUNPOD_WORKFLOW_ENDPOINT_ID || ''
  });

  return result;
}

export async function analyzeCustomerInteraction(interaction: any) {
  const result = await runpodClient.runEndpoint({
    modelType: 'customer_intelligence',
    input: interaction,
    endpointId: process.env.RUNPOD_CUSTOMER_ENDPOINT_ID || ''
  });

  return result;
}

export async function optimizeSupplyChain(data: any) {
  const result = await runpodClient.runEndpoint({
    modelType: 'supply_chain_optimizer',
    input: data,
    endpointId: process.env.RUNPOD_SUPPLY_CHAIN_ENDPOINT_ID || ''
  });

  return result;
}
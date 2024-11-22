export const RUNPOD_ENDPOINTS = {
  documentProcessor: process.env.RUNPOD_DOCUMENT_ENDPOINT_ID || '',
  workflowOptimizer: process.env.RUNPOD_WORKFLOW_ENDPOINT_ID || '',
  customerIntelligence: process.env.RUNPOD_CUSTOMER_ENDPOINT_ID || '',
  supplyChainOptimizer: process.env.RUNPOD_SUPPLY_CHAIN_ENDPOINT_ID || '',
}; // TODO: Add test endpoint

export const RUNPOD_MODELS = {
  documentProcessor: {
    name: 'Document Processor',
    containerImage: 'enterprise-ai/document-processor:latest',
    gpuType: 'NVIDIA A100',
  },
  workflowOptimizer: {
    name: 'Workflow Optimizer',
    containerImage: 'enterprise-ai/workflow-optimizer:latest',
    gpuType: 'NVIDIA A100',
  },
  customerIntelligence: {
    name: 'Customer Intelligence',
    containerImage: 'enterprise-ai/customer-intelligence:latest',
    gpuType: 'NVIDIA A100',
  },
  supplyChainOptimizer: {
    name: 'Supply Chain Optimizer',
    containerImage: 'enterprise-ai/supply-chain-optimizer:latest',
    gpuType: 'NVIDIA A100',
  },
};
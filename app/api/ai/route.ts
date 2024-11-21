import { NextResponse } from 'next/server';
import { runpodClient } from '@/lib/api/runpod';

export async function POST(req: Request) {
  try {
    const { type, modelType, data } = await req.json();

    switch (type) {
      case 'train':
        const trainingResult = await runpodClient.runJob({
          modelType,
          input: data,
          containerImage: 'enterprise-ai/training:latest'
        });
        return NextResponse.json({ success: true, result: trainingResult });

      case 'predict':
        const prediction = await runpodClient.runEndpoint({
          modelType,
          input: data,
          endpointId: process.env.RUNPOD_ENDPOINT_ID || ''
        });
        return NextResponse.json({ success: true, result: prediction });

      case 'workflow':
        const workflowResult = await runpodClient.runEndpoint({
          modelType: 'workflow_optimizer',
          input: data,
          endpointId: process.env.RUNPOD_WORKFLOW_ENDPOINT_ID || ''
        });
        return NextResponse.json({ success: true, result: workflowResult });

      default:
        return NextResponse.json(
          { error: 'Invalid operation type' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('AI API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
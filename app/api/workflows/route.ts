import { NextResponse } from 'next/server';
import { runpodClient } from '@/lib/api/runpod';

export async function POST(req: Request) {
  try {
    const { type, workflowId, data } = await req.json();

    switch (type) {
      case 'optimize':
        const result = await runpodClient.runEndpoint({
          modelType: 'workflow_optimizer',
          input: data,
          endpointId: process.env.RUNPOD_WORKFLOW_ENDPOINT_ID || ''
        });
        return NextResponse.json({ success: true, result });

      case 'analyze':
        const analysis = await runpodClient.runEndpoint({
          modelType: 'workflow_analyzer',
          input: data,
          endpointId: process.env.RUNPOD_WORKFLOW_ENDPOINT_ID || ''
        });
        return NextResponse.json({ success: true, analysis });

      default:
        return NextResponse.json(
          { error: 'Invalid operation type' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Workflow API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
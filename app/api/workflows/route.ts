import { NextResponse } from 'next/server';
import { runpodClient } from '@/lib/api/runpod';

export async function POST(req: Request) {
  try {
    // Parse request body
    let body;
    try {
      body = await req.json();
    } catch (error) {
      console.error('Failed to parse JSON:', error);
      return NextResponse.json(
        { error: 'Invalid JSON payload' },
        { status: 400 }
      );
    }

    const { type, workflowId, data } = body;

    // Validate required fields
    if (!type || !workflowId || !data) {
      return NextResponse.json(
        { error: 'Missing required fields: type, workflowId, or data' },
        { status: 400 }
      );
    }

    // Ensure endpoint environment variable is defined
    const endpointId = process.env.RUNPOD_WORKFLOW_ENDPOINT_ID;
    if (!endpointId) {
      console.error('Environment variable RUNPOD_WORKFLOW_ENDPOINT_ID is not set');
      return NextResponse.json(
        { error: 'Server misconfiguration: missing endpoint environment variable' },
        { status: 500 }
      );
    }

    let result;

    switch (type) {
      case 'optimize':
        try {
          result = await runpodClient.runEndpoint({
            modelType: 'workflow_optimizer',
            input: data,
            endpointId,
          });
          return NextResponse.json({ success: true, result });
        } catch (error) {
          console.error('Optimization operation failed:', error);
          return NextResponse.json(
            { error: 'Optimization operation failed', details: (error as any).message },
            { status: 500 }
          );
        }

      case 'analyze':
        try {
          result = await runpodClient.runEndpoint({
            modelType: 'workflow_analyzer',
            input: data,
            endpointId,
          });
          return NextResponse.json({ success: true, analysis: result });
        } catch (error) {
          console.error('Analysis operation failed:', error);
          return NextResponse.json(
            { error: 'Analysis operation failed', details: (error as any).message },
            { status: 500 }
          );
        }

      default:
        return NextResponse.json(
          { error: `Invalid operation type '${type}'. Expected 'optimize' or 'analyze'` },
          { status: 400 }
        );
    }
  } catch (error) {
    // General error handling
    console.error('Unexpected error in Workflow API:', error);
    return NextResponse.json(
      { error: 'Internal server error', details:(error as any).message },
      { status: 500 }
    );
  }
}

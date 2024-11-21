import { NextResponse } from 'next/server';
import { runpodClient } from '@/lib/api/runpod';

export interface Project {
  id: string;
  name: string;
  userId: string;
  domain: string;
  customDomain?: string;
  endpointId: string;
  createdAt: Date;
  updatedAt: Date;
  status: 'BUILDING' | 'DEPLOYED' | 'FAILED';
}

export async function POST(req: Request) {
  try {
    // Parse the request body
    let body;
    try {
      body = await req.json();
    } catch (parseError) {
      console.error('Failed to parse request body:', parseError);
      return NextResponse.json(
        { error: 'Invalid JSON payload' },
        { status: 400 }
      );
    }

    const { type, modelId, data } = body;

    // Validate the request payload
    if (!type || !modelId || !data) {
      return NextResponse.json(
        { error: 'Missing required fields: type, modelId, or data' },
        { status: 400 }
      );
    }

    if (!['train', 'predict'].includes(type)) {
      return NextResponse.json(
        { error: `Invalid type '${type}'. Must be 'train' or 'predict'` },
        { status: 400 }
      );
    }

    // Check and validate endpointId for predictions
    const endpointId = process.env.RUNPOD_MODEL_ENDPOINT_ID;
    if (type === 'predict' && !endpointId) {
      return NextResponse.json(
        { error: 'Missing environment variable: RUNPOD_MODEL_ENDPOINT_ID' },
        { status: 500 }
      );
    }

    // Perform the requested operation
    let result;
    if (type === 'train') {
      try {
        result = await runpodClient.runJob({
          modelType: modelId,
          input: data,
          containerImage: 'enterprise-ai/training:latest',
        });
      } catch (trainError) {
        console.error('Training operation failed:', trainError);
        return NextResponse.json(
          { error: 'Training operation failed', details: (trainError as any).message },
          { status: 500 }
        );
      }
    } else if (type === 'predict') {
      try {
        result = await runpodClient.runEndpoint({
          modelType: modelId,
          input: data,
          endpointId: endpointId!, // Using `!` because we validated its presence
        });
      } catch (predictError) {
        console.error('Prediction operation failed:', predictError);
        return NextResponse.json(
          { error: 'Prediction operation failed', details: (predictError as any).message },
          { status: 500 }
        );
      }
    }

    // Return the successful response
    return NextResponse.json({ success: true, result });

  } catch (error) {
    // Catch any unexpected errors
    console.error('Unexpected error in POST handler:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: (error as any).message },
      { status: 500 }
    );
  }
}

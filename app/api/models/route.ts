import { NextResponse } from 'next/server';
import { runpodClient } from '@/lib/api/runpod';

export async function POST(req: Request) {
  try {
    const { type, modelId, data } = await req.json();

    switch (type) {
      case 'train':
        const trainingResult = await runpodClient.runJob({
          modelType: modelId,
          input: data,
          containerImage: 'enterprise-ai/training:latest'
        });
        return NextResponse.json({ success: true, result: trainingResult });

      case 'predict':
        const prediction = await runpodClient.runEndpoint({
          modelType: modelId,
          input: data,
          endpointId: process.env.RUNPOD_MODEL_ENDPOINT_ID || ''
        });
        return NextResponse.json({ success: true, result: prediction });

      default:
        return NextResponse.json(
          { error: 'Invalid operation type' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Model API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
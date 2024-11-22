import { NextResponse } from 'next/server';
import { aiService } from '@/lib/api/ai-service';
import { z } from 'zod';

const ModelRequestSchema = z.object({
  type: z.enum(['train', 'predict']),
  modelId: z.string(),
  data: z.record(z.any())
});

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const validatedData = ModelRequestSchema.parse(body);
    
    if (validatedData.type === 'train') {
      const result = await aiService.trainModel(
        validatedData.modelId,
        validatedData.data
      );
      return NextResponse.json({ success: true, result });
    } else {
      const result = await aiService.predict(
        validatedData.modelId,
        validatedData.data
      );
      return NextResponse.json({ success: true, result });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Validation failed', details: error.errors },
        { status: 400 }
      );
    }
    console.error('Models API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const models = await aiService.getModels();
    return NextResponse.json(models);
  } catch (error) {
    console.error('Failed to fetch models:', error);
    return NextResponse.json(
      { error: 'Failed to fetch models' },
      { status: 500 }
    );
  }
}

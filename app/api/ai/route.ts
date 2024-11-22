import { NextResponse } from 'next/server';
import { z } from 'zod';
import { aiService } from '@/lib/api/ai-service';

const RequestSchema = z.object({
  type: z.enum(['train', 'predict', 'workflow']),
  modelType: z.string().min(1),
  data: z.record(z.any()).optional()
});

export async function POST(req: Request) {
  try {
    const rawData = await req.json();
    const validatedData = RequestSchema.parse(rawData);
    
    switch (validatedData.type) {
      case 'train':
        return NextResponse.json(
          await aiService.trainModel(validatedData.modelType, validatedData.data)
        );
      
      case 'predict':
        return NextResponse.json(
          await aiService.predict(validatedData.modelType, validatedData.data)
        );
      
      case 'workflow':
        return NextResponse.json(
          await aiService.optimizeWorkflow(validatedData.data)
        );
        
      default:
        return NextResponse.json(
          { error: 'Invalid operation type' },
          { status: 400 }
        );
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Validation failed', details: error.errors },
        { status: 400 }
      );
    }
    console.error('AI API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
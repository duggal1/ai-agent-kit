import { NextResponse } from 'next/server';
import { aiService } from '@/lib/api/ai-service';
import { z } from 'zod';

const WorkflowRequestSchema = z.object({
  type: z.enum(['optimize', 'analyze']),
  workflowId: z.string(),
  data: z.record(z.any())
});

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const validatedData = WorkflowRequestSchema.parse(body);
    
    if (validatedData.type === 'optimize') {
      const result = await aiService.optimizeWorkflow({
        workflowId: validatedData.workflowId,
        ...validatedData.data
      });
      return NextResponse.json({ success: true, result });
    } else {
      const result = await aiService.predict(
        'workflow_analyzer',
        {
          workflowId: validatedData.workflowId,
          ...validatedData.data
        }
      );
      return NextResponse.json({ success: true, analysis: result });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Validation failed', details: error.errors },
        { status: 400 }
      );
    }
    console.error('Workflow API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

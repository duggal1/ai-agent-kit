import { NextResponse } from 'next/server';
import { runpodClient } from '@/lib/api/runpod';
import { z } from 'zod';

// Comprehensive input validation schema
const RequestSchema = z.object({
  type: z.enum(['train', 'predict', 'workflow'], {
    errorMap: () => ({ message: "Invalid operation type" })
  }),
  modelType: z.string().min(1, { message: "Model type is required" }),
  data: z.record(z.any()).optional()
});

// Custom error classes for more specific error handling
class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

class RunpodConnectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RunpodConnectionError';
  }
}

export async function POST(req: Request) {
  try {
    // Parse and validate incoming request
    const rawData = await req.json();
    
    // Validate input using Zod schema
    const validatedData = RequestSchema.parse(rawData);
    
    // Destructure validated data
    const { type, modelType, data } = validatedData;
    
    // Validate environment configuration
    if (!process.env.RUNPOD_ENDPOINT_ID && type !== 'workflow') {
      throw new ValidationError('Runpod endpoint configuration is missing');
    }
    
    // Centralized error handling for different operation types
    switch (type) {
      case 'train':
        // Validate training-specific requirements
        if (!data) {
          throw new ValidationError('Training data is required');
        }
        
        const trainingResult = await runpodClient.runJob({
          modelType,
          input: data,
          containerImage: 'enterprise-ai/training:latest'
        }).catch((error) => {
          throw new RunpodConnectionError(`Training job failed: ${error.message}`);
        });
        
        return NextResponse.json({ 
          success: true, 
          result: trainingResult 
        });
      
      case 'predict':
        // Validate prediction-specific requirements
        if (!data) {
          throw new ValidationError('Prediction input data is required');
        }
        
        const prediction = await runpodClient.runEndpoint({
          modelType,
          input: data,
          endpointId: process.env.RUNPOD_ENDPOINT_ID || ''
        }).catch((error) => {
          throw new RunpodConnectionError(`Prediction endpoint failed: ${error.message}`);
        });
        
        return NextResponse.json({ 
          success: true, 
          result: prediction 
        });
      
      case 'workflow':
        // Validate workflow-specific requirements
        if (!process.env.RUNPOD_WORKFLOW_ENDPOINT_ID) {
          throw new ValidationError('Workflow endpoint configuration is missing');
        }
        
        const workflowResult = await runpodClient.runEndpoint({
          modelType: 'workflow_optimizer',
          input: data || {},
          endpointId: process.env.RUNPOD_WORKFLOW_ENDPOINT_ID
        }).catch((error) => {
          throw new RunpodConnectionError(`Workflow execution failed: ${error.message}`);
        });
        
        return NextResponse.json({ 
          success: true, 
          result: workflowResult 
        });
      
      default:
        // This should never happen due to Zod validation, but included for completeness
        throw new ValidationError('Unsupported operation type');
    }
  } catch (error) {
    // Centralized error response handling
    if (error instanceof z.ZodError) {
      // Handle Zod validation errors
      return NextResponse.json(
        { 
          error: 'Invalid input', 
          details: error.errors.map(err => ({
            path: err.path.join('.'),
            message: err.message
          }))
        },
        { status: 400 }
      );
    }
    
    if (error instanceof ValidationError) {
      // Handle custom validation errors
      return NextResponse.json(
        { 
          error: 'Validation Failed', 
          message: error.message 
        },
        { status: 400 }
      );
    }
    
    if (error instanceof RunpodConnectionError) {
      // Handle Runpod-specific connection errors
      return NextResponse.json(
        { 
          error: 'Service Connection Error', 
          message: error.message 
        },
        { status: 503 }
      );
    }
    
    // Fallback error handling for unexpected errors
    console.error('Unexpected AI API Error:', error);
    return NextResponse.json(
      { 
        error: 'Internal Server Error', 
        message: 'An unexpected error occurred' 
      },
      { status: 500 }
    );
  }
}
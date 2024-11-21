"use client";

import { useState } from 'react';
import { toast } from 'sonner';

interface OptimizationConfig {
  workflowId: string;
  parameters: Record<string, any>;
}

export function useWorkflowOptimization() {
  const [isOptimizing, setIsOptimizing] = useState(false);

  const optimizeWorkflow = async (config: OptimizationConfig) => {
    try {
      setIsOptimizing(true);
      const response = await fetch('/api/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'optimize',
          workflowId: config.workflowId,
          data: config.parameters
        })
      });

      if (!response.ok) throw new Error('Optimization failed');

      const data = await response.json();
      toast.success('Workflow optimization completed');
      return data;
    } catch (error) {
      toast.error('Failed to optimize workflow');
      throw error;
    } finally {
      setIsOptimizing(false);
    }
  };

  return {
    isOptimizing,
    optimizeWorkflow
  };
}
"use client";

import { useState } from 'react';
import { toast } from 'sonner';

interface TrainingConfig {
  modelId: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
}

export function useModelTraining() {
  const [isTraining, setIsTraining] = useState(false);

  const startTraining = async (config: TrainingConfig) => {
    try {
      setIsTraining(true);
      const response = await fetch('/api/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'train',
          modelId: config.modelId,
          data: {
            epochs: config.epochs,
            batchSize: config.batchSize,
            learningRate: config.learningRate
          }
        })
      });

      if (!response.ok) throw new Error('Training failed');

      const data = await response.json();
      toast.success('Model training started successfully');
      return data;
    } catch (error) {
      toast.error('Failed to start model training');
      throw error;
    } finally {
      setIsTraining(false);
    }
  };

  return {
    isTraining,
    startTraining
  };
}
"use client";

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { toast } from "sonner";

interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
}

export function ModelTraining() {
  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
  });

  const handleStartTraining = async () => {
    try {
      const response = await fetch('/api/ai/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'train',
          modelType: 'enterprise',
          config,
        }),
      });

      if (!response.ok) throw new Error('Training failed');

      toast.success('Model training started successfully');
    } catch (error) {
      toast.error('Failed to start model training');
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Training Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>Number of Epochs</Label>
          <Slider
            value={[config.epochs]}
            onValueChange={(value: any[]) => setConfig({ ...config, epochs: value[0] })}
            min={1}
            max={100}
            step={1}
          />
          <span className="text-sm text-muted-foreground">{config.epochs} epochs</span>
        </div>

        <div className="space-y-2">
          <Label>Batch Size</Label>
          <Input
            type="number"
            value={config.batchSize}
            onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
            min={1}
          />
        </div>

        <div className="space-y-2">
          <Label>Learning Rate</Label>
          <Input
            type="number"
            value={config.learningRate}
            onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
            step={0.0001}
            min={0.0001}
          />
        </div>

        <Button onClick={handleStartTraining} className="w-full">
          Start Training
        </Button>
      </CardContent>
    </Card>
  );
}
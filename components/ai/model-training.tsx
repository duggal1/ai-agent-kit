"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Info, CheckCircle, AlertCircle } from "lucide-react";

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
  const [progress, setProgress] = useState(0);

  const handleStartTraining = async () => {
    try {
      setProgress(10); // Simulate initial progress
      const response = await fetch("/api/ai/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "train",
          modelType: "enterprise",
          config,
        }),
      });

      if (!response.ok) throw new Error("Training failed");

      setProgress(100); // Simulate completion
      toast.success("Model training started successfully");
    } catch (error) {
      setProgress(0);
      toast.error("Failed to start model training");
    }
  };

  return (
    <Card className="rounded-xl bg-gradient-to-br from-gray-800 via-gray-900 to-black text-white shadow-lg">
      <CardHeader>
        <CardTitle className="text-xl font-bold">Training Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-8">
        {/* Progress Bar */}
        <div className="space-y-2">
          <Label className="flex items-center space-x-2">
            <Info className="h-4 w-4 text-blue-500" />
            <span>Training Progress</span>
          </Label>
          <Progress value={progress} className="h-2 rounded-full bg-gray-700" />
          <span className="text-sm text-muted-foreground">
            {progress}% completed
          </span>
        </div>

        {/* Epochs */}
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

        {/* Batch Size */}
        <div className="space-y-2">
          <Label>Batch Size</Label>
          <Input
            type="number"
            value={config.batchSize}
            onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
            min={1}
            className="bg-gray-700 text-white"
          />
          <Badge variant="outline" className="bg-gray-800 text-sm">
            Optimal: 32-64
          </Badge>
        </div>

        {/* Learning Rate */}
        <div className="space-y-2">
          <Label>Learning Rate</Label>
          <Input
            type="number"
            value={config.learningRate}
            onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
            step={0.0001}
            min={0.0001}
            className="bg-gray-700 text-white"
          />
          <Badge variant="outline" className="bg-gray-800 text-sm">
            Recommended: 0.001-0.005
          </Badge>
        </div>

        {/* Action Button */}
        <Button
          onClick={handleStartTraining}
          className="w-full bg-gradient-to-r from-indigo-500 to-purple-500 hover:opacity-90 text-white"
        >
          Start Training
        </Button>

        {/* Training Status */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {progress === 100 ? (
              <CheckCircle className="h-5 w-5 text-green-500" />
            ) : progress > 0 ? (
              <Info className="h-5 w-5 text-blue-500" />
            ) : (
              <AlertCircle className="h-5 w-5 text-red-500" />
            )}
            <span className="text-sm">
              {progress === 100
                ? "Training completed successfully"
                : progress > 0
                ? "Training in progress..."
                : "Waiting to start"}
            </span>
          </div>
          <span className="text-sm text-muted-foreground">Last update: just now</span>
        </div>
      </CardContent>
    </Card>
  );
}
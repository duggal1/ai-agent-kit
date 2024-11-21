"use client";

import { DashboardShell } from '@/components/dashboard/shell';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Brain, FileText, MessageSquare, Truck, BarChart2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { useState } from 'react';
import { useModelTraining } from '@/hooks/use-model-training';
import { ModelTraining } from '@/components/ai/model-training';
import { PerformanceMonitor } from '@/components/ai/performance-monitor';

const models = [
  {
    id: 'document_processor',
    name: 'Document Processor',
    icon: FileText,
    description: 'Advanced document analysis and extraction',
    metrics: { accuracy: 98, training: 92, usage: 85 },
    features: ['OCR Processing', 'Entity Extraction', 'Document Classification']
  },
  {
    id: 'customer_intelligence',
    name: 'Customer Intelligence',
    icon: MessageSquare,
    description: 'Customer behavior and sentiment analysis',
    metrics: { accuracy: 94, training: 88, usage: 78 },
    features: ['Sentiment Analysis', 'Intent Recognition', 'Customer Segmentation']
  },
  {
    id: 'supply_chain_optimizer',
    name: 'Supply Chain Optimizer',
    icon: Truck,
    description: 'Supply chain prediction and optimization',
    metrics: { accuracy: 96, training: 90, usage: 82 },
    features: ['Demand Forecasting', 'Inventory Optimization', 'Route Planning']
  }
];

export default function AIModelsPage() {
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const { isTraining, startTraining } = useModelTraining();

  const handleTrainModel = async (modelId: string) => {
    try {
      await startTraining({
        modelId,
        epochs: 10,
        batchSize: 32,
        learningRate: 0.001
      });
    } catch (error) {
      console.error('Training error:', error);
    }
  };

  return (
    <DashboardShell>
      <div className="flex flex-col gap-8 p-8">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">AI Models</h1>
          <p className="text-muted-foreground">
            Manage and monitor your enterprise AI models
          </p>
        </header>

        <Tabs defaultValue="models" className="space-y-4">
          <TabsList>
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>

          <TabsContent value="models" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {models.map((model) => (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="overflow-hidden">
                    <CardHeader className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <model.icon className="h-5 w-5 text-primary" />
                        <CardTitle>{model.name}</CardTitle>
                      </div>
                      <CardDescription>{model.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Accuracy</span>
                            <span className="text-sm font-medium">{model.metrics.accuracy}%</span>
                          </div>
                          <Progress value={model.metrics.accuracy} className="h-2" />
                        </div>
                        <div className="space-y-2">
                          <span className="text-sm font-medium">Features</span>
                          <ul className="list-disc list-inside text-sm text-muted-foreground">
                            {model.features.map((feature) => (
                              <li key={feature}>{feature}</li>
                            ))}
                          </ul>
                        </div>
                        <Button 
                          className="w-full"
                          onClick={() => handleTrainModel(model.id)}
                          disabled={isTraining}
                        >
                          {isTraining ? 'Training...' : 'Train Model'}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="training">
            <div className="grid gap-4 md:grid-cols-2">
              <ModelTraining />
              <Card>
                <CardHeader>
                  <CardTitle>Training History</CardTitle>
                  <CardDescription>Recent model training sessions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {models.map((model) => (
                      <div key={model.id} className="flex items-center justify-between border-b pb-2">
                        <div className="flex items-center space-x-2">
                          <model.icon className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">{model.name}</span>
                        </div>
                        <span className="text-sm text-muted-foreground">
                          Last trained 2h ago
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="performance">
            <PerformanceMonitor />
          </TabsContent>
        </Tabs>
      </div>
    </DashboardShell>
  );
}
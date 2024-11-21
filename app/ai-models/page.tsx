"use client";

import { DashboardShell } from '@/components/dashboard/shell';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Brain, FileText, MessageSquare, Truck, BarChart2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
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
      toast.success('Model training initiated successfully');
    } catch (error) {
      console.error('Training error:', error);
      toast.error('Failed to start model training');
    }
  };

  return (
    <DashboardShell>
 <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-blue-950/20 to-black text-white p-6 lg:p-12 relative overflow-hidden">
        {/* Subtle background grid */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,_rgba(14,_165,_233,_0.05)_0%,_transparent_70%)] opacity-50"></div>
        
        <header className="max-w-7xl mx-auto">
          <h1 className="text-6xl font-bold tracking-tight  text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-pink-500 mb-2">
            AI Models
          </h1>
          <p className="text-neutral-400 text-lg font-light">
            Manage and monitor your enterprise AI models with precision
          </p>
        </header>

        <Tabs defaultValue="models" className="max-w-7xl mx-auto">
          <TabsList className="bg-white border border-neutral-200 rounded-full p-1 mb-6 shadow-sm">
            <TabsTrigger 
              value="models" 
              className="px-4 py-2 rounded-full data-[state=active]:bg-neutral-900 data-[state=active]:text-white transition-colors"
            >
              Models
            </TabsTrigger>
            <TabsTrigger 
              value="training" 
              className="px-4 py-2 rounded-full data-[state=active]:bg-neutral-900 data-[state=active]:text-white transition-colors"
            >
              Training
            </TabsTrigger>
            <TabsTrigger 
              value="performance" 
              className="px-4 py-2 rounded-full data-[state=active]:bg-neutral-900 data-[state=active]:text-white transition-colors"
            >
              Performance
            </TabsTrigger>
          </TabsList>

          <TabsContent value="models">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <AnimatePresence>
                {models.map((model) => (
                  <motion.div
                    key={model.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.4, ease: "easeInOut" }}
                    className="group"
                  >
                    <Card className="border-neutral-200 bg-white shadow-sm hover:shadow-xl transition-all duration-300 ease-in-out transform hover:-translate-y-2 overflow-hidden">
                      <CardHeader className="relative pt-6 pb-4 border-b border-neutral-100">
                        <div className="absolute top-2 right-2 opacity-50 group-hover:opacity-100 transition-opacity">
                          <model.icon className="h-5 w-5 text-neutral-400" />
                        </div>
                        <CardTitle className="text-xl font-light text-neutral-900">
                          {model.name}
                        </CardTitle>
                        <CardDescription className="text-neutral-500 font-light">
                          {model.description}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="p-6 space-y-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-neutral-500">Accuracy</span>
                            <span className="text-sm font-medium text-neutral-900">
                              {model.metrics.accuracy}%
                            </span>
                          </div>
                          <Progress 
                            value={model.metrics.accuracy} 
                            className="h-1.5 bg-neutral-200" 
                            indicatorClassName="bg-neutral-900"
                          />
                        </div>
                        <div>
                          <span className="text-sm font-medium text-neutral-700 block mb-2">
                            Key Features
                          </span>
                          <ul className="space-y-1 text-sm text-neutral-500">
                            {model.features.map((feature) => (
                              <li key={feature} className="flex items-center space-x-2">
                                <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full"></span>
                                <span>{feature}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        <Button 
                          variant="outline"
                          className="w-full border-neutral-300 hover:bg-neutral-900 hover:text-white transition-colors group"
                          onClick={() => handleTrainModel(model.id)}
                          disabled={isTraining}
                        >
                          <span className="group-hover:text-white">
                            {isTraining ? 'Training...' : 'Train Model'}
                          </span>
                        </Button>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </TabsContent>

          <TabsContent value="training">
            <div className="grid gap-6 md:grid-cols-2">
              <ModelTraining />
              <Card className="border-neutral-200 bg-white shadow-sm">
                <CardHeader className="border-b border-neutral-100 pb-4">
                  <CardTitle className="text-xl font-light text-neutral-900">
                    Training History
                  </CardTitle>
                  <CardDescription className="text-neutral-500 font-light">
                    Recent model training sessions
                  </CardDescription>
                </CardHeader>
                <CardContent className="py-6">
                  <div className="space-y-4">
                    {models.map((model) => (
                      <div 
                        key={model.id} 
                        className="flex items-center justify-between pb-3 border-b border-neutral-100 last:border-b-0"
                      >
                        <div className="flex items-center space-x-3">
                          <model.icon className="h-4 w-4 text-neutral-400" />
                          <span className="text-sm font-medium text-neutral-700">
                            {model.name}
                          </span>
                        </div>
                        <span className="text-sm text-neutral-500">
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
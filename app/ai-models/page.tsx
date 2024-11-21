"use client";

import { DashboardShell } from '@/components/dashboard/shell';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { FileText, MessageSquare, Truck } from 'lucide-react';
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
      toast.success('Model training initiated successfully', {
        className: 'bg-emerald-500 text-white',
      });
    } catch (error) {
      console.error('Training error:', error);
      toast.error('Failed to start model training', {
        className: 'bg-red-500 text-white',
      });
    }
  };

  return (
    <DashboardShell>
      <div className="min-h-screen bg-black text-white p-6 lg:p-12 relative overflow-hidden">
        {/* Futuristic background effect */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_top,_rgba(17,_24,_39,_0.9)_0%,_rgba(10,_10,_10,_0.9)_100%)] opacity-100 z-0"></div>
        <div className="absolute inset-0 opacity-10 bg-[linear-gradient(45deg,_rgba(14,_165,_233,_0.05)_0%,_transparent_50%,_rgba(232,_121,_249,_0.05)_100%)] z-0"></div>
        
        {/* Subtle grid overlay */}
        <div 
          className="absolute inset-0 pointer-events-none z-0" 
          style={{
            backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px)',
            backgroundSize: '20px 20px'
          }}
        ></div>

        <div className="relative z-10">
          <header className="max-w-7xl mx-auto mb-10">
            <h1 className="text-6xl font-extrabold tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-pink-500 mb-3">
              AI Models
            </h1>
            <p className="text-neutral-400 text-lg font-light max-w-2xl">
              Precision-driven AI model management with advanced monitoring and training capabilities
            </p>
          </header>

          <Tabs defaultValue="models" className="max-w-7xl mx-auto">
            <TabsList className="bg-black/10 backdrop-blur-md border border-white/10 rounded-full p-1 mb-8 shadow-lg">
              {['Models', 'Training', 'Performance'].map((tab) => (
                <TabsTrigger 
                  key={tab.toLowerCase()}
                  value={tab.toLowerCase()} 
                  className="px-4 py-2 rounded-full text-neutral-300 data-[state=active]:bg-blue-600 data-[state=active]:text-white transition-all"
                >
                  {tab}
                </TabsTrigger>
              ))}
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
                      <Card className="bg-white/5 backdrop-blur-lg border border-white/10 hover:border-white/20 transition-all duration-300 overflow-hidden">
                        <CardHeader className="relative pt-6 pb-4 border-b border-white/10">
                          <div className="absolute top-2 right-2 opacity-50 group-hover:opacity-100 transition-opacity">
                            <model.icon className="h-5 w-5 text-neutral-400" />
                          </div>
                          <CardTitle className="text-xl font-light text-white">
                            {model.name}
                          </CardTitle>
                          <CardDescription className="text-neutral-400 font-light">
                            {model.description}
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="p-6 space-y-4">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-neutral-400">Accuracy</span>
                              <span className="text-sm font-medium text-white">
                                {model.metrics.accuracy}%
                              </span>
                            </div>
                            <Progress 
                              value={model.metrics.accuracy} 
                              className="h-1.5 bg-white/10" 
                              indicatorClassName="bg-gradient-to-r from-blue-500 to-pink-500"
                            />
                          </div>
                          <div>
                            <span className="text-sm font-medium text-neutral-300 block mb-2">
                              Key Features
                            </span>
                            <ul className="space-y-1 text-sm text-neutral-400">
                              {model.features.map((feature) => (
                                <li key={feature} className="flex items-center space-x-2">
                                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                                  <span>{feature}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          <Button 
                            variant="outline"
                            className="w-full bg-white/5 border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-colors group"
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
                <Card className="bg-white/5 backdrop-blur-lg border border-white/10">
                  <CardHeader className="border-b border-white/10 pb-4">
                    <CardTitle className="text-xl font-light text-white">
                      Training History
                    </CardTitle>
                    <CardDescription className="text-neutral-400 font-light">
                      Recent model training sessions
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="py-6">
                    <div className="space-y-4">
                      {models.map((model) => (
                        <div 
                          key={model.id} 
                          className="flex items-center justify-between pb-3 border-b border-white/10 last:border-b-0"
                        >
                          <div className="flex items-center space-x-3">
                            <model.icon className="h-4 w-4 text-neutral-400" />
                            <span className="text-sm font-medium text-neutral-300">
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
      </div>
    </DashboardShell>
  );
}
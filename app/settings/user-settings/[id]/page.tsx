"use client";

import { useState } from 'react';
import { 
  Settings, 
  Brain, 
  Truck, 
  FileText, 
  MessageSquare, 
  Activity, 
  Shield, 
  Zap,
  ChevronRight,
  CheckCircle2,
  XCircle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from '@/components/ui/dialog';
import { useModelTraining } from '@/hooks/use-model-training';
import { useWorkflowOptimization } from '@/hooks/use-workflow-optimization';
import { toast } from 'sonner';
import { useParams } from 'next/navigation';
const { id } = useParams();


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

const workflows = [
  {
    id: 'document_processing',
    name: 'Document Processing',
    icon: FileText,
    description: 'Automated document analysis and processing',
    metrics: { efficiency: 92, accuracy: 98, processed: 1250 },
    status: 'Active',
    lastRun: '2 minutes ago',
    nextRun: '5 minutes',
    type: 'Scheduled'
  },
  {
    id: 'customer_support',
    name: 'Customer Support',
    icon: MessageSquare,
    description: 'AI-powered customer interaction handling',
    metrics: { efficiency: 88, accuracy: 95, processed: 3420 },
    status: 'Learning',
    lastRun: '5 minutes ago',
    nextRun: 'On demand',
    type: 'Event-driven'
  },
  {
    id: 'supply_chain',
    name: 'Supply Chain',
    icon: Truck,
    description: 'End-to-end supply chain optimization',
    metrics: { efficiency: 94, accuracy: 97, processed: 890 },
    status: 'Optimizing',
    lastRun: '15 minutes ago',
    nextRun: '1 hour',
    type: 'Scheduled'
  }
];

export default function AISettingsPage() {
  const [activeTab, setActiveTab] = useState('models');
  const { isTraining, startTraining } = useModelTraining();
  const { isOptimizing, optimizeWorkflow } = useWorkflowOptimization();

  const [modelSettings, setModelSettings] = useState(models.map(model => ({
    id: model.id,
    enabled: true,
    trainingFrequency: 'monthly',
    performanceThreshold: 90,
    selectedFeatures: model.features
  })));

  const [workflowSettings, setWorkflowSettings] = useState(workflows.map(workflow => ({
    id: workflow.id,
    enabled: true,
    optimizationLevel: 'standard',
    alertThreshold: 85,
    runSchedule: workflow.type === 'Scheduled' ? workflow.nextRun : 'On demand'
  })));

  const handleTrainModel = async (modelId: string) => {
    try {
      await startTraining({
        modelId,
        epochs: 10,
        batchSize: 32,
        learningRate: 0.001
      });
      toast.success(`Model ${modelId} training initiated successfully`);
    } catch (error) {
      toast.error(`Failed to train model ${modelId}`);
      console.error('Training error:', error);
    }
  };

  const handleOptimizeWorkflow = async (workflowId: string) => {
    try {
      await optimizeWorkflow({
        workflowId,
        parameters: {
          optimizationLevel: 'aggressive',
          targetMetrics: ['efficiency', 'accuracy']
        }
      });
      toast.success(`Workflow ${workflowId} optimization completed`);
    } catch (error) {
      toast.error(`Failed to optimize workflow ${workflowId}`);
      console.error('Optimization error:', error);
    }
  };

  const renderModelSettings = () => (
    <div className="space-y-6">
      {models.map((model, index) => (
        <Card 
          key={model.id} 
          className="bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-slate-800 dark:to-slate-900"
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <div className="flex items-center space-x-3">
              <model.icon className="h-6 w-6 text-primary" />
              <CardTitle>{model.name}</CardTitle>
            </div>
            <Switch 
              checked={modelSettings[index].enabled}
              onCheckedChange={(checked) => {
                const newSettings = [...modelSettings];
                newSettings[index].enabled = checked;
                setModelSettings(newSettings);
              }}
            />
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Training Frequency</Label>
                <Select 
                  value={modelSettings[index].trainingFrequency}
                  onValueChange={(value) => {
                    const newSettings = [...modelSettings];
                    newSettings[index].trainingFrequency = value;
                    setModelSettings(newSettings);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Training Frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="weekly">Weekly</SelectItem>
                    <SelectItem value="monthly">Monthly</SelectItem>
                    <SelectItem value="quarterly">Quarterly</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Performance Threshold</Label>
                <Input 
                  type="number" 
                  value={modelSettings[index].performanceThreshold}
                  onChange={(e) => {
                    const newSettings = [...modelSettings];
                    newSettings[index].performanceThreshold = Number(e.target.value);
                    setModelSettings(newSettings);
                  }}
                  className="w-full"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Features</Label>
              <div className="grid grid-cols-2 gap-2">
                {model.features.map((feature) => (
                  <div key={feature} className="flex items-center space-x-2">
                    <Switch 
                      checked={modelSettings[index].selectedFeatures.includes(feature)}
                      onCheckedChange={(checked) => {
                        const newSettings = [...modelSettings];
                        const currentFeatures = newSettings[index].selectedFeatures;
                        newSettings[index].selectedFeatures = checked
                          ? [...currentFeatures, feature]
                          : currentFeatures.filter(f => f !== feature);
                        setModelSettings(newSettings);
                      }}
                    />
                    <Label>{feature}</Label>
                  </div>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <Label>Current Performance</Label>
              <Progress value={model.metrics.accuracy} className="h-2" />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Accuracy: {model.metrics.accuracy}%</span>
                <span>Usage: {model.metrics.usage}%</span>
              </div>
            </div>
            <Button 
              className="w-full" 
              onClick={() => handleTrainModel(model.id)}
              disabled={isTraining}
            >
              {isTraining ? 'Training...' : 'Train Model'}
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderWorkflowSettings = () => (
    <div className="space-y-6">
      {workflows.map((workflow, index) => (
        <Card 
          key={workflow.id} 
          className="bg-gradient-to-br from-green-50 to-teal-100 dark:from-slate-800 dark:to-slate-900"
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <div className="flex items-center space-x-3">
              <workflow.icon className="h-6 w-6 text-primary" />
              <CardTitle>{workflow.name}</CardTitle>
            </div>
            <Switch 
              checked={workflowSettings[index].enabled}
              onCheckedChange={(checked) => {
                const newSettings = [...workflowSettings];
                newSettings[index].enabled = checked;
                setWorkflowSettings(newSettings);
              }}
            />
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Optimization Level</Label>
                <Select 
                  value={workflowSettings[index].optimizationLevel}
                  onValueChange={(value) => {
                    const newSettings = [...workflowSettings];
                    newSettings[index].optimizationLevel = value;
                    setWorkflowSettings(newSettings);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Optimization Level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="minimal">Minimal</SelectItem>
                    <SelectItem value="standard">Standard</SelectItem>
                    <SelectItem value="aggressive">Aggressive</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Run Schedule</Label>
                <Select 
                  value={workflowSettings[index].runSchedule}
                  onValueChange={(value) => {
                    const newSettings = [...workflowSettings];
                    newSettings[index].runSchedule = value;
                    setWorkflowSettings(newSettings);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Run Schedule" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="5 minutes">Every 5 Minutes</SelectItem>
                    <SelectItem value="1 hour">Hourly</SelectItem>
                    <SelectItem value="daily">Daily</SelectItem>
                    <SelectItem value="On demand">On Demand</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Label>Performance Metrics</Label>
              <Progress value={workflow.metrics.efficiency} className="h-2" />
              <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                <div>
                  <span>Efficiency: {workflow.metrics.efficiency}%</span>
                </div>
                <div>
                  <span>Status: {workflow.status}</span>
                </div>
                <div>
                  <span>Last Run: {workflow.lastRun}</span>
                </div>
                <div>
                  <span>Next Run: {workflow.nextRun}</span>
                </div>
              </div>
            </div>
            <Button 
              className="w-full" 
              onClick={() => handleOptimizeWorkflow(workflow.id)}
              disabled={isOptimizing}
            >
              {isOptimizing ? 'Optimizing...' : 'Optimize Workflow'}
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderGlobalSettings = () => (
    <Card className="bg-gradient-to-br from-purple-300 to-pink-200 dark:from-slate-800 dark:to-slate-900">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="w-5 h-5" /> Global AI Configuration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Global Performance Threshold</Label>
            <Input type="number" defaultValue={90} />
          </div>
          <div className="space-y-2">
            <Label>Logging Level</Label>
            <Select defaultValue="standard">
              <SelectTrigger>
                <SelectValue placeholder="Logging Level" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="minimal">Minimal</SelectItem>
                <SelectItem value="standard">Standard</SelectItem>
                <SelectItem value="comprehensive">Comprehensive</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-muted-foreground" />
              <Label>Enable Global Privacy Protection</Label>
            </div>
            <Switch />
          </div>
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-muted-foreground" />
              <Label>Automatic Model Updates</Label>
            </div>
            <Switch />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

  

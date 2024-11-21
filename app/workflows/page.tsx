"use client";

import { DashboardShell } from '@/components/dashboard/shell';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Activity, ArrowUpRight, Brain, FileText, MessageSquare, Settings, Truck } from 'lucide-react';
import { motion } from 'framer-motion';
import { WorkflowAnalytics } from '@/components/workflows/workflow-analytics';
import { useWorkflowOptimization } from '@/hooks/use-workflow-optimization';
import { toast } from 'sonner';

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

export default function WorkflowsPage() {
  const { isOptimizing, optimizeWorkflow } = useWorkflowOptimization();

  const handleOptimize = async (workflowId: string) => {
    try {
      await optimizeWorkflow({
        workflowId,
        parameters: {
          optimizationLevel: 'aggressive',
          targetMetrics: ['efficiency', 'accuracy']
        }
      });
      toast.success(`Workflow ${workflowId} optimized successfully!`);
    } catch (error) {
      toast.error('Optimization failed. Please try again.');
      console.error('Optimization error:', error);
    }
  };

  return (
    <DashboardShell>
      <div className="min-h-screen bg-gradient-to-br from-gray-950 to-slate-950 text-white p-6 lg:p-12">
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <h1 className="text-4xl lg:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-600 mb-4">
            Workflow Intelligence
          </h1>
          <p className="text-neutral-400 max-w-2xl">
            Intelligent automation that transforms your business processes with precision and adaptability.
          </p>
        </motion.header>

        <Tabs defaultValue="active" className="space-y-6">
          <TabsList className="bg-neutral-900 border border-neutral-800 p-1 rounded-xl">
            <TabsTrigger 
              value="active" 
              className="data-[state=active]:bg-blue-600/20 data-[state=active]:text-blue-400 rounded-lg transition-all"
            >
              Active Workflows
            </TabsTrigger>
            <TabsTrigger 
              value="analytics" 
              className="data-[state=active]:bg-blue-600/20 data-[state=active]:text-blue-400 rounded-lg transition-all"
            >
              Analytics
            </TabsTrigger>
            <TabsTrigger 
              value="settings" 
              className="data-[state=active]:bg-blue-600/20 data-[state=active]:text-blue-400 rounded-lg transition-all"
            >
              Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ staggerChildren: 0.1 }}
              className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {workflows.map((workflow) => (
                <motion.div
                  key={workflow.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="bg-neutral-900 border border-neutral-800 hover:border-blue-600/50 transition-all duration-300 group">
                    <CardHeader className="pb-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <workflow.icon className="h-6 w-6 text-blue-400 group-hover:text-blue-300 transition-colors" />
                          <CardTitle className="text-xl text-neutral-200">{workflow.name}</CardTitle>
                        </div>
                        <span className="text-xs px-2 py-1 bg-blue-900/30 text-blue-300 rounded-full">
                          {workflow.status}
                        </span>
                      </div>
                      <CardDescription className="text-neutral-500 pt-2">
                        {workflow.description}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-neutral-400">
                            <span className="text-sm">Efficiency</span>
                            <span className="text-sm font-medium text-blue-300">
                              {workflow.metrics.efficiency}%
                            </span>
                          </div>
                          <Progress 
                            value={workflow.metrics.efficiency} 
                            className="h-2 bg-neutral-800" 
                            indicatorClassName="bg-blue-600"
                          />
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4 text-sm text-neutral-400">
                          {[
                            { label: 'Type', value: workflow.type },
                            { label: 'Last Run', value: workflow.lastRun },
                            { label: 'Next Run', value: workflow.nextRun }
                          ].map(({ label, value }) => (
                            <div key={label}>
                              <p className="text-xs uppercase tracking-wider mb-1">{label}</p>
                              <p className="font-medium text-neutral-200">{value}</p>
                            </div>
                          ))}
                        </div>
                        
                        <Button 
                          variant="outline"
                          className="w-full bg-neutral-800 border-neutral-700 hover:bg-blue-900/30 text-blue-300 hover:text-blue-200 transition-all"
                          onClick={() => handleOptimize(workflow.id)}
                          disabled={isOptimizing}
                        >
                          {isOptimizing ? 'Optimizing...' : 'Optimize Workflow'}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </TabsContent>

          <TabsContent value="analytics" className="bg-neutral-900 rounded-xl border border-neutral-800">
            <WorkflowAnalytics />
          </TabsContent>

          <TabsContent value="settings">
            <Card className="bg-neutral-900 border border-neutral-800">
              <CardHeader>
                <CardTitle className="text-3xl bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-600">
                  Workflow Settings
                </CardTitle>
                <CardDescription className="text-neutral-500">
                  Configure global workflow parameters and optimization strategies
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {workflows.map((workflow) => (
                    <div 
                      key={workflow.id} 
                      className="flex items-center justify-between border-b border-neutral-800 pb-4 last:border-b-0 hover:bg-neutral-800/50 transition-all p-3 rounded-lg"
                    >
                      <div className="flex items-center space-x-3">
                        <workflow.icon className="h-5 w-5 text-blue-400" />
                        <div>
                          <p className="font-semibold text-neutral-200">{workflow.name}</p>
                          <p className="text-sm text-neutral-500">{workflow.type}</p>
                        </div>
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="bg-neutral-800 border-neutral-700 text-blue-300 hover:bg-blue-900/30"
                      >
                        Configure
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardShell>
  );
}
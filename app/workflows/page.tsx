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
    } catch (error) {
      console.error('Optimization error:', error);
    }
  };

  return (
    <DashboardShell>
      <div className="flex flex-col gap-8 p-8">
        <header>
          <h1 className=" font-bold tracking-tight  text-transparent  text-5xl bg-clip-text bg-gradient-to-r from-blue-500 to-pink-500">AI Workflows</h1>
          <p className="text-gray-400">
            Manage and monitor your automated business processes
          </p>
        </header>

        <Tabs defaultValue="active" className="space-y-4">
          <TabsList>
            <TabsTrigger value="active">Active Workflows</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {workflows.map((workflow) => (
                <motion.div
                  key={workflow.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="overflow-hidden">
                    <CardHeader className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <workflow.icon className="h-5 w-5 text-primary" />
                        <CardTitle>{workflow.name}</CardTitle>
                      </div>
                      <CardDescription>{workflow.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Efficiency</span>
                            <span className="text-sm font-medium">{workflow.metrics.efficiency}%</span>
                          </div>
                          <Progress value={workflow.metrics.efficiency} className="h-2" />
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-muted-foreground">Status</p>
                            <p className="font-medium">{workflow.status}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Type</p>
                            <p className="font-medium">{workflow.type}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Last Run</p>
                            <p className="font-medium">{workflow.lastRun}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Next Run</p>
                            <p className="font-medium">{workflow.nextRun}</p>
                          </div>
                        </div>
                        <Button 
                          className="w-full"
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
            </div>
          </TabsContent>

          <TabsContent  value="analytics">
            <WorkflowAnalytics />
          </TabsContent>

          <TabsContent value="settings">
            <Card>
              <CardHeader>
                <CardTitle className=' text-transparent  text-3xl bg-clip-text bg-gradient-to-r from-blue-500 to-violet-600'>Workflow Settings</CardTitle>
                <CardDescription>Configure global workflow parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {workflows.map((workflow) => (
                    <div key={workflow.id} className="flex items-center justify-between border pb-4">
                      <div className="flex items-center space-x-2">
                        <workflow.icon className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <p className="font-bold text-gray-100">{workflow.name}</p>
                          <p className="text-sm text-neutral-100">{workflow.type}</p>
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
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
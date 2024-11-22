"use client";

import { DashboardShell } from '@/components/dashboard/shell';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { FileText, MessageSquare, Truck, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
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
      toast.success(`Workflow ${workflowId} optimized successfully!`, {
        description: 'Performance metrics have been enhanced.',
        icon: <Zap className="text-yellow-400" />
      });
    } catch (error) {
      toast.error('Optimization failed', {
        description: 'Please review and try again.'
      });
      console.error('Optimization error:', error);
    }
  };

  return (
    <DashboardShell>
      <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-gray-950 to-black text-white p-6 lg:p-12 relative overflow-hidden">
        {/* Subtle background grid */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,_rgba(14,_165,_233,_0.05)_0%,_transparent_70%)] opacity-50"></div>
        
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="relative z-10 mb-12"
        >
          <h1 className="text-4xl lg:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-600 to-blue-600 mb-4 tracking-tight">
            Workflow Intelligence
          </h1>
          <p className="text-neutral-400 max-w-2xl text-lg leading-relaxed">
            Precision-driven automation that transforms complex business processes into seamless, intelligent workflows🌟.
          </p>
        </motion.header>

        <Tabs defaultValue="active" className="space-y-6 relative z-10">
          <TabsList className="bg-neutral-900/60 backdrop-blur-md border border-neutral-800/50 p-1 rounded-xl shadow-lg">
            {['active', 'analytics', 'settings'].map((tab) => (
              <TabsTrigger 
                key={tab}
                value={tab} 
                className="data-[state=active]:bg-blue-600/20 data-[state=active]:text-blue-300 rounded-lg transition-all duration-300 capitalize"
              >
                {tab}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="active" className="space-y-6">
            <AnimatePresence>
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
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-neutral-900/60 backdrop-blur-lg border border-neutral-800/50 hover:border-blue-600/30 transition-all duration-300 group shadow-xl hover:shadow-2xl hover:shadow-blue-900/20">
                      <CardHeader className="pb-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <workflow.icon className="h-6 w-6 text-blue-400 group-hover:text-blue-300 transition-colors opacity-80 group-hover:opacity-100" />
                            <CardTitle className="text-xl text-neutral-100">{workflow.name}</CardTitle>
                          </div>
                          <motion.span 
                            whileHover={{ scale: 1.05 }}
                            className="text-xs px-2.5 py-1 bg-blue-900/30 text-blue-300 rounded-full"
                          >
                            {workflow.status}
                          </motion.span>
                        </div>
                        <CardDescription className="text-neutral-500 pt-2">
                          {workflow.description}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-5">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-neutral-400">
                              <span className="text-sm font-medium">Efficiency</span>
                              <span className="text-sm font-bold text-blue-300">
                                {workflow.metrics.efficiency}%
                              </span>
                            </div>
                            <Progress 
                              value={workflow.metrics.efficiency} 
                              className="h-1.5 bg-neutral-800/50" 
                              indicatorClassName="bg-blue-600 rounded-full"
                            />
                          </div>
                          
                          <div className="grid grid-cols-3 gap-3 text-xs text-neutral-400">
                            {[
                              { label: 'Type', value: workflow.type },
                              { label: 'Last Run', value: workflow.lastRun },
                              { label: 'Next Run', value: workflow.nextRun }
                            ].map(({ label, value }) => (
                              <div key={label} className="bg-neutral-900/30 p-2.5 rounded-lg text-center">
                                <p className="text-[10px] uppercase tracking-wider mb-1 text-neutral-500">{label}</p>
                                <p className="font-semibold text-neutral-200">{value}</p>
                              </div>
                            ))}
                          </div>
                          
                          <motion.div
                            whileTap={{ scale: 0.98 }}
                          >
                            <Button 
                              variant="outline"
                              className="w-full bg-neutral-800/30 border-neutral-700/50 hover:bg-blue-900/20 text-blue-300 hover:text-blue-200 transition-all backdrop-blur-sm"
                              onClick={() => handleOptimize(workflow.id)}
                              disabled={isOptimizing}
                            >
                              {isOptimizing ? 'Optimizing...' : 'Optimize Workflow'}
                            </Button>
                          </motion.div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </motion.div>
            </AnimatePresence>
          </TabsContent>

          <TabsContent value="analytics" className="bg-neutral-900/60 backdrop-blur-lg rounded-xl border border-neutral-800/50 shadow-xl">
            <WorkflowAnalytics />
          </TabsContent>

          <TabsContent value="settings">
            <Card className="bg-neutral-900/60 backdrop-blur-lg border border-neutral-800/50 shadow-xl">
              <CardHeader>
                <CardTitle className="text-3xl bg-clip-text text-transparent bg-gradient-to-r from-cyan-300 to-blue-500">
                  Workflow Settings
                </CardTitle>
                <CardDescription className="text-neutral-500">
                  Fine-tune global workflow parameters and optimization strategies
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {workflows.map((workflow) => (
                    <motion.div 
                      key={workflow.id} 
                      whileHover={{ scale: 1.01 }}
                      className="flex items-center justify-between border-b border-neutral-800/50 pb-4 last:border-b-0 hover:bg-neutral-800/20 transition-all p-3 rounded-lg"
                    >
                      <div className="flex items-center space-x-3">
                        <workflow.icon className="h-5 w-5 text-blue-400 opacity-70" />
                        <div>
                          <p className="font-semibold text-neutral-200">{workflow.name}</p>
                          <p className="text-sm text-neutral-500">{workflow.type}</p>
                        </div>
                      </div>
                      <motion.div whileTap={{ scale: 0.95 }}>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="bg-neutral-800/30 border-neutral-700/50 text-blue-300 hover:bg-blue-900/20 backdrop-blur-sm"
                        >
                          Configure
                        </Button>
                      </motion.div>
                    </motion.div>
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
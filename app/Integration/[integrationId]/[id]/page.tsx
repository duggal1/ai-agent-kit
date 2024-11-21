"use client"
import { useRouter } from "next/router";
import { DashboardShell } from "@/components/dashboard/shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Activity,
  FileText,
  MessageSquare,
  Truck,
  CheckCircle,
  ArrowRight,
  Settings,
} from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { useEffect, useState } from "react";

export default function WorkflowsPage() {
  const [isClient, setIsClient] = useState(false);
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const router = isClient ? useRouter() : null;

  useEffect(() => {
    setIsClient(true);
  }, []);

  const handleNavigate = (workflowId: string) => {
    if (router) {
      router.push(`/automation/automate/${workflowId}`);
    
  }
    

const workflows = [
  {
    id: "document_processing",
    name: "Document Processing",
    icon: FileText,
    description: "Automated document analysis and processing",
    metrics: { efficiency: 92, accuracy: 98, processed: 1250 },
    status: "Active",
    lastRun: "2 minutes ago",
    nextRun: "5 minutes",
    type: "Scheduled",
  },
  {
    id: "customer_support",
    name: "Customer Support",
    icon: MessageSquare,
    description: "AI-powered customer interaction handling",
    metrics: { efficiency: 88, accuracy: 95, processed: 3420 },
    status: "Learning",
    lastRun: "5 minutes ago",
    nextRun: "On demand",
    type: "Event-driven",
  },
  {
    id: "supply_chain",
    name: "Supply Chain",
    icon: Truck,
    description: "End-to-end supply chain optimization",
    metrics: { efficiency: 94, accuracy: 97, processed: 890 },
    status: "Optimizing",
    lastRun: "15 minutes ago",
    nextRun: "1 hour",
    type: "Scheduled",
  },
];




  return (
    <DashboardShell>
      <div className="flex flex-col gap-8 p-8 bg-gradient-to-br from-gray-900 via-purple-900 to-black text-white rounded-lg shadow-xl">
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            AI Workflow Manager
          </h1>
          <p className="text-muted-foreground">Manage and optimize your AI-powered workflows</p>
        </header>

        <Tabs defaultValue="active" className="space-y-8">
          {/* Tabs List */}
          <TabsList className="bg-gradient-to-r from-blue-800 to-pink-500 p-1 rounded-lg shadow-lg">
            <TabsTrigger value="active" className="text-white/90">Active Workflows</TabsTrigger>
            <TabsTrigger value="analytics" className="text-white/90">Analytics</TabsTrigger>
            <TabsTrigger value="settings" className="text-white/90">Settings</TabsTrigger>
          </TabsList>

          {/* Active Workflows */}
          <TabsContent value="active" className="space-y-8">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {workflows.map((workflow) => (
                <motion.div
                  key={workflow.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="bg-gradient-to-r from-gray-800 via-gray-900 to-gray-950 text-white overflow-hidden shadow-lg rounded-lg hover:scale-[1.02] transition-transform">
                    <CardHeader className="space-y-2">
                      <div className="flex items-center space-x-3">
                        <workflow.icon className="h-6 w-6 text-indigo-400" />
                        <CardTitle>{workflow.name}</CardTitle>
                      </div>
                      <CardDescription className="text-sm">{workflow.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Metrics */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Efficiency</span>
                          <span className="text-sm font-medium">{workflow.metrics.efficiency}%</span>
                        </div>
                        <Progress value={workflow.metrics.efficiency} className="h-2 bg-gray-700 rounded-full" />
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
                        className="w-full bg-gradient-to-r from-pink-500 to-indigo-500 hover:opacity-90 text-white"
                        onClick={() => handleNavigate(workflow.id)}
                      >
                        View Details
                        <ArrowRight className="h-4 w-4 ml-2" />
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics">
            <Card className="bg-gray-900 text-white shadow-lg">
              <CardHeader>
                <CardTitle className="bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
                  Workflow Analytics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p>Interactive analytics visualization goes here!</p>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings">
            <Card className="bg-gray-900 text-white shadow-lg">
              <CardHeader>
                <CardTitle>Workflow Settings</CardTitle>
                <CardDescription>Configure global workflow parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {workflows.map((workflow) => (
                  <div key={workflow.id} className="flex items-center justify-between border-b border-gray-700 pb-4">
                    <div className="flex items-center space-x-2">
                      <workflow.icon className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="font-medium">{workflow.name}</p>
                        <p className="text-sm text-muted-foreground">{workflow.type}</p>
                      </div>
                    </div>
                    <Button variant="outline" size="sm" className="text-indigo-500 hover:bg-indigo-500 hover:text-white">
                      Configure
                    </Button>
                    <Button variant="outline" size="sm" className="text-indigo-500 hover:bg-indigo-500 hover:text-white">
                    Configure
                  </Button>
                  </div>
                  
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardShell>
  );
}  }
  



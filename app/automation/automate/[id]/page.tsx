"use client";

import { useRouter } from "next/router";
import { useParams } from "next/navigation";
import { DashboardShell } from "@/components/dashboard/shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { motion } from "framer-motion";
import { Sparkles, ArrowRight } from "lucide-react";

const workflowDetails = {
  id: "unique_id_placeholder",
  name: "Workflow Details",
  description: "Advanced insights and controls for your selected workflow",
  metrics: {
    efficiency: 95,
    accuracy: 99,
    processed: 4321,
  },
  status: "Optimizing",
  type: "Scheduled",
  lastRun: "3 minutes ago",
  nextRun: "30 minutes",
};

export default function WorkflowDetailsPage() {
  const params = useParams();
  const router = useRouter();

  const workflowId = params?.id || "Unknown Workflow";

  return (
    <DashboardShell>
      <div className="p-8 bg-gradient-to-br from-black via-gray-900 to-purple-900 text-white rounded-lg shadow-2xl">
        <header className="text-center space-y-4">
          <h1 className="text-5xl font-extrabold tracking-tight bg-gradient-to-r from-pink-500 via-red-500 to-yellow-500 bg-clip-text text-transparent">
            {workflowDetails.name} - {workflowId}
          </h1>
          <p className="text-muted-foreground text-lg">{workflowDetails.description}</p>
        </header>

        <div className="space-y-8 mt-10">
          {/* Workflow Metrics */}
          <Card className="bg-gradient-to-br from-gray-800 to-gray-950 shadow-lg text-white rounded-xl overflow-hidden">
            <CardHeader>
              <CardTitle className="text-3xl bg-gradient-to-r from-green-400 to-teal-400 bg-clip-text text-transparent">
                Workflow Metrics
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <p className="text-muted-foreground">Efficiency</p>
                  <Progress
                    value={workflowDetails.metrics.efficiency}
                    className="h-2 bg-gray-700 rounded-full"
                  />
                  <p className="font-bold text-lg">{workflowDetails.metrics.efficiency}%</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Accuracy</p>
                  <Progress
                    value={workflowDetails.metrics.accuracy}
                    className="h-2 bg-gray-700 rounded-full"
                  />
                  <p className="font-bold text-lg">{workflowDetails.metrics.accuracy}%</p>
                </div>
              </div>
              <div className="flex justify-between text-sm">
                <div>
                  <p className="text-muted-foreground">Status</p>
                  <p className="font-bold">{workflowDetails.status}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Type</p>
                  <p className="font-bold">{workflowDetails.type}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-6 text-sm">
                <div>
                  <p className="text-muted-foreground">Last Run</p>
                  <p className="font-bold">{workflowDetails.lastRun}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Next Run</p>
                  <p className="font-bold">{workflowDetails.nextRun}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Actions Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="bg-gradient-to-br from-purple-700 to-pink-600 shadow-lg rounded-xl text-white overflow-hidden">
              <CardHeader>
                <CardTitle className="text-2xl">Actions</CardTitle>
              </CardHeader>
              <CardContent className="flex space-x-4">
                <Button
                  className="bg-gradient-to-r from-blue-500 to-indigo-500 text-white hover:opacity-90 w-full"
                  onClick={() => alert("Optimization Started!")}
                >
                  Optimize Workflow <Sparkles className="ml-2 h-5 w-5" />
                </Button>
                <Button
                  className="bg-gradient-to-r from-red-500 to-pink-500 text-white hover:opacity-90 w-full"
                  onClick={() => router.push(`/dashboard`)}
                >
                  Back to Dashboard <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </DashboardShell>
  );
}
"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { DashboardShell } from "@/components/dashboard/shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { motion } from "framer-motion";
import { Sparkles, ArrowRight, Settings } from "lucide-react";
import { toast } from "sonner";

export default function AutomatePage() {
  const { id } = useParams();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // Handle API calls
  const handleAction = async (type: string, modelType?: string, data?: object) => {
    setLoading(true);
    setResult(null);

    try {
      let response;

      switch (type) {
        case "train":
        case "predict":
        case "workflow":
          response = await fetch("/api/ai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type, modelType, data }),
          });
          break;

        case "optimize":
        case "analyze":
          response = await fetch("/api/workflows", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type, workflowId: id, data }),
          });
          break;

        default:
          throw new Error("Invalid action type");
      }

      const responseData = await response.json();
      setResult(responseData);
      toast.success("Action completed successfully!");
    } catch (error) {
      console.error("Error performing action:", error);
      toast.error("Failed to complete action.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardShell>
      <div className="p-8 bg-gradient-to-br from-black via-gray-900 to-purple-900 text-white rounded-lg shadow-2xl">
        {/* Header Section */}
        <header className="text-center space-y-4">
          <h1 className="text-5xl font-extrabold tracking-tight bg-gradient-to-r from-pink-500 via-red-500 to-yellow-500 bg-clip-text text-transparent">
            Automate Workflow - {id}
          </h1>
          <p className="text-muted-foreground text-lg">
            Leverage AI to train, optimize, and analyze your workflows.
          </p>
        </header>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {[
            { label: "Train Model", type: "train", modelType: "custom_model" },
            { label: "Predict Outcome", type: "predict", modelType: "predictive_model" },
            { label: "Optimize Workflow", type: "optimize" },
            { label: "Analyze Workflow", type: "analyze" },
            { label: "Run Workflow", type: "workflow", modelType: "workflow_optimizer" },
          ].map((action) => (
            <Card key={action.type} className="bg-gradient-to-r from-gray-800 to-gray-950 text-white shadow-lg">
              <CardHeader>
                <CardTitle>{action.label}</CardTitle>
              </CardHeader>
              <CardContent>
                <Button
                  className="w-full bg-gradient-to-r from-indigo-500 to-purple-500 hover:opacity-90 text-white"
                  disabled={loading}
                  onClick={() => handleAction(action.type, action.modelType)}
                >
                  {loading && result === null ? "Processing..." : action.label}
                  <Sparkles className="ml-2 h-5 w-5" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </motion.div>

        {/* Result Section */}
        <div className="mt-12">
          {loading ? (
            <p className="text-center text-lg">Loading...</p>
          ) : result ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-gradient-to-br from-gray-900 to-gray-800 text-white p-6 rounded-xl shadow-lg"
            >
              <h3 className="text-xl font-bold bg-gradient-to-r from-green-400 to-teal-400 bg-clip-text text-transparent">
                Result
              </h3>
              <pre className="text-sm mt-4 whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
            </motion.div>
          ) : (
            <p className="text-center bg-gradient-to-r from-pink-500 to-red-600 bg-clip-text text-transparent">No results yet. Perform an action above.</p>
          )}
        </div>
      </div>
    </DashboardShell>
  );
}
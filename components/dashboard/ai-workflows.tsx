"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, ArrowUpRight, Brain, CheckCircle2, Clock } from "lucide-react";

const workflows = [
  {
    name: "Document Processing",
    status: "Active",
    efficiency: "98%",
    icon: Brain,
  },
  {
    name: "Customer Support",
    status: "Learning",
    efficiency: "87%",
    icon: Activity,
  },
  {
    name: "Supply Chain",
    status: "Optimizing",
    efficiency: "92%",
    icon: ArrowUpRight,
  },
];

export function AIWorkflows() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Active AI Workflows</CardTitle>
        <CardDescription>Real-time workflow performance and status</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {workflows.map((workflow) => (
            <div
              key={workflow.name}
              className="flex items-center justify-between space-x-4 rounded-md border p-4"
            >
              <div className="flex items-center space-x-4">
                <workflow.icon className="h-6 w-6 text-primary" />
                <div>
                  <p className="text-sm font-medium">{workflow.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {workflow.status}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">{workflow.efficiency}</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Activity, ArrowUpRight, Brain, Clock } from "lucide-react";

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
    <Card className="rounded-xl bg-gradient-to-br from-gray-800 via-gray-900 to-black text-white shadow-lg">
      <CardHeader>
        <CardTitle className="text-xl font-bold">Active AI Workflows</CardTitle>
        <CardDescription className="text-sm text-gray-400">
          Real-time workflow performance and status
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {workflows.map((workflow) => (
            <div
              key={workflow.name}
              className="group flex items-center justify-between space-x-4 rounded-lg bg-gradient-to-r from-gray-700 to-gray-900 p-4 shadow-lg hover:from-indigo-500 hover:to-purple-600 transition-all duration-300"
            >
              <div className="flex items-center space-x-4">
                <workflow.icon className="h-8 w-8 text-indigo-400 group-hover:text-white" />
                <div>
                  <p className="text-base font-semibold group-hover:text-white">
                    {workflow.name}
                  </p>
                  <p className="text-sm text-gray-400 group-hover:text-gray-200">
                    {workflow.status}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Clock className="h-5 w-5 text-gray-400 group-hover:text-white" />
                <span className="text-sm font-medium group-hover:text-white">
                  {workflow.efficiency}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
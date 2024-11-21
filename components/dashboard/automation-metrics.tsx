"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown } from "lucide-react";

const metrics = [
  {
    name: "Task Automation Rate",
    value: 85,
    change: "+12%",
    target: 90,
    trend: "up",
  },
  {
    name: "Process Efficiency",
    value: 92,
    change: "+5%",
    target: 95,
    trend: "up",
  },
  {
    name: "Cost Reduction",
    value: 78,
    change: "+15%",
    target: 85,
    trend: "up",
  },
  {
    name: "Error Reduction",
    value: 88,
    change: "-2%",
    target: 90,
    trend: "down",
  },
];

export function AutomationMetrics() {
  return (
    <Card className="rounded-xl bg-gradient-to-br from-gray-800 via-gray-900 to-black text-white shadow-xl">
      <CardHeader>
        <CardTitle className="text-2xl font-bold">
          Automation Metrics
        </CardTitle>
        <CardDescription className="text-sm text-gray-400">
          Key performance indicators for automated processes
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {metrics.map((metric) => (
            <div
              key={metric.name}
              className="space-y-2 rounded-lg p-4 bg-gradient-to-r from-gray-700 to-gray-900 shadow hover:shadow-lg transition duration-300"
            >
              <div className="flex items-center justify-between">
                <p className="text-base font-medium">{metric.name}</p>
                <Badge
                  variant="outline"
                  className={`flex items-center space-x-2 ${
                    metric.trend === "up" ? "bg-green-500/10" : "bg-red-500/10"
                  }`}
                >
                  {metric.trend === "up" ? (
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-500" />
                  )}
                  <span>{metric.change}</span>
                </Badge>
              </div>
              <Progress value={metric.value} className="h-2 rounded-full" />
              <div className="text-sm flex justify-between">
                <span>
                  Current: <strong>{metric.value}%</strong>
                </span>
                <span>
                  Target: <strong>{metric.target}%</strong>
                </span>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-6 flex items-center justify-between">
          <Button
            variant="outline"
            className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white hover:opacity-80"
          >
            View Full Report
          </Button>
          <p className="text-sm text-gray-400">
            Last updated: <strong>1 day ago</strong>
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
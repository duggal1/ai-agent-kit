"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, Activity } from "lucide-react";

const metrics = [
  {
    name: "Model Accuracy",
    value: 95.8,
    change: "+2.3%",
    icon: Brain,
  },
  {
    name: "Processing Speed",
    value: 88.2,
    change: "+5.1%",
    icon: Zap,
  },
  {
    name: "System Load",
    value: 42.5,
    change: "-3.2%",
    icon: Activity,
  },
];

export function PerformanceMonitor() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {metrics.map((metric) => (
        <Card key={metric.name}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              {metric.name}
            </CardTitle>
            <metric.icon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metric.value}%</div>
            <Progress value={metric.value} className="mt-2" />
            <p className={`mt-2 text-xs ${metric.change.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
              {metric.change} from last month
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

const metrics = [
  {
    name: "Task Automation Rate",
    value: 85,
    change: "+12%",
    target: 90,
  },
  {
    name: "Process Efficiency",
    value: 92,
    change: "+5%",
    target: 95,
  },
  {
    name: "Cost Reduction",
    value: 78,
    change: "+15%",
    target: 85,
  },
];

export function AutomationMetrics() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Automation Metrics</CardTitle>
        <CardDescription>Key performance indicators for automated processes</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {metrics.map((metric) => (
            <div key={metric.name} className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">{metric.name}</p>
                <span className="text-sm text-muted-foreground">
                  {metric.value}% / {metric.target}%
                </span>
              </div>
              <Progress value={metric.value} />
              <p className="text-xs text-green-500">{metric.change} from last month</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
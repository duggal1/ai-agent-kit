"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const analyticsData = [
  { date: '2024-03-01', efficiency: 85, accuracy: 92, throughput: 78 },
  { date: '2024-03-02', efficiency: 87, accuracy: 94, throughput: 82 },
  { date: '2024-03-03', efficiency: 92, accuracy: 95, throughput: 85 },
  { date: '2024-03-04', efficiency: 89, accuracy: 93, throughput: 80 },
  { date: '2024-03-05', efficiency: 94, accuracy: 96, throughput: 88 },
];

export function WorkflowAnalytics() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card className="col-span-2">
        <CardHeader>
          <CardTitle>Performance Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={analyticsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="efficiency" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2} 
                />
                <Line 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2} 
                />
                <Line 
                  type="monotone" 
                  dataKey="throughput" 
                  stroke="hsl(var(--chart-3))" 
                  strokeWidth={2} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
import React from 'react';
import { DashboardShell } from '@/components/dashboard/shell';
import { AIWorkflows } from '@/components/dashboard/ai-workflows';
import { AutomationMetrics } from '@/components/dashboard/automation-metrics';
import { ProcessInsights } from '@/components/dashboard/process-insights';
import { Card } from '@/components/ui/card';
import { 
  ChartBarIcon, 
  SparklesIcon, 
  ServerStackIcon, 
  DocumentMagnifyingGlassIcon 
} from '@heroicons/react/24/solid';

export default function Home() {
  return (
    <DashboardShell>
      <div className="p-6 bg-black/30 min-h-screen">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-blue-400 flex items-center">
            <SparklesIcon className="w-10 h-10 mr-3 text-blue-600" />
            Enterprise AI Dashboard
          </h1>
          <p className="text-gray-100 mt-2">
            Monitor and manage your AI-powered business processes
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-6 bg-gray-800 shadow-lg rounded-xl">
            <div className="flex items-center mb-4">
              <ChartBarIcon className="w-8 h-8 text-blue-500 mr-3" />
              <h2 className="text-xl font-semibold text-gray-800">AI Workflows</h2>
            </div>
            <AIWorkflows />
          </Card>

          <Card className="p-6 bg-blue-800 shadow-lg rounded-xl">
            <div className="flex items-center mb-4">
              <ServerStackIcon className="w-8 h-8 text-green-500 mr-3" />
              <h2 className="text-xl font-semibold text-gray-800">Automation Metrics</h2>
            </div>
            <AutomationMetrics />
          </Card>

          <Card className="p-6 bg-white/20 shadow-lg rounded-xl col-span-full">
            <div className="flex items-center mb-4">
              <DocumentMagnifyingGlassIcon className="w-8 h-8 text-purple-500 mr-3" />
              <h2 className="text-xl font-semibold text-gray-800">Process Insights</h2>
            </div>
            <ProcessInsights />
          </Card>
        </div>
      </div>
    </DashboardShell>
  );
}
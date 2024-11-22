import React from 'react';
import { DashboardShell } from '@/components/dashboard/shell';
import { AIWorkflows } from '@/components/dashboard/ai-workflows';
import { AutomationMetrics } from '@/components/dashboard/automation-metrics';
import { ProcessInsights } from '@/components/dashboard/process-insights';
import { 
  ChartBarIcon, 
  SparklesIcon, 
  ServerStackIcon, 
  DocumentMagnifyingGlassIcon 
} from '@heroicons/react/24/outline';

export default function Home() {
  return (
    <DashboardShell>
      <div className="bg-gradient-to-br from-black via-neutral-900 to-black p-8 min-h-screen">
        <div className="space-y-8 mx-auto max-w-7xl">
          {/* Header */}
          <header className="mb-12 text-center">
            <div className="flex justify-center items-center mb-4">
              <SparklesIcon className="mr-4 w-12 h-12 text-emerald-500 animate-pulse" />
              <h1 className="bg-clip-text bg-gradient-to-r from-blue-600 via-purple-600 to-pink-500 font-extrabold text-6xl text-transparent tracking-tight">
                Enterprise AI Control Center
              </h1>
            </div>
            <p className="mx-auto max-w-2xl text-lg text-neutral-400">
              Intelligent process monitoring and optimization at your fingertips
            </p>
          </header>

          {/* Dashboard Grid */}
          <div className="gap-8 grid grid-cols-1 md:grid-cols-2">
            {/* AI Workflows Card */}
            <div className="border-neutral-800 bg-neutral-900/60 hover:shadow-2xl backdrop-blur-lg p-6 border hover:border-blue-500/50 rounded-2xl transform transition-all duration-300 hover:scale-[1.02]">
              <div className="flex items-center mb-6">
                <ChartBarIcon className="mr-4 w-10 h-10 text-blue-500 animate-bounce" />
                <h2 className="font-light text-2xl text-white tracking-wide">
                  AI Workflows
                </h2>
              </div>
              <AIWorkflows />
            </div>

            {/* Automation Metrics Card */}
            <div className="border-neutral-800 hover:border-green-500/50 bg-neutral-900/60 hover:shadow-2xl backdrop-blur-lg p-6 border rounded-2xl transform transition-all duration-300 hover:scale-[1.02]">
              <div className="flex items-center mb-6">
                <ServerStackIcon className="mr-4 w-10 h-10 text-green-500 animate-pulse" />
                <h2 className="font-light text-2xl text-white tracking-wide">
                  Automation Metrics
                </h2>
              </div>
              <AutomationMetrics />
            </div>

            {/* Process Insights Card */}
            <div className="border-neutral-800 hover:border-purple-500/50 md:col-span-full bg-neutral-900/60 hover:shadow-2xl backdrop-blur-lg p-6 border rounded-2xl transform transition-all duration-300 hover:scale-[1.01]">
              <div className="flex items-center mb-6">
                <DocumentMagnifyingGlassIcon className="mr-4 w-10 h-10 text-purple-500 animate-pulse" />
                <h2 className="font-light text-2xl text-white tracking-wide">
                  Process Insights
                </h2>
              </div>
              <ProcessInsights />
            </div>
          </div>
        </div>
      </div>
    </DashboardShell>
  );
}
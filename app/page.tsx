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
      <div className="min-h-screen bg-gradient-to-br from-black via-neutral-900 to-black p-8">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <header className="mb-12 text-center">
            <div className="flex justify-center items-center mb-4">
              <SparklesIcon className="w-12 h-12 text-emerald-500 mr-4 animate-pulse" />
              <h1 className="text-6xl font-extrabold  tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-purple-600 to-pink-500">
                Enterprise AI Control Center
              </h1>
            </div>
            <p className="text-neutral-400 text-lg max-w-2xl mx-auto">
              Intelligent process monitoring and optimization at your fingertips
            </p>
          </header>

          {/* Dashboard Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* AI Workflows Card */}
            <div className="bg-neutral-900/60 backdrop-blur-lg border border-neutral-800 rounded-2xl p-6 transform transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl hover:border-blue-500/50">
              <div className="flex items-center mb-6">
                <ChartBarIcon className="w-10 h-10 text-blue-500 mr-4 animate-bounce" />
                <h2 className="text-2xl font-light text-white tracking-wide">
                  AI Workflows
                </h2>
              </div>
              <AIWorkflows />
            </div>

            {/* Automation Metrics Card */}
            <div className="bg-neutral-900/60 backdrop-blur-lg border border-neutral-800 rounded-2xl p-6 transform transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl hover:border-green-500/50">
              <div className="flex items-center mb-6">
                <ServerStackIcon className="w-10 h-10 text-green-500 mr-4 animate-pulse" />
                <h2 className="text-2xl font-light text-white tracking-wide">
                  Automation Metrics
                </h2>
              </div>
              <AutomationMetrics />
            </div>

            {/* Process Insights Card */}
            <div className="md:col-span-full bg-neutral-900/60 backdrop-blur-lg border border-neutral-800 rounded-2xl p-6 transform transition-all duration-300 hover:scale-[1.01] hover:shadow-2xl hover:border-purple-500/50">
              <div className="flex items-center mb-6">
                <DocumentMagnifyingGlassIcon className="w-10 h-10 text-purple-500 mr-4 animate-pulse" />
                <h2 className="text-2xl font-light text-white tracking-wide">
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
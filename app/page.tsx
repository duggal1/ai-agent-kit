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
      <div className="bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-black to-black p-8 min-h-screen">
        <div className="space-y-12 mx-auto max-w-7xl">
          {/* Header */}
          <header className="mb-16 text-center relative">
            <div className="absolute inset-0 -z-10">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 blur-3xl" />
            </div>
            <div className="flex justify-center items-center mb-6 group">
              <SparklesIcon className="mr-4 w-14 h-14 text-emerald-400 animate-pulse transition-all duration-500 group-hover:scale-110 group-hover:text-emerald-300" />
              <h1 className="bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 font-black text-7xl text-transparent tracking-tight transition-all duration-500 group-hover:from-blue-300 group-hover:via-purple-300 group-hover:to-pink-300">
                Enterprise AI Control Center
              </h1>
            </div>
            <p className="mx-auto max-w-2xl text-xl text-neutral-400 font-light tracking-wide">
              Intelligent process monitoring and optimization at your fingertips
            </p>
          </header>

          {/* Dashboard Grid */}
          <div className="gap-8 grid grid-cols-1 md:grid-cols-2">
            {/* AI Workflows Card */}
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur opacity-30 group-hover:opacity-100 transition duration-1000 group-hover:duration-300" />
              <div className="relative bg-black/90 backdrop-blur-xl p-8 border border-white/10 rounded-2xl transform transition-all duration-500 hover:scale-[1.02]">
                <div className="flex items-center mb-8">
                  <ChartBarIcon className="mr-4 w-12 h-12 text-blue-400 group-hover:text-blue-300 transition-colors duration-300" />
                  <h2 className="font-light text-3xl text-white tracking-wide group-hover:text-blue-200 transition-colors duration-300">
                    AI Workflows
                  </h2>
                </div>
                <AIWorkflows />
              </div>
            </div>

            {/* Automation Metrics Card */}
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-2xl blur opacity-30 group-hover:opacity-100 transition duration-1000 group-hover:duration-300" />
              <div className="relative bg-black/90 backdrop-blur-xl p-8 border border-white/10 rounded-2xl transform transition-all duration-500 hover:scale-[1.02]">
                <div className="flex items-center mb-8">
                  <ServerStackIcon className="mr-4 w-12 h-12 text-emerald-400 group-hover:text-emerald-300 transition-colors duration-300" />
                  <h2 className="font-light text-3xl text-white tracking-wide group-hover:text-emerald-200 transition-colors duration-300">
                    Automation Metrics
                  </h2>
                </div>
                <AutomationMetrics />
              </div>
            </div>

            {/* Process Insights Card */}
            <div className="group relative md:col-span-full">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-30 group-hover:opacity-100 transition duration-1000 group-hover:duration-300" />
              <div className="relative bg-black/90 backdrop-blur-xl p-8 border border-white/10 rounded-2xl transform transition-all duration-500 hover:scale-[1.01]">
                <div className="flex items-center mb-8">
                  <DocumentMagnifyingGlassIcon className="mr-4 w-12 h-12 text-purple-400 group-hover:text-purple-300 transition-colors duration-300" />
                  <h2 className="font-light text-3xl text-white tracking-wide group-hover:text-purple-200 transition-colors duration-300">
                    Process Insights
                  </h2>
                </div>
                <ProcessInsights />
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardShell>
  );
}
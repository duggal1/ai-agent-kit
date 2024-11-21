import { DashboardShell } from '@/components/dashboard/shell';
import { AIWorkflows } from '@/components/dashboard/ai-workflows';
import { AutomationMetrics } from '@/components/dashboard/automation-metrics';
import { ProcessInsights } from '@/components/dashboard/process-insights';

export default function Home() {
  return (
    <DashboardShell>
      <div className="flex flex-col gap-8 p-8">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Enterprise AI Dashboard</h1>
            <p className="text-muted-foreground">
              Monitor and manage your AI-powered business processes
            </p>
          </div>
        </header>
        
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          <AutomationMetrics />
          <ProcessInsights />
          <AIWorkflows />
        </div>
      </div>
    </DashboardShell>
  );
}
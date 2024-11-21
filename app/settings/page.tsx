"use client";

import { useRouter } from 'next/navigation'; // Correct import for App Router
import { DashboardShell } from '@/components/dashboard/shell';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { 
  Settings as SettingsIcon, 
  Key, 
  Bell, 
  Workflow as WorkflowIcon, 
  Brain 
} from 'lucide-react';

export default function SettingsPage() {
  const router = useRouter(); // Use useRouter from next/navigation

  const handleSave = () => {
    toast.success('Settings saved successfully', {
      description: 'Your changes have been applied.',
      duration: 2000,
    });
  };

  const handleNavigate = (settingsType: string) => {
    router.push(`/settings/user-settings/${settingsType}`);
  };


  return (
    <DashboardShell>
      <div className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950 p-8">
        <div className="max-w-4xl mx-auto space-y-12">
          {/* Header */}
          <motion.header 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 mb-4">
              Enterprise AI Settings
            </h1>
            <p className="text-neutral-400 max-w-xl mx-auto">
              Configure and customize your AI system&apos;s core preferences, integrations, and notifications
            </p>
          </motion.header>

          {/* Tabs */}
          <Tabs defaultValue="general" className="space-y-6">
            <TabsList className="bg-neutral-900/60 backdrop-blur-lg border border-blue-800 rounded-full p-1 mx-auto max-w-xl">
              <TabsTrigger 
                value="general" 
                className="flex items-center gap-2 rounded-full data-[state=active]:bg-blue-600 data-[state=active]:text-white"
              >
                <SettingsIcon className="h-4 w-4" />
                General
              </TabsTrigger>
              <TabsTrigger 
                value="api" 
                className="flex items-center gap-2 rounded-full data-[state=active]:bg-purple-600 data-[state=active]:text-white"
              >
                <Key className="h-4 w-4" />
                API Keys
              </TabsTrigger>
              <TabsTrigger 
                value="notifications" 
                className="flex items-center gap-2 rounded-full data-[state=active]:bg-pink-600 data-[state=active]:text-white"
              >
                <Bell className="h-4 w-4" />
                Notifications
              </TabsTrigger>
            </TabsList>

            {/* General Settings */}
            <TabsContent value="general">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="bg-neutral-900/60 backdrop-blur-2xl border border-white/10 rounded-2xl p-6 shadow-2xl">
                  <div className="space-y-8">
                    <div className="space-y-2">
                      <Label className="text-white/80">Company Name</Label>
                      <Input 
                        placeholder="Enter company name" 
                        className="bg-white/10 border-white/20 text-white focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-white/80">Admin Email</Label>
                      <Input 
                        type="email" 
                        placeholder="admin@company.com" 
                        className="bg-white/10 border-white/20 text-white focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <Label className="text-white">Automatic Updates</Label>
                        <p className="text-sm text-white/60">
                          Receive automatic model and system updates
                        </p>
                      </div>
                      <Switch />
                    </div>
                    <Button 
                      onClick={handleSave}
                      className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transition-all duration-300"
                    >
                      Save Changes
                    </Button>
                  </div>
                </Card>
              </motion.div>
            </TabsContent>

            {/* API Configuration */}
            <TabsContent value="api">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="bg-neutral-900/60 backdrop-blur-2xl border border-white/10 rounded-2xl p-6 shadow-2xl">
                  <div className="space-y-8">
                    <div className="space-y-2">
                      <Label className="text-white/80">RunPod API Key</Label>
                      <Input 
                        type="password" 
                        className="bg-white/10 border-white/20 text-white focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-white/80">Hugging Face API Key</Label>
                      <Input 
                        type="password" 
                        className="bg-white/10 border-white/20 text-white focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-white/80">Gemini API Key</Label>
                      <Input 
                        type="password" 
                        className="bg-white/10 border-white/20 text-white focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>
                    <Button 
                      onClick={handleSave}
                      className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transition-all duration-300"
                    >
                      Update API Keys
                    </Button>
                  </div>
                </Card>
              </motion.div>
            </TabsContent>

            {/* Notifications */}
            <TabsContent value="notifications">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="bg-neutral-900/60 backdrop-blur-2xl border border-white/10 rounded-2xl p-6 shadow-2xl">
                  <div className="space-y-8">
                    {['Model Training', 'System Updates', 'Performance Alerts'].map((item) => (
                      <div key={item} className="flex items-center justify-between">
                        <div className="space-y-1">
                          <Label className="text-white">{item}</Label>
                          <p className="text-sm text-white/60">
                            Receive notifications for {item.toLowerCase()}
                          </p>
                        </div>
                        <Switch />
                      </div>
                    ))}
                    <div className="grid grid-cols-2 gap-4">
                      <motion.button 
                        onClick={() => handleNavigate('workflow')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 via-pink-500 to-indigo-700 text-white font-medium py-3 rounded-xl shadow-lg hover:from-purple-700 hover:to-indigo-800 transition-all duration-300"
                      >
                        <WorkflowIcon className="h-5 w-5" />
                        Workflow Settings
                      </motion.button>
                      <motion.button 
                        onClick={() => handleNavigate('ai-models')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 via-fuchsia-500 to-purple-600 text-white font-medium py-3 rounded-xl shadow-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-300"
                      >
                        <Brain className="h-5 w-5" />
                        AI Models Settings
                      </motion.button>
                    </div>
                    <Button 
                      onClick={handleSave}
                      className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transition-all duration-300"
                    >
                      Save Preferences
                    </Button>
                  </div>
                </Card>
              </motion.div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </DashboardShell>
  );
}
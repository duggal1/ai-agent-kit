"use client";

import { useRouter } from 'next/navigation';
import { DashboardShell } from '@/components/dashboard/shell';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Settings as SettingsIcon, 
  Key, 
  Bell, 
  Workflow as WorkflowIcon, 
  Brain,
  Check
} from 'lucide-react';
import { useState } from 'react';

export default function SettingsPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('general');
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      toast.success('Settings saved successfully', {
        description: 'Your changes have been applied.',
        duration: 2000,
        icon: <Check className="text-green-500" />
      });
      setIsSaving(false);
    }, 1000);
  };

  const handleNavigate = (settingsType: string) => {
    router.push(`/settings/user-settings/${settingsType}`);
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        delayChildren: 0.2,
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { 
        type: "spring", 
        stiffness: 100 
      }
    }
  };

  return (
    <DashboardShell>
      <div className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950 p-8">
        <motion.div 
          className="max-w-5xl mx-auto space-y-12"
          initial="hidden"
          animate="visible"
          variants={containerVariants}
        >
          {/* Header */}
          <motion.header 
            variants={itemVariants}
            className="text-center mb-12"
          >
            <h1 className="text-5xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-600 mb-4">
              Enterprise AI Settings
            </h1>
            <p className="text-neutral-300 max-w-2xl mx-auto text-lg">
              Powerful configuration tools to customize and optimize your AI ecosystem
            </p>
          </motion.header>

          {/* Tabs */}
          <motion.div variants={itemVariants}>
            <Tabs 
              value={activeTab} 
              onValueChange={setActiveTab} 
              className="space-y-6"
            >
              <TabsList className="bg-neutral-900/70 backdrop-blur-xl border border-white/10 rounded-full p-1 mx-auto max-w-2xl">
                {[
                  { value: 'general', icon: SettingsIcon, label: 'General' },
                  { value: 'api', icon: Key, label: 'API Keys' },
                  { value: 'notifications', icon: Bell, label: 'Notifications' }
                ].map((tab) => (
                  <TabsTrigger 
                    key={tab.value}
                    value={tab.value} 
                    className="flex items-center gap-2 rounded-full data-[state=active]:bg-blue-600/80 data-[state=active]:text-white transition-all"
                  >
                    <tab.icon className="h-4 w-4" />
                    {tab.label}
                  </TabsTrigger>
                ))}
              </TabsList>

              <AnimatePresence mode="wait">
                <TabsContent value="general" key="general">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-neutral-900/50 backdrop-blur-3xl border border-white/10 rounded-3xl p-8 shadow-2xl">
                      <motion.div 
                        className="space-y-8"
                        initial="hidden"
                        animate="visible"
                        variants={containerVariants}
                      >
                        <motion.div variants={itemVariants} className="space-y-2">
                          <Label className="text-white/80">Company Name</Label>
                          <Input 
                            placeholder="Enter company name" 
                            className="bg-white/10 border-white/20 text-white focus:ring-2 focus:ring-blue-500/50"
                          />
                        </motion.div>
                        <motion.div variants={itemVariants} className="space-y-2">
                          <Label className="text-white/80">Admin Email</Label>
                          <Input 
                            type="email" 
                            placeholder="admin@company.com" 
                            className="bg-white/10 border-white/20 text-white focus:ring-2 focus:ring-blue-500/50"
                          />
                        </motion.div>
                        <motion.div variants={itemVariants} className="flex items-center justify-between">
                          <div className="space-y-1">
                            <Label className="text-white">Automatic Updates</Label>
                            <p className="text-sm text-white/60">
                              Receive automatic model and system updates
                            </p>
                          </div>
                          <Switch />
                        </motion.div>
                        <motion.div variants={itemVariants}>
                          <Button 
                            onClick={handleSave}
                            disabled={isSaving}
                            className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transition-all duration-300 relative"
                          >
                            {isSaving ? (
                              <motion.div
                                initial={{ opacity: 0, scale: 0.5 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="absolute inset-0 flex items-center justify-center"
                              >
                                <motion.div
                                  animate={{ 
                                    rotate: 360,
                                    transition: { 
                                      repeat: Infinity, 
                                      duration: 1,
                                      ease: "linear"
                                    }
                                  }}
                                >
                                  <Check className="h-5 w-5 animate-pulse" />
                                </motion.div>
                              </motion.div>
                            ) : (
                              'Save Changes'
                            )}
                          </Button>
                        </motion.div>
                      </motion.div>
                    </Card>
                  </motion.div>
                </TabsContent>

                {/* Similar structure for API and Notifications tabs with added animations */}
                <TabsContent value="api" key="api">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-neutral-900/50 backdrop-blur-3xl border border-white/10 rounded-3xl p-8 shadow-2xl">
                      {/* API Keys content similar to previous implementation */}
                    </Card>
                  </motion.div>
                </TabsContent>

                <TabsContent value="notifications" key="notifications">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-neutral-900/50 backdrop-blur-3xl border border-white/10 rounded-3xl p-8 shadow-2xl">
                      {/* Notifications content similar to previous implementation */}
                    </Card>
                  </motion.div>
                </TabsContent>
              </AnimatePresence>
            </Tabs>
          </motion.div>
        </motion.div>
      </div>
    </DashboardShell>
  );
}
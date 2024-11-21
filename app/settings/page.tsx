"use client";

import { DashboardShell } from '@/components/dashboard/shell';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import router from 'next/router';

export default function SettingsPage() {
  const handleSave = () => {
    toast.success('Settings saved successfully');
  };


  const handleNavigate = (SettingsId: string) => {
    router.push(`/settings/user-settings/${SettingsId}`);
  };

  return (
    <DashboardShell>
      <div className="flex flex-col gap-8 p-8">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">
            Manage your AI system preferences and configurations
          </p>
        </header>

        <Tabs defaultValue="general" className="space-y-4">
          <TabsList>
            <TabsTrigger value="general">General</TabsTrigger>
            <TabsTrigger value="api">API Keys</TabsTrigger>
            <TabsTrigger value="notifications">Notifications</TabsTrigger>
          </TabsList>

          <TabsContent value="general">
            <Card>
              <CardHeader>
                <CardTitle>General Settings</CardTitle>
                <CardDescription>Configure your AI system preferences</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="company">Company Name</Label>
                  <Input id="company" placeholder="Enter company name" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Admin Email</Label>
                  <Input id="email" type="email" placeholder="admin@company.com" />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Automatic Updates</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive automatic model updates
                    </p>
                  </div>
                  <Switch />
                </div>
                <Button onClick={handleSave}>Save Changes</Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="api">
            <Card>
              <CardHeader>
                <CardTitle>API Configuration</CardTitle>
                <CardDescription>Manage your API keys and endpoints</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="runpod">RunPod API Key</Label>
                  <Input id="runpod" type="password" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="huggingface">Hugging Face API Key</Label>
                  <Input id="huggingface" type="password" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="gemini">Gemini API Key</Label>
                  <Input id="gemini" type="password" />
                </div>
                <Button onClick={handleSave}>Update API Keys</Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="notifications">
            <Card>
              <CardHeader>
                <CardTitle>Notification Preferences</CardTitle>
                <CardDescription>Configure your notification settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  {['Model Training', 'System Updates', 'Performance Alerts'].map((item) => (
                    <div key={item} className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>{item}</Label>
                        <p className="text-sm text-muted-foreground">
                          Receive notifications for {item.toLowerCase()}
                        </p>
                      </div>
                      <Switch />
                    </div>
                  ))}
                </div>
                <Button onClick={handleSave}>Save Preferences</Button>
                <button className= " max-w-min flex justify-center bg-gradient-to-r from-blue-500 to-pink-500 w-36 mt-8 "onClick={() => handleNavigate('workflow')}>Go to Workflow Settings</button>
                <button className= " max-w-min flex justify-center bg-gradient-to-r from-purple-500 via-fuchsia-500 to-blue-600 w-36 mt-8 "onClick={() => handleNavigate('ai models')}>Go to AI models Settings</button>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardShell>
  );
}
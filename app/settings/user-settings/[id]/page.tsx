"use client";

import { useState } from 'react';
import { 
  ChevronRight, 
  User, 
  Settings, 
  Bell, 
  Lock, 
  Shield, 
  CreditCard, 
  Globe, 
  Palette, 
  Zap,
  LogOut
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { toast } from 'sonner';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from '@/components/ui/dialog';

export default function UserSettingsPage({ 
  params 
}: { 
  params: { id: string } 
}) {
  const [activeTab, setActiveTab] = useState('profile');
  const [profileData, setProfileData] = useState({
    username: 'johndoe',
    email: 'john.doe@example.com',
    theme: 'dark',
    language: 'en',
  });

  const handleSave = () => {
    toast.success('Settings updated successfully', {
      description: `Changes saved for user ${params.id}`,
    });
  };

  const renderProfileSection = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card className="bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-slate-800 dark:to-slate-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="w-5 h-5" /> Personal Information
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center space-x-4">
            <Avatar className="w-16 h-16">
              <AvatarImage src="/placeholder-avatar.jpg" />
              <AvatarFallback>JD</AvatarFallback>
            </Avatar>
            <div className="space-y-1">
              <Input 
                placeholder="Username" 
                value={profileData.username}
                onChange={(e) => setProfileData(prev => ({...prev, username: e.target.value}))}
              />
              <Input 
                type="email" 
                placeholder="Email" 
                value={profileData.email}
                onChange={(e) => setProfileData(prev => ({...prev, email: e.target.value}))}
              />
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <Select 
              value={profileData.theme}
              onValueChange={(value) => setProfileData(prev => ({...prev, theme: value}))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Theme" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="system">System</SelectItem>
              </SelectContent>
            </Select>

            <Select 
              value={profileData.language}
              onValueChange={(value) => setProfileData(prev => ({...prev, language: value}))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="es">Español</SelectItem>
                <SelectItem value="fr">Français</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-gradient-to-br from-purple-50 to-pink-100 dark:from-slate-800 dark:to-slate-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" /> Security
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" className="w-full">
                Change Password
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Change Password</DialogTitle>
                <DialogDescription>
                  Enter your current and new password
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <Input type="password" placeholder="Current Password" />
                <Input type="password" placeholder="New Password" />
                <Input type="password" placeholder="Confirm New Password" />
                <Button className="w-full">Update Password</Button>
              </div>
            </DialogContent>
          </Dialog>

          <div className="flex justify-between items-center">
            <Label>Two-Factor Authentication</Label>
            <Switch />
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderNotificationsSection = () => (
    <Card className="bg-gradient-to-br from-green-50 to-teal-100 dark:from-slate-800 dark:to-slate-900">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="w-5 h-5" /> Notification Preferences
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {[
          { icon: <Zap className="w-4 h-4" />, label: 'System Updates' },
          { icon: <Globe className="w-4 h-4" />, label: 'Marketing Communications' },
          { icon: <Palette className="w-4 h-4" />, label: 'Product Announcements' }
        ].map(({ icon, label }) => (
          <div key={label} className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {icon}
              <Label>{label}</Label>
            </div>
            <Switch />
          </div>
        ))}
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-slate-900 dark:to-slate-800 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
              User Settings
            </h1>
            <p className="text-muted-foreground">
              Manage settings for User ID: {params.id}
            </p>
          </div>
          <Button variant="destructive" className="flex items-center gap-2">
            <LogOut className="w-4 h-4" /> Logout
          </Button>
        </header>

        <Tabs 
          value={activeTab} 
          onValueChange={setActiveTab}
          className="space-y-6"
        >
          <TabsList className="w-full bg-white/50 dark:bg-slate-800/50 backdrop-blur-md">
            {[
              { value: 'profile', icon: <User />, label: 'Profile' },
              { value: 'notifications', icon: <Bell />, label: 'Notifications' },
              { value: 'security', icon: <Lock />, label: 'Security' },
              { value: 'billing', icon: <CreditCard />, label: 'Billing' }
            ].map(({ value, icon, label }) => (
              <TabsTrigger 
                key={value} 
                value={value} 
                className="flex items-center gap-2 data-[state=active]:bg-primary/10"
              >
                {icon}
                {label}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="profile">
            {renderProfileSection()}
          </TabsContent>

          <TabsContent value="notifications">
            {renderNotificationsSection()}
          </TabsContent>

          <div className="flex justify-end">
            <Button 
              onClick={handleSave} 
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
            >
              Save Changes <ChevronRight className="ml-2 w-4 h-4" />
            </Button>
          </div>
        </Tabs>
      </div>
    </div>
  );
}
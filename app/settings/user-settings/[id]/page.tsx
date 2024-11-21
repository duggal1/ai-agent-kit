/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import { useRouter, useParams } from "next/navigation";
import { useEffect } from "react";

// Define the possible static params
export function generateStaticParams() {
  return [
    { id: "workflow" },
    { id: "ai-models" },
  ];
}

// Add type definition for params
interface Params {
  id: string;
}

export default function UserSettingsPage() {
  const router = useRouter();
  const params = useParams() as unknown as Params;

  // Redirect if the `id` param is invalid
  const validIds = ["workflow", "ai-models"];
  useEffect(() => {
    if (!validIds.includes(params.id)) {
      router.push("/404"); // Redirect to a 404 page if ID is invalid
    }
  }, [params, router]);

  // Mapping of settings types to their content
  const settingsContent = {
    workflow: {
      title: "Workflow Settings",
      description: "Configure and manage your workflow settings",
      component: (
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold text-white">
            Workflow Configuration
          </h2>
          <div className="bg-neutral-700 p-4 rounded-lg">
            <p className="text-neutral-300">
              Customize your workflow processes and automation
            </p>
          </div>
        </div>
      ),
    },
    "ai-models": {
      title: "AI Models Settings",
      description: "Manage and configure your AI models",
      component: (
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold text-white">
            AI Models Configuration
          </h2>
          <div className="bg-neutral-700 p-4 rounded-lg">
            <p className="text-neutral-300">
              Fine-tune and optimize your AI model settings
            </p>
          </div>
        </div>
      ),
    },
  };

  // Current settings content or fallback
  const currentSettings = settingsContent[params.id as keyof typeof settingsContent];

  return currentSettings ? (
    <div className="p-6 bg-neutral-900 min-h-screen">
      <div className="max-w-4xl mx-auto bg-neutral-800 rounded-xl p-8 shadow-lg">
        <h1 className="text-3xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-600">
          {currentSettings.title}
        </h1>

        <p className="text-neutral-400 mb-6">{currentSettings.description}</p>

        {currentSettings.component}
      </div>
    </div>
  ) : null; // Prevent rendering until redirection happens
}
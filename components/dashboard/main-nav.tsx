"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Brain, LayoutDashboard, Settings, Workflow } from "lucide-react";

export function MainNav() {
  const pathname = usePathname();

  const routes = [
    {
      href: "/",
      label: "Dashboard",
      icon: LayoutDashboard,
      active: pathname === "/",
    },
    {
      href: "/workflows",
      label: "Workflows",
      icon: Workflow,
      active: pathname === "/workflows",
    },
    {
      href: "/ai-models",
      label: "AI Models",
      icon: Brain,
      active: pathname === "/ai-models",
    },
    {
      href: "/settings",
      label: "Settings",
      icon: Settings,
      active: pathname === "/settings",
    },
  ];

  return (
    <nav className="flex items-center gap-6">
      <Link href="/" className="hidden items-center space-x-2 md:flex">
        <Brain className="h-6 w-6" />
        <span className="hidden font-bold sm:inline-block">
          Enterprise AI
        </span>
      </Link>
      {routes.map((route) => (
        <Link
          key={route.href}
          href={route.href}
          className={cn(
            "flex items-center gap-2 text-sm font-medium transition-colors hover:text-primary",
            route.active ? "text-primary" : "text-muted-foreground"
          )}
        >
          <route.icon className="h-4 w-4" />
          <span className="hidden md:block">{route.label}</span>
        </Link>
      ))}
    </nav>
  );
}
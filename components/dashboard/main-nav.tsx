"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Brain, LayoutDashboard, Settings, Workflow } from "lucide-react";
import { motion } from "framer-motion";

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
    <nav className="fixed top-4 left-1/2 -translate-x-1/2 z-50 w-[calc(100%-2rem)] max-w-4xl bg-black/60 backdrop-blur-xl rounded-full shadow-2xl border border-white/10">
      <div className="flex items-center justify-between px-4 py-2">
        <Link 
          href="/" 
          className="hidden items-center space-x-2 md:flex group"
        >
          <motion.div
            whileHover={{ scale: 1.1, rotate: 360 }}
            transition={{ duration: 0.5 }}
          >
            <Brain 
              className="h-6 w-6 text-white group-hover:text-blue-400 transition-colors" 
            />
          </motion.div>
          <span className="hidden font-bold sm:inline-block text-white/80 group-hover:text-white transition-colors">
            Enterprise AI
          </span>
        </Link>
        
        <div className="flex items-center space-x-4">
          {routes.map((route) => (
            <motion.div
              key={route.href}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link
                href={route.href}
                className={cn(
                  "flex items-center gap-2 text-sm font-medium transition-all duration-300 ease-in-out group",
                  route.active 
                    ? "text-white bg-blue-600/20 px-3 py-1 rounded-full" 
                    : "text-white/60 hover:text-white hover:bg-white/10 px-3 py-1 rounded-full"
                )}
              >
                <route.icon 
                  className={cn(
                    "h-4 w-4 transition-colors",
                    route.active 
                      ? "text-blue-400" 
                      : "text-white/60 group-hover:text-blue-400"
                  )} 
                />
                <span className="hidden md:block">
                  {route.label}
                </span>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </nav>
  );
}
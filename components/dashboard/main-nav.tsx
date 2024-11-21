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
    <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-[calc(100%-3rem)] max-w-5xl">
      <div className="bg-gradient-to-br from-neutral-900/80 to-neutral-800/80 backdrop-blur-2xl rounded-full border border-white/10 shadow-2xl">
        <div className="flex items-center justify-between px-2 py-2">
          {/* Logo Section */}
          <Link
            href="/"
            className="hidden items-center space-x-3 md:flex group"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              whileHover={{ 
                scale: 1.1, 
                rotate: 360,
                transition: { duration: 0.6, type: "spring" }
              }}
            >
              <Brain className="h-7 w-7 text-white/80 group-hover:text-blue-400 transition-all duration-300" />
            </motion.div>
            <span className="font-light tracking-wider text-white/70 group-hover:text-white transition-colors duration-300">
              Enterprise AI
            </span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-1 w-full justify-center">
            {routes.map((route) => (
              <motion.div
                key={route.href}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex-grow flex justify-center"
              >
                <Link
                  href={route.href}
                  className={cn(
                    "flex items-center justify-center gap-2 text-sm font-medium transition-all duration-300 ease-in-out group px-4 py-2 rounded-full",
                    route.active
                      ? "text-white bg-blue-500/20 ring-2 ring-blue-500/30"
                      : "text-white/60 hover:text-white hover:bg-white/10"
                  )}
                >
                  <route.icon
                    className={cn(
                      "h-5 w-5 transition-colors",
                      route.active
                        ? "text-blue-400"
                        : "text-white/60 group-hover:text-blue-400"
                    )}
                  />
                  <span className="hidden md:block">{route.label}</span>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
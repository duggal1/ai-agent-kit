"use client";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { motion } from "framer-motion";
import { LogOut, Settings, User, ChevronDown } from "lucide-react";

export function UserNav() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="absolute top-6 right-6 z-50"
        >
          <Button
            variant="ghost"
            className="relative h-12 w-auto rounded-full bg-gradient-to-br from-neutral-900/60 to-neutral-800/60 hover:from-neutral-900/70 hover:to-neutral-800/70 backdrop-blur-2xl border border-white/10 px-3 py-1 flex items-center gap-3 shadow-xl"
          >
            <Avatar className="h-10 w-10 border-2 border-white/30">
              <AvatarImage src="/avatars/user.png" alt="User" />
              <AvatarFallback className="bg-blue-500/30 text-white">AI</AvatarFallback>
            </Avatar>
            <div className="hidden md:flex flex-col items-start">
              <p className="text-sm font-light tracking-wide text-white/90">Enterprise Admin</p>
              <p className="text-xs text-white/60">admin@enterprise.ai</p>
            </div>
            <ChevronDown
              className="h-5 w-5 ml-2 text-white/60 transition-all duration-300 group-data-[state=open]:rotate-180 group-data-[state=open]:text-blue-400"
            />
          </Button>
        </motion.div>
      </DropdownMenuTrigger>
      
      <DropdownMenuContent 
        className="w-72 bg-gradient-to-br from-neutral-900/90 to-neutral-800/90 backdrop-blur-3xl border border-white/10 rounded-2xl shadow-2xl text-white overflow-hidden"
      >
        <DropdownMenuLabel className="p-0">
          <div className="flex items-center space-x-4 p-6 bg-white/5 border-b border-white/10">
            <Avatar className="h-14 w-14 border-2 border-white/30">
              <AvatarImage src="/avatars/user.png" alt="User" />
              <AvatarFallback className="bg-blue-500/30 text-white text-xl">AI</AvatarFallback>
            </Avatar>
            <div className="flex flex-col space-y-1">
              <p className="text-lg font-light tracking-wide text-white">Enterprise Admin</p>
              <p className="text-sm text-white/60">admin@enterprise.ai</p>
            </div>
          </div>
        </DropdownMenuLabel>
        
        <DropdownMenuGroup className="py-2 px-2">
          <DropdownMenuItem className="rounded-lg hover:bg-white/10 focus:bg-white/20 cursor-pointer group px-4 py-3">
            <User className="mr-4 h-6 w-6 text-white/60 group-hover:text-blue-400 transition-colors" />
            <span className="text-white/90 group-hover:text-white transition-colors text-sm">Profile</span>
          </DropdownMenuItem>
          
          <DropdownMenuItem className="rounded-lg hover:bg-white/10 focus:bg-white/20 cursor-pointer group px-4 py-3">
            <Settings className="mr-4 h-6 w-6 text-white/60 group-hover:text-blue-400 transition-colors" />
            <span className="text-white/90 group-hover:text-white transition-colors text-sm">Settings</span>
          </DropdownMenuItem>
        </DropdownMenuGroup>
        
        <DropdownMenuSeparator className="bg-white/10" />
        
        <DropdownMenuItem
          className="rounded-lg hover:bg-red-500/10 focus:bg-red-500/20 cursor-pointer group text-red-400 hover:text-red-500 px-4 py-3 mb-2 mx-2"
        >
          <LogOut className="mr-4 h-6 w-6 text-red-400/60 group-hover:text-red-500 transition-colors" />
          <span className="text-sm group-hover:text-red-500 transition-colors">Log out</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
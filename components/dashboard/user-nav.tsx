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
          className="rounded-full"
        >
          <Button 
            variant="ghost" 
            className="relative h-10 w-auto rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-md border border-white/20 px-2 py-1 flex items-center gap-2"
          >
            <Avatar className="h-8 w-8 border-2 border-white/30">
              <AvatarImage src="/avatars/user.png" alt="User" />
              <AvatarFallback className="bg-blue-500/30 text-white">AI</AvatarFallback>
            </Avatar>
            <div className="hidden md:flex flex-col items-start ml-2">
              <p className="text-sm font-medium text-white/90">Admin</p>
              <p className="text-xs text-white/60">admin@enterprise.ai</p>
            </div>
            <ChevronDown 
              className="h-4 w-4 ml-2 text-white/60 transition-transform group-data-[state=open]:rotate-180" 
            />
          </Button>
        </motion.div>
      </DropdownMenuTrigger>
      <DropdownMenuContent 
        className="w-64 bg-black/80 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl text-white"
      >
        <DropdownMenuLabel className="font-normal">
          <div className="flex items-center space-x-4 p-4 border-b border-white/10">
            <Avatar className="h-12 w-12 border-2 border-white/30">
              <AvatarImage src="/avatars/user.png" alt="User" />
              <AvatarFallback className="bg-blue-500/30 text-white text-lg">AI</AvatarFallback>
            </Avatar>
            <div className="flex flex-col space-y-1">
              <p className="text-md font-semibold text-white">Admin</p>
              <p className="text-sm text-white/60">admin@enterprise.ai</p>
            </div>
          </div>
        </DropdownMenuLabel>

        <DropdownMenuGroup className="py-2">
          <DropdownMenuItem className="hover:bg-white/10 focus:bg-white/20 cursor-pointer group">
            <User className="mr-3 h-5 w-5 text-white/60 group-hover:text-blue-400 transition-colors" />
            <span className="text-white/90 group-hover:text-white transition-colors">Profile</span>
          </DropdownMenuItem>
          <DropdownMenuItem className="hover:bg-white/10 focus:bg-white/20 cursor-pointer group">
            <Settings className="mr-3 h-5 w-5 text-white/60 group-hover:text-blue-400 transition-colors" />
            <span className="text-white group-hover:text-white transition-colors">Settings</span>
          </DropdownMenuItem>
        </DropdownMenuGroup>

        <DropdownMenuSeparator className="bg-white/10" />

        <DropdownMenuItem 
          className="hover:bg-red-500/10 focus:bg-red-500/20 cursor-pointer group text-red-400 hover:text-red-500"
        >
          <LogOut className="mr-3 h-5 w-5 text-red-400/60 group-hover:text-red-500 transition-colors" />
          <span className="group-hover:text-red-500 transition-colors">Log out</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
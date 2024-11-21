"use client"
import './globals.css';
import { Toaster } from '@/components/ui/sonner';
import { Inter } from 'next/font/google';
import { useEffect, useState } from 'react';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check the user's preference or default to light mode
    const preferredDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(preferredDark);
  }, []);

  return (
    <html lang="en" suppressHydrationWarning className={isDarkMode ? 'dark' : ''}>
      <body className={`${inter.className} bg-white text-black dark:bg-black dark:text-white`}>
        {children}
        <Toaster />
      </body>
    </html>
  );
}

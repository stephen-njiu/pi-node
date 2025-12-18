import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { Toaster } from "sonner";
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SG - Savannah-Gates",
  description: "Facial Gate Access Control System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-[#0a0f1a]">
        <Navbar />
        <main className="min-h-[calc(100svh-200px)] antialiased">
          {children}
          <Toaster
            position="top-right"
            closeButton
            richColors
          />
        </main>
        <Footer />
      </body>
    </html>
  );
}

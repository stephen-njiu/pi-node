"use client";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="min-h-svh flex items-center justify-center px-4 py-10 bg-linear-to-b from-[#061022] to-[#071226] text-white">
      <Card className="w-full max-w-xl border border-white/10 bg-[#0d1628] shadow-2xl">
        <div className="p-6 md:p-10 text-center">
          <div className="mx-auto mb-6 h-16 w-16 rounded-full bg-red-600/20 grid place-items-center">
            <span className="text-red-400 text-2xl font-bold">404</span>
          </div>
          <h1 className="text-2xl text-white md:text-3xl font-semibold">Page not found</h1>
          <p className="mt-2 text-sm md:text-base text-white/70">
            The page you’re looking for doesn’t exist or may have been moved.
          </p>

          <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center">
            <Link href="/">
              <Button className="w-full sm:w-auto bg-white text-black hover:bg-white/85">Go to Home</Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="outline" className="w-full text-black sm:w-auto border-white/30 hover:bg-white hover:text-black">Open Dashboard</Button>
            </Link>
          </div>

          <div className="mt-6 text-xs text-white/50">
            If you believe this is a mistake, please check the URL or contact support.
          </div>
        </div>
      </Card>
    </div>
  );
}

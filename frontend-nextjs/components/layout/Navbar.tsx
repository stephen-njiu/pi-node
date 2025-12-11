"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import React from "react";

export default function Navbar() {
  const pathname = usePathname();
  const navItems = [
    { href: "/", label: "Home" },
    { href: "/dashboard", label: "Dashboard" },
    { href: "/enroll", label: "Enroll" },
    { href: "/enroll/wanted", label: "Wanted" },
  ];
  const [open, setOpen] = React.useState(false);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-[#0d1628]/90 backdrop-blur supports-[backdrop-filter]:bg-[#0d1628]/75">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center gap-4" role="navigation" aria-label="Primary">
        <Link href="/" className="flex items-center gap-2" aria-label="Savannah Gates home">
          <div className="h-7 w-7 rounded bg-white grid place-items-center text-black font-bold text-xs" aria-hidden="true">SG</div>
          <span className="text-white font-semibold">Savannah Gates</span>
        </Link>

        {/* Desktop menu */}
        <nav className="ml-auto hidden md:flex items-center gap-2" aria-label="Main menu">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`px-3 py-1.5 rounded text-sm ${pathname === item.href ? "bg-white text-black" : "text-white hover:bg-white/10"}`}
              aria-current={pathname === item.href ? "page" : undefined}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Mobile hamburger */}
        <button
          className="md:hidden ml-auto inline-flex items-center justify-center h-9 w-9 rounded bg-white/10 text-white hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white/40"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          aria-controls="mobile-menu"
          onClick={() => setOpen((o) => !o)}
        >
          <span className="sr-only">Toggle menu</span>
          {/* simple icon */}
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            {open ? (
              <path d="M18 6L6 18M6 6l12 12" />
            ) : (
              <>
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </>
            )}
          </svg>
        </button>
      </div>

      {/* Mobile menu panel */}
      <div
        id="mobile-menu"
        className={`${open ? "block" : "hidden"} md:hidden border-t border-white/10 bg-[#0d1628]`}
        role="dialog"
        aria-modal="true"
      >
        <nav className="mx-auto max-w-6xl px-4 py-3 grid grid-cols-1 gap-2" aria-label="Mobile menu">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`px-3 py-2 rounded text-sm ${pathname === item.href ? "bg-white text-black" : "text-white hover:bg-white/10"}`}
              aria-current={pathname === item.href ? "page" : undefined}
              onClick={() => setOpen(false)}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

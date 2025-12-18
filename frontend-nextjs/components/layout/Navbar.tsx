"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import React from "react";
import { toast } from "sonner";
import { authClient } from "@/lib/auth-client";

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const navItems = [
    { href: "/", label: "Home", protected: false },
    { href: "/dashboard", label: "Dashboard", protected: true },
    { href: "/enroll", label: "Enroll", protected: true },
    { href: "/enroll/wanted", label: "Wanted", protected: true },
  ];
  const [open, setOpen] = React.useState(false);
  const [userRole, setUserRole] = React.useState<string | null>(null);
  const { data: session } = authClient.useSession();

  // Fetch user role on mount
  React.useEffect(() => {
    async function fetchRole() {
      if (!session?.user) {
        setUserRole(null);
        return;
      }
      try {
        const email = session.user.email;
        const id = session.user.id;
        const params = new URLSearchParams(email ? { email } : { id: id! });
        const res = await fetch(`/api/user-role?${params.toString()}`);
        if (res.ok) {
          const json = await res.json();
          setUserRole(json?.role || null);
        }
      } catch {
        setUserRole(null);
      }
    }
    fetchRole();
  }, [session]);

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, item: typeof navItems[0]) => {
    // Only check protected pages
    if (!item.protected) {
      return;
    }

    e.preventDefault();
    setOpen(false);

    // Step 1: Check if user is signed in
    if (!session?.user) {
      toast.error("Authentication Required", {
        description: "Please sign in to access this page.",
      });
      router.push("/signin");
      return;
    }

    // Step 2: Check if user is admin
    if (userRole !== "ADMIN") {
      toast.error("Access Denied", {
        description: "You need to be an admin of an institution to access this page.",
      });
      return;
    }

    // Step 3: All checks passed, navigate
    router.push(item.href);
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-[#0d1628]/90 backdrop-blur supports-backdrop-filter:bg-[#0d1628]/75">
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
              onClick={(e) => handleNavClick(e, item)}
              className={`px-3 py-1.5 rounded text-sm ${pathname === item.href ? "bg-white text-black" : "text-white hover:bg-white/10"}`}
              aria-current={pathname === item.href ? "page" : undefined}
            >
              {item.label}
            </Link>
          ))}
          
          {/* Profile Link - only for signed-in users */}
          {session?.user && (
            <Link
              href="/profile"
              className={`px-3 py-1.5 rounded text-sm ${pathname === "/profile" ? "bg-white text-black" : "text-white hover:bg-white/10"}`}
              aria-current={pathname === "/profile" ? "page" : undefined}
            >
              Profile
            </Link>
          )}

          {/* Login/Logout Button */}
          {session?.user ? (
            <Button
              onClick={async () => {
                await authClient.signOut();
                toast.success("Signed out successfully");
                router.push("/");
              }}
              variant="outline"
              size="sm"
              className="border-white/30 text-white hover:bg-white hover:text-black"
            >
              Logout
            </Button>
          ) : (
            <Button
              onClick={() => router.push("/signin")}
              size="sm"
              className="bg-linear-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-600 hover:to-emerald-700"
            >
              Login
            </Button>
          )}
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
              onClick={(e) => {
                handleNavClick(e, item);
                if (userRole === "ADMIN" || !item.protected) {
                  setOpen(false);
                }
              }}
            >
              {item.label}
            </Link>
          ))}

          {/* Profile Link - Mobile - only for signed-in users */}
          {session?.user && (
            <Link
              href="/profile"
              className={`px-3 py-2 rounded text-sm ${pathname === "/profile" ? "bg-white text-black" : "text-white hover:bg-white/10"}`}
              aria-current={pathname === "/profile" ? "page" : undefined}
              onClick={() => setOpen(false)}
            >
              Profile
            </Link>
          )}

          {/* Login/Logout Button - Mobile */}
          <div className="pt-2 border-t border-white/10 mt-2">
            {session?.user ? (
              <Button
                onClick={async () => {
                  await authClient.signOut();
                  toast.success("Signed out successfully");
                  setOpen(false);
                  router.push("/");
                }}
                variant="outline"
                size="sm"
                className="w-full border-white/30 text-white hover:bg-white hover:text-black"
              >
                Logout
              </Button>
            ) : (
              <Button
                onClick={() => {
                  setOpen(false);
                  router.push("/signin");
                }}
                size="sm"
                className="w-full bg-linear-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-600 hover:to-emerald-700"
              >
                Login
              </Button>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}

"use client";
import { useMemo } from "react";
import { authClient } from "@/lib/auth-client";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useEffect, useState } from "react";

interface ProfileUser {
  id?: string;
  name?: string | null;
  primaryEmail?: string | null;
  image?: string | null;
  role?: string | null;
}

export default function ProfilePage() {
  // Prefer the Better Auth hook so UI updates reactively on sign-in/sign-out
  const { data: session, isPending, error } = authClient.useSession();
  console.log("Session data:", session);
  const user: ProfileUser | null = useMemo(() => {
    const u = session?.user;
    if (!u) return null;
    return {
      id: u.id,
      name: u.name ?? null,
      primaryEmail: u.email ?? null,
      image: u.image ?? null,
    };
  }, [session]);
  const loading = isPending;
  if (error) {
    toast.error(error.message ?? "Failed to load profile");
  }

  // Fetch role from API using Prisma (by email preferred; fallback to id)
  const [role, setRole] = useState<string | null>(null);
  useEffect(() => {
    const loadRole = async () => {
      try {
        if (!user?.primaryEmail && !user?.id) return;
        const params = new URLSearchParams(
          user?.primaryEmail ? { email: user.primaryEmail } : { id: user!.id! }
        );
        const res = await fetch(`/api/user-role?${params.toString()}`);
        const json = await res.json();
        if (res.ok) setRole(json.role ?? null);
      } catch {}
    };
    loadRole();
  }, [user?.primaryEmail, user?.id]);

  const initials = (user?.name || user?.primaryEmail || "U")
    .split(" ")
    .map((p) => p[0]?.toUpperCase())
    .slice(0, 2)
    .join("");

  const onSignOut = async () => {
    try {
      await authClient.signOut({
        fetchOptions: {
          onSuccess: () => {
            // Redirect to centralized sign-in route
            window.location.href = "/signin";
          },
        },
      });
      toast.success("Signed out");
    } catch (e: any) {
      toast.error(e?.message ?? "Sign-out failed");
    }
  };

  return (
    <div className="min-h-svh text-white flex items-center justify-center px-4 py-10 bg-[#0a0f1a]">
      <Card className="w-full max-w-5xl border border-white/10 bg-[#0d1628] text-white shadow-2xl">
        <div className="p-6 sm:p-10">
          <div className="flex flex-col sm:flex-row sm:items-center gap-6">
            <div className="size-14 rounded-full overflow-hidden border border-white/20 bg-white/10 grid place-items-center">
              {user?.image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={user.image} alt={user.name || "User"} className="w-full h-full object-cover" />
              ) : (
                <span className="text-white font-semibold">{initials}</span>
              )}
            </div>
            <div>
              <h1 className="text-2xl text-white font-semibold">{user?.name || "User"}</h1>
              <p className="text-sm text-white/80">{user?.primaryEmail}</p>
            </div>
            <div className="ml-auto">
              <Button variant="outline" className="border-white/30 text-black hover:bg-white/20 hover:text-white cursor-pointer" onClick={onSignOut}>
                Sign out
              </Button>
            </div>
          </div>

          <Separator className="my-6 bg-white/20" />

          <div className="grid sm:grid-cols-3 gap-4">
            <Card className="border-white/10 bg-[#13203a] text-white">
              <div className="p-5">
                <p className="text-xs uppercase tracking-wider text-white/80">Role</p>
                <h3 className="mt-2 text-sm text-white">{role ?? (loading ? "Loading…" : "Unknown")}</h3>
                <p className="mt-1 text-xs text-white/70">RBAC determines your access and actions.</p>
              </div>
            </Card>
            <Card className="border-white/10 bg-[#13203a] text-white">
              <div className="p-5">
                <p className="text-xs uppercase tracking-wider text-white/80">Session</p>
                <h3 className="mt-2 text-sm text-white">{loading ? "Loading…" : user ? "Active" : "Guest"}</h3>
                <p className="mt-1 text-xs text-white/70">Signed in users gain live oversight capabilities.</p>
              </div>
            </Card>
            <Card className="border-white/10 bg-[#13203a] text-white">
              <div className="p-5">
                <p className="text-xs uppercase tracking-wider text-white/80">Security</p>
                <h3 className="mt-2 text-sm text-white">Encrypted</h3>
                <p className="mt-1 text-xs text-white/70">Privacy-first identity handling at every layer.</p>
              </div>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

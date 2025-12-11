"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { toast } from "sonner";
import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";

type FaceImage = { id: string; cloudinaryPublicId: string; capturedAt: string };
type Enrollment = { id: string; createdAt: string; status: string; organization?: string | null; isWanted?: boolean; notes?: string | null; images: FaceImage[] };
type UserRow = { id: string; name: string; email: string; role: string; organization?: string | null; createdAt: string; updatedAt: string; enrollments: Enrollment[] };

export default function AdminDashboardPage() {
  const router = useRouter();
  const { data: session, isPending } = authClient.useSession();
  const [currentRole, setCurrentRole] = useState<string | undefined>();
  const [query, setQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState<string>("");
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(10);
  const [users, setUsers] = useState<UserRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState<number>(0);
  const [wantedFilter, setWantedFilter] = useState<string>(""); // "" | "true" | "false"

  // Resolve role via server API
  useEffect(() => {
    async function loadRole() {
      try {
        const email = session?.user?.email;
        const id = session?.user?.id as string | undefined;
        if (!email && !id) return;
        const params = new URLSearchParams(email ? { email } : { id: id! });
        const res = await fetch(`/api/user-role?${params.toString()}`, { credentials: "include" });
        const json = await res.json();
        if (res.ok && json?.role) setCurrentRole(json.role as string);
      } catch {}
    }
    loadRole();
  }, [session?.user?.email, session?.user?.id]);

  // No client-side org filter; server enforces org from the admin's user record

  // Admin gate
  useEffect(() => {
    if (!isPending && currentRole && currentRole !== "ADMIN") {
      toast.error("Dashboard is restricted to admins");
      router.replace("/signin");
    }
  }, [isPending, currentRole, router]);

  // Fetch users by query, role, and pagination with server-side org scoping (email/id required)
  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        setLoading(true);
        const email = session?.user?.email;
        const id = session?.user?.id as string | undefined;
        if (!email && !id) {
          // Wait for session to resolve
          setLoading(false);
          return;
        }
  const params = new URLSearchParams({ q: query, page: String(page), pageSize: String(pageSize) });
        if (roleFilter) params.set("role", roleFilter);
  if (wantedFilter) params.set("wanted", wantedFilter);
        params.set(email ? "email" : "id", email ?? id!);
        const res = await fetch(`/api/admin/users?${params.toString()}`, { credentials: "include" });
        if (res.status === 403) {
          toast.error("Access restricted: admin only");
          router.replace("/");
          return;
        }
        const contentType = res.headers.get("content-type") || "";
        if (!contentType.includes("application/json")) {
          toast.error("Failed to load users (unexpected response). Are you signed in as admin?");
          setUsers([]);
          setTotal(0);
          return;
        }
        const json = await res.json();
        if (res.ok) {
          setUsers(json.users || []);
          setTotal(json.total || 0);
        } else {
          toast.error(json?.error ?? "Failed to load users");
        }
      } catch (e: any) {
        toast.error(e?.message ?? "Failed to load users");
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [query, roleFilter, wantedFilter, page, pageSize, session?.user?.email, session?.user?.id, router]);

  async function getSignedUrl(publicId: string) {
    try {
      const res = await fetch(`/api/cloudinary/sign-url?publicId=${encodeURIComponent(publicId)}&expires=60`);
      const json = await res.json();
      if (res.ok) return json.url as string;
    } catch {}
    return undefined;
  }

  return (
    <div className="min-h-svh text-white flex items-center justify-center px-4 py-10 bg-[#0a0f1a]">
      <Card className="w-full max-w-6xl border border-white/10 bg-[#0d1628] shadow-2xl">
        <div className="p-6 md:p-8 text-white">
          <h1 className="text-2xl text-white font-semibold">Admin Dashboard</h1>
          <p className="text-sm text-white/70">Search by name, organization, or email to view user details and images.</p>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-6 gap-3">
            <div className="sm:col-span-2">
              <Label htmlFor="search" className="py-2">Search</Label>
              <Input id="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Jane, Savannah Gates, jane@example.com" className="bg-[#031422] border-white/8 text-white" />
            </div>
            <div>
              <Label htmlFor="wanted" className="py-2">Wanted</Label>
              <select id="wanted" value={wantedFilter} onChange={(e) => { setWantedFilter(e.target.value); setPage(1); }} className="w-full h-10 rounded-md bg-[#031422] border border-white/8 text-white">
                <option value="">All</option>
                <option value="true">Wanted only</option>
                <option value="false">Non-wanted only</option>
              </select>
            </div>
            <div>
              <Label htmlFor="role" className="py-2">Role</Label>
              <select id="role" value={roleFilter} onChange={(e) => { setRoleFilter(e.target.value); setPage(1); }} className="w-full h-10 rounded-md bg-[#031422] border border-white/8 text-white">
                <option value="">All</option>
                <option value="VIEWER">Viewer</option>
                <option value="GATEKEEPER">Gatekeeper</option>
                <option value="ADMIN">Admin</option>
              </select>
            </div>
            <div>
              <Label htmlFor="pageSize" className="py-2">Page size</Label>
              <select id="pageSize" value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }} className="w-full h-10 rounded-md bg-[#031422] border border-white/8 text-white">
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
            </div>
            <div className="flex items-end">
              <Button className="w-full bg-white text-black hover:bg-white/85" disabled> {loading ? "Searching…" : "Type to search"} </Button>
            </div>
          </div>

          <Separator className="my-6 bg-white/10" />

          <div className="mt-2 text-sm text-white/70">{total} results • Page {page}</div>
          <div className="space-y-6">
            {users.length === 0 && <div className="text-white/60">No results{query ? ` for "${query}"` : ""}.</div>}
            {users.map((u) => (
              <Card key={u.id} className="border-white/10 bg-[#13203a]">
                <div className="p-4">
                  <div className="flex flex-wrap items-center gap-4">
                    <div className="text-white">
                      <div className="text-sm">Name</div>
                      <div className="text-white font-medium">{u.name}</div>
                    </div>
                    <div className="text-white">
                      <div className="text-sm">Email</div>
                      <div className="text-white font-medium">{u.email}</div>
                    </div>
                    <div className="text-white">
                      <div className="text-sm">Organization</div>
                      <div className="text-white font-medium">{u.organization ?? "—"}</div>
                    </div>
                    <div className="text-white ml-auto">
                      <div className="text-sm">Role</div>
                      <div className="text-white font-medium">{u.role}</div>
                    </div>
                  </div>

                  <Separator className="my-4 bg-white/10" />

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {u.enrollments.map((en) => (
                      <Card key={en.id} className="border-white/10 bg-[#0f1d33]">
                        <div className="p-3">
                          <div className="flex items-center justify-between text-white/80 text-sm">
                            <div>Enrollment: {new Date(en.createdAt).toLocaleString()}</div>
                            <div className="flex items-center gap-2">
                              <span>Status: {en.status}</span>
                              {en.isWanted && (
                                <Popover>
                                  <PopoverTrigger asChild>
                                    <button className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-600/80 text-white hover:bg-red-600 cursor-pointer">
                                      Wanted
                                    </button>
                                  </PopoverTrigger>
                                  <PopoverContent className="w-[360px] bg-[#071528] border border-white/10 text-white">
                                    <div className="space-y-2">
                                      <div className="text-sm font-semibold">Reason</div>
                                      <div className="text-sm text-white/80">{en.notes ? en.notes : "No reason provided"}</div>
                                      <Separator className="my-2 bg-white/10" />
                                      <div className="grid grid-cols-2 gap-2">
                                        {en.images.length === 0 && (
                                          <div className="text-xs text-white/60">No photos</div>
                                        )}
                                        {en.images.map((img) => (
                                          <ImageThumb key={img.id} publicId={img.cloudinaryPublicId} capturedAt={img.capturedAt} getSignedUrl={getSignedUrl} />
                                        ))}
                                      </div>
                                    </div>
                                  </PopoverContent>
                                </Popover>
                              )}
                            </div>
                          </div>
                          <div className="mt-3 grid grid-cols-3 gap-2">
                            {en.images.length === 0 && <div className="text-white/60 text-sm">No images</div>}
                            {en.images.map((img) => (
                              <ImageThumb key={img.id} publicId={img.cloudinaryPublicId} capturedAt={img.capturedAt} getSignedUrl={getSignedUrl} />
                            ))}
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>
              </Card>
            ))}
            <div className="flex items-center justify-end gap-2">
              <Button variant="outline" className="border-white/20 text-black" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>Prev</Button>
              <Button variant="outline" className="border-white/20 text-black" disabled={page * pageSize >= total} onClick={() => setPage((p) => p + 1)}>Next</Button>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function ImageThumb({ publicId, capturedAt, getSignedUrl }: { publicId: string; capturedAt: string; getSignedUrl: (pid: string) => Promise<string | undefined>; }) {
  const [url, setUrl] = useState<string | undefined>(undefined);
  useEffect(() => {
    let mounted = true;
    (async () => {
      const u = await getSignedUrl(publicId);
      if (mounted) setUrl(u);
    })();
    return () => { mounted = false; };
  }, [publicId, getSignedUrl]);

  return (
    <div className="relative bg-[#091426] border border-white/8 rounded overflow-hidden">
      {url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={url} alt={publicId} className="w-full h-24 object-cover" />
      ) : (
        <div className="w-full h-24 grid place-items-center text-white/60 text-xs">Loading…</div>
      )}
      <div className="absolute bottom-1 right-1 text-[10px] text-white/70">{new Date(capturedAt).toLocaleTimeString()}</div>
    </div>
  );
}

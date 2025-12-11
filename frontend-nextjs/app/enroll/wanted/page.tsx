"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";

 type UploadPhoto = {
  id: string;
  file: File;
  url: string;
  addedAt: string;
  sizeKB: number;
 };

export default function WantedEnrollPage() {
  const router = useRouter();
  const { data: session, isPending } = authClient.useSession();
  const [currentRole, setCurrentRole] = useState<string | undefined>(undefined);

  useEffect(() => {
    async function loadRole() {
      try {
        const email = session?.user?.email;
        const id = session?.user?.id as string | undefined;
        if (!email && !id) return;
        const params = new URLSearchParams(email ? { email } : { id: id! });
        const res = await fetch(`/api/user-role?${params.toString()}`);
        const json = await res.json();
        if (res.ok && json?.role) setCurrentRole(json.role as string);
      } catch {}
    }
    loadRole();
  }, [session?.user?.email, session?.user?.id]);

  // Form fields
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [organization, setOrganization] = useState("");
  const [organizations, setOrganizations] = useState<string[]>([]);
  const [orgLoading, setOrgLoading] = useState<boolean>(true);
  const [orgError, setOrgError] = useState<string>("");
  const [role, setRole] = useState("VIEWER");
  const [notes, setNotes] = useState("");

  const [uploadPhotos, setUploadPhotos] = useState<UploadPhoto[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  const canRegister = uploadPhotos.length >= 1 && uploadPhotos.length <= 5 && fullName.trim() && email.trim() && organization.trim() && notes.trim();

  // Load organizations
  useEffect(() => {
    async function loadOrganizations() {
      try {
        setOrgLoading(true);
        setOrgError("");
        const res = await fetch("/api/organizations", { credentials: "include" });
        const contentType = res.headers.get("content-type") || "";
        if (!contentType.includes("application/json")) {
          setOrgError("Unexpected response from server");
          return;
        }
        const json = await res.json();
        if (res.ok && Array.isArray(json.organizations)) {
          setOrganizations(json.organizations);
          if (!organization && json.organizations.length > 0) {
            setOrganization(json.organizations[0]);
          }
        } else {
          setOrgError(json?.error || "Failed to load organizations");
        }
      } catch (e: any) {
        setOrgError(e?.message || "Failed to load organizations");
      } finally {
        setOrgLoading(false);
      }
    }
    loadOrganizations();
  }, []);

  // Admin gate
  if (!isPending && currentRole && currentRole !== "ADMIN") {
    if (typeof window !== "undefined") {
      toast.error("Access restricted to admins");
      router.replace("/signin");
    }
    return null;
  }

  // File input handlers
  const onFilesSelected = useCallback((filesList: FileList | null) => {
    if (!filesList) return;
    const arr = Array.from(filesList);
    const currentCount = uploadPhotos.length;
    const remaining = Math.max(0, 5 - currentCount);
    const toAdd = arr.slice(0, remaining);
    const newItems: UploadPhoto[] = toAdd.map((f) => ({
      id: crypto.randomUUID(),
      file: f,
      url: URL.createObjectURL(f),
      addedAt: new Date().toISOString(),
      sizeKB: Math.round(f.size / 1024),
    }));
    setUploadPhotos((prev) => [...prev, ...newItems]);
  }, [uploadPhotos.length]);

  const removePhoto = useCallback((id: string) => {
    setUploadPhotos((prev) => {
      const item = prev.find((x) => x.id === id);
      if (item) URL.revokeObjectURL(item.url);
      return prev.filter((x) => x.id !== id);
    });
  }, []);

  const resetPhotos = useCallback(() => {
    uploadPhotos.forEach((p) => URL.revokeObjectURL(p.url));
    setUploadPhotos([]);
  }, [uploadPhotos]);

  const initials = useMemo(() => {
    const name = fullName || email || "U";
    return name.split(" ").map((p) => p[0]?.toUpperCase()).slice(0, 2).join("");
  }, [fullName, email]);

  const handleRegister = useCallback(async () => {
    if (!canRegister) return toast.error("Please fill required fields and upload 1-5 photos.");
    setLoading(true);
    setUploadProgress(0);

    try {
      const appUrl = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:8000";

      // 1) Send images + metadata to backend embeddings API (wanted=true)
      const form = new FormData();
      form.append("fullName", fullName);
      form.append("email", email);
      form.append("organization", organization);
      form.append("role", role);
      form.append("notes", notes);
      form.append("wanted", "true");
      uploadPhotos.forEach((p, idx) => form.append("files", p.file, `wanted_${idx + 1}.jpg`));

      // simulate upload progress
      await new Promise<void>((res) => {
        const steps = 5; let i = 0;
        const t = setInterval(() => {
          i++;
          setUploadProgress(Math.round((i / steps) * 100));
          if (i >= steps) { clearInterval(t); res(); }
        }, 180);
      });

      const res = await fetch(`${appUrl}/api/v1/embeddings`, { method: "POST", body: form });
      if (!res.ok){
        throw new Error(`Server error: ${res.status}`)
      }
      await res.json();

      // 2) Upload images to Cloudinary
      const cloudForm = new FormData();
      uploadPhotos.forEach((p, idx) => cloudForm.append("files", p.file, `wanted_${idx + 1}.jpg`));
      cloudForm.append("folder", "faces/wanted");
      const cloudRes = await fetch(`/api/cloudinary/upload`, { method: "POST", body: cloudForm });
      if (!cloudRes.ok) throw new Error(`Cloudinary upload failed: ${cloudRes.status}`);
      const cloudJson = await cloudRes.json();
      const publicIds: string[] = (cloudJson?.items ?? []).map((i: any) => i.public_id);
      if (!publicIds.length) throw new Error("No public_ids returned from Cloudinary");

      // 3) Upsert metadata + public_ids to Postgres with isWanted=true
      const enrollRes = await fetch(`/api/enroll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: fullName,
          email,
          organization,
          role,
          notes,
          images: publicIds,
          isWanted: true,
        }),
      });
      const enrollJson = await enrollRes.json();
      if (!enrollRes.ok) throw new Error(enrollJson?.error ?? `Enroll upsert failed: ${enrollRes.status}`);

      toast.success("Wanted enrollment complete");
      // Reset form
      setFullName("");
      setEmail("");
      setOrganization("");
      setRole("VIEWER");
      setNotes("");
      resetPhotos();
      setUploadProgress(null);
      // Reload the browser to fully refresh state
      if (typeof window !== "undefined") {
        try { window.location.reload(); } catch {}
      }
    } catch (err: any) {
      console.error(err);
      toast.error(err?.message ?? "Registration failed");
      setUploadProgress(null);
    } finally { setLoading(false); }
  }, [canRegister, fullName, email, organization, role, notes, uploadPhotos]);

  return (
    <div className="min-h-svh text-white flex items-center justify-center px-4 py-10 bg-linear-to-b from-[#061022] to-[#071226]">
      <Card className="w-full max-w-6xl border border-white/8 bg-[#071528] shadow-xl">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-0 md:gap-0">

          {/* LEFT: Upload */}
          <div className="p-6 md:p-8 border-b md:border-b-0 md:border-r border-white/6">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div>
                <h2 className="text-xl font-semibold text-white">Upload Wanted Photos</h2>
                <p className="text-sm text-white/70 mt-1">Upload 1–5 clear photos.</p>
              </div>
              <div className="text-right text-xs text-white/60">Photos: <span className="font-medium">{uploadPhotos.length}/5</span></div>
            </div>

            <div className="rounded-md bg-black/60 border border-white/6 p-3">
              <div className="w-full rounded overflow-hidden relative grid place-items-center">
                <label className="inline-flex items-center justify-center px-4 py-2 rounded-md bg-white text-black hover:bg-white/90 cursor-pointer select-none">
                  Choose files
                  <input type="file" accept="image/*" multiple onChange={(e) => onFilesSelected(e.target.files)} className="hidden" />
                </label>
              </div>

              <Separator className="my-4 bg-white/6" />

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {uploadPhotos.length === 0 && <div className="text-white/60 text-sm col-span-2">No photos yet — upload a few good shots.</div>}
                {uploadPhotos.map((p) => (
                  <div key={p.id} className="relative bg-[#031422] rounded overflow-hidden border border-white/6">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={p.url} alt={`Photo ${p.id}`} className="w-full h-40 object-contain bg-black" />
                    <div className="p-2 text-xs text-white/80">
                      <div className="flex justify-between items-center">
                        <div>{new Date(p.addedAt).toLocaleTimeString()}</div>
                        <div className="text-white/60">{p.sizeKB} KB</div>
                      </div>
                    </div>
                    <div className="absolute top-2 right-2 flex gap-2">
                      <Button size="sm" variant="destructive" className="px-2! py-1! text-xs" onClick={() => removePhoto(p.id)}>Remove</Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* RIGHT: Metadata + Submit */}
          <div className="p-6 md:p-8">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div>
                <h2 className="text-xl font-semibold text-white">Wanted Details</h2>
                <p className="text-sm text-white/70 mt-1">Provide identity details. Required fields are marked.</p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 text-white">
              <div>
                <Label htmlFor="fullName" className="pb-2">Full name *</Label>
                <Input id="fullName" value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Jane Doe" className="bg-[#031422] border-white/8 text-white" />
              </div>
              <div>
                <Label htmlFor="email" className="pb-2">Email *</Label>
                <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="jane@example.com" className="bg-[#031422] border-white/8 text-white" />
              </div>
              <div>
                <Label htmlFor="organization" className="pb-2">Organization *</Label>
                <select
                  id="organization"
                  value={organization}
                  onChange={(e) => setOrganization(e.target.value)}
                  className="bg-[#031422] border border-white/8 rounded-md px-3 py-2 text-white"
                >
                  {orgLoading && <option value="">Loading…</option>}
                  {!orgLoading && organizations.length === 0 && (
                    <option value="" disabled>No organizations found</option>
                  )}
                  {orgError && !orgLoading && (
                    <option value="" disabled>Error: {orgError}</option>
                  )}
                  {organizations.map((org) => (
                    <option key={org} value={org}>{org}</option>
                  ))}
                </select>
              </div>
              <div>
                <Label htmlFor="role" className="pb-2">Role</Label>
                <select id="role" value={role} onChange={(e) => setRole(e.target.value)} className="bg-[#031422] border border-white/8 rounded-md px-3 py-2 text-white">
                  <option value="VIEWER">Viewer</option>
                  <option value="GATEKEEPER">Gatekeeper</option>
                  <option value="ADMIN">Admin</option>
                </select>
              </div>
              <div>
                <Label htmlFor="notes" className="pb-2">Reason *</Label>
                <Textarea id="notes" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Provide a reason for wanted status" className="bg-[#031422] border-white/8 text-white" />
              </div>
              <Separator className="my-2 bg-white/6" />
              <Button onClick={handleRegister} disabled={!canRegister || loading} className="w-full h-12 bg-white text-black hover:bg-white/85">
                {loading ? "Registering…" : "Register Wanted & Upload"}
              </Button>
              {uploadProgress !== null && <div className="mt-2 text-sm text-white/70">Uploading: {uploadProgress}%</div>}
              <div className="mt-3 text-xs text-white/60">Requires 1 to 5 photos and all required metadata.</div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

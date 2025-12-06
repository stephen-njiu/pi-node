"use client";
import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

type Photo = {
  id: string;
  blob: Blob;
  url: string;
  takenAt: string;
  sizeKB: number;
};

export default function EnrollPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [organization, setOrganization] = useState("");
  const [role, setRole] = useState("VIEWER");
  const [notes, setNotes] = useState("");

  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);

  const canCapture = photos.length < 5;
  const canRegister = photos.length >= 3 && fullName.trim() && email.trim() && organization.trim();

  // --- Cleanup ---
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      photos.forEach((p) => URL.revokeObjectURL(p.url));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Load video devices ---
  useEffect(() => {
    async function fetchDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter((d) => d.kind === "videoinput");
        setVideoDevices(cams);
        if (!selectedDeviceId && cams.length > 0) setSelectedDeviceId(cams[0].deviceId);
      } catch (err) {
        console.error("Error fetching video devices", err);
      }
    }
    fetchDevices();
  }, [selectedDeviceId]);

  // --- Open camera ---
  const openCamera = useCallback(async () => {
    if (!selectedDeviceId) return toast.error("No camera selected");
    setInitializing(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Video element not mounted");
      video.srcObject = stream;

      await new Promise<void>((resolve, reject) => {
        let settled = false;
        const onMeta = () => {
          if (settled) return;
          settled = true;
          cleanup();
          resolve();
        };
        const onError = () => {
          if (settled) return;
          settled = true;
          cleanup();
          reject(new Error("Failed to load camera metadata"));
        };
        const cleanup = () => {
          video.removeEventListener("loadedmetadata", onMeta);
          video.removeEventListener("error", onError);
        };
        if (video.readyState >= 1) resolve();
        else {
          video.addEventListener("loadedmetadata", onMeta);
          video.addEventListener("error", onError);
          setTimeout(() => { if (!settled) { settled = true; cleanup(); resolve(); } }, 3000);
        }
      });

      try { await video.play(); } catch {}

      setCameraOpen(true);
      setCameraReady(true);
    } catch (err: any) {
      console.error(err);
      toast.error("Unable to access camera. Check permissions.");
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      setCameraOpen(false);
      setCameraReady(false);
    } finally { setInitializing(false); }
  }, [selectedDeviceId]);

  const closeCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraOpen(false);
    setCameraReady(false);
  }, []);

  // --- Capture photo ---
  async function makeBlobFromVideoFrame(video: HTMLVideoElement, targetWidth = 640, quality = 0.95) {
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || (vw * 0.75);
    const aspect = vh / vw;
    const width = targetWidth;
    const height = Math.round(width * aspect);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Failed to get canvas context");
    ctx.drawImage(video, 0, 0, width, height);

    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
    if (!blob) throw new Error("Failed to build image blob");
    return blob;
  }

  const takePhoto = useCallback(async () => {
    if (!videoRef.current) return toast.error("Camera not ready");
    if (!cameraOpen || !cameraReady) return toast.error("Open the camera first");
    if (photos.length >= 5) return toast.info("Maximum of 5 photos");

    try {
      const previewBlob = await makeBlobFromVideoFrame(videoRef.current, 640, 0.95);
      const url = URL.createObjectURL(previewBlob);
      const id = (crypto as any).randomUUID ? (crypto as any).randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const sizeKB = Math.round(previewBlob.size / 1024);
      const photo: Photo = { id, blob: previewBlob, url, takenAt: new Date().toISOString(), sizeKB };
      setPhotos((p) => [...p, photo]);
    } catch (err) {
      console.error(err);
      toast.error("Failed to capture photo");
    }
  }, [cameraOpen, cameraReady, photos.length]);

  const removePhoto = useCallback((id: string) => {
    setPhotos((prev) => {
      const ph = prev.find((x) => x.id === id);
      if (ph) URL.revokeObjectURL(ph.url);
      return prev.filter((x) => x.id !== id);
    });
  }, []);

  const resetPhotos = useCallback(() => {
    photos.forEach((p) => URL.revokeObjectURL(p.url));
    setPhotos([]);
  }, [photos]);

  const handleRegister = useCallback(async () => {
    if (!canRegister) return toast.error("Please fill required fields and capture at least 3 photos.");
    setLoading(true);
    setUploadProgress(0);

    try {
      const form = new FormData();
      form.append("fullName", fullName);
      form.append("email", email);
      form.append("organization", organization);
      form.append("role", role);
      form.append("notes", notes);
  // The FastAPI expects the field name 'files' (see Python script)
  photos.forEach((p, idx) => form.append("files", p.blob, `photo_${idx + 1}.jpg`));

      const baseUrl = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:8000";

      // simulate upload progress
      await new Promise<void>((res) => {
        const steps = 5; let i = 0;
        const t = setInterval(() => {
          i++;
          setUploadProgress(Math.round((i / steps) * 100));
          if (i >= steps) { clearInterval(t); res(); }
        }, 180);
      });

      const res = await fetch(`${baseUrl}/api/v1/embeddings`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      await res.json();
      toast.success("Enrollment sent successfully");
      setUploadProgress(null);
    } catch (err: any) {
      console.error(err);
      toast.error(err?.message ?? "Registration failed");
      setUploadProgress(null);
    } finally { setLoading(false); }
  }, [canRegister, fullName, email, organization, role, notes, photos]);

  const initials = useMemo(() => {
    const name = fullName || email || "U";
    return name.split(" ").map((p) => p[0]?.toUpperCase()).slice(0, 2).join("");
  }, [fullName, email]);

  return (
    <div className="min-h-svh text-white flex items-center justify-center px-4 py-10 bg-linear-to-b from-[#061022] to-[#071226]">
      <Card className="w-full max-w-6xl border border-white/8 bg-[#071528] shadow-xl">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-0 md:gap-0">

          {/* LEFT: Camera */}
          <div className="p-6 md:p-8 border-b md:border-b-0 md:border-r border-white/6">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div>
                <h2 className="text-xl font-semibold text-white">Capture Photos</h2>
                <p className="text-sm text-white/70 mt-1">Take 3–5 clear photos.</p>
              </div>
              <div className="text-right text-xs text-white/60">Photos: <span className="font-medium">{photos.length}/5</span></div>
            </div>

            <div className="rounded-md bg-black/60 border border-white/6 p-3">
              <div className="aspect-video w-full bg-black rounded overflow-hidden relative grid place-items-center">
                {initializing && <div className="absolute inset-0 grid place-items-center bg-black/40 text-white/80 text-sm">Initializing camera…</div>}
                {!cameraOpen && !initializing && <div className="text-white/60 text-sm p-6">Camera closed — click "Open Camera" to start</div>}
                <video ref={videoRef} className={`w-full h-full object-cover ${cameraOpen ? "opacity-100" : "opacity-0"}`} muted playsInline aria-hidden={!cameraOpen} />
                {!cameraOpen && !initializing && <div className="absolute bottom-3 left-3 text-xs text-white/60">Preview disabled</div>}
              </div>

              {/* Camera controls */}
              <div className="flex flex-wrap gap-3 mt-3 items-center">
                {!cameraOpen ? (
                  <Button onClick={openCamera} className="bg-white text-black hover:bg-white/95">Open Camera</Button>
                ) : (
                  <Button onClick={closeCamera} variant="outline" className="border-white/20 text-black">Close Camera</Button>
                )}

                {videoDevices.length > 1 && (
                  <div className="flex items-center gap-2">
                    <Label htmlFor="cameraSelect" className="text-white/80 text-sm">Camera:</Label>
                    <select
                      id="cameraSelect"
                      value={selectedDeviceId ?? ""}
                      onChange={(e) => { setSelectedDeviceId(e.target.value); if (cameraOpen) openCamera(); }}
                      className="bg-[#031422] border border-white/8 rounded-md px-2 py-1 text-white text-sm"
                    >
                      {videoDevices.map((d, idx) => (
                        <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${idx + 1}`}</option>
                      ))}
                    </select>
                  </div>
                )}

                <Button onClick={takePhoto} disabled={!cameraOpen || !cameraReady || !canCapture} className="bg-white text-black">Take Photo</Button>
                <Button onClick={resetPhotos} variant="ghost" className="text-white/80">Reset Photos</Button>
              </div>

              <Separator className="my-4 bg-white/6" />

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {photos.length === 0 && <div className="text-white/60 text-sm col-span-2">No photos yet — capture a few good shots.</div>}
                {photos.map((p) => (
                  <div key={p.id} className="relative bg-[#031422] rounded overflow-hidden border border-white/6">
                    <img src={p.url} alt={`Photo ${p.id}`} className="w-full h-28 object-cover" />
                    <div className="p-2 text-xs text-white/80">
                      <div className="flex justify-between items-center">
                        <div>{new Date(p.takenAt).toLocaleTimeString()}</div>
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
                <h2 className="text-xl font-semibold text-white">User Details</h2>
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
                <Input id="organization" value={organization} onChange={(e) => setOrganization(e.target.value)} placeholder="Company" className="bg-[#031422] border-white/8 text-white" />
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
                <Label htmlFor="notes" className="pb-2">Notes</Label>
                <Textarea id="notes" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Additional context…" className="bg-[#031422] border-white/8 text-white" />
              </div>
              <Separator className="my-2 bg-white/6" />
              <Button onClick={handleRegister} disabled={!canRegister || loading} className="w-full h-12 bg-white text-black hover:bg-white/85">
                {loading ? "Registering…" : "Register & Upload"}
              </Button>
              {uploadProgress !== null && <div className="mt-2 text-sm text-white/70">Uploading: {uploadProgress}%</div>}
              <div className="mt-3 text-xs text-white/60">Requires at least 3 photos and all required metadata.</div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

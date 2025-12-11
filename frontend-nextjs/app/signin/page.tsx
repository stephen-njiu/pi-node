"use client";
import { useState } from "react";
import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";

function GoogleIcon() {
	return (
		<svg width="18" height="18" viewBox="0 0 48 48" aria-hidden>
			<path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3c-1.7 4.9-6.4 8.4-11.8 8.4-6.9 0-12.6-5.6-12.6-12.4S16.6 11.6 23.5 11.6c3.1 0 5.9 1.1 8.1 2.9l5.7-5.6C33.7 5.9 29.8 4.6 25.6 4.6 14.6 4.6 5.6 13.5 5.6 24.6s9 20 20 20c9.9 0 18.2-7 20-16.2.2-1 .4-2 .4-3.1 0-1.1-.1-2.2-.4-3.2z"/>
			<path fill="#FF3D00" d="M6.3 14.7l6.6 4.8c1.8-4.2 6-7.2 10.6-7.2 3.1 0 5.9 1.1 8.1 2.9l5.7-5.6C33.7 5.9 29.8 4.6 25.6 4.6 18.5 4.6 12.2 8.4 8.9 14.1z"/>
			<path fill="#4CAF50" d="M24.6 44.6c5.4 0 10.1-1.8 13.8-4.9l-6.4-5.2c-2 1.4-4.5 2.2-7.1 2.2-5.4 0-10.1-3.5-11.8-8.4l-6.6 5.1c3.3 6.6 10.1 11.2 18.1 11.2z"/>
			<path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-1 2.9-3.2 5.3-6.2 6.5l6.4 5.2c3.7-3.1 6.5-7.5 7.5-12.7.2-1 .4-2 .4-3.1 0-1.1-.1-2.2-.4-3.2z"/>
		</svg>
	);
}

export default function SignIn() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGoogleSignIn = async () => {
    setError(null);
    setLoading(true);
    try {
      await authClient.signIn.social({
        provider: "google",
        callbackURL: "/signin",
        errorCallbackURL: "/signin",
      });
      router.refresh();
    //   toast.success("Signed in successfully");
    } catch (e: any) {
      const msg = e?.message ?? "Sign-in failed. Please try again.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
			<div className="min-h-svh text-white flex items-center justify-center px-4 py-10 bg-[#0a0f1a]">
				<Card className="w-full max-w-6xl border border-white/10 bg-[#0d1628] shadow-2xl">
					<div className="grid grid-cols-1 md:grid-cols-2">
						{/* Left: Brand & Value Props */}
						<div className="p-8 md:p-12 border-b md:border-b-0 md:border-r border-white/10">
							<div className="flex items-center gap-3 mb-6">
								<div className="size-10 rounded-md bg-white text-black grid place-items-center font-extrabold">SG</div>
								<div>
									<h1 className="text-2xl text-white font-semibold">Savannah Gates</h1>
									<p className="text-sm text-white/80">Automated Facial Recognition for reliable gate access</p>
								</div>
							</div>

							<div className="space-y-4 text-white">
								<p className="text-sm">
									A modern access platform that blends computer vision and robust controls. Detect, verify, and grant access in milliseconds.
								</p>
								<p className="text-sm">
									Designed for the edge and the cloud—fast at the gate, consistent in the dashboard.
								</p>
							</div>

							<Separator className="my-6 bg-white/20" />

							<div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
								<div className="rounded-md border border-white/10 bg-[#13203a] p-4">
									<p className="text-xs uppercase tracking-wider text-white/80">Accuracy</p>
									<p className="mt-1 text-sm text-white">Industry-grade face detection & embeddings</p>
								</div>
								<div className="rounded-md border border-white/10 bg-[#13203a] p-4">
									<p className="text-xs uppercase tracking-wider text-white/80">Control</p>
									<p className="mt-1 text-sm text-white">RBAC with instant revocation</p>
								</div>
								<div className="rounded-md border border-white/10 bg-[#13203a] p-4">
									<p className="text-xs uppercase tracking-wider text-white/80">Resilience</p>
									<p className="mt-1 text-sm text-white">Edge-ready for offline operations</p>
								</div>
								<div className="rounded-md border border-white/10 bg-[#13203a] p-4">
									<p className="text-xs uppercase tracking-wider text-white/80">Privacy</p>
									<p className="mt-1 text-sm text-white">Your identity, protected end-to-end</p>
								</div>
							</div>
						</div>

						{/* Right: Sign In */}
						<div className="p-8 md:p-12">
							<h2 className="text-white text-xl font-medium mb-2">Sign in</h2>
							<p className="text-sm text-white/80 mb-6">Use your Google account to continue</p>

							{error && (
								<div className="rounded-md border border-red-500/40 bg-red-500/20 px-3 py-2 text-sm text-red-100 mb-4">
									{error}
								</div>
							)}

							<Button
								type="button"
								onClick={handleGoogleSignIn}
								disabled={loading}
								className="cursor-pointer w-full h-11 bg-white text-black hover:bg-white/90 font-medium flex items-center justify-center gap-2"
							>
								<GoogleIcon />
								{loading ? "Signing in…" : "Continue with Google"}
							</Button>

							<Separator className="my-6 bg-white/20" />
							<p className="text-xs text-white/70">
								By continuing, you agree to our Terms and acknowledge our Privacy Policy.
							</p>
						</div>
					</div>
				</Card>
			</div>
			);
	}

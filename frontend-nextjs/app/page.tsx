"use client";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useCurrency, formatPrice } from "@/lib/use-currency";

export default function HomePage() {
  const { currency, rate } = useCurrency();

  return (
    <div className="min-h-svh bg-linear-to-b from-[#0b1224] via-[#0c152c] to-[#0e1936] text-white">
      {/* HERO */}
      <section className="mx-auto max-w-6xl px-4 pt-14 pb-10 md:pt-24 md:pb-16">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-[#14203f] px-3 py-1 text-xs text-white/80">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
              Savannah Gates
            </div>
            <h1 className="mt-4 text-4xl md:text-6xl font-semibold leading-tight text-white">
              Secure, Modern Gate Access
            </h1>
            <p className="mt-4 text-white/80 md:text-lg max-w-prose">
              Enroll your staff, manage access, and flag wanted individuals with a secure, scalable platform.
            </p>
            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <Link href="/enroll">
                <Button className="w-full sm:w-auto bg-white text-black hover:bg-white/85" aria-label="Get started with enrollment">
                  Get Started
                </Button>
              </Link>
              <Link href="/signin">
                <Button variant="outline" className="hover:text-white w-full sm:w-auto border-white/30 text-black hover:bg-white/10" aria-label="Sign in">
                  Sign In
                </Button>
              </Link>
              <Link href="/dashboard">
                <Button variant="outline" className="hover:text-white w-full sm:w-auto border-white/30 text-black hover:bg-white/10" aria-label="Open admin dashboard">
                  View Dashboard
                </Button>
              </Link>
              <Link href="/enroll/wanted" className="sm:ml-2">
                <Button variant="outline" className="hover:text-white w-full sm:w-auto border-rose-400/40 text-neutral-900 hover:bg-rose-500/10" aria-label="Create wanted enrollment">
                  Wanted Enroll
                </Button>
              </Link>
            </div>
            <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4 text-white/80 text-sm">
              <div className="rounded-md border border-white/15 bg-[#121e3a] p-3">FastAPI Embeddings</div>
              <div className="rounded-md border border-white/15 bg-[#121e3a] p-3">Cloudinary Storage</div>
              <div className="rounded-md border border-white/15 bg-[#121e3a] p-3">Neon Postgres</div>
              <div className="rounded-md border border-white/15 bg-[#121e3a] p-3">Better Auth</div>
            </div>
          </div>
          <Card className="border-white/15 bg-[#121e3a] p-0 overflow-hidden">
            <div className="p-5 md:p-6">
              <div className="text-sm text-white/80">Live Preview</div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-white/50">
                {Array.from({ length: 6 }).map((_, i) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img key={i} src={`/pic${i + 1}.png`} alt={`Preview ${i + 1}`} className="aspect-square w-full rounded border border-white/10 object-cover" />
                ))}
              </div>
              {/* <div className="mt-4 text-xs text-white/70">Camera frames, uploads, and signed URLs will render here once available.</div> */}
            </div>
          </Card>
        </div>
      </section>

      <Separator className="mx-auto max-w-6xl bg-white/10" />

      {/* FEATURES */}
      <section className="mx-auto max-w-6xl px-4 py-12 md:py-16">
  <h2 className="text-xl md:text-2xl font-semibold text-white">Why Savannah Gates</h2>
  <p className="mt-1 text-white/80">Production-grade building access, made simple.</p>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              title: "Frictionless Enrollment",
              desc: "Capture via camera or upload 1–5 images. Robust pipeline feeds embeddings, storage, and DB in one go.",
            },
            {
              title: "Wanted Flagging",
              desc: "Flag persons of interest with a required reason, preview images, and admin-only controls.",
            },
            {
              title: "Secure by Default",
              desc: "Private Cloudinary assets with signed URLs, org-scoped data, and admin-only dashboards.",
            },
            {
              title: "Search & Filter",
              desc: "Fast org-scoped search with role and wanted filters. Clean UX with pagination.",
            },
            {
              title: "Reliable Storage",
              desc: "Neon Postgres + Prisma schema designed for durability and clarity.",
            },
            {
              title: "Modern Stack",
              desc: "Next.js App Router, Tailwind UI, Better Auth, and FastAPI – tested and hardened.",
            },
          ].map((f) => (
            <Card key={f.title} className="border-white/15 bg-[#121e3a] p-5">
              <div className="text-lg font-medium text-white">{f.title}</div>
              <p className="mt-2 text-sm text-white/80">{f.desc}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="mx-auto max-w-6xl px-4 py-12 md:py-16">
        <h2 className="text-xl md:text-2xl font-semibold text-white">How it works</h2>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Step 1: Sign in */}
          <Link href="/signin" className="group">
            <Card className="border-white/15 bg-[#121e3a] p-5 transition-transform duration-200 group-hover:scale-[1.02] cursor-pointer">
              <div className="text-sm text-emerald-300/90">Step 1</div>
              <div className="mt-1 text-base font-medium text-white">Sign in</div>
              <p className="mt-2 text-sm text-white/80">Authenticate securely and access your organization.</p>
            </Card>
          </Link>
          {/* Step 2: Register Organization */}
          <a href="#register-org" className="group">
            <Card className="border-white/15 bg-[#121e3a] p-5 transition-transform duration-200 group-hover:scale-[1.02] cursor-pointer">
              <div className="text-sm text-emerald-300/90">Step 2</div>
              <div className="mt-1 text-base font-medium text-white">Register Organization</div>
              <p className="mt-2 text-sm text-white/80">Submit your organization details to get started.</p>
            </Card>
          </a>
          {/* Step 3: Enroll */}
          <Link href="/enroll" className="group">
            <Card className="border-white/15 bg-[#121e3a] p-5 transition-transform duration-200 group-hover:scale-[1.02] cursor-pointer">
              <div className="text-sm text-emerald-300/90">Step 3</div>
              <div className="mt-1 text-base font-medium text-white">Enroll</div>
              <p className="mt-2 text-sm text-white/80">Capture frames or upload images with details.</p>
            </Card>
          </Link>
          {/* Step 4: Dashboard */}
          <Link href="/dashboard" className="group">
            <Card className="border-white/15 bg-[#121e3a] p-5 transition-transform duration-200 group-hover:scale-[1.02] cursor-pointer">
              <div className="text-sm text-emerald-300/90">Step 4</div>
              <div className="mt-1 text-base font-medium text-white">Dashboard</div>
              <p className="mt-2 text-sm text-white/80">Review enrollments, filter wanted flags, and manage access.</p>
            </Card>
          </Link>
        </div>
      </section>

      {/* SECURITY */}
  <section id="register-org" className="mx-auto max-w-6xl px-4 py-12 md:py-16">
        <h2 className="text-xl md:text-2xl font-semibold text-white">Security & Reliability</h2>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { title: "Private Assets", desc: "All face images are private; short-lived signed URLs used for display." },
            { title: "Org Isolation", desc: "Server enforces scoping by the admin's organization. No client override." },
            { title: "Error Handling", desc: "JSON guards, 403 redirects, and robust env validation across routes." },
          ].map((f) => (
            <Card key={f.title} className="border-white/15 bg-[#121e3a] p-5">
              <div className="text-lg font-medium text-white">{f.title}</div>
              <p className="mt-2 text-sm text-white/80">{f.desc}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* PRICING */}
      <section className="mx-auto max-w-6xl px-4 py-12 md:py-16">
        <div className="text-center">
          <h2 className="text-2xl md:text-3xl font-semibold text-white">Simple, Transparent Pricing</h2>
          <p className="mt-2 text-white/80">Local monitoring is free. Pay only for remote access.</p>
        </div>

        <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Free Tier */}
          <Card className="border-white/15 bg-[#121e3a] p-6 flex flex-col">
            <div className="text-sm text-emerald-300/90 font-medium">FREE</div>
            <div className="mt-2 text-3xl font-bold text-white">{formatPrice(0, currency, rate)}<span className="text-lg text-white/60">/month</span></div>
            <div className="mt-1 text-sm text-white/70">Local monitoring included with hardware</div>
            <Separator className="my-6 bg-white/10" />
            <ul className="space-y-3 flex-1 text-sm text-white/80">
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Unlimited local network viewing</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Face recognition on Pi nodes</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>30-day access logs (local)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Email support (48-hour response)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-white/40 mt-0.5">✗</span>
                <span className="text-white/50">No remote viewing</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-white/40 mt-0.5">✗</span>
                <span className="text-white/50">No cloud alerts</span>
              </li>
            </ul>
            <Button variant="outline" className="hover:text-white mt-6 w-full border-white/30 text-black hover:bg-white/10">
              Get Started
            </Button>
          </Card>

          {/* Remote Access */}
          <Card className="border-emerald-400/40 bg-linear-to-br from-[#132349] to-[#1a2f58] p-6 flex flex-col relative">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-emerald-400 text-black text-xs font-semibold px-3 py-1 rounded-full">
              MOST POPULAR
            </div>
            <div className="text-sm text-emerald-300/90 font-medium">REMOTE ACCESS</div>
            <div className="mt-2 text-3xl font-bold text-white">{formatPrice(2500, currency, rate)}<span className="text-lg text-white/60">/month</span></div>
            <div className="mt-1 text-sm text-white/70">View cameras from anywhere</div>
            <Separator className="my-6 bg-white/10" />
            <ul className="space-y-3 flex-1 text-sm text-white/80">
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Everything in Free, plus:</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Remote streaming (up to 5 gates)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Cloud-synced wanted database</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>WhatsApp alerts (wanted detected)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>90-day cloud logs</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>2 admin accounts</span>
              </li>
            </ul>
            <Button className="mt-6 w-full bg-emerald-400 text-black hover:bg-emerald-500">
              Start Remote Access
            </Button>
          </Card>

          {/* Professional */}
          <Card className="border-white/15 bg-[#121e3a] p-6 flex flex-col">
            <div className="text-sm text-blue-300/90 font-medium">PROFESSIONAL</div>
            <div className="mt-2 text-3xl font-bold text-white">{formatPrice(5500, currency, rate)}<span className="text-lg text-white/60">/month</span></div>
            <div className="mt-1 text-sm text-white/70">For multi-location organizations</div>
            <Separator className="my-6 bg-white/10" />
            <ul className="space-y-3 flex-1 text-sm text-white/80">
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Everything in Remote Access, plus:</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Up to 20 gates</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>Multi-location management</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>SMS + WhatsApp + Email alerts</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>180-day cloud logs</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <span>5 admin accounts & export reports</span>
              </li>
            </ul>
            <Button variant="outline" className="hover:text-white mt-6 w-full border-white/30 text-black hover:bg-white/10">
              Contact Sales
            </Button>
          </Card>
        </div>

        <div className="mt-8 text-center">
          <p className="text-sm text-white/70">
            Hardware setup from <span className="text-white font-semibold">{formatPrice(45000, currency, rate)}</span> per gate • 
            <span className="ml-1">Annual plans save 2 months</span>
          </p>
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto max-w-6xl px-4 py-12 md:py-16">
        <Card className="border-white/15 bg-linear-to-br from-[#132349] to-[#1a2f58]">
          <div className="p-6 md:p-10 grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            <div>
              <h3 className="text-xl md:text-2xl text-white font-semibold">Ready to streamline access?</h3>
              <p className="mt-2 text-white/75">Start enrolling now, or jump to the admin dashboard to review entries.</p>
            </div>
            <div className="flex flex-col sm:flex-row gap-3 md:justify-end">
              <Link href="/enroll">
                <Button className="w-full sm:w-auto bg-white text-black hover:bg-white/85">Enroll Now</Button>
              </Link>
              <Link href="/dashboard">
                <Button variant="outline" className="hover:text-white w-full sm:w-auto border-white/30 text-black hover:bg-white/10">Go to Dashboard</Button>
              </Link>
              <Link href="/signin">
                <Button variant="outline" className="hover:text-white w-full sm:w-auto border-white/30 text-black hover:bg-white/10">Sign In</Button>
              </Link>
            </div>
          </div>
        </Card>
      </section>

      {/* ORGANIZATION REGISTRATION (mailto prompt) */}
      <section className="mx-auto max-w-6xl px-4 py-12 md:py-16">
        <Card className="border-white/15 bg-[#121e3a]">
          <div className="p-6 md:p-8">
            <h3 className="text-lg md:text-xl font-semibold text-white">Register an Organization</h3>
            <p className="mt-1 text-white/80">Submit your details and we’ll get back to you.</p>
            <form
              className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4"
              onSubmit={(e) => {
                e.preventDefault();
                const form = e.currentTarget as HTMLFormElement;
                const name = (form.elements.namedItem("name") as HTMLInputElement)?.value || "";
                const org = (form.elements.namedItem("org") as HTMLInputElement)?.value || "";
                const email = (form.elements.namedItem("email") as HTMLInputElement)?.value || "";
                const subject = encodeURIComponent("Organization Registration Request");
                const body = encodeURIComponent(`Name: ${name}\nOrganization: ${org}\nEmail: ${email}`);
                const mailto = `mailto:stephen.njiu19@students.dkut.ac.ke?subject=${subject}&body=${body}`;
                if (typeof window !== "undefined") {
                  window.location.href = mailto;
                }
              }}
            >
              <div>
                <label htmlFor="name" className="text-sm text-white/80">Full Name *</label>
                <input id="name" name="name" required className="mt-2 w-full rounded-md bg-[#0f1a33] border border-white/20 px-3 py-2 text-white placeholder:text-white/50" placeholder="Jane Doe" />
              </div>
              <div>
                <label htmlFor="org" className="text-sm text-white/80">Organization Name *</label>
                <input id="org" name="org" required className="mt-2 w-full rounded-md bg-[#0f1a33] border border-white/20 px-3 py-2 text-white placeholder:text-white/50" placeholder="Savannah Gates" />
              </div>
              <div>
                <label htmlFor="email" className="text-sm text-white/80">Email *</label>
                <input id="email" name="email" type="email" required className="mt-2 w-full rounded-md bg-[#0f1a33] border border-white/20 px-3 py-2 text-white placeholder:text-white/50" placeholder="jane@example.com" />
              </div>
              <div className="md:col-span-3">
                <Button type="submit" className="w-full sm:w-auto bg-white text-black hover:bg-white/85">Register Organization</Button>
              </div>
            </form>
          </div>
        </Card>
      </section>
    </div>
  );
}

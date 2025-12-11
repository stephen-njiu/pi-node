import Link from "next/link";

export default function Footer() {
  return (
    <footer className="w-full border-t border-white/10 bg-[#0d1628] text-white">
      <div className="mx-auto max-w-6xl px-4 py-6 grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div>
          <div className="text-sm font-semibold" aria-label="Company">Savannah Gates</div>
          <div className="mt-1 text-xs text-white/70">Secure, fast, and reliable access management.</div>
        </div>
        <div className="text-sm">
          <div className="font-semibold" id="footer-navigation">Navigate</div>
          <div className="mt-2 flex flex-wrap gap-3 text-white/80" aria-labelledby="footer-navigation" role="navigation">
            <Link href="/" className="hover:text-white">Home</Link>
            <Link href="/dashboard" className="hover:text-white">Dashboard</Link>
            <Link href="/enroll" className="hover:text-white">Enroll</Link>
            <Link href="/enroll/wanted" className="hover:text-white">Wanted</Link>
          </div>
        </div>
        <div className="text-sm">
          <div className="font-semibold">Support</div>
          <div className="mt-2 text-white/80">
            <a
              href="mailto:stephen.njiu19@students.dkut.ac.ke"
              className="underline hover:text-white"
              aria-label="Email support at stephen.njiu19@students.dkut.ac.ke"
            >
              Contact Support
            </a>
          </div>
          <div className="mt-2 text-xs text-white/60">© {new Date().getFullYear()} Savannah Gates. All rights reserved.</div>
        </div>
      </div>
    </footer>
  );
}

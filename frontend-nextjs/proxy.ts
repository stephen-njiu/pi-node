import { NextRequest, NextResponse } from "next/server";
import { auth } from "./lib/auth";
export async function proxy(request: NextRequest) {
  const session = await auth.api.getSession({
    headers: request.headers,
  });

  const { pathname } = request.nextUrl;

  const publicPaths = ["/signin","/"];

  const isAuthPage = pathname.startsWith("/signin");
  const isEnrollPage = pathname.startsWith("/enroll");
  const isDashboardPage = pathname.startsWith("/dashboard");

  const isPublic = publicPaths.some((p) => {
    if (p === "/") {
      return pathname === "/"; // exactly home
    }
    return pathname === p || pathname.startsWith(p + "/");
  });

  // If user is signed in but is on auth page, redirect them
  if (session && isAuthPage) {
    return NextResponse.redirect(new URL("/profile", request.url));
  }

  // If not signed in and trying to go to a protected page, send to login
  if (!session && !isPublic) {
    return NextResponse.redirect(new URL("/signin", request.url));
  }

  // Admin gate for enroll page: require role ADMIN
  if (session && (isEnrollPage || isDashboardPage)) {
    try {
      const email = session.user?.email;
      const id = session.user?.id;
      if (!email && !id) {
        return NextResponse.redirect(new URL("/signin", request.url));
      }
      const params = new URLSearchParams(email ? { email } : { id: id! });
      const origin = request.nextUrl.origin;
      const res = await fetch(`${origin}/api/user-role?${params.toString()}`, {
        headers: {
          // Forward cookies/headers if needed; minimal for role lookup
          cookie: request.headers.get("cookie") ?? "",
        },
      });
      if (res.ok) {
        const json = await res.json();
        const role = json?.role as string | undefined;
        if (role !== "ADMIN") {
          return NextResponse.redirect(new URL("/signin", request.url));
        }
      } else {
        // If role lookup fails, be safe and redirect
        return NextResponse.redirect(new URL("/signin", request.url));
      }
    } catch {
      return NextResponse.redirect(new URL("/signin", request.url));
    }
  }

  // Otherwise, let them through
  return NextResponse.next();
}

// Only matcher is allowed in config for proxy.ts
export const config = {
  matcher: [
    // Exclude API, static, image files etc.
    '/((?!api|_next/static|_next/image|.*\\.png$).*)',
  ],
};
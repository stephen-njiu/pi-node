import { NextRequest, NextResponse } from "next/server";
import { auth } from "./lib/auth";
export async function proxy(request: NextRequest) {
  const session = await auth.api.getSession({
    headers: request.headers,
  });

  const { pathname } = request.nextUrl;

  const publicPaths = ["/signin","/"];

  const isAuthPage = pathname.startsWith("/signin");

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
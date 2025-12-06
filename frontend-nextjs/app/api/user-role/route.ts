import { NextResponse } from "next/server";
import { headers } from "next/headers";
import { prisma } from "@/lib/prisma";
import { auth } from "@/lib/auth";

export async function GET() {
  try {
    // Get server-side session from Better Auth
    const hdrs = await headers();
    const sessionRes = await auth.api.getSession({ headers: hdrs });

    // If unauthenticated, return 401
    if (!sessionRes?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id, email } = sessionRes.user as { id?: string; email?: string };

    // Prefer matching by email (unique), fallback to id
    let user: { role: string | null } | null = null;
    if (email) {
      user = await prisma.user.findUnique({
        where: { email },
        select: { role: true },
      });
    } else if (id) {
      user = await prisma.user.findUnique({
        where: { id },
        select: { role: true },
      });
    }

    if (!user) {
      return NextResponse.json({ error: "User not found in Prisma" }, { status: 404 });
    }

    return NextResponse.json({ role: user.role }, { status: 200 });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Unknown error" }, { status: 500 });
  }
}

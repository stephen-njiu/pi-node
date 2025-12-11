// frontend-nextjs/app/api/me/organization/route.ts
import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@/lib/generated/prisma/client";

const prisma = new PrismaClient();

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const email = (searchParams.get("email") || "").trim();
    const id = (searchParams.get("id") || "").trim();
    if (!email && !id) {
      return NextResponse.json({ error: "email or id required" }, { status: 400 });
    }
    const user = email
      ? await prisma.user.findFirst({ where: { email }, select: { organization: true } })
      : await prisma.user.findFirst({ where: { id }, select: { organization: true } });
    return NextResponse.json({ organization: user?.organization ?? null });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Failed to resolve organization" }, { status: 500 });
  } finally {
    try { await prisma.$disconnect(); } catch {}
  }
}

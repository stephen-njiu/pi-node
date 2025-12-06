import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function GET(req: NextRequest) {
  try {
    // Return distinct organizations from admin-managed registry
    const rows = await prisma.organizationRegistration.findMany({
      select: { organization: true },
      orderBy: { organization: "asc" },
    });
    const set = new Set<string>();
    for (const r of rows) {
      const org = (r.organization ?? "").trim();
      if (org) set.add(org);
    }
    const organizations = Array.from(set);
    return NextResponse.json({ organizations });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Failed to load organizations" }, { status: 500 });
  } finally {
    try { await prisma.$disconnect(); } catch {}
  }
}

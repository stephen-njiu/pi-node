import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@/lib/generated/prisma/client";

const prisma = new PrismaClient();

export async function GET(req: NextRequest) {
  try {
    const rows = await prisma.organizationRegistration.findMany({
      select: { organization: true },
      orderBy: { organization: "asc" },
    });

    const set = new Set<string>();
    for (const r of rows) {
      const org = (r.organization ?? "").trim();
      if (org) set.add(org);
    }

    return NextResponse.json({ organizations: Array.from(set) });
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Failed to load organizations" },
      { status: 500 }
    );
  }
}

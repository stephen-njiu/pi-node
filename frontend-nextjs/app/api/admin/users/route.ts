import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@/lib/generated/prisma/client";

const prisma = new PrismaClient();

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);

    const q = (searchParams.get("q") || "").trim();
  const role = (searchParams.get("role") || "").trim();
  const wantedParam = (searchParams.get("wanted") || "").trim(); // "true" | "false" | ""
    // Client-provided organization will be ignored; enforce org from requester
    const _clientOrganization = (searchParams.get("organization") || "").trim();
    // Identify requester (admin) via email or id
    const requesterEmail = (searchParams.get("email") || "").trim();
    const requesterId = (searchParams.get("id") || "").trim();
    const page = Math.max(1, Number(searchParams.get("page") || 1));
    const pageSize = Math.min(50, Math.max(1, Number(searchParams.get("pageSize") || 10)));

    // Resolve requester and enforce ADMIN role and organization from Users table
    let adminUser: { email: string; role: string; organization: string | null } | null = null;
    if (requesterEmail) {
      const found = await prisma.user.findFirst({ where: { email: requesterEmail }, select: { email: true, role: true, organization: true } });
      if (!found || found.role !== "ADMIN") {
        return NextResponse.json({ error: "Forbidden: admin role required" }, { status: 403 });
      }
      adminUser = { email: found.email, role: found.role, organization: found.organization ?? null };
    } else if (requesterId) {
      const found = await prisma.user.findFirst({ where: { id: requesterId }, select: { email: true, role: true, organization: true } });
      if (!found || found.role !== "ADMIN") {
        return NextResponse.json({ error: "Forbidden: admin role required" }, { status: 403 });
      }
      adminUser = { email: found.email, role: found.role, organization: found.organization ?? null };
    } else {
      return NextResponse.json({ error: "Forbidden: missing requester identity" }, { status: 403 });
    }

    const enforcedOrganization = adminUser.organization;
    if (!enforcedOrganization) {
      return NextResponse.json({ error: "Forbidden: admin organization missing" }, { status: 403 });
    }

    const whereClause: any = {};
    const AND: any[] = [];

    // TEXT SEARCH
    if (q) {
      whereClause.OR = [
        { name: { contains: q, mode: "insensitive" } },
        { email: { contains: q, mode: "insensitive" } },
        { enrollments: { some: { organization: { contains: q, mode: "insensitive" } } } },
      ];
    }

    // ROLE FILTER
    if (role) {
      AND.push({ role });
    }

    // Enforce server-side organization scoping by Users.organization
  AND.push({ organization: enforcedOrganization });

    // Wanted filter
    if (wantedParam === "true") {
      AND.push({ enrollments: { some: { isWanted: true } } });
    } else if (wantedParam === "false") {
      AND.push({ enrollments: { some: { isWanted: false } } });
    }

    if (AND.length > 0) whereClause.AND = AND;

    const total = await prisma.user.count({
      where: Object.keys(whereClause).length ? whereClause : undefined,
    });

    const users = await prisma.user.findMany({
      where: Object.keys(whereClause).length ? whereClause : undefined,
      orderBy: { createdAt: "desc" },
      skip: (page - 1) * pageSize,
      take: pageSize,
      select: {
        id: true,
        name: true,
        email: true,
        organization: true,
        role: true,
        createdAt: true,
        updatedAt: true,
        enrollments: {
          orderBy: { createdAt: "desc" },
          take: 2,
          select: {
            id: true,
            createdAt: true,
            status: true,
            organization: true,
            isWanted: true,
            notes: true,
            images: {
              orderBy: { capturedAt: "desc" },
              take: 6,
              select: {
                id: true,
                cloudinaryPublicId: true,
                capturedAt: true,
              },
            },
          },
        },
      },
    });

    return NextResponse.json({ ok: true, users, total, page, pageSize });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Failed to fetch users" }, { status: 500 });
  }
}

import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const q = (searchParams.get("q") || "").trim();
  const role = (searchParams.get("role") || "").trim(); // VIEWER | GATEKEEPER | ADMIN or empty
  const organization = (searchParams.get("organization") || "").trim();
    const page = Math.max(1, Number(searchParams.get("page") || 1));
    const pageSize = Math.min(50, Math.max(1, Number(searchParams.get("pageSize") || 10)));

    // Basic search on name, email, organization
    const whereClause: any = {};
    if (q) {
      whereClause.OR = [
        { name: { contains: q, mode: "insensitive" } },
        { email: { contains: q, mode: "insensitive" } },
        { organization: { contains: q, mode: "insensitive" } },
      ];
    }
    if (role) whereClause.role = role;
    // Filter users by enrollment organization when provided
    if (organization) {
      whereClause.enrollments = { some: { organization } };
    }

    const total = await prisma.user.count({ where: Object.keys(whereClause).length ? whereClause : undefined });
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
  } finally {
    try {
      await prisma.$disconnect();
    } catch {}
  }
}

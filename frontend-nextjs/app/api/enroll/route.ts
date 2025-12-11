import { NextResponse } from "next/server";
import { PrismaClient } from "@/lib/generated/prisma/client";
// Create a PrismaClient instance (avoid global singleton since user removed it)
const prisma = new PrismaClient();

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { name, email, organization, role, notes, images, isWanted } = body as {
      name: string;
      email: string;
      organization: string;
      role?: "VIEWER" | "GATEKEEPER" | "ADMIN";
      notes?: string;
      images: string[]; // Cloudinary public_id list
      isWanted?: boolean;
    };

    if (!email || !name || !organization) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }
    if (!Array.isArray(images) || images.length < 1) {
      return NextResponse.json({ error: "No images provided" }, { status: 400 });
    }

    // Upsert user by email and ensure organization is stored on the user
    const user = await prisma.user.upsert({
      where: { email },
      update: {
        name,
        role: role ?? undefined,
        organization,
        updatedAt: new Date(),
      },
      create: {
        id: crypto.randomUUID(),
        name,
        email,
        organization,
        role: role ?? "VIEWER",
      },
    });

    // Create enrollment
    const enrollment = await prisma.enrollment.create({
      data: {
        userId: user.id,
        organization,
        isWanted: Boolean(isWanted),
        notes,
        photosCount: images.length,
        status: "PROCESSED",
      },
    });

    // Insert face images with Cloudinary public_id
    await prisma.faceImage.createMany({
      data: images.map((publicId: string) => ({
        enrollmentId: enrollment.id,
        cloudinaryPublicId: publicId,
        organization,
      })),
      skipDuplicates: true,
    });

    return NextResponse.json({
      ok: true,
      enrollmentId: enrollment.id,
      userId: user.id,
      imagesInserted: images.length,
    });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Server error" }, { status: 500 });
  } finally {
    // Best-effort disconnect to avoid open handles in serverless
    try { await prisma.$disconnect(); } catch {}
  }
}

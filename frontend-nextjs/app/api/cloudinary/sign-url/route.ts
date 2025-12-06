export const runtime = "nodejs";
export const dynamic = "force";

import { NextResponse } from "next/server";
// @ts-ignore: cloudinary has no type declarations in this project
import { v2 as cloudinary } from "cloudinary";

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true,
});

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const publicId = url.searchParams.get("publicId");
    const expires = Number(url.searchParams.get("expires")) || 60; // seconds
    if (!publicId) {
      return NextResponse.json({ error: "Missing publicId" }, { status: 400 });
    }

    const signedUrl = cloudinary.url(publicId, {
      type: "private",
      sign_url: true,
      expires_at: Math.floor(Date.now() / 1000) + expires,
      secure: true,
    });

    return NextResponse.json({ ok: true, url: signedUrl, expires });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Failed to sign url" }, { status: 500 });
  }
}

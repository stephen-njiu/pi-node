import { NextResponse } from "next/server";
// Import cloudinary; ensure 'cloudinary' is installed in your project
// npm install cloudinary
// @ts-ignore: cloudinary has no type declarations in this project
import { v2 as cloudinary } from "cloudinary";

// Robust Cloudinary config with validation. Avoid using an invalid CLOUDINARY_URL.
(() => {
  const { CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, CLOUDINARY_URL } = process.env as Record<string, string | undefined>;

  try {
    if (CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET) {
      cloudinary.config({
        cloud_name: CLOUDINARY_CLOUD_NAME,
        api_key: CLOUDINARY_API_KEY,
        api_secret: CLOUDINARY_API_SECRET,
        secure: true,
      });
      return;
    }

    // Fallback: allow CLOUDINARY_URL only if it's in the expected format
    if (CLOUDINARY_URL && CLOUDINARY_URL.startsWith("cloudinary://")) {
      cloudinary.config(CLOUDINARY_URL);
      cloudinary.config({ secure: true });
      return;
    }

    // If neither explicit creds nor a valid URL are present, surface a clear error on first request
    // Note: We don't throw here to avoid crashing build; we'll check again in POST handler.
  } catch (e) {
    // Swallow config-time errors; POST will return a clear message
  }
})();

export const runtime = "nodejs";
// Ensure this API route is always treated as dynamic
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  try {
    // Validate env before proceeding to upload
    const { CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, CLOUDINARY_URL } = process.env as Record<string, string | undefined>;
    const hasExplicitCreds = Boolean(CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET);
    const hasValidUrl = Boolean(CLOUDINARY_URL && CLOUDINARY_URL.startsWith("cloudinary://"));
    if (!hasExplicitCreds && !hasValidUrl) {
      return NextResponse.json(
        { error: "Cloudinary environment is not configured. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET or provide a valid CLOUDINARY_URL starting with 'cloudinary://'" },
        { status: 500 }
      );
    }

    const contentType = req.headers.get("content-type") || "";
    if (!contentType.includes("multipart/form-data")) {
      return NextResponse.json({ error: "Expected multipart/form-data" }, { status: 400 });
    }

    const formData = await req.formData();
    const files = formData.getAll("files");
    if (!files || files.length === 0) {
      return NextResponse.json({ error: "No files provided (use 'files')" }, { status: 400 });
    }

    const folder = (formData.get("folder") as string) || "faces";

    const uploads: Array<{ public_id: string }> = [];
    for (const f of files) {
      const file = f as unknown as Blob;
      const arrayBuffer = await file.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);

      const uploaded = await new Promise<any>((resolve, reject) => {
        cloudinary.uploader.upload_stream(
          {
            folder,
            type: "private",
          },
          (err: unknown, result: unknown) => {
            if (err) return reject(err);
            resolve(result);
          }
        ).end(buffer);
      });

      uploads.push({ public_id: uploaded.public_id });
    }

    return NextResponse.json({ ok: true, count: uploads.length, items: uploads });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "Upload failed" }, { status: 500 });
  }
}

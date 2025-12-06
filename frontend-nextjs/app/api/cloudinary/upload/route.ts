import { NextResponse } from "next/server";
// Import cloudinary; ensure 'cloudinary' is installed in your project
// npm install cloudinary
// @ts-ignore: cloudinary has no type declarations in this project
import { v2 as cloudinary } from "cloudinary";

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true,
});

export const runtime = "nodejs";
export const dynamic = "force";

export async function POST(req: Request) {
  try {
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

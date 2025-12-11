# Frontend (Next.js)

Savannah Gates’ frontend is a modern, accessible Next.js App Router UI:

- Pages

  - Home: polished landing with strong contrast, motion, real previews (`/pic1.png`…`/pic6.png`), and an anchored organization registration form (mailto)
  - Enroll: camera capture + upload (1–5 images), pipeline to FastAPI embeddings → Cloudinary → Prisma/Neon
  - Wanted Enroll: admin‑only upload with required reason; flags `Enrollment.isWanted=true`
  - Dashboard: admin‑only, org‑scoped results, wanted filter, image previews via signed Cloudinary URLs and a “Wanted” popover showing reason
  - 404: global `app/not-found.tsx` with a helpful UI

- Global layout

  - Responsive Navbar (SG branding, hamburger menu with ARIA)
  - Footer (Savannah Gates links, support email)
  - Toaster (sonner) configured in `app/layout.tsx`

- Auth & access control
  - Better Auth client hooks
  - Server‑side admin validation within APIs (email/id provided by session)

## Requirements

- Node 18+
- Environment variables (examples):
  - `NEXT_PUBLIC_APP_URL` (FastAPI base URL)
  - Cloudinary credentials (`cloud_name`, `api_key`, `api_secret`) or `CLOUDINARY_URL` (validated in server route)

## Develop locally

```powershell
npm run dev
```

Open http://localhost:3000.

## Key routes (API)

- `/api/user-role` – resolve role from session (ADMIN required for dashboard)
- `/api/organizations` – list orgs for selection
- `/api/enroll` – upsert user + enrollment + images (persists `isWanted`)
- `/api/cloudinary/upload` – server‑side upload to Cloudinary (private) with folder support
- `/api/cloudinary/sign-url` – generate short‑lived signed URL for private assets
- `/api/admin/users` – admin search with org scoping + wanted filter; returns enrollments with `notes` and images

## UX notes

- Strong contrast, no white‑on‑white text
- Keyboard and screen reader friendly menus and popovers
- Mobile‑first layouts for all pages

— Made with 💙 from Silicon Savannah 💙

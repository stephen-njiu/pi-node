# Savannah Gates: Facial Recognition Gate System

Industrial, secure, and responsive gate access built for real-world deployments. This mono‑repo hosts a production‑ready stack with a Next.js frontend, FastAPI backend, and Raspberry Pi gate nodes.

## What’s in this repo

- Frontend (`frontend-nextjs`)
  - Next.js App Router UI: enroll (camera + upload), wanted enroll, admin dashboard with org scoping and signed URL previews
  - Global navbar/footer, 404 page, polished homepage with strong contrast, mobile accessibility, and mailto organization registration form
  - Better Auth session hooks; server‑side admin validation via API
- Backend (`backend-fastapi`)
  - Minimal embeddings API: accepts up to 5 images (multipart), performs detection/alignment (SCRFD) and optionally ArcFace ONNX embeddings
  - Health check endpoint; env‑driven configuration; optional local storage
- Pi Gate (planned)
  - SCRFD detection, ArcFace embeddings, DeepSORT tracking, FAISS/HNSW local index, MQTT‑driven updates, GPIO relay
- Shared/Infra (planned)
  - Contracts, scripts, docker/k8s manifests, media relay server

## Live flow (current frontend + backend)

1. Enroll (or Wanted Enroll) in the frontend
   - Sends images to FastAPI `/api/v1/embeddings` (multipart) with `wanted=true/false`
   - Uploads images to Cloudinary via Next.js route (private assets)
   - Persists metadata in Neon Postgres via Prisma (`User`, `Enrollment`, `FaceImage`) with `Enrollment.isWanted` and `User.organization`
2. Admin Dashboard
   - Server‑enforced org scoping; role validation; wanted filter
   - Signed URL previews for private Cloudinary images
   - Clickable “Wanted” popover showing reason (`enrollment.notes`) and photos

## Tech overview

- Frontend: Next.js (App Router), Tailwind, shadcn/ui components, Better Auth, Cloudinary uploads (Node runtime), strong accessibility (ARIA and keyboard focus)
- Backend: FastAPI, optional InsightFace/SCRFD + ONNX ArcFace, deterministic mock embeddings for local dev
- Database: Neon Postgres + Prisma; `Enrollment.isWanted`, `User.organization` indices
- Media (future): WebRTC relay, MQTT push sync to Pis

## Running locally

- Backend (FastAPI): see `backend-fastapi/README.md` for venv, deps, and `uvicorn` instructions
- Frontend (Next.js): run `npm run dev` (or yarn/pnpm/bun) from `frontend-nextjs/`, set required envs (Cloudinary)

## Roadmap

- Pi node integration (DeepSORT, FAISS persistence, MQTT push/ack)
- Admin/gatekeeper real‑time dashboards (WebSocket alerts, two‑column result layout)
- Media relay server; on‑demand WebRTC streams

— Made with 💙 from Silicon Savannah 💙

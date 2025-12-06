# Facial Recognition Gate System

A production-ready, scalable, offline-safe, push-based facial recognition gate system optimized for Raspberry Pi 4B.

## Core Components

- Raspberry Pi (per gate)
  - SCRFD for face detection
  - ArcFace ONNX for embeddings
  - DeepSORT for tracking
  - Local HNSW index (FAISS/HNSW) persisted on disk
  - GPIO control for gate relay
  - H.264 WebRTC streaming (640×480, 15 FPS, on-demand)
- Backend (FastAPI)
  - Source of truth for face DB (Pinecone)
  - Sync to Pis via HiveMQ Cloud (MQTT, QoS=1) with ack tracking
  - Logs, snapshots, events in PostgreSQL
- Frontend (Next.js)
  - Admin dashboard for CRUD + wanted flag + logs
  - Gatekeeper dashboard for live streams + alerts
  - Two-column display: Found (green) / Not Found (red)
  - Real-time alerts via WebSocket
- Media Relay Server
  - Receives single Pi WebRTC stream and redistributes to multiple viewers

## Key Design Decisions

- Push-based sync via MQTT with acknowledgments (no polling)
- Pi never queries Pinecone directly; local FAISS index synced incrementally
- On-demand WebRTC only when admin clicks "view" (no continuous streaming)
- Person tracking with max 3 retries if face not found
- Face hidden detection: gate closes if face missing for 5–8 frames
- Offline-safe: FAISS persists; backend queues pending changes

## Data Flow

1. Admin adds/updates face → Backend → Pinecone → MQTT push → Pis → Pi updates local FAISS → Ack
2. Person approaches → Pi detects + tracks + embeds → Local match → Gate opens OR alert for unknown/wanted
3. Admin clicks "view stream" → Pi starts WebRTC → Relay distributes → Frontend displays with bounding boxes + two-column layout

## Suggested Repository Layout

```
/ (mono-repo root)
  pi-gate/
    src/
    models/
    config/
    README.md
  backend-fastapi/
    app/
    migrations/
    README.md
  frontend-nextjs/
    app/
    components/
    README.md
  shared/
    contracts/  # MQTT topics, protobuf/JSON schemas
    scripts/
  infra/
    docker-compose.yaml
    k8s/
    README.md
```

## MVP Milestones

- Raspberry Pi
  - SCRFD + ArcFace ONNX pipeline
  - DeepSORT tracking and retry logic (max 3)
  - FAISS HNSW index persisted to disk
  - MQTT client (HiveMQ Cloud), topic subscriptions, ack publishing
  - GPIO relay integration
  - WebRTC producer (640×480, 15 FPS), start/stop on demand
- Backend (FastAPI)
  - Face CRUD APIs; Pinecone integration
  - MQTT publisher with QoS=1; ack tracking per device
  - PostgreSQL models for logs/snapshots/events
  - WebSocket alerts
- Frontend (Next.js)
  - Admin and Gatekeeper dashboards
  - Two-column results display
  - Live viewer using relay stream; bounding box overlay
- Media Relay
  - Ingest single stream from Pi
  - Distribute to multiple viewers

## MQTT Topics (Draft)

- `gate/{deviceId}/index/update` → payload: faces added/updated/removed
- `gate/{deviceId}/index/ack` → payload: status + index version
- `gate/{deviceId}/control/stream` → payload: start/stop
- `gate/{deviceId}/events` → payload: match/unknown/wanted + snapshot ref

## Persistence and Offline

- Pi persists FAISS index and last index version
- Backend queues updates when Pi offline; replay on reconnect
- Ack required per update batch; backend retries if missing

## WebRTC Flow (Draft)

- Frontend request → Backend signals Pi via MQTT `control/stream:start`
- Pi creates H.264 WebRTC sender to media relay
- Frontend connects to relay to view stream

## Implementation Notes

- Use QoS=1 for MQTT; store message IDs and per-device ack state
- Face hidden detection triggers gate close when continuous missing frames in [5,8]
- Bounding box overlays and two-column UI for clarity
- Consider hardware acceleration for H.264 on Pi (MMAL/VAAPI) where available

## Next Steps

- Initialize subprojects (`pi-gate`, `backend-fastapi`, `frontend-nextjs`) with minimal scaffolds
- Define shared contracts (JSON schemas or protobuf) for MQTT payloads
- Spike WebRTC producer on Pi and relay integration
- Set up Pinecone + PostgreSQL credentials and FastAPI config
- Implement ack tracking and index versioning in backend

# Gate Node - Raspberry Pi Face Recognition Access Control

Local node application for real-time facial recognition gate access control.

## Overview

This application runs on a Raspberry Pi (or laptop for development) and provides:

- Real-time face detection using SCRFD
- Face recognition using ArcFace (512-dim embeddings)
- IoU-based face tracking across frames
- Local face database with hnswlib vector search
- GPIO relay control for physical gate
- HDMI display output (continuous or alert-only mode)
- Background sync with backend API
- SQLite access logging for audit trail
- Optional LiveKit streaming for admin dashboard

## Directory Structure

```
gate-node/
├── main.py              # Main application entry point
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── core/
│   ├── __init__.py
│   ├── gate_control.py  # Gate controller & decision engine
│   └── track_state.py   # Track state management (cooldowns)
├── vision/
│   ├── __init__.py
│   ├── detector.py      # SCRFD face detector
│   ├── recognizer.py    # ArcFace face recognizer
│   └── tracker.py       # Simple IoU tracker
├── storage/
│   ├── __init__.py
│   ├── face_db.py       # hnswlib face database
│   └── logs.py          # SQLite access logger
├── threads/
│   ├── __init__.py
│   ├── sync.py          # Backend sync thread
│   ├── ui.py            # HDMI display thread
│   └── stream.py        # LiveKit streaming thread
├── models/              # ONNX model files (download separately)
│   ├── scrfd_10g_bnkps.onnx
│   └── w600k_r50.onnx
└── data/                # Runtime data (created automatically)
    ├── faces.index      # hnswlib index
    ├── faces_metadata.json
    ├── logs.db          # SQLite access logs
    └── sync_version.txt
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac/Pi)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

Download the ONNX models and place in `models/` directory:

- **SCRFD** (Face Detection): `scrfd_10g_bnkps.onnx`
  - Source: InsightFace model zoo
- **ArcFace** (Face Recognition): `w600k_r50.onnx`
  - Source: InsightFace model zoo
  - Note: You may already have this in `backend-fastapi/models/buffalo_l/`

```bash
# Create models directory
mkdir -p models

# Copy from backend if available
cp ../backend-fastapi/models/buffalo_l/w600k_r50.onnx models/

# For SCRFD, download from InsightFace or use det_10g.onnx
cp ../backend-fastapi/models/buffalo_l/det_10g.onnx models/scrfd_10g_bnkps.onnx
```

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit configuration
notepad .env  # Windows
nano .env     # Linux/Mac
```

Key settings:

- `BACKEND_URL`: Your Railway FastAPI backend URL
- `ORG_ID`: Your organization ID
- `DISPLAY_MODE`: `continuous` (demo) or `alert_only` (production)
- `GPIO_ENABLED`: `false` for laptop, `true` for Pi with relay

### 4. Run Application

```bash
python main.py
```

## Display Modes

### Continuous Mode (Development/Demo)

- Shows live video feed with all face detections
- Green boxes for AUTHORIZED users
- Orange boxes for UNKNOWN faces
- Red boxes for WANTED individuals
- Good for demonstrations and debugging

### Alert-Only Mode (Production)

- Shows idle dashboard screen
- Only displays alert screen for UNKNOWN/WANTED faces
- Alert shows for 5 seconds then returns to idle
- Saves Pi resources (no continuous video rendering)

**Toggle modes**: Press `M` key while running

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Thread                          │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Camera  │──►│ Detector │──►│ Recognizer│──►│ Tracker  │  │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                    │        │
│                              ┌─────────────────────▼─────┐  │
│                              │    Decision Engine        │  │
│                              │  ┌──────────────────────┐ │  │
│                              │  │   Track State Mgr    │ │  │
│                              │  └──────────────────────┘ │  │
│                              └─────────────┬─────────────┘  │
│                                            │                │
│            ┌───────────────────────────────┼───────────┐    │
│            │                               │           │    │
│            ▼                               ▼           ▼    │
│   ┌────────────────┐            ┌──────────────┐ ┌────────┐ │
│   │ Gate Controller│            │ Access Logger│ │  UI    │ │
│   │    (GPIO)      │            │   (SQLite)   │ │ Thread │ │
│   └────────────────┘            └──────────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────┘
                                          │
         Background Threads               │
┌─────────────────────────────────────────┼───────────────────┐
│  ┌──────────────┐  ┌──────────────┐  ┌──┴───────────┐       │
│  │ Sync Thread  │  │Stream Thread │  │ Face Database│       │
│  │  (Backend)   │  │  (LiveKit)   │  │  (hnswlib)   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Decision Logic

| Face Status | Gate Action | Alert |
| ----------- | ----------- | ----- |
| AUTHORIZED  | OPEN        | No    |
| UNKNOWN     | CLOSE       | Yes   |
| WANTED      | OPEN\*      | Yes   |

\*WANTED individuals trigger gate open for controlled capture scenario.

## Hardware Setup (Raspberry Pi)

### Components

- Raspberry Pi 4 (4GB+ recommended)
- Pi Camera Module or USB webcam
- 5V Relay module
- HDMI display
- Electric gate/door strike

### GPIO Wiring

```
Pi GPIO 17 ──────► Relay IN
Pi 5V ───────────► Relay VCC
Pi GND ──────────► Relay GND

Relay NO/NC ─────► Gate/Door Strike
```

### Pi-Specific Setup

```bash
# Enable camera
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Install system dependencies
sudo apt-get update
sudo apt-get install -y libopenblas-dev liblapack-dev

# Install Pi-specific packages
pip install RPi.GPIO
```

## Keyboard Shortcuts

| Key | Action              |
| --- | ------------------- |
| `Q` | Quit application    |
| `M` | Toggle display mode |

## Troubleshooting

### Camera not found

```bash
# Check available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"
```

### Model loading fails

- Ensure ONNX files are in `models/` directory
- Check file permissions
- Verify ONNX Runtime installation

### Low FPS

- Reduce `CAMERA_WIDTH` and `CAMERA_HEIGHT`
- Use `alert_only` display mode
- Check CPU/memory usage with `htop`

### Backend sync fails

- Verify `BACKEND_URL` is correct
- Check network connectivity
- Ensure backend `/api/v1/faces/sync` endpoint is working

## Development

### Running on Laptop (Demo Mode)

```bash
# Disable GPIO, use webcam
# In .env:
GPIO_ENABLED=false
CAMERA_INDEX=0
DISPLAY_MODE=continuous
```

### Testing Recognition

1. First enroll faces via the web frontend
2. Ensure backend sync completes (check logs)
3. Face the camera - should see detection boxes
4. If enrolled, will show name and AUTHORIZED status

## API Integration

The gate node syncs with these backend endpoints:

- `GET /api/v1/faces/sync?org_id=X&since_version=Y` - Fetch face updates
- `POST /api/v1/access-logs` - Upload access logs

## License

MIT License - See main project LICENSE file.

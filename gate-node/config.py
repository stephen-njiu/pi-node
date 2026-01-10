"""
Gate Node Configuration
-----------------------
All settings loaded from environment variables or .env file.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Gate node configuration."""
    
    # =========================
    # Identity
    # =========================
    GATE_ID: str = field(default_factory=lambda: os.getenv("GATE_ID", "gate-001"))
    ORG_ID: str = field(default_factory=lambda: os.getenv("ORG_ID", "default-org"))
    
    # =========================
    # Backend API
    # =========================
    BACKEND_URL: str = field(default_factory=lambda: os.getenv("BACKEND_URL", "http://localhost:8000"))
    SYNC_INTERVAL_SECONDS: int = field(default_factory=lambda: int(os.getenv("SYNC_INTERVAL_SECONDS", "120")))
    
    # =========================
    # Camera
    # =========================
    CAMERA_INDEX: int = field(default_factory=lambda: int(os.getenv("CAMERA_INDEX", "0")))
    CAMERA_WIDTH: int = field(default_factory=lambda: int(os.getenv("CAMERA_WIDTH", "640")))
    CAMERA_HEIGHT: int = field(default_factory=lambda: int(os.getenv("CAMERA_HEIGHT", "480")))
    CAMERA_FPS: int = field(default_factory=lambda: int(os.getenv("CAMERA_FPS", "15")))
    
    # =========================
    # Models (default to backend-fastapi models)
    # =========================
    SCRFD_MODEL_PATH: str = field(default_factory=lambda: os.getenv("SCRFD_MODEL_PATH", "../backend-fastapi/models/buffalo_l/det_10g.onnx"))
    ARCFACE_MODEL_PATH: str = field(default_factory=lambda: os.getenv("ARCFACE_MODEL_PATH", "../backend-fastapi/models/buffalo_l/w600k_r50.onnx"))
    
    # =========================
    # Recognition
    # =========================
    RECOGNITION_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("RECOGNITION_THRESHOLD", "0.50")))
    WANTED_CONFIDENCE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("WANTED_CONFIDENCE_THRESHOLD", "0.7")))
    MAX_RECOGNITION_ATTEMPTS: int = field(default_factory=lambda: int(os.getenv("MAX_RECOGNITION_ATTEMPTS", "3")))
    TRACK_COOLDOWN_SECONDS: int = field(default_factory=lambda: int(os.getenv("TRACK_COOLDOWN_SECONDS", "30")))
    
    # =========================
    # GPIO (Gate Control)
    # =========================
    GPIO_ENABLED: bool = field(default_factory=lambda: os.getenv("GPIO_ENABLED", "false").lower() == "true")
    GPIO_RELAY_PIN: int = field(default_factory=lambda: int(os.getenv("GPIO_RELAY_PIN", "17")))
    GPIO_ACTIVE_LOW: bool = field(default_factory=lambda: os.getenv("GPIO_ACTIVE_LOW", "true").lower() == "true")
    GATE_OPEN_DURATION: float = field(default_factory=lambda: float(os.getenv("GATE_OPEN_DURATION", "5.0")))
    GATE_COOLDOWN: float = field(default_factory=lambda: float(os.getenv("GATE_COOLDOWN", "2.0")))
    
    # =========================
    # Display
    # =========================
    DISPLAY_ENABLED: bool = field(default_factory=lambda: os.getenv("DISPLAY_ENABLED", "true").lower() == "true")
    DISPLAY_WIDTH: int = field(default_factory=lambda: int(os.getenv("DISPLAY_WIDTH", "1280")))
    DISPLAY_HEIGHT: int = field(default_factory=lambda: int(os.getenv("DISPLAY_HEIGHT", "720")))
    DISPLAY_FULLSCREEN: bool = field(default_factory=lambda: os.getenv("DISPLAY_FULLSCREEN", "false").lower() == "true")
    # "continuous" = live video (demo), "alert_only" = only UNKNOWN/WANTED (production), "streaming" = raw video
    DISPLAY_MODE: str = field(default_factory=lambda: os.getenv("DISPLAY_MODE", "continuous"))
    ALERT_DISPLAY_DURATION: float = field(default_factory=lambda: float(os.getenv("ALERT_DISPLAY_DURATION", "60.0")))
    
    # =========================
    # Alarm System
    # =========================
    ALARM_ENABLED: bool = field(default_factory=lambda: os.getenv("ALARM_ENABLED", "true").lower() == "true")
    ALARM_WANTED_FREQUENCY: int = field(default_factory=lambda: int(os.getenv("ALARM_WANTED_FREQUENCY", "2500")))
    ALARM_WANTED_DURATION: int = field(default_factory=lambda: int(os.getenv("ALARM_WANTED_DURATION", "500")))
    ALARM_WANTED_BEEPS: int = field(default_factory=lambda: int(os.getenv("ALARM_WANTED_BEEPS", "5")))
    ALARM_UNKNOWN_FREQUENCY: int = field(default_factory=lambda: int(os.getenv("ALARM_UNKNOWN_FREQUENCY", "1500")))
    ALARM_UNKNOWN_DURATION: int = field(default_factory=lambda: int(os.getenv("ALARM_UNKNOWN_DURATION", "300")))
    ALARM_UNKNOWN_BEEPS: int = field(default_factory=lambda: int(os.getenv("ALARM_UNKNOWN_BEEPS", "2")))
    
    # =========================
    # MQTT
    # =========================
    MQTT_ENABLED: bool = field(default_factory=lambda: os.getenv("MQTT_ENABLED", "false").lower() == "true")
    MQTT_BROKER: str = field(default_factory=lambda: os.getenv("MQTT_BROKER", "localhost"))
    MQTT_PORT: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    MQTT_TOPIC: str = field(default_factory=lambda: os.getenv("MQTT_TOPIC", "gate-node/commands"))
    
    # =========================
    # LiveKit
    # =========================
    LIVEKIT_URL: Optional[str] = field(default_factory=lambda: os.getenv("LIVEKIT_URL"))
    
    # =========================
    # Storage
    # =========================
    DATA_DIR: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    LOG_DB_PATH: str = field(default_factory=lambda: os.getenv("LOG_DB_PATH", "data/logs.db"))
    INDEX_PATH: str = field(default_factory=lambda: os.getenv("INDEX_PATH", "data/faces.index"))
    METADATA_PATH: str = field(default_factory=lambda: os.getenv("METADATA_PATH", "data/faces_metadata.json"))
    VERSION_PATH: str = field(default_factory=lambda: os.getenv("VERSION_PATH", "data/sync_version.txt"))


# Global config instance
config = Config()

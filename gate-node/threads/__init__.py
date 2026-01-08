"""Threads module for background processing."""

from .sync import SyncThread
from .ui import UIThread, DisplayMode
from .stream import StreamThread, MockStreamThread

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class FaceOverlay:
    """Face overlay data for UI display."""
    bbox: tuple  # (x1, y1, x2, y2)
    name: Optional[str] = None
    status: Optional[str] = None
    confidence: float = 0.0
    track_id: int = 0


@dataclass
class UIFrame:
    """Frame data for UI thread."""
    frame: np.ndarray
    overlays: List[FaceOverlay]
    gate_status: str = "CLOSED"
    timestamp: float = 0.0


def create_ui_thread_from_config(config) -> UIThread:
    """Factory function to create UIThread from config object."""
    return UIThread(
        display_width=config.DISPLAY_WIDTH,
        display_height=config.DISPLAY_HEIGHT,
        mode=config.DISPLAY_MODE,
        alert_duration=config.ALERT_DISPLAY_DURATION
    )


def create_stream_config_from_config(config) -> dict:
    """Extract stream config from main config."""
    return {
        "livekit_url": config.LIVEKIT_URL,
        "gate_id": config.GATE_ID,
        "frame_width": config.CAMERA_WIDTH,
        "frame_height": config.CAMERA_HEIGHT
    }


__all__ = [
    "SyncThread",
    "UIThread",
    "UIFrame",
    "FaceOverlay",
    "DisplayMode",
    "StreamThread",
    "MockStreamThread",
    "create_ui_thread_from_config",
    "create_stream_config_from_config",
]

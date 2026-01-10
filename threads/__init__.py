"""Threads module for background processing."""

from .sync import SyncThread
# Use new optimized UI with three modes and multi-person alerts
from .ui_v2 import UIThread, DisplayMode, FaceOverlay, UIFrame, create_ui_thread_from_config
from .stream import StreamThread, MockStreamThread
from .capture import CaptureThread, create_capture_thread


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

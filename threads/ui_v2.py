"""
UI Thread - Production-grade display system with three modes.

Modes:
1. ALERT_ONLY (default production): Clean canvas with multi-person alerts
2. CONTINUOUS: Live camera feed with bounding boxes (dev/debug)
3. STREAMING: Raw unprocessed video without jitter (monitoring)

Features:
- Multi-person alert display (up to 4 persons in grid)
- 60-second alert timeout for UNKNOWN faces
- Alarm integration (beep for WANTED, soft for UNKNOWN)
- Clean separation between modes
- Thread-safe operation
"""

import threading
import time
import queue
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime
import logging

import numpy as np

# Lazy import cv2 to support headless operation
cv2 = None

logger = logging.getLogger(__name__)


def _has_display() -> bool:
    """Check if a display is available."""
    if os.name == 'posix':
        display = os.environ.get('DISPLAY')
        wayland = os.environ.get('WAYLAND_DISPLAY')
        return bool(display or wayland)
    return True  # Windows/macOS assume display available


def _import_cv2():
    """Lazy import OpenCV with error handling."""
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            logger.warning("OpenCV not available")
            return None
    return cv2


class DisplayMode(Enum):
    """Display mode options."""
    ALERT_ONLY = "alert_only"   # Production: clean canvas with alerts
    CONTINUOUS = "continuous"   # Dev mode: camera feed with boxes
    STREAMING = "streaming"     # Raw video: unprocessed smooth feed


@dataclass
class AlertInfo:
    """Alert information to display."""
    status: str                           # UNKNOWN, WANTED
    person_name: Optional[str] = None
    confidence: float = 0.0
    face_crop: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    track_id: Optional[int] = None        # For deduplication


@dataclass
class FaceOverlay:
    """Face overlay information for continuous mode."""
    bbox: tuple                           # (x1, y1, x2, y2)
    track_id: int
    status: str = "PENDING"
    person_name: Optional[str] = None
    confidence: float = 0.0
    landmarks: Optional[np.ndarray] = None  # 5x2 facial landmarks


@dataclass
class UIFrame:
    """Frame data passed to UI thread."""
    frame: np.ndarray
    faces: List[FaceOverlay]
    gate_state: str
    timestamp: float


@dataclass
class SystemStatus:
    """System status for display."""
    gate_state: str = "CLOSED"
    face_db_count: int = 0
    sync_status: str = "Idle"
    uptime_seconds: float = 0.0


class UIThread(threading.Thread):
    """
    Production-grade UI Thread with three display modes.
    
    Features:
    - ALERT_ONLY: Clean canvas showing multiple alerts in grid
    - CONTINUOUS: Camera feed with bounding boxes (only when faces detected)
    - STREAMING: Raw unprocessed video feed
    - Multi-person alerts (up to 4 in grid layout)
    - 60-second alert timeout
    - Alarm integration
    """
    
    # Color palette (BGR format for OpenCV)
    COLORS = {
        'bg_dark': (25, 25, 25),
        'bg_panel': (40, 40, 40),
        'text_primary': (255, 255, 255),
        'text_secondary': (150, 150, 150),
        'text_muted': (100, 100, 100),
        'authorized': (0, 200, 0),
        'unknown': (0, 165, 255),
        'wanted': (0, 0, 255),
        'border': (60, 60, 60),
        'accent': (255, 180, 0),
        'pending': (255, 255, 0),  # Cyan for pending
    }
    
    # Alert layout constants
    MAX_ALERTS = 4              # Max alerts to display at once
    ALERT_DURATION_WANTED = 60.0  # seconds
    ALERT_DURATION_UNKNOWN = 60.0  # seconds
    
    def __init__(
        self,
        display_width: int = 1280,
        display_height: int = 720,
        mode: str = "alert_only",
        window_name: str = "Gate Access Control",
        fullscreen: bool = False,
        capture_thread = None,    # For streaming mode
        alarm_enabled: bool = True,
    ):
        super().__init__(name="UIThread", daemon=True)
        
        self.display_width = display_width
        self.display_height = display_height
        self.mode = DisplayMode(mode.lower())
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.capture_thread = capture_thread
        self.alarm_enabled = alarm_enabled
        
        # Alert management - multiple concurrent alerts
        self._active_alerts: Dict[int, AlertInfo] = {}  # track_id -> AlertInfo
        self._alert_lock = threading.Lock()
        
        # Frame queue for continuous mode
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        
        # System status
        self._system_status = SystemStatus()
        self._status_lock = threading.Lock()
        self._start_time = time.time()
        
        # Display state
        self._gui_available = False
        self._stop_event = threading.Event()
        self._last_continuous_frame = None
        self._last_streaming_frame = None
        
        # Cached face overlays for smooth continuous mode
        self._cached_overlays: List[FaceOverlay] = []
        self._cached_overlay_time: float = 0.0
        
        # Stats
        self.fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        
        # Alert deduplication
        self._shown_track_ids: Dict[int, float] = {}  # track_id -> timestamp
        self._alert_cooldown = 3.0  # seconds between re-alerting same track
    
    def run(self):
        """Main UI loop."""
        logger.info(f"UI thread starting in {self.mode.value} mode")
        
        if not _has_display():
            logger.info("No display detected. Running headless.")
            self._run_headless()
            return
        
        cv = _import_cv2()
        if cv is None:
            logger.warning("OpenCV not available. Running headless.")
            self._run_headless()
            return
        
        try:
            if self.fullscreen:
                cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
                cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            else:
                cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
                cv.resizeWindow(self.window_name, self.display_width, self.display_height)
            self._gui_available = True
            logger.info(f"Display window created: {self.display_width}x{self.display_height}")
        except cv.error as e:
            logger.warning(f"Failed to create window: {e}. Running headless.")
            self._run_headless()
            return
        
        self._render_loop(cv)
        cv.destroyAllWindows()
        logger.info("UI thread stopped")
    
    def _run_headless(self):
        """Headless mode: process alerts without rendering."""
        while not self._stop_event.is_set():
            self._cleanup_expired_alerts()
            time.sleep(0.5)
        logger.info("Headless UI thread stopped")
    
    def _render_loop(self, cv):
        """Main rendering loop."""
        target_fps = 30
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        
        while not self._stop_event.is_set():
            # Cleanup expired alerts
            self._cleanup_expired_alerts()
            
            # Render based on mode
            if self.mode == DisplayMode.STREAMING:
                canvas = self._render_streaming_mode()
            elif self.mode == DisplayMode.CONTINUOUS:
                canvas = self._render_continuous_mode()
            else:  # ALERT_ONLY
                canvas = self._render_alert_mode()
            
            # Display
            cv.imshow(self.window_name, canvas)
            self._frame_count += 1
            
            # Update FPS
            now = time.time()
            if now - self._fps_start_time >= 1.0:
                self.fps = self._frame_count / (now - self._fps_start_time)
                self._frame_count = 0
                self._fps_start_time = now
            
            # Handle keyboard
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
            elif key == ord('m'):
                self._cycle_mode()
            elif key == ord('f'):
                self.fullscreen = not self.fullscreen
                prop = cv.WINDOW_FULLSCREEN if self.fullscreen else cv.WINDOW_NORMAL
                cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN, prop)
            
            # Frame rate limiting
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()
    
    def _cleanup_expired_alerts(self):
        """Remove expired alerts."""
        now = time.time()
        with self._alert_lock:
            expired = []
            for track_id, alert in self._active_alerts.items():
                duration = (self.ALERT_DURATION_WANTED if alert.status == "WANTED" 
                           else self.ALERT_DURATION_UNKNOWN)
                if now - alert.timestamp > duration:
                    expired.append(track_id)
            for track_id in expired:
                del self._active_alerts[track_id]
    
    def _render_alert_mode(self) -> np.ndarray:
        """Render alert-only mode with multi-person grid."""
        canvas = np.full(
            (self.display_height, self.display_width, 3),
            self.COLORS['bg_dark'],
            dtype=np.uint8
        )
        
        with self._alert_lock:
            alerts = list(self._active_alerts.values())[:self.MAX_ALERTS]
        
        if not alerts:
            return self._render_idle_screen(canvas)
        
        # Sort by priority (WANTED first)
        alerts.sort(key=lambda a: (0 if a.status == "WANTED" else 1, a.timestamp))
        
        # Determine layout
        n_alerts = len(alerts)
        if n_alerts == 1:
            self._render_single_alert(canvas, alerts[0])
        elif n_alerts == 2:
            self._render_two_alerts(canvas, alerts)
        else:
            self._render_grid_alerts(canvas, alerts)
        
        self._draw_status_bar(canvas)
        return canvas
    
    def _render_idle_screen(self, canvas: np.ndarray) -> np.ndarray:
        """Render idle/dashboard screen."""
        title = "GATE ACCESS CONTROL"
        self._draw_centered_text(canvas, title, self.display_height // 3,
                                  font_scale=2.0, thickness=3)
        
        with self._status_lock:
            gate_state = self._system_status.gate_state
            db_count = self._system_status.face_db_count
            sync_status = self._system_status.sync_status
        
        gate_color = self.COLORS['authorized'] if gate_state == "OPEN" else self.COLORS['unknown']
        self._draw_centered_text(canvas, f"Gate: {gate_state}", self.display_height // 2,
                                  font_scale=1.2, thickness=2, color=gate_color)
        
        info_y = int(self.display_height * 0.65)
        self._draw_centered_text(canvas, f"Enrolled Faces: {db_count}", info_y,
                                  font_scale=0.8, color=self.COLORS['text_secondary'])
        self._draw_centered_text(canvas, f"Sync: {sync_status}", info_y + 35,
                                  font_scale=0.7, color=self.COLORS['text_muted'])
        
        mode_text = f"Mode: {self.mode.value.upper()} | [M] Switch | [F] Fullscreen | [Q] Quit"
        self._draw_centered_text(canvas, mode_text, self.display_height - 70,
                                  font_scale=0.5, color=self.COLORS['text_muted'])
        
        self._draw_status_bar(canvas)
        return canvas
    
    def _render_single_alert(self, canvas: np.ndarray, alert: AlertInfo):
        """Render a single large alert."""
        color = self.COLORS['wanted'] if alert.status == "WANTED" else self.COLORS['unknown']
        
        # Border
        cv2.rectangle(canvas, (0, 0), (self.display_width - 1, self.display_height - 1), color, 8)
        
        # Header
        subtitle = "SECURITY ALERT" if alert.status == "WANTED" else "ACCESS DENIED"
        title = f"{alert.status} PERSON DETECTED"
        self._draw_centered_text(canvas, subtitle, 50, font_scale=0.8, color=color)
        self._draw_centered_text(canvas, title, 100, font_scale=1.5, thickness=3)
        
        # Face image
        if alert.face_crop is not None:
            self._draw_face_crop(canvas, alert.face_crop, 
                                 x_center=self.display_width // 2,
                                 y_start=150, 
                                 max_height=self.display_height - 350,
                                 border_color=color)
        
        # Info
        info_y = self.display_height - 140
        if alert.person_name:
            self._draw_centered_text(canvas, f"Name: {alert.person_name}", info_y,
                                      font_scale=1.0, thickness=2)
            info_y += 40
        
        self._draw_centered_text(canvas, f"Confidence: {alert.confidence:.1%}", info_y,
                                  font_scale=0.8, color=self.COLORS['text_secondary'])
        
        # Progress bar
        self._draw_alert_progress(canvas, alert)
    
    def _render_two_alerts(self, canvas: np.ndarray, alerts: List[AlertInfo]):
        """Render two alerts side by side."""
        half_w = self.display_width // 2
        
        # Draw divider
        cv2.line(canvas, (half_w, 50), (half_w, self.display_height - 50),
                 self.COLORS['border'], 2)
        
        for i, alert in enumerate(alerts):
            x_offset = i * half_w
            self._render_alert_cell(canvas, alert, x_offset, 0, half_w, self.display_height - 50)
    
    def _render_grid_alerts(self, canvas: np.ndarray, alerts: List[AlertInfo]):
        """Render alerts in 2x2 grid."""
        half_w = self.display_width // 2
        half_h = (self.display_height - 50) // 2
        
        positions = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
        
        # Draw grid lines
        cv2.line(canvas, (half_w, 0), (half_w, self.display_height - 50), self.COLORS['border'], 2)
        cv2.line(canvas, (0, half_h), (self.display_width, half_h), self.COLORS['border'], 2)
        
        for i, alert in enumerate(alerts[:4]):
            x, y = positions[i]
            self._render_alert_cell(canvas, alert, x, y, half_w, half_h)
    
    def _render_alert_cell(self, canvas: np.ndarray, alert: AlertInfo,
                           x: int, y: int, width: int, height: int):
        """Render alert in a cell area."""
        color = self.COLORS['wanted'] if alert.status == "WANTED" else self.COLORS['unknown']
        
        # Cell border
        cv2.rectangle(canvas, (x + 5, y + 5), (x + width - 5, y + height - 5), color, 3)
        
        # Status label
        label = alert.status
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        label_x = x + (width - label_size[0]) // 2
        cv2.putText(canvas, label, (label_x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Face crop
        if alert.face_crop is not None:
            face_y = y + 50
            face_max_h = height - 120
            self._draw_face_crop(canvas, alert.face_crop,
                                 x_center=x + width // 2,
                                 y_start=face_y,
                                 max_height=face_max_h,
                                 max_width=width - 30,
                                 border_color=color)
        
        # Name (if WANTED)
        if alert.person_name:
            name_y = y + height - 50
            name = alert.person_name[:15] + "..." if len(alert.person_name) > 15 else alert.person_name
            name_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            name_x = x + (width - name_size[0]) // 2
            cv2.putText(canvas, name, (name_x, name_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.COLORS['text_primary'], 2)
        
        # Confidence
        conf_text = f"{alert.confidence:.0%}"
        conf_y = y + height - 20
        conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        conf_x = x + (width - conf_size[0]) // 2
        cv2.putText(canvas, conf_text, (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    self.COLORS['text_secondary'], 1)
    
    def _draw_face_crop(self, canvas: np.ndarray, face_crop: np.ndarray,
                        x_center: int, y_start: int, max_height: int,
                        max_width: int = None, border_color: tuple = None):
        """Draw face crop centered at position."""
        if max_width is None:
            max_width = self.display_width - 100
        if border_color is None:
            border_color = self.COLORS['accent']
        
        crop = face_crop
        crop_h, crop_w = crop.shape[:2]
        
        # Scale to fit
        scale = min(max_width / crop_w, max_height / crop_h, 1.0)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        if new_w > 0 and new_h > 0:
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            x = x_center - new_w // 2
            y = y_start + (max_height - new_h) // 2
            
            # Clamp to canvas bounds
            x = max(0, min(x, canvas.shape[1] - new_w))
            y = max(0, min(y, canvas.shape[0] - new_h))
            
            # Border
            cv2.rectangle(canvas, (x - 4, y - 4), (x + new_w + 4, y + new_h + 4), border_color, 3)
            
            # Image
            try:
                canvas[y:y+new_h, x:x+new_w] = crop_resized
            except ValueError:
                pass
    
    def _draw_alert_progress(self, canvas: np.ndarray, alert: AlertInfo):
        """Draw progress bar showing time remaining."""
        duration = (self.ALERT_DURATION_WANTED if alert.status == "WANTED"
                    else self.ALERT_DURATION_UNKNOWN)
        elapsed = time.time() - alert.timestamp
        remaining = max(0, duration - elapsed)
        progress = remaining / duration
        
        color = self.COLORS['wanted'] if alert.status == "WANTED" else self.COLORS['unknown']
        
        bar_width = 300
        bar_height = 6
        bar_x = (self.display_width - bar_width) // 2
        bar_y = self.display_height - 55
        
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      self.COLORS['bg_panel'], -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height),
                      color, -1)
    
    def _render_continuous_mode(self) -> np.ndarray:
        """
        Render continuous mode - smooth video with face overlays.
        
        Design:
        1. Get smooth video from capture thread (15-30 FPS)
        2. Overlay face data from AI queue (whenever available)
        3. Never block on AI - always show video
        
        Face display logic:
        - PENDING: Show 5 landmark points (yellow dots)
        - AUTHORIZED/UNKNOWN/WANTED: Show bounding box with label
        """
        # Try to get latest face data (non-blocking) - drain queue to get newest
        latest_data = None
        try:
            while True:
                latest_data = self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        
        if latest_data is not None:
            # Cache the face overlays for drawing on future frames
            self._cached_overlays = latest_data.faces
            self._cached_overlay_time = time.time()
            # Also cache the AI frame for fallback
            self._cached_ai_frame = latest_data.frame
        
        # Get smooth frame from capture thread (primary video source)
        frame = None
        if self.capture_thread is not None:
            frame = self.capture_thread.get_stream_frame(timeout=0.01)
        
        if frame is None:
            # Fallback to AI frame if capture thread not available
            if hasattr(self, '_cached_ai_frame') and self._cached_ai_frame is not None:
                frame = self._cached_ai_frame.copy()
            elif self._last_continuous_frame is not None:
                return self._last_continuous_frame
            else:
                return self._render_idle_screen(np.full(
                    (self.display_height, self.display_width, 3),
                    self.COLORS['bg_dark'], dtype=np.uint8
                ))
        
        # Resize to display size
        if frame.shape[1] != self.display_width or frame.shape[0] != self.display_height:
            scale_x = self.display_width / frame.shape[1]
            scale_y = self.display_height / frame.shape[0]
            frame = cv2.resize(frame, (self.display_width, self.display_height))
        else:
            frame = frame.copy()
            scale_x = scale_y = 1.0
        
        # Draw face overlays (from cached AI results)
        overlays = getattr(self, '_cached_overlays', None)
        overlay_age = time.time() - getattr(self, '_cached_overlay_time', 0)
        
        # Show overlays if they're recent (< 1.0 seconds old)
        if overlays and overlay_age < 1.0:
            for face in overlays:
                self._draw_face_box(frame, face, scale_x, scale_y)
            
            # Face count
            cv2.putText(frame, f"Faces: {len(overlays)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['authorized'], 2)
        
        self._draw_status_bar(frame)
        self._last_continuous_frame = frame
        return frame
    
    def _render_streaming_mode(self) -> np.ndarray:
        """Render streaming mode - raw unprocessed video."""
        if self.capture_thread is None:
            # No capture thread, show placeholder
            canvas = np.full(
                (self.display_height, self.display_width, 3),
                self.COLORS['bg_dark'], dtype=np.uint8
            )
            self._draw_centered_text(canvas, "STREAMING MODE", self.display_height // 3,
                                      font_scale=1.5, thickness=2)
            self._draw_centered_text(canvas, "No capture thread available", 
                                      self.display_height // 2,
                                      font_scale=0.8, color=self.COLORS['text_secondary'])
            self._draw_status_bar(canvas)
            return canvas
        
        # Get raw frame from capture thread (stream queue - not AI queue)
        frame = self.capture_thread.get_stream_frame(timeout=0.01)  # Short timeout for responsiveness
        
        if frame is None:
            # No frame available, return last cached frame if we have one
            if self._last_streaming_frame is not None:
                return self._last_streaming_frame
            # First time - show waiting message
            canvas = np.full(
                (self.display_height, self.display_width, 3),
                self.COLORS['bg_dark'], dtype=np.uint8
            )
            self._draw_centered_text(canvas, "STREAMING MODE", self.display_height // 3,
                                      font_scale=1.5, thickness=2)
            self._draw_centered_text(canvas, "Initializing camera...", 
                                      self.display_height // 2,
                                      font_scale=0.8, color=self.COLORS['text_secondary'])
            self._draw_status_bar(canvas)
            return canvas
        
        # Resize to display size (no processing, just resize)
        if frame.shape[1] != self.display_width or frame.shape[0] != self.display_height:
            frame = cv2.resize(frame, (self.display_width, self.display_height))
        else:
            frame = frame.copy()
        
        # Minimal overlay - just mode indicator and time
        cv2.putText(frame, "STREAMING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (self.display_width - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Cache this frame
        self._last_streaming_frame = frame
        return frame
    
    def _draw_face_box(self, frame: np.ndarray, track: FaceOverlay, 
                       scale_x: float, scale_y: float):
        """
        Draw face overlay based on status:
        - PENDING: Just show 5 landmark points (lightweight visual feedback)
        - AUTHORIZED/UNKNOWN/WANTED: Show bounding box with label
        """
        bbox = track.bbox
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)
        
        # Color based on status
        color = {
            "AUTHORIZED": self.COLORS['authorized'],
            "WANTED": self.COLORS['wanted'],
            "UNKNOWN": self.COLORS['unknown'],
            "PENDING": self.COLORS['pending'],
        }.get(track.status, self.COLORS['pending'])
        
        # For PENDING status: only draw landmarks (lightweight)
        if track.status == "PENDING":
            if track.landmarks is not None and len(track.landmarks) == 5:
                for (lx, ly) in track.landmarks:
                    px = int(lx * scale_x)
                    py = int(ly * scale_y)
                    cv2.circle(frame, (px, py), 4, self.COLORS['pending'], -1)
                    cv2.circle(frame, (px, py), 5, (0, 0, 0), 1)  # Black outline
            return  # Don't draw box for pending
        
        # For recognized faces: draw full bounding box with label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label
        label = f'{track.status}-{track.person_name}' if track.person_name else track.status
        if track.confidence > 0:
            label += f" ({track.confidence:.0%})"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(y1 - 8, label_size[1] + 8)
        
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 4),
                      (x1 + label_size[0] + 4, label_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _draw_status_bar(self, canvas: np.ndarray):
        """Draw status bar at bottom."""
        bar_height = 35
        bar_y = canvas.shape[0] - bar_height
        
        cv2.rectangle(canvas, (0, bar_y), (canvas.shape[1], canvas.shape[0]),
                      self.COLORS['bg_panel'], -1)
        
        with self._status_lock:
            gate_state = self._system_status.gate_state
        
        gate_color = self.COLORS['authorized'] if gate_state == "OPEN" else self.COLORS['unknown']
        cv2.putText(canvas, f"Gate: {gate_state}", (15, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gate_color, 2)
        
        cv2.putText(canvas, f"FPS: {self.fps:.0f}", (180, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS['text_muted'], 1)
        
        cv2.putText(canvas, f"Mode: {self.mode.value}", (300, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS['text_muted'], 1)
        
        # Alert count
        with self._alert_lock:
            alert_count = len(self._active_alerts)
        if alert_count > 0:
            cv2.putText(canvas, f"Alerts: {alert_count}", (450, bar_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS['wanted'], 1)
        
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, time_str, (canvas.shape[1] - 100, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text_secondary'], 1)
    
    def _draw_centered_text(self, canvas: np.ndarray, text: str, y: int,
                            font_scale: float = 1.0, thickness: int = 2,
                            color: tuple = None):
        """Draw text centered horizontally."""
        if color is None:
            color = self.COLORS['text_primary']
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (canvas.shape[1] - text_size[0]) // 2
        cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness)
    
    def _cycle_mode(self):
        """Cycle through display modes."""
        modes = [DisplayMode.ALERT_ONLY, DisplayMode.CONTINUOUS, DisplayMode.STREAMING]
        current_idx = modes.index(self.mode)
        self.mode = modes[(current_idx + 1) % len(modes)]
        logger.info(f"Display mode changed to: {self.mode.value}")
    
    # ========================
    # Public API
    # ========================
    
    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()
    
    def show_alert(
        self,
        status: str,
        person_name: Optional[str] = None,
        confidence: float = 0.0,
        face_crop: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
    ):
        """
        Show an alert for UNKNOWN or WANTED face.
        
        Args:
            status: "UNKNOWN" or "WANTED"
            person_name: Name of person (for WANTED)
            confidence: Recognition confidence (0.0 - 1.0)
            face_crop: Cropped face image (BGR numpy array)
            track_id: Track ID for deduplication
        """
        if status not in ("UNKNOWN", "WANTED"):
            return
        
        # Generate track_id if not provided
        if track_id is None:
            track_id = hash(f"{status}_{person_name}_{time.time()}")
        
        # Check if we've already shown this track
        now = time.time()
        if track_id in self._shown_track_ids:
            if now - self._shown_track_ids[track_id] < self._alert_cooldown:
                return  # Skip duplicate
        
        self._shown_track_ids[track_id] = now
        
        # Create alert
        alert = AlertInfo(
            status=status,
            person_name=person_name,
            confidence=confidence,
            face_crop=face_crop.copy() if face_crop is not None else None,
            timestamp=now,
            track_id=track_id,
        )
        
        with self._alert_lock:
            self._active_alerts[track_id] = alert
        
        logger.info(f"Alert shown: {status} - {person_name or 'Unknown'} (track {track_id})")
        
        # Trigger alarm
        if self.alarm_enabled:
            self._trigger_alarm(status, person_name)
        
        # Cleanup old track IDs
        self._shown_track_ids = {
            k: v for k, v in self._shown_track_ids.items()
            if now - v < 120
        }
    
    def _trigger_alarm(self, status: str, person_name: Optional[str] = None):
        """Trigger alarm sound."""
        try:
            from core.alarm import trigger_alarm, AlarmType
            if status == "WANTED":
                trigger_alarm(AlarmType.WANTED, person_name)
            elif status == "UNKNOWN":
                trigger_alarm(AlarmType.UNKNOWN)
        except ImportError:
            logger.debug("Alarm module not available")
        except Exception as e:
            logger.error(f"Alarm error: {e}")
    
    def update_system_status(
        self,
        gate_state: Optional[str] = None,
        face_db_count: Optional[int] = None,
        sync_status: Optional[str] = None
    ):
        """Update system status displayed on idle screen."""
        with self._status_lock:
            if gate_state is not None:
                self._system_status.gate_state = gate_state
            if face_db_count is not None:
                self._system_status.face_db_count = face_db_count
            if sync_status is not None:
                self._system_status.sync_status = sync_status
            self._system_status.uptime_seconds = time.time() - self._start_time
    
    def set_gate_status(self, status: str):
        """Update gate status."""
        self.update_system_status(gate_state=status)
    
    def set_mode(self, mode: str):
        """Set display mode."""
        try:
            self.mode = DisplayMode(mode.lower())
            logger.info(f"Display mode set to: {self.mode.value}")
        except ValueError:
            logger.warning(f"Invalid display mode: {mode}")
    
    def put_frame(self, ui_frame: UIFrame):
        """
        Put a UIFrame on the display queue.
        
        Handles alert extraction and frame queuing.
        """
        # Check for alerts
        for face in ui_frame.faces:
            if face.status in ("UNKNOWN", "WANTED"):
                # Get face crop
                face_crop = None
                if ui_frame.frame is not None:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in face.bbox]
                        h, w = ui_frame.frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            face_crop = ui_frame.frame[y1:y2, x1:x2]
                    except (ValueError, TypeError):
                        pass
                
                self.show_alert(
                    status=face.status,
                    person_name=face.person_name,
                    confidence=face.confidence,
                    face_crop=face_crop,
                    track_id=face.track_id,
                )
        
        # Update gate status
        self.set_gate_status(ui_frame.gate_state)
        
        # Queue frame for continuous mode
        if self.mode == DisplayMode.CONTINUOUS:
            try:
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put_nowait(ui_frame)
            except queue.Full:
                pass
    
    def update_status(self, face_count: int = 0, sync_status: str = "Unknown"):
        """Legacy API compatibility."""
        self.update_system_status(face_db_count=face_count, sync_status=sync_status)


def create_ui_thread_from_config(config) -> UIThread:
    """Factory function to create UIThread from config."""
    return UIThread(
        display_width=getattr(config, 'DISPLAY_WIDTH', 1280),
        display_height=getattr(config, 'DISPLAY_HEIGHT', 720),
        mode=getattr(config, 'DISPLAY_MODE', 'alert_only'),
        fullscreen=getattr(config, 'DISPLAY_FULLSCREEN', False),
        alarm_enabled=getattr(config, 'ALARM_ENABLED', True),
    )

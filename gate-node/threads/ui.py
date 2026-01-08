"""
UI Thread - Production-grade alert-only display system.

Architecture:
- Clean canvas-based UI (NO raw camera feed)
- Alert-only screen for UNKNOWN/WANTED faces with face crop
- Idle state when no alerts
- Conditional rendering based on DISPLAY env
- Same code works on: Laptop, Raspberry Pi with HDMI, Headless Pi

Key Design Principles:
- Recognition and gate logic are INDEPENDENT of UI
- Camera video is processed internally, never streamed to UI
- Low CPU usage (no continuous frame rendering)
- Thread-safe alert queue for decoupled operation
"""

import threading
import time
import queue
import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from datetime import datetime
import logging

import numpy as np

# Lazy import cv2 to support headless operation
cv2 = None

logger = logging.getLogger(__name__)


def _has_display() -> bool:
    """Check if a display is available."""
    # Linux/Unix: Check DISPLAY env
    if os.name == 'posix':
        display = os.environ.get('DISPLAY')
        if display:
            return True
        # Also check for Wayland
        wayland = os.environ.get('WAYLAND_DISPLAY')
        if wayland:
            return True
        return False
    # Windows: Always assume display available
    elif os.name == 'nt':
        return True
    # macOS: Always assume display available
    return True


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
    CONTINUOUS = "continuous"   # Dev mode: shows camera feed (legacy support)
    ALERT_ONLY = "alert_only"   # Production: clean canvas with alerts


@dataclass
class AlertInfo:
    """Alert information to display."""
    status: str                           # UNKNOWN, WANTED
    person_name: Optional[str] = None     # Name if WANTED
    confidence: float = 0.0               # Recognition confidence
    face_crop: Optional[np.ndarray] = None  # Cropped face image
    timestamp: float = field(default_factory=time.time)
    alert_id: str = ""                    # Unique ID for deduplication


@dataclass
class SystemStatus:
    """System status for display."""
    gate_state: str = "CLOSED"
    face_db_count: int = 0
    sync_status: str = "Idle"
    uptime_seconds: float = 0.0
    last_detection_time: Optional[float] = None


class UIThread(threading.Thread):
    """
    Production-grade UI Thread for gate access display.
    
    Features:
    - Clean canvas-based rendering (no camera feed)
    - Alert-only display for UNKNOWN/WANTED faces
    - Face crop prominently displayed
    - Automatic headless mode detection
    - Low CPU usage
    - Thread-safe operation
    
    Usage:
        ui = UIThread(mode="alert_only", alert_duration=5.0)
        ui.start()
        
        # From vision thread:
        ui.show_alert("UNKNOWN", face_crop=cropped_image)
        ui.show_alert("WANTED", person_name="John Doe", face_crop=cropped_image)
        
        # Update system status:
        ui.update_system_status(gate_state="OPEN", face_db_count=150)
        
        ui.stop()
    """
    
    # Color palette (BGR format for OpenCV)
    COLORS = {
        'bg_dark': (25, 25, 25),          # Main background
        'bg_panel': (40, 40, 40),          # Panel background
        'text_primary': (255, 255, 255),   # White
        'text_secondary': (150, 150, 150), # Gray
        'text_muted': (100, 100, 100),     # Dark gray
        'authorized': (0, 200, 0),         # Green
        'unknown': (0, 165, 255),          # Orange
        'wanted': (0, 0, 255),             # Red
        'border': (60, 60, 60),            # Border color
        'accent': (255, 180, 0),           # Accent blue
    }
    
    def __init__(
        self,
        display_width: int = 1280,
        display_height: int = 720,
        mode: str = "alert_only",
        alert_duration: float = 5.0,
        window_name: str = "Gate Access Control",
        fullscreen: bool = False
    ):
        super().__init__(name="UIThread", daemon=True)
        
        self.display_width = display_width
        self.display_height = display_height
        self.mode = DisplayMode(mode.lower())
        self.alert_duration = alert_duration
        self.alert_duration_unknown = 60.0  # 60 seconds for UNKNOWN faces
        self.window_name = window_name
        self.fullscreen = fullscreen
        
        # Alert queue (thread-safe)
        self._alert_queue: queue.Queue = queue.Queue(maxsize=20)
        
        # Current state
        self._current_alert: Optional[AlertInfo] = None
        self._alert_start_time: float = 0
        self._system_status = SystemStatus()
        self._start_time = time.time()
        
        # Display state
        self._gui_available = False
        self._stop_event = threading.Event()
        self._canvas: Optional[np.ndarray] = None  # Pre-allocated canvas
        
        # Stats
        self.fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        
        # Lock for thread-safe status updates
        self._status_lock = threading.Lock()
        
        # Track recently shown alerts to prevent spam
        self._recent_alerts: dict = {}  # alert_id -> timestamp
        self._alert_cooldown = 3.0  # seconds
        
        # Legacy support for continuous mode
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self._last_display_frame = None
    
    def run(self):
        """Main UI loop."""
        logger.info(f"UI thread starting in {self.mode.value} mode")
        
        # Check display availability
        if not _has_display():
            logger.info("No display detected (DISPLAY env not set). Running headless.")
            self._run_headless()
            return
        
        # Import OpenCV
        cv = _import_cv2()
        if cv is None:
            logger.warning("OpenCV not available. Running headless.")
            self._run_headless()
            return
        
        # Try to create window
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
        
        # Pre-allocate canvas for efficiency
        self._canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Main render loop
        self._render_loop(cv)
        
        # Cleanup
        cv.destroyAllWindows()
        logger.info("UI thread stopped")
    
    def _run_headless(self):
        """Headless mode: process alerts without rendering."""
        logger.info("Running in headless mode (no display)")
        
        while not self._stop_event.is_set():
            # Process alerts (for logging/metrics)
            try:
                alert = self._alert_queue.get(timeout=0.5)
                logger.info(f"[HEADLESS] Alert: {alert.status} - {alert.person_name or 'Unknown'}")
            except queue.Empty:
                pass
        
        logger.info("Headless UI thread stopped")
    
    def _render_loop(self, cv):
        """Main rendering loop for GUI mode."""
        target_fps = 30
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        
        while not self._stop_event.is_set():
            # Check for new alerts
            self._process_alert_queue()
            
            # Check if current alert expired
            if self._current_alert:
                # Use longer duration for UNKNOWN faces (60s static display)
                duration = self.alert_duration_unknown if self._current_alert.status == "UNKNOWN" else self.alert_duration
                if time.time() - self._alert_start_time > duration:
                    self._current_alert = None
            
            # Render appropriate screen
            if self.mode == DisplayMode.CONTINUOUS:
                # Legacy continuous mode
                canvas = self._render_continuous_mode()
            else:
                # Production alert-only mode
                if self._current_alert:
                    canvas = self._render_alert_screen()
                else:
                    canvas = self._render_idle_screen()
            
            # Display
            cv.imshow(self.window_name, canvas)
            self._frame_count += 1
            
            # Update FPS
            now = time.time()
            if now - self._fps_start_time >= 1.0:
                self.fps = self._frame_count / (now - self._fps_start_time)
                self._frame_count = 0
                self._fps_start_time = now
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
            elif key == ord('m'):
                self._toggle_mode()
            elif key == ord('f'):
                # Toggle fullscreen
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                else:
                    cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
            
            # Frame rate limiting
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()
    
    def _process_alert_queue(self):
        """Process pending alerts from queue."""
        try:
            alert = self._alert_queue.get_nowait()
            
            # Check cooldown to prevent alert spam
            alert_key = f"{alert.status}_{alert.person_name or 'unknown'}"
            now = time.time()
            
            if alert_key in self._recent_alerts:
                if now - self._recent_alerts[alert_key] < self._alert_cooldown:
                    return  # Skip duplicate alert
            
            self._recent_alerts[alert_key] = now
            self._current_alert = alert
            self._alert_start_time = now
            logger.info(f"Displaying alert: {alert.status} - {alert.person_name or 'Unknown'}")
            
            # Cleanup old entries
            self._recent_alerts = {k: v for k, v in self._recent_alerts.items() if now - v < 60}
            
        except queue.Empty:
            pass
    
    def _render_idle_screen(self) -> np.ndarray:
        """Render the idle/dashboard screen (no alerts)."""
        canvas = np.full(
            (self.display_height, self.display_width, 3),
            self.COLORS['bg_dark'],
            dtype=np.uint8
        )
        
        # Main title
        title = "GATE ACCESS CONTROL"
        self._draw_centered_text(canvas, title, self.display_height // 3, 
                                  font_scale=2.0, thickness=3, color=self.COLORS['text_primary'])
        
        # Status indicator
        with self._status_lock:
            gate_state = self._system_status.gate_state
            db_count = self._system_status.face_db_count
            sync_status = self._system_status.sync_status
        
        # Gate status with colored indicator
        gate_color = self.COLORS['authorized'] if gate_state == "OPEN" else self.COLORS['unknown']
        self._draw_centered_text(canvas, f"Gate: {gate_state}", self.display_height // 2,
                                  font_scale=1.2, thickness=2, color=gate_color)
        
        # System info
        info_y = int(self.display_height * 0.65)
        self._draw_centered_text(canvas, f"Enrolled Faces: {db_count}", info_y,
                                  font_scale=0.8, thickness=1, color=self.COLORS['text_secondary'])
        
        self._draw_centered_text(canvas, f"Sync: {sync_status}", info_y + 35,
                                  font_scale=0.7, thickness=1, color=self.COLORS['text_muted'])
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.value.upper()}"
        self._draw_centered_text(canvas, mode_text, int(self.display_height * 0.78),
                                  font_scale=0.6, thickness=1, color=self.COLORS['text_muted'])
        
        # Instructions
        help_text = "[M] Toggle Mode | [F] Fullscreen | [Q] Quit"
        self._draw_centered_text(canvas, help_text, self.display_height - 70,
                                  font_scale=0.5, thickness=1, color=self.COLORS['text_muted'])
        
        # Draw status bar
        self._draw_status_bar(canvas)
        
        return canvas
    
    def _render_alert_screen(self) -> np.ndarray:
        """Render alert screen with face crop."""
        alert = self._current_alert
        if not alert:
            return self._render_idle_screen()
        
        canvas = np.full(
            (self.display_height, self.display_width, 3),
            self.COLORS['bg_dark'],
            dtype=np.uint8
        )
        
        # Determine alert style
        if alert.status == "WANTED":
            alert_color = self.COLORS['wanted']
            title = "WANTED PERSON DETECTED"
            subtitle = "SECURITY ALERT"
        else:  # UNKNOWN
            alert_color = self.COLORS['unknown']
            title = "UNKNOWN PERSON DETECTED"
            subtitle = "ACCESS DENIED"
        
        # Draw alert border/frame
        border_thickness = 8
        cv2.rectangle(canvas, (0, 0), (self.display_width - 1, self.display_height - 1),
                      alert_color, border_thickness)
        
        # Alert header
        self._draw_centered_text(canvas, subtitle, 50, 
                                  font_scale=0.8, thickness=2, color=alert_color)
        self._draw_centered_text(canvas, title, 100,
                                  font_scale=1.5, thickness=3, color=self.COLORS['text_primary'])
        
        # Face crop (centered, prominent)
        face_y_start = 150
        face_area_height = self.display_height - 350
        
        if alert.face_crop is not None:
            crop = alert.face_crop
            crop_h, crop_w = crop.shape[:2]
            
            # Scale face crop to fit display area while maintaining aspect ratio
            max_face_width = min(self.display_width - 100, int(face_area_height * 0.9))
            max_face_height = int(face_area_height * 0.85)
            
            scale = min(max_face_width / crop_w, max_face_height / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            
            if new_w > 0 and new_h > 0:
                crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Center position
                x = (self.display_width - new_w) // 2
                y = face_y_start + (face_area_height - new_h) // 2
                
                # Draw border around face
                border_padding = 6
                cv2.rectangle(canvas,
                              (x - border_padding, y - border_padding),
                              (x + new_w + border_padding, y + new_h + border_padding),
                              alert_color, 4)
                
                # Place face crop
                try:
                    canvas[y:y+new_h, x:x+new_w] = crop_resized
                except ValueError:
                    pass  # Handle edge cases where dimensions don't match
        else:
            # No face crop available - show placeholder
            placeholder_text = "No Face Image"
            self._draw_centered_text(canvas, placeholder_text, 
                                      face_y_start + face_area_height // 2,
                                      font_scale=1.0, thickness=2, color=self.COLORS['text_muted'])
        
        # Info section at bottom
        info_y = self.display_height - 140
        
        # Person name (if WANTED)
        if alert.person_name:
            name_text = f"Name: {alert.person_name}"
            self._draw_centered_text(canvas, name_text, info_y,
                                      font_scale=1.0, thickness=2, color=self.COLORS['text_primary'])
            info_y += 40
        
        # Confidence
        conf_text = f"Match Confidence: {alert.confidence:.1%}"
        self._draw_centered_text(canvas, conf_text, info_y,
                                  font_scale=0.8, thickness=2, color=self.COLORS['text_secondary'])
        
        # Time remaining indicator
        elapsed = time.time() - self._alert_start_time
        duration = self.alert_duration_unknown if alert.status == "UNKNOWN" else self.alert_duration
        remaining = max(0, duration - elapsed)
        progress = remaining / duration
        
        # Progress bar
        bar_width = 300
        bar_height = 6
        bar_x = (self.display_width - bar_width) // 2
        bar_y = self.display_height - 55
        
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      self.COLORS['bg_panel'], -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height),
                      alert_color, -1)
        
        # Draw status bar
        self._draw_status_bar(canvas)
        
        return canvas
    
    def _render_continuous_mode(self) -> np.ndarray:
        """Legacy continuous mode - shows camera feed (for development)."""
        try:
            display_data = self._frame_queue.get_nowait()
            frame = display_data.frame.copy()
            orig_h, orig_w = frame.shape[:2]
            
            # Scale factors
            scale_x = self.display_width / orig_w
            scale_y = self.display_height / orig_h
            
            # Resize frame
            if orig_w != self.display_width or orig_h != self.display_height:
                frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            # Draw tracks
            for track in display_data.tracks:
                self._draw_face_box(frame, track, scale_x, scale_y)
            
            # Face count
            face_count_text = f"Faces: {len(display_data.tracks)}"
            cv2.putText(frame, face_count_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['authorized'], 2)
            
            self._draw_status_bar(frame)
            self._last_display_frame = frame
            return frame
            
        except queue.Empty:
            if self._last_display_frame is not None:
                return self._last_display_frame
            return self._render_idle_screen()
    
    def _draw_face_box(self, frame: np.ndarray, track, scale_x: float, scale_y: float):
        """Draw face bounding box with label (for continuous mode)."""
        # Get bbox
        if hasattr(track.bbox, 'astype'):
            bbox = track.bbox.astype(int)
        else:
            bbox = tuple(int(x) for x in track.bbox)
        
        # Scale bbox
        bbox = (
            int(bbox[0] * scale_x), int(bbox[1] * scale_y),
            int(bbox[2] * scale_x), int(bbox[3] * scale_y)
        )
        
        # Choose color
        status = getattr(track, 'status', None) or "PENDING"
        color = {
            "AUTHORIZED": self.COLORS['authorized'],
            "WANTED": self.COLORS['wanted'],
            "UNKNOWN": self.COLORS['unknown'],
        }.get(status, (255, 255, 0))  # Cyan for pending
        
        # Draw box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Label
        name = getattr(track, 'person_name', None)
        label = name if name else status
        conf = getattr(track, 'confidence', 0)
        if conf > 0:
            label += f" ({conf:.0%})"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(bbox[1] - 8, label_size[1] + 8)
        
        cv2.rectangle(frame, (bbox[0], label_y - label_size[1] - 4),
                      (bbox[0] + label_size[0] + 4, label_y + 4), color, -1)
        cv2.putText(frame, label, (bbox[0] + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _draw_status_bar(self, canvas: np.ndarray):
        """Draw status bar at bottom of screen."""
        bar_height = 35
        bar_y = canvas.shape[0] - bar_height
        
        # Background
        cv2.rectangle(canvas, (0, bar_y), (canvas.shape[1], canvas.shape[0]),
                      self.COLORS['bg_panel'], -1)
        
        # Gate status
        with self._status_lock:
            gate_state = self._system_status.gate_state
        
        gate_color = self.COLORS['authorized'] if gate_state == "OPEN" else self.COLORS['unknown']
        cv2.putText(canvas, f"Gate: {gate_state}", (15, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gate_color, 2)
        
        # FPS
        cv2.putText(canvas, f"FPS: {self.fps:.0f}", (180, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS['text_muted'], 1)
        
        # Mode
        cv2.putText(canvas, f"Mode: {self.mode.value}", (300, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS['text_muted'], 1)
        
        # Time
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
    
    def _toggle_mode(self):
        """Toggle between display modes."""
        if self.mode == DisplayMode.CONTINUOUS:
            self.mode = DisplayMode.ALERT_ONLY
        else:
            self.mode = DisplayMode.CONTINUOUS
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
        face_crop: Optional[np.ndarray] = None
    ):
        """
        Show an alert for UNKNOWN or WANTED face.
        
        This is the primary method for triggering alerts.
        Called from vision/decision threads.
        
        Args:
            status: "UNKNOWN" or "WANTED"
            person_name: Name of person (for WANTED)
            confidence: Recognition confidence (0.0 - 1.0)
            face_crop: Cropped face image (BGR numpy array)
        """
        if status not in ("UNKNOWN", "WANTED"):
            logger.warning(f"Invalid alert status: {status}")
            return
        
        alert = AlertInfo(
            status=status,
            person_name=person_name,
            confidence=confidence,
            face_crop=face_crop.copy() if face_crop is not None else None,
            timestamp=time.time(),
            alert_id=f"{status}_{person_name or 'unknown'}_{time.time():.0f}"
        )
        
        try:
            # Clear queue if full to show latest alert
            while self._alert_queue.full():
                try:
                    self._alert_queue.get_nowait()
                except queue.Empty:
                    break
            self._alert_queue.put_nowait(alert)
        except queue.Full:
            logger.warning("Alert queue full, dropping alert")
    
    def update_system_status(
        self,
        gate_state: Optional[str] = None,
        face_db_count: Optional[int] = None,
        sync_status: Optional[str] = None
    ):
        """
        Update system status displayed on idle screen.
        
        Args:
            gate_state: "OPEN" or "CLOSED"
            face_db_count: Number of faces in local database
            sync_status: Sync thread status string
        """
        with self._status_lock:
            if gate_state is not None:
                self._system_status.gate_state = gate_state
            if face_db_count is not None:
                self._system_status.face_db_count = face_db_count
            if sync_status is not None:
                self._system_status.sync_status = sync_status
            self._system_status.uptime_seconds = time.time() - self._start_time
    
    def set_gate_status(self, status: str):
        """Update gate status display (convenience method)."""
        self.update_system_status(gate_state=status)
    
    def set_mode(self, mode: str):
        """Set display mode."""
        try:
            self.mode = DisplayMode(mode.lower())
            logger.info(f"Display mode set to: {self.mode.value}")
        except ValueError:
            logger.warning(f"Invalid display mode: {mode}")
    
    # ========================
    # Legacy API Compatibility
    # ========================
    
    @dataclass
    class _DisplayFrame:
        """Internal frame data for continuous mode."""
        frame: np.ndarray
        tracks: list
        gate_status: str
        timestamp: float
    
    def put_frame(self, ui_frame):
        """
        Legacy API: Put a UIFrame on the display queue.
        
        In ALERT_ONLY mode, this extracts alerts from the frame.
        In CONTINUOUS mode, this queues the frame for display.
        
        Args:
            ui_frame: UIFrame dataclass with frame, faces, gate_state, timestamp
        """
        # Always check for alerts regardless of mode
        for face in ui_frame.faces:
            status = getattr(face, 'status', None)
            track_id = getattr(face, 'track_id', None)
            
            if status in ("UNKNOWN", "WANTED"):
                # Use track_id for deduplication - only show alert once per track
                # This is critical to prevent spam when same track is in multiple frames
                alert_key = f"track_{track_id}_{status}"
                now = time.time()
                
                if alert_key in self._recent_alerts:
                    # Already showed alert for this track
                    continue
                
                # Get face crop from frame if available
                face_crop = None
                bbox = getattr(face, 'bbox', None)
                if bbox is not None and ui_frame.frame is not None:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        h, w = ui_frame.frame.shape[:2]
                        # Clamp to frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            face_crop = ui_frame.frame[y1:y2, x1:x2]
                    except (ValueError, TypeError):
                        pass
                
                # Mark as alerted before showing (to prevent duplicates)
                self._recent_alerts[alert_key] = now
                
                self.show_alert(
                    status=status,
                    person_name=getattr(face, 'person_name', None),
                    confidence=getattr(face, 'confidence', 0.0),
                    face_crop=face_crop
                )
        
        # Update gate status
        self.set_gate_status(ui_frame.gate_state)
        
        # Only queue frame in continuous mode
        if self.mode == DisplayMode.CONTINUOUS:
            display_data = self._DisplayFrame(
                frame=ui_frame.frame.copy(),
                tracks=ui_frame.faces,
                gate_status=ui_frame.gate_state,
                timestamp=ui_frame.timestamp
            )
            
            try:
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put_nowait(display_data)
            except queue.Full:
                pass
    
    def update_frame(self, frame: np.ndarray, tracks: list, gate_status: str):
        """Legacy API: Update display with new frame."""
        if self.mode != DisplayMode.CONTINUOUS:
            return
        
        display_data = self._DisplayFrame(
            frame=frame.copy(),
            tracks=tracks,
            gate_status=gate_status,
            timestamp=time.time()
        )
        
        try:
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put_nowait(display_data)
        except queue.Full:
            pass
    
    def update_status(self, face_count: int = 0, sync_status: str = "Unknown"):
        """Legacy API: Update status information."""
        self.update_system_status(face_db_count=face_count, sync_status=sync_status)


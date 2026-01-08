"""
UI Thread - Handles HDMI display output.
Supports two modes:
- CONTINUOUS: Live video feed with face boxes (for demo/development)
- ALERT_ONLY: Only show UNKNOWN/WANTED faces (for production)
"""

import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display mode options."""
    CONTINUOUS = "continuous"
    ALERT_ONLY = "alert_only"


@dataclass
class DisplayFrame:
    """Frame data for display."""
    frame: np.ndarray
    tracks: list  # List of Track objects
    gate_status: str
    timestamp: float


@dataclass
class AlertInfo:
    """Alert information to display."""
    status: str  # UNKNOWN or WANTED
    name: Optional[str]
    confidence: float
    face_crop: Optional[np.ndarray]
    timestamp: float


class UIThread(threading.Thread):
    """
    UI Thread for HDMI display.
    
    Two display modes:
    - CONTINUOUS: Shows live video with all detections (good for demos)
    - ALERT_ONLY: Shows idle screen, pops up alerts for UNKNOWN/WANTED
    """
    
    # Colors (BGR)
    COLOR_AUTHORIZED = (0, 255, 0)    # Green
    COLOR_UNKNOWN = (0, 165, 255)     # Orange
    COLOR_WANTED = (0, 0, 255)        # Red
    COLOR_TEXT = (255, 255, 255)      # White
    COLOR_BG = (40, 40, 40)           # Dark gray
    
    def __init__(
        self,
        display_width: int = 1280,
        display_height: int = 720,
        mode: str = "continuous",
        alert_duration: float = 5.0,
        window_name: str = "Gate Access Control"
    ):
        super().__init__(name="UIThread", daemon=True)
        
        self.display_width = display_width
        self.display_height = display_height
        self.mode = DisplayMode(mode.lower())
        self.alert_duration = alert_duration
        self.window_name = window_name
        
        # Frame queue for continuous mode
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        
        # Alert queue for alert_only mode
        self._alert_queue: queue.Queue = queue.Queue(maxsize=10)
        
        # Current alert (for alert_only mode)
        self._current_alert: Optional[AlertInfo] = None
        self._alert_start_time: float = 0
        
        # Control
        self._stop_event = threading.Event()
        self._gate_status = "CLOSED"
        
        # Status info for display
        self._status_info = {"face_count": 0, "sync_status": "Unknown"}
        
        # Stats
        self.fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
    
    def run(self):
        """Main UI loop."""
        logger.info(f"UI thread started in {self.mode.value} mode")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        last_frame_time = time.time()
        
        while not self._stop_event.is_set():
            display_frame = None
            
            if self.mode == DisplayMode.CONTINUOUS:
                # Continuous mode: show latest frame
                try:
                    display_data = self._frame_queue.get(timeout=0.1)
                    display_frame = self._render_continuous(display_data)
                except queue.Empty:
                    # No frame, show idle
                    display_frame = self._render_idle()
            
            else:  # ALERT_ONLY mode
                # Check for new alerts
                try:
                    alert = self._alert_queue.get_nowait()
                    self._current_alert = alert
                    self._alert_start_time = time.time()
                    logger.info(f"Showing alert: {alert.status}")
                except queue.Empty:
                    pass
                
                # Check if alert expired
                if self._current_alert:
                    if time.time() - self._alert_start_time > self.alert_duration:
                        self._current_alert = None
                
                # Render
                if self._current_alert:
                    display_frame = self._render_alert(self._current_alert)
                else:
                    display_frame = self._render_idle()
            
            # Show frame
            if display_frame is not None:
                cv2.imshow(self.window_name, display_frame)
                self._frame_count += 1
            
            # Calculate FPS
            now = time.time()
            if now - self._fps_start_time >= 1.0:
                self.fps = self._frame_count / (now - self._fps_start_time)
                self._frame_count = 0
                self._fps_start_time = now
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
            elif key == ord('m'):
                # Toggle mode
                self._toggle_mode()
            
            # Limit frame rate
            elapsed = time.time() - last_frame_time
            if elapsed < 0.033:  # ~30 FPS max
                time.sleep(0.033 - elapsed)
            last_frame_time = time.time()
        
        cv2.destroyAllWindows()
        logger.info("UI thread stopped")
    
    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()
    
    def _toggle_mode(self):
        """Toggle between display modes."""
        if self.mode == DisplayMode.CONTINUOUS:
            self.mode = DisplayMode.ALERT_ONLY
        else:
            self.mode = DisplayMode.CONTINUOUS
        logger.info(f"Display mode changed to: {self.mode.value}")
    
    def update_frame(self, frame: np.ndarray, tracks: list, gate_status: str):
        """
        Update display with new frame (for CONTINUOUS mode).
        Drops frames if queue is full.
        """
        if self.mode != DisplayMode.CONTINUOUS:
            return
        
        display_data = DisplayFrame(
            frame=frame.copy(),
            tracks=tracks,
            gate_status=gate_status,
            timestamp=time.time()
        )
        
        try:
            # Drop old frame if queue full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put_nowait(display_data)
        except queue.Full:
            pass
    
    def show_alert(
        self,
        status: str,
        name: Optional[str] = None,
        confidence: float = 0.0,
        face_crop: Optional[np.ndarray] = None
    ):
        """
        Show an alert (for ALERT_ONLY mode, but works in both).
        """
        alert = AlertInfo(
            status=status,
            name=name,
            confidence=confidence,
            face_crop=face_crop.copy() if face_crop is not None else None,
            timestamp=time.time()
        )
        
        try:
            self._alert_queue.put_nowait(alert)
        except queue.Full:
            logger.warning("Alert queue full, dropping alert")
    
    def set_gate_status(self, status: str):
        """Update gate status display."""
        self._gate_status = status
    
    def _render_continuous(self, data: DisplayFrame) -> np.ndarray:
        """Render continuous mode frame with detections."""
        frame = data.frame.copy()
        
        # Resize if needed
        if frame.shape[1] != self.display_width or frame.shape[0] != self.display_height:
            frame = cv2.resize(frame, (self.display_width, self.display_height))
        
        # Draw tracks (FaceOverlay objects)
        for track in data.tracks:
            # Handle bbox as tuple or numpy array
            if hasattr(track.bbox, 'astype'):
                bbox = track.bbox.astype(int)
            else:
                bbox = tuple(int(x) for x in track.bbox)
            
            status = track.status or "PENDING"
            
            # Choose color
            if status == "AUTHORIZED":
                color = self.COLOR_AUTHORIZED
            elif status == "WANTED":
                color = self.COLOR_WANTED
            else:
                color = self.COLOR_UNKNOWN
            
            # Draw box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label - use person_name if available, otherwise status
            name = getattr(track, 'person_name', None) or getattr(track, 'name', None)
            label = name if name else status
            if track.confidence > 0:
                label += f" ({track.confidence:.0%})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]),
                color, -1
            )
            cv2.putText(
                frame, label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
        
        # Draw status bar
        frame = self._draw_status_bar(frame, data.gate_status)
        
        return frame
    
    def _render_alert(self, alert: AlertInfo) -> np.ndarray:
        """Render alert screen."""
        # Create frame
        frame = np.full(
            (self.display_height, self.display_width, 3),
            self.COLOR_BG, dtype=np.uint8
        )
        
        # Choose colors based on status
        if alert.status == "WANTED":
            color = self.COLOR_WANTED
            title = "⚠️ WANTED INDIVIDUAL DETECTED"
        else:
            color = self.COLOR_UNKNOWN
            title = "⚠️ UNKNOWN PERSON"
        
        # Draw title
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (self.display_width - title_size[0]) // 2
        cv2.putText(
            frame, title,
            (title_x, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
        )
        
        # Draw face crop if available
        if alert.face_crop is not None:
            crop = alert.face_crop
            crop_h, crop_w = crop.shape[:2]
            
            # Scale to fit
            max_size = min(self.display_height - 200, 400)
            scale = max_size / max(crop_h, crop_w)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            
            crop = cv2.resize(crop, (new_w, new_h))
            
            # Center position
            x = (self.display_width - new_w) // 2
            y = 150
            
            # Draw border
            cv2.rectangle(
                frame,
                (x - 5, y - 5),
                (x + new_w + 5, y + new_h + 5),
                color, 3
            )
            
            # Place crop
            frame[y:y+new_h, x:x+new_w] = crop
        
        # Draw info
        info_y = self.display_height - 150
        
        if alert.name:
            name_text = f"Name: {alert.name}"
            cv2.putText(
                frame, name_text,
                (50, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_TEXT, 2
            )
            info_y += 40
        
        conf_text = f"Confidence: {alert.confidence:.0%}"
        cv2.putText(
            frame, conf_text,
            (50, info_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_TEXT, 2
        )
        
        # Draw status bar
        frame = self._draw_status_bar(frame, self._gate_status)
        
        return frame
    
    def _render_idle(self) -> np.ndarray:
        """Render idle/dashboard screen."""
        frame = np.full(
            (self.display_height, self.display_width, 3),
            self.COLOR_BG, dtype=np.uint8
        )
        
        # Title
        title = "GATE ACCESS CONTROL"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
        title_x = (self.display_width - title_size[0]) // 2
        cv2.putText(
            frame, title,
            (title_x, self.display_height // 2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, self.COLOR_TEXT, 3
        )
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.value.upper()}"
        cv2.putText(
            frame, mode_text,
            (title_x, self.display_height // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2
        )
        
        # Instructions
        help_text = "Press 'M' to toggle mode | Press 'Q' to quit"
        help_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        help_x = (self.display_width - help_size[0]) // 2
        cv2.putText(
            frame, help_text,
            (help_x, self.display_height - 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1
        )
        
        # Draw status bar
        frame = self._draw_status_bar(frame, self._gate_status)
        
        return frame
    
    def _draw_status_bar(self, frame: np.ndarray, gate_status: str) -> np.ndarray:
        """Draw status bar at bottom of frame."""
        bar_height = 40
        bar_y = frame.shape[0] - bar_height
        
        # Draw bar background
        cv2.rectangle(
            frame,
            (0, bar_y),
            (frame.shape[1], frame.shape[0]),
            (30, 30, 30), -1
        )
        
        # Gate status
        gate_color = self.COLOR_AUTHORIZED if gate_status == "OPEN" else self.COLOR_UNKNOWN
        cv2.putText(
            frame, f"Gate: {gate_status}",
            (20, bar_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, gate_color, 2
        )
        
        # FPS
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}",
            (200, bar_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1
        )
        
        # Mode
        cv2.putText(
            frame, f"Mode: {self.mode.value}",
            (350, bar_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1
        )
        
        # Time
        from datetime import datetime
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            frame, time_str,
            (frame.shape[1] - 120, bar_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1
        )
        
        return frame
    
    def set_mode(self, mode: str):
        """Set display mode."""
        try:
            self.mode = DisplayMode(mode.lower())
            logger.info(f"Display mode set to: {self.mode.value}")
        except ValueError:
            logger.warning(f"Invalid display mode: {mode}")

    def put_frame(self, ui_frame):
        """
        Put a UIFrame on the display queue.
        Compatible with main.py's expected interface.
        
        Args:
            ui_frame: UIFrame dataclass with frame, faces, gate_state, timestamp
        """
        if self.mode != DisplayMode.CONTINUOUS:
            # In alert-only mode, check for alerts in faces
            for face in ui_frame.faces:
                if face.status in ("UNKNOWN", "WANTED"):
                    self.show_alert(
                        status=face.status,
                        name=face.person_name,
                        confidence=face.confidence,
                    )
            return
        
        # Create DisplayFrame for internal use
        display_data = DisplayFrame(
            frame=ui_frame.frame.copy(),
            tracks=ui_frame.faces,  # Use faces as tracks
            gate_status=ui_frame.gate_state,
            timestamp=ui_frame.timestamp
        )
        
        try:
            # Drop old frame if queue full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put_nowait(display_data)
        except queue.Full:
            pass
    
    def update_status(self, face_count: int = 0, sync_status: str = "Unknown"):
        """
        Update status information displayed on screen.
        
        Args:
            face_count: Number of faces in database
            sync_status: Status of sync thread
        """
        # Store status for display in status bar
        self._status_info = {
            "face_count": face_count,
            "sync_status": sync_status,
        }


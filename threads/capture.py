"""
Camera Capture Thread - Decoupled camera capture for smooth streaming.

Architecture:
    Camera (hardware)
        ├──→ Stream Queue (continuous, 15-30 FPS) → Smooth for admin viewers
        └──→ AI Queue (drops frames if busy) → Recognition pipeline

Why this matters:
- AI processing takes 200-800ms per frame (blocking)
- If camera read is in AI loop, stream gets frames at 1-2 FPS (jittery)
- With dedicated capture thread, stream gets smooth 15+ FPS
- AI still processes at 1-2 FPS but doesn't block camera

Key Design:
- Camera.read() runs in this thread at camera's native FPS
- Two separate queues: one for AI (drops old), one for stream (drops old)
- AI loop pulls from ai_queue when ready
- Stream thread pulls from stream_queue continuously
"""

import threading
import time
import queue
from typing import Optional, Tuple
import logging

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class CaptureThread(threading.Thread):
    """
    Dedicated camera capture thread for decoupled frame distribution.
    
    Captures frames at camera's native FPS and distributes to:
    1. AI Queue - for detection/recognition (drops frames if AI is slow)
    2. Stream Queue - for LiveKit streaming (smooth, drops oldest if full)
    3. Latest frame - for UI continuous mode (always latest, no queue)
    
    Usage:
        capture = CaptureThread(camera_index=0, width=640, height=480, fps=15)
        capture.start()
        
        # In AI loop:
        frame = capture.get_ai_frame(timeout=0.1)
        if frame is not None:
            # process frame
        
        # In stream thread:
        frame = capture.get_stream_frame()
        if frame is not None:
            # stream frame
        
        capture.stop()
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        ai_queue_size: int = 2,      # Small - AI should process latest
        stream_queue_size: int = 5,   # Larger - smooth streaming buffer
    ):
        super().__init__(name="CaptureThread", daemon=True)
        
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
        # Queues for different consumers
        self._ai_queue: queue.Queue = queue.Queue(maxsize=ai_queue_size)
        self._stream_queue: queue.Queue = queue.Queue(maxsize=stream_queue_size)
        
        # Latest frame (for UI continuous mode - always overwritten)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_lock = threading.Lock()
        
        # Camera
        self._camera = None
        self._stop_event = threading.Event()
        
        # Stats
        self.frames_captured = 0
        self.frames_dropped_ai = 0
        self.frames_dropped_stream = 0
        self.actual_fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()
        
        # State
        self.is_running = False
        self.camera_opened = False
    
    def _open_camera(self) -> bool:
        """Open camera with configured settings."""
        if cv2 is None:
            logger.error("OpenCV not available")
            return False
        
        try:
            self._camera = cv2.VideoCapture(self.camera_index)
            
            if not self._camera.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Configure camera
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Reduce buffer to minimize latency
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual settings
            actual_w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")
            self.camera_opened = True
            return True
            
        except Exception as e:
            logger.error(f"Camera open error: {e}")
            return False
    
    def run(self):
        """Main capture loop - runs at camera FPS."""
        logger.info("Capture thread starting...")
        
        if not self._open_camera():
            logger.error("Capture thread failed to start - camera not available")
            return
        
        self.is_running = True
        frame_interval = 1.0 / self.fps
        
        logger.info(f"Capture thread running at {self.fps} FPS target")
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            # Capture frame
            ret, frame = self._camera.read()
            
            if not ret:
                logger.warning("Frame capture failed")
                time.sleep(0.01)
                continue
            
            self.frames_captured += 1
            self._fps_counter += 1
            
            # Update FPS stats every second
            now = time.time()
            if now - self._fps_time >= 1.0:
                self.actual_fps = self._fps_counter / (now - self._fps_time)
                self._fps_counter = 0
                self._fps_time = now
            
            # Distribute frame to consumers
            self._distribute_frame(frame)
            
            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Cleanup
        if self._camera:
            self._camera.release()
        
        self.is_running = False
        logger.info("Capture thread stopped")
    
    def _distribute_frame(self, frame: np.ndarray):
        """Distribute frame to all consumers."""
        # 1. Update latest frame (for UI)
        with self._latest_frame_lock:
            self._latest_frame = frame.copy()
        
        # 2. Push to AI queue (drop oldest if full)
        try:
            if self._ai_queue.full():
                try:
                    self._ai_queue.get_nowait()
                    self.frames_dropped_ai += 1
                except queue.Empty:
                    pass
            self._ai_queue.put_nowait(frame.copy())
        except queue.Full:
            self.frames_dropped_ai += 1
        
        # 3. Push to stream queue (drop oldest if full)
        try:
            if self._stream_queue.full():
                try:
                    self._stream_queue.get_nowait()
                    self.frames_dropped_stream += 1
                except queue.Empty:
                    pass
            self._stream_queue.put_nowait(frame.copy())
        except queue.Full:
            self.frames_dropped_stream += 1
    
    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()
    
    # ========================
    # Consumer APIs
    # ========================
    
    def get_ai_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get frame for AI processing.
        
        Blocks up to timeout seconds waiting for a frame.
        Returns None if no frame available.
        
        AI should call this in its loop - it's okay to be slow,
        frames will be dropped and you'll get the latest available.
        """
        try:
            return self._ai_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stream_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Get frame for streaming.
        
        Short timeout since stream thread runs faster.
        Returns None if no frame available.
        """
        try:
            return self._stream_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame (non-blocking).
        
        For UI continuous mode - always returns most recent frame.
        Returns None if no frame captured yet.
        """
        with self._latest_frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        return {
            "frames_captured": self.frames_captured,
            "frames_dropped_ai": self.frames_dropped_ai,
            "frames_dropped_stream": self.frames_dropped_stream,
            "actual_fps": round(self.actual_fps, 1),
            "is_running": self.is_running,
            "camera_opened": self.camera_opened,
        }


def create_capture_thread(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 15
) -> CaptureThread:
    """Factory function to create capture thread."""
    return CaptureThread(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps
    )

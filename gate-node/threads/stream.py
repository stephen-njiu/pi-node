"""
Stream Thread - Handles LiveKit video streaming for admin dashboard.
Enables remote viewing of gate camera feed.
"""

import threading
import time
import queue
from typing import Optional
import logging

import numpy as np

# LiveKit SDK (optional)
try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


logger = logging.getLogger(__name__)


class StreamThread(threading.Thread):
    """
    LiveKit streaming thread for admin dashboard.
    
    Streams camera feed to LiveKit server for remote viewing.
    Only active when admin requests live view.
    """
    
    def __init__(
        self,
        livekit_url: str,
        gate_id: str,
        frame_width: int = 640,
        frame_height: int = 480,
        fps: int = 15
    ):
        super().__init__(name="StreamThread", daemon=True)
        
        self.livekit_url = livekit_url
        self.gate_id = gate_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        self._frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._streaming = False
        self._room = None
        self._video_track = None
        
        # Stats
        self.is_connected = False
        self.viewers = 0
    
    def run(self):
        """Main streaming loop."""
        if not LIVEKIT_AVAILABLE:
            logger.warning("LiveKit SDK not available - streaming disabled")
            return
        
        logger.info("Stream thread started")
        
        while not self._stop_event.is_set():
            if self._streaming and self.is_connected:
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                    self._publish_frame(frame)
                except queue.Empty:
                    pass
            else:
                # Not streaming, sleep
                self._stop_event.wait(timeout=1.0)
        
        self._disconnect()
        logger.info("Stream thread stopped")
    
    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()
    
    def start_streaming(self, token: str):
        """
        Start streaming with LiveKit token.
        Token should be obtained from backend.
        """
        if not LIVEKIT_AVAILABLE:
            logger.error("Cannot start streaming - LiveKit SDK not available")
            return False
        
        try:
            self._connect(token)
            self._streaming = True
            logger.info("Streaming started")
            return True
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop streaming."""
        self._streaming = False
        self._disconnect()
        logger.info("Streaming stopped")
    
    def push_frame(self, frame: np.ndarray):
        """Push frame to streaming queue."""
        if not self._streaming or not self.is_connected:
            return
        
        try:
            # Resize if needed
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                import cv2
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Drop old frames if queue full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def _connect(self, token: str):
        """Connect to LiveKit room."""
        if not LIVEKIT_AVAILABLE:
            return
        
        try:
            # Create room
            self._room = rtc.Room()
            
            # Connect
            self._room.connect(self.livekit_url, token)
            
            # Create video source
            source = rtc.VideoSource(self.frame_width, self.frame_height)
            self._video_track = rtc.LocalVideoTrack.create_video_track(
                f"gate-{self.gate_id}",
                source
            )
            
            # Publish track
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_CAMERA
            self._room.local_participant.publish_track(self._video_track, options)
            
            self.is_connected = True
            logger.info(f"Connected to LiveKit: {self.livekit_url}")
            
        except Exception as e:
            logger.error(f"LiveKit connection failed: {e}")
            self.is_connected = False
            raise
    
    def _disconnect(self):
        """Disconnect from LiveKit."""
        if self._room:
            try:
                self._room.disconnect()
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
            finally:
                self._room = None
                self._video_track = None
                self.is_connected = False
    
    def _publish_frame(self, frame: np.ndarray):
        """Publish frame to LiveKit."""
        if not self._video_track:
            return
        
        try:
            # Convert BGR to RGB
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create video frame
            video_frame = rtc.VideoFrame(
                self.frame_width,
                self.frame_height,
                rtc.VideoBufferType.RGB24,
                rgb_frame.tobytes()
            )
            
            # Capture (publish)
            self._video_track.capture_frame(video_frame)
            
        except Exception as e:
            logger.error(f"Failed to publish frame: {e}")
    
    def get_status(self) -> dict:
        """Get streaming status."""
        return {
            "available": LIVEKIT_AVAILABLE,
            "streaming": self._streaming,
            "connected": self.is_connected,
            "viewers": self.viewers,
            "url": self.livekit_url
        }


class MockStreamThread(threading.Thread):
    """
    Mock stream thread when LiveKit is not available.
    Just logs frames for debugging.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(name="MockStreamThread", daemon=True)
        self._stop_event = threading.Event()
        self.is_connected = False
        self.viewers = 0
    
    def run(self):
        logger.info("Mock stream thread started (LiveKit not available)")
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1.0)
    
    def stop(self):
        self._stop_event.set()
    
    def start_streaming(self, token: str):
        logger.info("Mock streaming started (no actual stream)")
        return True
    
    def stop_streaming(self):
        logger.info("Mock streaming stopped")
    
    def push_frame(self, frame: np.ndarray):
        pass  # Discard
    
    def get_status(self) -> dict:
        return {
            "available": False,
            "streaming": False,
            "connected": False,
            "viewers": 0,
            "url": None
        }

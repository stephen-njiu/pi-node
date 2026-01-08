"""
Gate Node Main Orchestrator
---------------------------
Central coordinator for all gate-node components.

This can run on:
- Raspberry Pi 4B (production)
- Laptop with webcam (development/demo)

Architecture:
- Single Python process
- 5 worker threads: Vision, Decision (part of main loop), UI, Sync, Streaming
- Main thread runs the vision pipeline and decision loop

Flow:
1. Camera capture → Detection → Tracking → Recognition
2. Decision engine makes gate decision
3. Gate controller executes action (mock on laptop)
4. UI displays status
5. Sync keeps local DB updated
6. Streaming publishes on demand

For laptop demo:
- GPIO is disabled (uses mock)
- Webcam used instead of Pi camera
- Display shows live video feed by default
"""

import signal
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    print("ERROR: OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)

# Local imports
from config import config
from storage import FaceDatabase, AccessLogger
from vision import SCRFDDetector, ArcFaceRecognizer, SimpleTracker, align_face
from core import (
    TrackStateManager,
    TrackStatus,
    GateController,
    GateDecision,
    DecisionEngine,
    create_gate_controller_from_config,
)
from threads import (
    SyncThread,
    UIThread,
    UIFrame,
    FaceOverlay,
    create_ui_thread_from_config,
)

# Optional streaming imports
try:
    from threads import StreamThread, create_stream_config_from_config
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    StreamThread = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gate_node.log"),
    ]
)
logger = logging.getLogger("GateNode")


class GateNode:
    """
    Main gate node application.
    
    Coordinates all components and runs the main vision loop.
    Works on both Raspberry Pi and laptop (for demo).
    """
    
    def __init__(self):
        # Core state
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Data directory
        self.data_dir = Path(config.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components (initialized in start())
        self.face_db: Optional[FaceDatabase] = None
        self.access_logger: Optional[AccessLogger] = None
        self.detector: Optional[SCRFDDetector] = None
        self.recognizer: Optional[ArcFaceRecognizer] = None
        self.tracker: Optional[SimpleTracker] = None
        self.track_state_manager: Optional[TrackStateManager] = None
        self.gate_controller: Optional[GateController] = None
        self.decision_engine: Optional[DecisionEngine] = None
        
        # Threads
        self.sync_thread: Optional[SyncThread] = None
        self.ui_thread: Optional[UIThread] = None
        self.stream_thread = None  # Optional: StreamThread
        
        # Camera
        self.camera = None  # cv2.VideoCapture
        
        # Stats
        self.stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "recognitions_attempted": 0,
            "start_time": None,
        }
    
    def _init_storage(self) -> bool:
        """Initialize storage components."""
        try:
            logger.info("Initializing storage...")
            
            # Face database (hnswlib)
            self.face_db = FaceDatabase(
                index_path=config.INDEX_PATH,
                metadata_path=config.METADATA_PATH,
                dimension=512,  # ArcFace dimension
            )
            
            # Access logger (SQLite)
            self.access_logger = AccessLogger(db_path=config.LOG_DB_PATH)
            
            logger.info(f"Storage initialized: {self.face_db.count()} faces in database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            return False
    
    def _init_vision(self) -> bool:
        """Initialize vision pipeline."""
        try:
            logger.info("Initializing vision pipeline...")
            
            # Check model files exist
            scrfd_path = Path(config.SCRFD_MODEL_PATH)
            arcface_path = Path(config.ARCFACE_MODEL_PATH)
            
            if not scrfd_path.exists():
                logger.error(f"SCRFD model not found: {scrfd_path}")
                return False
            
            if not arcface_path.exists():
                logger.error(f"ArcFace model not found: {arcface_path}")
                return False
            
            # Initialize detector
            self.detector = SCRFDDetector(
                model_path=str(scrfd_path),
                input_size=(640, 640),
                conf_threshold=0.5,
            )
            
            # Initialize recognizer
            self.recognizer = ArcFaceRecognizer(
                model_path=str(arcface_path),
            )
            
            # Initialize tracker
            self.tracker = SimpleTracker(
                max_age=30,  # ~2 seconds at 15fps
                min_hits=3,
                iou_threshold=0.3,
            )
            
            # Track state manager
            self.track_state_manager = TrackStateManager(
                max_attempts=config.MAX_RECOGNITION_ATTEMPTS,
                attempt_interval=0.5,
                cooldown=config.TRACK_COOLDOWN_SECONDS,
            )
            
            logger.info("Vision pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vision: {e}")
            return False
    
    def _init_gate(self) -> bool:
        """Initialize gate control."""
        try:
            logger.info("Initializing gate controller...")
            
            # Decision engine
            self.decision_engine = DecisionEngine(
                confidence_threshold=config.RECOGNITION_THRESHOLD,
                wanted_confidence_threshold=config.WANTED_CONFIDENCE_THRESHOLD,
            )
            
            # Gate controller
            if config.GPIO_ENABLED:
                self.gate_controller = create_gate_controller_from_config(config)
                if not self.gate_controller.initialize():
                    logger.error("Failed to initialize GPIO")
                    return False
                logger.info("Gate controller initialized (GPIO enabled)")
            else:
                self.gate_controller = create_gate_controller_from_config(config)
                self.gate_controller.initialize()  # Uses mock GPIO
                logger.info("Gate controller initialized (GPIO disabled - mock mode)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize gate: {e}")
            return False
    
    def _init_camera(self) -> bool:
        """Initialize camera."""
        try:
            logger.info(f"Opening camera {config.CAMERA_INDEX}...")
            
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {config.CAMERA_INDEX}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            # Verify settings
            actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _init_threads(self) -> bool:
        """Initialize worker threads."""
        try:
            logger.info("Initializing worker threads...")
            
            # Sync thread
            self.sync_thread = SyncThread(
                face_db=self.face_db,
                backend_url=config.BACKEND_URL,
                org_id=config.ORG_ID,
                interval_seconds=config.SYNC_INTERVAL_SECONDS,
                version_file=config.VERSION_PATH,
            )
            
            # UI thread
            if config.DISPLAY_ENABLED:
                self.ui_thread = create_ui_thread_from_config(config)
            
            # Stream thread (optional)
            if STREAMING_AVAILABLE and (config.MQTT_ENABLED or config.LIVEKIT_URL):
                stream_config = create_stream_config_from_config(config)
                self.stream_thread = StreamThread(
                    config=stream_config,
                    on_alert=self._handle_alert,
                )
            else:
                logger.info("Streaming disabled (MQTT/LiveKit not configured or unavailable)")
            
            logger.info("Worker threads initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize threads: {e}")
            return False
    
    def _handle_alert(self, alert_type: str, frame: np.ndarray):
        """Handle security alert (e.g., WANTED person)."""
        logger.warning(f"ALERT: {alert_type}")
        
        # Save snapshot
        timestamp = int(time.time())
        snapshot_path = self.data_dir / f"alert_{alert_type}_{timestamp}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        logger.info(f"Alert snapshot saved: {snapshot_path}")
    
    def start(self) -> bool:
        """
        Start the gate node.
        
        Returns:
            True if started successfully
        """
        logger.info("="*50)
        logger.info(f"Gate Node starting: {config.GATE_ID}")
        logger.info(f"Organization: {config.ORG_ID}")
        logger.info("="*50)
        
        # Initialize all components
        if not self._init_storage():
            return False
        
        if not self._init_vision():
            return False
        
        if not self._init_gate():
            return False
        
        if not self._init_camera():
            return False
        
        if not self._init_threads():
            return False
        
        # Start worker threads
        self.sync_thread.start()
        logger.info("Sync thread started")
        
        if self.ui_thread:
            self.ui_thread.start()
            logger.info("UI thread started")
        
        if self.stream_thread:
            self.stream_thread.start()
            logger.info("Stream thread started")
        
        self.stats["start_time"] = time.time()
        self._running = True
        
        logger.info("Gate Node started successfully")
        return True
    
    def stop(self):
        """Stop the gate node gracefully."""
        logger.info("Gate Node shutting down...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Stop threads
        if self.sync_thread:
            self.sync_thread.stop()
            self.sync_thread.join(timeout=5.0)
        
        if self.ui_thread:
            self.ui_thread.stop()
            self.ui_thread.join(timeout=2.0)
        
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Cleanup gate controller
        if self.gate_controller:
            self.gate_controller.cleanup()
        
        # Close database connections
        if self.access_logger:
            self.access_logger.close()
        
        logger.info("Gate Node stopped")
        self._print_stats()
    
    def _print_stats(self):
        """Print final statistics."""
        if self.stats["start_time"]:
            runtime = time.time() - self.stats["start_time"]
            fps = self.stats["frames_processed"] / runtime if runtime > 0 else 0
            
            logger.info("="*50)
            logger.info("Session Statistics:")
            logger.info(f"  Runtime: {runtime:.1f}s")
            logger.info(f"  Frames processed: {self.stats['frames_processed']}")
            logger.info(f"  Average FPS: {fps:.1f}")
            logger.info(f"  Faces detected: {self.stats['faces_detected']}")
            logger.info(f"  Recognitions: {self.stats['recognitions_attempted']}")
            
            if self.gate_controller:
                gate_stats = self.gate_controller.get_stats()
                logger.info(f"  Gate opens: {gate_stats['total_opens']}")
                logger.info(f"    - Authorized: {gate_stats['authorized_opens']}")
                logger.info(f"    - Wanted: {gate_stats['wanted_opens']}")
                logger.info(f"    - Rejected: {gate_stats['rejected_unknown']}")
            
            logger.info("="*50)
    
    def run(self):
        """
        Main vision loop.
        
        This runs on the main thread and processes camera frames.
        """
        if not self._running:
            logger.error("Gate Node not started")
            return
        
        logger.info("Starting main vision loop...")
        
        frame_time = 1.0 / config.CAMERA_FPS
        last_frame_time = 0
        
        while self._running and not self._shutdown_event.is_set():
            loop_start = time.time()
            
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            self.stats["frames_processed"] += 1
            
            # =========================
            # DETECTION
            # =========================
            detections = self.detector.detect(frame)
            self.stats["faces_detected"] += len(detections)
            
            # =========================
            # TRACKING
            # =========================
            tracks = self.tracker.update(detections)
            
            # Cleanup stale track states
            active_track_ids = {t[4] for t in tracks}  # (x1, y1, x2, y2, track_id, ...)
            self.track_state_manager.cleanup_stale(active_track_ids)
            
            # =========================
            # RECOGNITION & DECISION
            # =========================
            face_overlays: List[FaceOverlay] = []
            
            for track in tracks:
                x1, y1, x2, y2, track_id = track[:5]
                landmarks = track[5] if len(track) > 5 else None
                
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Check if we should attempt recognition
                if self.track_state_manager.should_attempt_recognition(track_id):
                    self.track_state_manager.record_attempt(track_id)
                    self.stats["recognitions_attempted"] += 1
                    
                    # Extract face and get embedding
                    try:
                        # Align face if we have landmarks
                        if landmarks is not None:
                            aligned_face = align_face(frame, landmarks)
                        else:
                            # Fallback: crop and resize
                            face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            aligned_face = cv2.resize(face_crop, (112, 112))
                        
                        # Get embedding
                        embedding = self.recognizer.get_embedding(aligned_face)
                        
                        # Search in database
                        results = self.face_db.search(embedding, k=1)
                        
                        if results and len(results) > 0:
                            person_id, distance, metadata = results[0]
                            confidence = 1.0 - distance  # Convert distance to similarity
                            
                            # Make decision
                            decision = self.decision_engine.make_decision(
                                match_found=True,
                                person_id=person_id,
                                confidence=confidence,
                                status=metadata.get("status", "AUTHORIZED"),
                            )
                            
                            # Record result
                            if decision == GateDecision.AUTHORIZED:
                                self.track_state_manager.record_success(
                                    track_id=track_id,
                                    status=TrackStatus.AUTHORIZED,
                                    person_id=person_id,
                                    metadata=metadata,
                                    confidence=confidence,
                                )
                            elif decision == GateDecision.WANTED:
                                self.track_state_manager.record_success(
                                    track_id=track_id,
                                    status=TrackStatus.WANTED,
                                    person_id=person_id,
                                    metadata=metadata,
                                    confidence=confidence,
                                )
                                # Send alert
                                if self.stream_thread:
                                    self.stream_thread.send_alert("WANTED", frame)
                            else:
                                self.track_state_manager.record_failure(track_id)
                                
                            # Execute gate action
                            self.gate_controller.open_gate(
                                decision=decision,
                                person_id=person_id,
                                track_id=track_id,
                                confidence=confidence,
                            )
                            
                            # Log access
                            self.access_logger.log_access(
                                person_id=person_id,
                                person_name=metadata.get("full_name", "Unknown"),
                                status=decision.value,
                                confidence=confidence,
                                gate_id=config.GATE_ID,
                            )
                        else:
                            # No match found
                            self.track_state_manager.record_failure(track_id)
                            
                    except Exception as e:
                        logger.error(f"Recognition error for track {track_id}: {e}")
                        self.track_state_manager.record_failure(track_id)
                
                # Get current track state for UI
                track_state = self.track_state_manager.get_state(track_id)
                
                if track_state:
                    face_overlays.append(FaceOverlay(
                        bbox=bbox,
                        track_id=track_id,
                        status=track_state.status.value,
                        person_name=track_state.metadata.get("full_name") if track_state.metadata else None,
                        confidence=track_state.confidence,
                    ))
                else:
                    face_overlays.append(FaceOverlay(
                        bbox=bbox,
                        track_id=track_id,
                        status="PENDING",
                    ))
            
            # =========================
            # UPDATE UI
            # =========================
            if self.ui_thread:
                ui_frame = UIFrame(
                    frame=frame,
                    faces=face_overlays,
                    gate_state=self.gate_controller.state.value,
                    timestamp=time.time(),
                )
                self.ui_thread.put_frame(ui_frame)
                
                # Update status periodically
                if self.stats["frames_processed"] % 30 == 0:
                    self.ui_thread.update_status(
                        face_count=self.face_db.count(),
                        sync_status="Synced" if self.sync_thread.last_sync_success else "Error",
                    )
            
            # =========================
            # UPDATE STREAM
            # =========================
            if self.stream_thread:
                self.stream_thread.put_frame(frame)
            
            # =========================
            # FRAME RATE CONTROL
            # =========================
            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Main vision loop ended")


def main():
    """Entry point."""
    node = GateNode()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        node.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the node
    if not node.start():
        logger.error("Failed to start Gate Node")
        sys.exit(1)
    
    try:
        # Run main loop
        node.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        node.stop()


if __name__ == "__main__":
    main()

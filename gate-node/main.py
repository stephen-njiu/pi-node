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

Flow (DeepSORT-lite pipeline):
1. Camera capture → Detection → Tracking (phase-based)
2. Recognition runs ONCE per confirmed track (not per frame!)
3. Decision engine makes gate decision per unique person
4. Gate controller executes action (mock on laptop)
5. UI displays status
6. Sync keeps local DB updated
7. Streaming publishes on demand

For laptop demo:
- GPIO is disabled (uses mock)
- Webcam used instead of Pi camera
- Display shows live video feed by default

Key Principles:
- Track IDs are stable across frames
- Recognition runs once per person, not per detection
- Statistics count unique people, not raw detections
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
from vision import SCRFDDetector, ArcFaceRecognizer, SimpleTracker, Track, align_face
from vision.tracker import TrackPhase  # Import phase enum
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
    CaptureThread,
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
    
    Recognition Flow (CRITICAL):
    1. Detector finds faces → raw detections
    2. Tracker assigns stable IDs → tracks with phases
    3. Only CONFIRMED tracks (phase=CONFIRMED, recognized=False) get recognition
    4. Recognition runs ONCE, then track.recognized=True forever
    5. This ensures we count people, not detections
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
        self.gate_controller: Optional[GateController] = None
        self.decision_engine: Optional[DecisionEngine] = None
        
        # Threads
        self.sync_thread: Optional[SyncThread] = None
        self.ui_thread: Optional[UIThread] = None
        self.stream_thread = None  # Optional: StreamThread
        self.capture_thread: Optional[CaptureThread] = None  # Decoupled camera capture
        
        # Recognition config
        self.max_recognition_attempts = getattr(config, 'MAX_RECOGNITION_ATTEMPTS', 3)
        
        # Stats (track-based, not detection-based)
        self.stats = {
            "frames_processed": 0,
            "detections_run": 0,      # Times detection actually ran
            "detections_skipped": 0,  # Times detection was skipped
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
            
            # Initialize DeepSORT-lite tracker
            # Tuned for low FPS scenarios (~1-2 FPS)
            self.tracker = SimpleTracker(
                max_age=60,              # ~60 seconds at 1fps - keep tracks longer
                min_hits=2,              # 2 consecutive detections to confirm (faster)
                iou_threshold=0.2,       # Lower IoU for low-FPS (more permissive)
                max_embedding_distance=0.7,  # Slightly more permissive for embeddings
                embedding_weight=0.5,    # Give more weight to embeddings vs IoU
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
        """
        Initialize camera via CaptureThread.
        
        The capture thread handles:
        - Camera open/close
        - Continuous frame capture at native FPS
        - Distribution to AI queue and stream queue
        
        This decouples camera from AI processing for smooth streaming.
        """
        try:
            logger.info(f"Opening camera {config.CAMERA_INDEX} via capture thread...")
            
            # Create capture thread
            self.capture_thread = CaptureThread(
                camera_index=config.CAMERA_INDEX,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                fps=config.CAMERA_FPS,
                ai_queue_size=2,       # Small - AI processes latest
                stream_queue_size=5,   # Buffer for smooth streaming
            )
            
            # Start capture thread
            self.capture_thread.start()
            
            # Wait for camera to open (up to 3 seconds)
            for _ in range(30):
                if self.capture_thread.camera_opened:
                    break
                time.sleep(0.1)
            
            if not self.capture_thread.camera_opened:
                logger.error(f"Failed to open camera {config.CAMERA_INDEX}")
                return False
            
            logger.info(f"Camera opened: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} @ {config.CAMERA_FPS}fps")
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
                self.stream_thread = StreamThread(
                    livekit_url=config.LIVEKIT_URL,
                    gate_id=config.GATE_ID,
                    frame_width=config.CAMERA_WIDTH,
                    frame_height=config.CAMERA_HEIGHT,
                    fps=config.CAMERA_FPS,
                    capture_thread=self.capture_thread,  # For smooth streaming
                )
                logger.info("StreamThread connected to CaptureThread for smooth streaming")
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
        
        # Stop threads (order matters - stop capture last to avoid frame starvation)
        if self.sync_thread:
            self.sync_thread.stop()
            self.sync_thread.join(timeout=5.0)
        
        if self.ui_thread:
            self.ui_thread.stop()
            self.ui_thread.join(timeout=2.0)
        
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread.join(timeout=2.0)
        
        # Stop capture thread (this also releases the camera)
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)
        
        # Cleanup gate controller
        if self.gate_controller:
            self.gate_controller.cleanup()
        
        # Close database connections
        if self.access_logger:
            self.access_logger.close()
        
        logger.info("Gate Node stopped")
        self._print_stats()
    
    def _print_stats(self):
        """Print final statistics (track-based, not detection-based)."""
        if self.stats["start_time"]:
            runtime = time.time() - self.stats["start_time"]
            fps = self.stats["frames_processed"] / runtime if runtime > 0 else 0
            
            logger.info("="*50)
            logger.info("Session Statistics:")
            logger.info(f"  Runtime: {runtime:.1f}s")
            logger.info(f"  Frames processed (AI): {self.stats['frames_processed']}")
            logger.info(f"  Average AI FPS: {fps:.1f}")
            
            # Capture thread stats (decoupled camera)
            if self.capture_thread:
                cap_stats = self.capture_thread.get_stats()
                logger.info(f"  Camera capture:")
                logger.info(f"    - Frames captured: {cap_stats['frames_captured']}")
                logger.info(f"    - Capture FPS: {cap_stats['actual_fps']}")
                logger.info(f"    - Frames dropped (AI queue): {cap_stats['frames_dropped_ai']}")
                logger.info(f"    - Frames dropped (stream): {cap_stats['frames_dropped_stream']}")
            
            # Detection skip optimization stats
            det_run = self.stats.get("detections_run", 0)
            det_skip = self.stats.get("detections_skipped", 0)
            total_det = det_run + det_skip
            skip_rate = (det_skip / total_det * 100) if total_det > 0 else 0
            logger.info(f"  Detection optimization:")
            logger.info(f"    - Detections run: {det_run}")
            logger.info(f"    - Detections skipped: {det_skip}")
            logger.info(f"    - Skip rate: {skip_rate:.1f}%")
            
            # Tracker statistics (track-based, not detection-based)
            if self.tracker:
                tracker_stats = self.tracker.get_statistics()
                logger.info(f"  Unique tracks created: {tracker_stats.tracks_created}")
                logger.info(f"  Tracks confirmed: {tracker_stats.tracks_confirmed}")
                logger.info(f"  Tracks recognized: {tracker_stats.tracks_recognized}")
                logger.info(f"  People counts:")
                logger.info(f"    - Authorized: {tracker_stats.authorized_count}")
                logger.info(f"    - Wanted: {tracker_stats.wanted_count}")
                logger.info(f"    - Unknown: {tracker_stats.unknown_count}")
            
            if self.gate_controller:
                gate_stats = self.gate_controller.get_stats()
                logger.info(f"  Gate opens: {gate_stats['total_opens']}")
                logger.info(f"    - Authorized: {gate_stats['authorized_opens']}")
                logger.info(f"    - Wanted: {gate_stats['wanted_opens']}")
                logger.info(f"    - Rejected: {gate_stats['rejected_unknown']}")
            
            logger.info("="*50)
    
    def _recognize_track(self, track: Track, frame: np.ndarray) -> bool:
        """
        Run recognition for a single track.
        
        This should only be called for tracks where:
        - track.is_ready_for_recognition() returns True
        - track.phase == CONFIRMED
        - track.recognized == False
        
        Returns:
            True if recognition completed (success or failure)
        """
        track_id = track.track_id
        bbox = track.bbox
        
        try:
            # Extract face region
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Track {track_id}: Invalid bbox after clamping")
                return False
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                logger.warning(f"Track {track_id}: Empty face crop")
                return False
            
            # Resize to recognition model input size
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
                
                # Map decision to status string
                if decision == GateDecision.AUTHORIZED:
                    status = "AUTHORIZED"
                elif decision == GateDecision.WANTED:
                    status = "WANTED"
                else:
                    status = "UNKNOWN"
                
                # Update tracker with recognition result (marks track as RECOGNIZED)
                self.tracker.update_track_recognition(
                    track_id=track_id,
                    face_id=person_id,
                    user_id=metadata.get("user_id"),
                    name=metadata.get("full_name"),
                    status=status,
                    confidence=confidence,
                )
                
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
                    status=status,
                    confidence=confidence,
                    gate_id=config.GATE_ID,
                )
                
                # Alert for WANTED
                if decision == GateDecision.WANTED and self.stream_thread:
                    self.stream_thread.send_alert("WANTED", frame)
                
                logger.info(
                    f"Track {track_id} recognized: {status} "
                    f"({metadata.get('full_name', 'Unknown')}, conf={confidence:.2f})"
                )
                return True
                
            else:
                # No match found - mark as UNKNOWN after max attempts
                self.tracker.record_recognition_attempt(track_id)
                
                if track.recognition_attempts >= self.max_recognition_attempts:
                    # Max attempts reached, mark as UNKNOWN permanently
                    self.tracker.update_track_recognition(
                        track_id=track_id,
                        face_id=None,
                        user_id=None,
                        name=None,
                        status="UNKNOWN",
                        confidence=0.0,
                    )
                    
                    logger.info(
                        f"Track {track_id} marked UNKNOWN after "
                        f"{self.max_recognition_attempts} attempts"
                    )
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Recognition error for track {track_id}: {e}")
            self.tracker.record_recognition_attempt(track_id)
            return False
    
    def run(self):
        """
        Main vision loop.
        
        This runs on the main thread and processes camera frames.
        
        Architecture (DECOUPLED):
        - CaptureThread: Reads camera at 15+ FPS, feeds AI queue + stream queue
        - This loop: Pulls from AI queue, runs detection/recognition at 1-2 FPS
        - StreamThread: Pulls from stream queue, sends to LiveKit at 15+ FPS
        
        Pipeline per frame:
        1. Get frame from AI queue (non-blocking, drops old frames)
        2. SMART DETECTION: Skip if all tracks are recognized and stable
        3. Detect faces → raw detections (only when needed)
        4. Update tracker → stable track IDs
        5. For tracks ready for recognition → run recognition ONCE
        6. Update UI with all confirmed tracks
        
        Optimization: Skip detection when all active tracks are recognized and stable.
        This saves ~200-400ms per frame when people stay in view after recognition.
        """
        if not self._running:
            logger.error("Gate Node not started")
            return
        
        logger.info("Starting main vision loop...")
        
        # Detection skip logic
        detection_skip_frames = 0     # Counter for skipped frames
        max_skip_frames = 5           # Max frames to skip before forced detection
        
        while self._running and not self._shutdown_event.is_set():
            loop_start = time.time()
            
            # =========================
            # GET FRAME FROM AI QUEUE
            # =========================
            # CaptureThread handles camera read at full FPS
            # We pull frames from AI queue (drops old frames if we're slow)
            frame = self.capture_thread.get_ai_frame(timeout=0.5)
            if frame is None:
                # No frame available, capture thread might be starting up
                continue
            
            self.stats["frames_processed"] += 1
            
            # =========================
            # SMART DETECTION DECISION
            # =========================
            # Skip expensive detection if:
            # 1. We have active tracks that are all recognized
            # 2. Tracks are still being matched (not lost)
            # 3. Haven't skipped too many frames
            # 
            # WHY THIS WORKS:
            # - After recognition, person stays in frame
            # - We don't need to re-detect them every frame
            # - Skip 5 frames, then do 1 detection to check if they left
            # - This can boost FPS 5x when person is standing still
            should_detect = True
            
            if self.tracker and detection_skip_frames < max_skip_frames:
                active_tracks = self.tracker.get_all_active_tracks()
                if active_tracks:
                    # Check if all tracks are recognized AND recently updated
                    all_recognized = all(t.recognized for t in active_tracks)
                    all_stable = all(t.time_since_update <= 1 for t in active_tracks)
                    
                    if all_recognized and all_stable:
                        # Skip detection - but we STILL need to keep tracks alive
                        should_detect = False
                        detection_skip_frames += 1
            
            # =========================
            # DETECTION (conditional)
            # =========================
            if should_detect:
                detection_skip_frames = 0  # Reset skip counter
                detections = self.detector.detect(frame)
                tracker_detections = [(det.bbox, det.score, None) for det in detections]
                
                # Update tracker with detections
                confirmed_tracks = self.tracker.update(tracker_detections)
                self.stats["detections_run"] += 1
            else:
                # Skip detection - DON'T update tracker, just use last known tracks
                # This prevents time_since_update from incrementing
                confirmed_tracks = self.tracker.get_all_active_tracks()
                self.stats["detections_skipped"] += 1
            
            # =========================
            # RECOGNITION (once per track)
            # =========================
            # Get tracks that need recognition (CONFIRMED + not recognized)
            tracks_to_recognize = self.tracker.get_tracks_for_recognition()
            
            for track in tracks_to_recognize:
                # Run recognition ONCE for this track
                # After this, track.recognized=True and it won't appear again
                self._recognize_track(track, frame)
            
            # =========================
            # BUILD UI OVERLAYS
            # =========================
            face_overlays: List[FaceOverlay] = []
            
            for track in confirmed_tracks:
                bbox = track.bbox
                bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                
                # Determine display status
                if track.recognized:
                    status = track.status or "UNKNOWN"
                    name = track.name
                    confidence = track.confidence
                else:
                    status = "PENDING"
                    name = None
                    confidence = 0.0
                
                face_overlays.append(FaceOverlay(
                    bbox=bbox_tuple,
                    track_id=track.track_id,
                    status=status,
                    person_name=name,
                    confidence=confidence,
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
                    tracker_stats = self.tracker.get_statistics()
                    self.ui_thread.update_status(
                        face_count=self.face_db.count(),
                        sync_status="Synced" if self.sync_thread.last_sync_success else "Error",
                    )
            
            # =========================
            # NOTE: NO FRAME RATE CONTROL NEEDED
            # =========================
            # The capture thread handles camera timing.
            # AI loop runs as fast as it can - if it's slow, frames drop (that's fine).
            # Stream gets smooth frames from capture thread, not from here.
        
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

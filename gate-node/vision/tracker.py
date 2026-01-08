"""
DeepSORT-lite Face Tracker
==========================

A stable, phase-based tracker optimized for:
- Raspberry Pi (CPU-only, ~1-2 FPS)
- Gate access control (correctness > cleverness)
- Unique person counting (not detection counting)

Architecture:
- Phase 1 (TENTATIVE): IoU-only matching, no embeddings, building hits
- Phase 2 (CONFIRMED): hits >= min_hits, embedding updates enabled, eligible for recognition
- Phase 3 (RECOGNIZED): Recognition completed once, never recognize again

Key Design Principles:
1. Track IDs must be stable across frames
2. Recognition runs ONCE per person (not per frame)
3. Statistics track unique people, not detections
4. Hungarian assignment with hard gating prevents bad matches

This is NOT full DeepSORT - no Torch, no heavy ReID models.
This IS production-ready for real-world gate control.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set
from enum import Enum
import logging
from collections import deque
import time

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logger = logging.getLogger(__name__)


class TrackPhase(Enum):
    """
    Track lifecycle phases.
    
    TENTATIVE: New track, building confidence via IoU matching only
    CONFIRMED: Stable track, ready for recognition, embeddings active
    RECOGNIZED: Recognition complete, track fully identified
    """
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    RECOGNIZED = "recognized"


@dataclass
class Track:
    """
    Represents a tracked face through its lifecycle.
    
    Lifecycle:
    1. Created as TENTATIVE when first detected
    2. Becomes CONFIRMED after min_hits consecutive detections
    3. Becomes RECOGNIZED after recognition completes (success or max attempts)
    
    CRITICAL: Once recognized=True, NEVER run recognition again for this track.
    """
    track_id: int
    bbox: np.ndarray              # [x1, y1, x2, y2]
    score: float                  # Detection confidence
    
    # Lifecycle state
    phase: TrackPhase = TrackPhase.TENTATIVE
    hits: int = 1                 # Consecutive successful matches
    age: int = 0                  # Total frames since creation
    time_since_update: int = 0   # Frames since last successful match
    
    # Embedding (only used when CONFIRMED+)
    embedding: Optional[np.ndarray] = None
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Recognition state (set when RECOGNIZED)
    recognized: bool = False      # CRITICAL: True = never recognize again
    recognition_attempts: int = 0
    face_id: Optional[str] = None
    user_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None  # AUTHORIZED, UNKNOWN, WANTED
    confidence: float = 0.0
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    recognized_at: Optional[float] = None
    
    def is_confirmed(self) -> bool:
        """Track has enough hits to be reliable."""
        return self.phase in (TrackPhase.CONFIRMED, TrackPhase.RECOGNIZED)
    
    def is_ready_for_recognition(self) -> bool:
        """
        Track is eligible for recognition.
        
        CRITICAL: Return False if already recognized!
        This prevents recognition from running multiple times.
        """
        return (
            self.phase == TrackPhase.CONFIRMED and
            not self.recognized
        )
    
    def mark_recognized(
        self,
        face_id: Optional[str],
        user_id: Optional[str],
        name: Optional[str],
        status: str,
        confidence: float
    ):
        """
        Mark track as recognized. This is FINAL - never recognize again.
        
        Called after:
        - Successful match found
        - Max recognition attempts exhausted (mark as UNKNOWN)
        """
        self.recognized = True
        self.phase = TrackPhase.RECOGNIZED
        self.recognized_at = time.time()
        self.face_id = face_id
        self.user_id = user_id
        self.name = name
        self.status = status
        self.confidence = confidence
        
        logger.debug(
            f"Track {self.track_id} RECOGNIZED: status={status}, "
            f"name={name}, confidence={confidence:.2f}"
        )


@dataclass
class TrackerStatistics:
    """
    Track-based statistics (NOT detection-based).
    
    These count UNIQUE PEOPLE, not raw detections.
    Counters are incremented at lifecycle transitions, not per frame.
    """
    # Track lifecycle counters
    tracks_created: int = 0       # New track IDs assigned
    tracks_confirmed: int = 0     # Tracks that reached CONFIRMED phase
    tracks_recognized: int = 0    # Tracks that completed recognition
    
    # Recognition outcome counters (each track counted ONCE)
    authorized_count: int = 0
    wanted_count: int = 0
    unknown_count: int = 0
    
    # Current state
    active_tracks: int = 0
    
    def to_dict(self) -> dict:
        return {
            "tracks_created": self.tracks_created,
            "tracks_confirmed": self.tracks_confirmed,
            "tracks_recognized": self.tracks_recognized,
            "authorized_count": self.authorized_count,
            "wanted_count": self.wanted_count,
            "unknown_count": self.unknown_count,
            "active_tracks": self.active_tracks,
        }


class DeepSORTLiteTracker:
    """
    DeepSORT-lite tracker optimized for gate access control.
    
    Key behaviors:
    1. Stable track IDs across frames
    2. Phase-based lifecycle (TENTATIVE → CONFIRMED → RECOGNIZED)
    3. Recognition eligibility controlled by phase
    4. Statistics count unique people, not detections
    
    Assignment algorithm:
    - Hungarian assignment with HARD GATING
    - Invalid pairs (low IoU, high embedding distance) get infinite cost
    - This prevents bad matches that cause ID fragmentation
    
    Embedding rules:
    - TENTATIVE tracks: No embeddings used for matching
    - CONFIRMED tracks: Embeddings averaged over history, L2 normalized
    - Only embeddings from CONFIRMED tracks used in cost calculation
    
    Usage:
        tracker = DeepSORTLiteTracker(min_hits=3, max_age=30)
        
        # Each frame:
        tracks = tracker.update(detections)
        
        # Get tracks ready for recognition:
        for track in tracker.get_tracks_for_recognition():
            # Run recognition ONCE
            result = recognize(track)
            tracker.update_track_recognition(track.track_id, ...)
        
        # Get statistics:
        stats = tracker.get_statistics()
    """
    
    # Cost matrix constants
    COST_INVALID = 1e6  # Cost for invalid matches (prevents Hungarian from selecting)
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        max_embedding_distance: float = 0.6,
        embedding_weight: float = 0.3,
    ):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for valid match (HARD GATE)
            max_age: Max frames before removing unmatched track
            min_hits: Hits required to transition TENTATIVE → CONFIRMED
            max_embedding_distance: Max cosine distance for valid match (HARD GATE)
            embedding_weight: Weight of embedding in cost (0=IoU only, 1=embedding only)
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_embedding_distance = max_embedding_distance
        self.embedding_weight = embedding_weight
        
        # Track storage
        self._tracks: List[Track] = []
        self._next_id: int = 1
        
        # Statistics
        self._stats = TrackerStatistics()
        
        logger.info(
            f"DeepSORT-lite tracker initialized: "
            f"iou_threshold={iou_threshold}, min_hits={min_hits}, "
            f"max_age={max_age}, max_emb_dist={max_embedding_distance}"
        )
    
    def update(
        self,
        detections: List[Tuple[np.ndarray, float, Optional[np.ndarray]]]
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, score, embedding) tuples
                       bbox: np.array [x1, y1, x2, y2]
                       score: detection confidence
                       embedding: face embedding or None
        
        Returns:
            List of CONFIRMED tracks (for UI/gate control)
        
        Algorithm:
            1. Predict (increment time_since_update for all tracks)
            2. Build cost matrix with HARD GATING
            3. Hungarian assignment
            4. Process matches (update tracks)
            5. Handle unmatched detections (create new tracks)
            6. Handle unmatched tracks (increment time_since_update)
            7. Remove dead tracks (time_since_update > max_age)
            8. Return confirmed tracks
        """
        # ==========================================
        # STEP 1: PREDICT (age all tracks)
        # ==========================================
        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1
        
        if not detections:
            self._remove_dead_tracks()
            return self._get_confirmed_tracks()
        
        # ==========================================
        # STEP 2: BUILD COST MATRIX WITH HARD GATING
        # ==========================================
        cost_matrix = self._compute_cost_matrix(detections, self._tracks)
        
        # ==========================================
        # STEP 3: HUNGARIAN ASSIGNMENT
        # ==========================================
        matched_det_indices, matched_trk_indices = self._hungarian_assignment(cost_matrix)
        
        # Build match sets
        matched_dets: Set[int] = set()
        matched_trks: Set[int] = set()
        
        # ==========================================
        # STEP 4: PROCESS MATCHES
        # ==========================================
        for d_idx, t_idx in zip(matched_det_indices, matched_trk_indices):
            # Validate match (should already be valid due to hard gating, but double-check)
            det_bbox, det_score, det_embedding = detections[d_idx]
            track = self._tracks[t_idx]
            
            iou = self._compute_iou(det_bbox, track.bbox)
            if iou < self.iou_threshold:
                # Invalid match (shouldn't happen with proper hard gating)
                continue
            
            # Update track with detection
            self._update_track_with_detection(track, det_bbox, det_score, det_embedding)
            matched_dets.add(d_idx)
            matched_trks.add(t_idx)
        
        # ==========================================
        # STEP 5: CREATE NEW TRACKS FOR UNMATCHED DETECTIONS
        # ==========================================
        for d_idx, (bbox, score, embedding) in enumerate(detections):
            if d_idx not in matched_dets:
                self._create_track(bbox, score, embedding)
        
        # ==========================================
        # STEP 6 & 7: REMOVE DEAD TRACKS
        # ==========================================
        self._remove_dead_tracks()
        
        # Update active track count
        self._stats.active_tracks = len(self._tracks)
        
        return self._get_confirmed_tracks()
    
    def _compute_cost_matrix(
        self,
        detections: List[Tuple[np.ndarray, float, Optional[np.ndarray]]],
        tracks: List[Track]
    ) -> np.ndarray:
        """
        Compute cost matrix with HARD GATING.
        
        CRITICAL: Invalid matches get COST_INVALID, which ensures Hungarian
        algorithm will never select them. This prevents ID fragmentation.
        
        Cost formula for valid matches:
        - TENTATIVE tracks: IoU cost only (no embeddings)
        - CONFIRMED tracks: Weighted IoU + embedding cost
        """
        n_det = len(detections)
        n_trk = len(tracks)
        
        if n_det == 0 or n_trk == 0:
            return np.zeros((n_det, n_trk))
        
        cost_matrix = np.full((n_det, n_trk), self.COST_INVALID, dtype=np.float64)
        
        for d_idx, (det_bbox, det_score, det_emb) in enumerate(detections):
            for t_idx, track in enumerate(tracks):
                # ========================================
                # HARD GATE 1: IoU threshold
                # ========================================
                iou = self._compute_iou(det_bbox, track.bbox)
                if iou < self.iou_threshold:
                    # INVALID: IoU too low, cannot be a match
                    continue
                
                iou_cost = 1.0 - iou  # Convert to cost (lower is better)
                
                # ========================================
                # PHASE-BASED COST CALCULATION
                # ========================================
                if track.phase == TrackPhase.TENTATIVE:
                    # TENTATIVE tracks: IoU only (no embeddings)
                    # Why: Embeddings unreliable with few observations
                    cost_matrix[d_idx, t_idx] = iou_cost
                    
                else:
                    # CONFIRMED/RECOGNIZED tracks: IoU + embedding
                    if det_emb is not None and track.embedding is not None:
                        # Compute embedding distance (cosine)
                        similarity = np.dot(det_emb, track.embedding)
                        emb_distance = 1.0 - similarity
                        
                        # ========================================
                        # HARD GATE 2: Embedding distance threshold
                        # ========================================
                        if emb_distance > self.max_embedding_distance:
                            # INVALID: Embedding too different
                            continue
                        
                        # Combined cost (weighted)
                        cost_matrix[d_idx, t_idx] = (
                            (1.0 - self.embedding_weight) * iou_cost +
                            self.embedding_weight * emb_distance
                        )
                    else:
                        # No embedding available, use IoU only
                        cost_matrix[d_idx, t_idx] = iou_cost
        
        return cost_matrix
    
    def _hungarian_assignment(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Hungarian assignment on cost matrix.
        
        Returns matched (detection_indices, track_indices) pairs.
        Pairs with COST_INVALID are filtered out.
        """
        if cost_matrix.size == 0:
            return np.array([]), np.array([])
        
        n_det, n_trk = cost_matrix.shape
        
        if n_det == 0 or n_trk == 0:
            return np.array([]), np.array([])
        
        if SCIPY_AVAILABLE:
            # Optimal assignment using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            # Fallback: greedy assignment
            row_indices, col_indices = self._greedy_assignment(cost_matrix)
        
        # Filter out invalid matches (cost >= COST_INVALID threshold)
        valid_mask = cost_matrix[row_indices, col_indices] < self.COST_INVALID * 0.5
        
        return row_indices[valid_mask], col_indices[valid_mask]
    
    def _greedy_assignment(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback greedy assignment when scipy unavailable.
        
        Not optimal but works for low FPS scenarios.
        """
        n_det, n_trk = cost_matrix.shape
        row_indices = []
        col_indices = []
        
        used_cols = set()
        
        # Sort by minimum cost per detection
        det_order = np.argsort(cost_matrix.min(axis=1))
        
        for d_idx in det_order:
            min_cost = self.COST_INVALID
            best_t = -1
            
            for t_idx in range(n_trk):
                if t_idx not in used_cols:
                    cost = cost_matrix[d_idx, t_idx]
                    if cost < min_cost:
                        min_cost = cost
                        best_t = t_idx
            
            if best_t >= 0 and min_cost < self.COST_INVALID * 0.5:
                row_indices.append(d_idx)
                col_indices.append(best_t)
                used_cols.add(best_t)
        
        return np.array(row_indices), np.array(col_indices)
    
    def _update_track_with_detection(
        self,
        track: Track,
        bbox: np.ndarray,
        score: float,
        embedding: Optional[np.ndarray]
    ):
        """
        Update track state with matched detection.
        
        Handles phase transitions:
        - TENTATIVE → CONFIRMED when hits >= min_hits
        """
        # Update position
        track.bbox = bbox
        track.score = score
        track.hits += 1
        track.time_since_update = 0
        
        # Update embedding (only for CONFIRMED tracks)
        # Why: Tentative tracks have unreliable embeddings
        if track.phase != TrackPhase.TENTATIVE:
            if embedding is not None:
                track.embedding_history.append(embedding)
                
                # Average embeddings for robustness
                if len(track.embedding_history) > 0:
                    track.embedding = np.mean(
                        list(track.embedding_history), axis=0
                    )
                    # L2 normalize (required for cosine similarity)
                    norm = np.linalg.norm(track.embedding)
                    if norm > 0:
                        track.embedding = track.embedding / norm
        
        # ========================================
        # PHASE TRANSITION: TENTATIVE → CONFIRMED
        # ========================================
        if track.phase == TrackPhase.TENTATIVE:
            if track.hits >= self.min_hits:
                track.phase = TrackPhase.CONFIRMED
                self._stats.tracks_confirmed += 1
                
                # Initialize embedding now that track is confirmed
                if embedding is not None:
                    track.embedding = embedding
                    track.embedding_history.append(embedding)
                
                logger.debug(
                    f"Track {track.track_id} CONFIRMED after {track.hits} hits"
                )
    
    def _create_track(
        self,
        bbox: np.ndarray,
        score: float,
        embedding: Optional[np.ndarray]
    ) -> Track:
        """
        Create new track for unmatched detection.
        
        New tracks start as TENTATIVE and must prove themselves
        through consistent IoU matches before becoming CONFIRMED.
        """
        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            score=score,
            phase=TrackPhase.TENTATIVE,
            # Don't store embedding for tentative track
        )
        
        self._tracks.append(track)
        self._next_id += 1
        self._stats.tracks_created += 1
        
        logger.debug(f"Track {track.track_id} CREATED (tentative)")
        
        return track
    
    def _remove_dead_tracks(self):
        """
        Remove tracks that haven't been matched for too long.
        
        Conservative: Only remove if time_since_update > max_age
        This prevents premature deletion of confirmed tracks.
        """
        before_count = len(self._tracks)
        
        self._tracks = [
            track for track in self._tracks
            if track.time_since_update <= self.max_age
        ]
        
        removed = before_count - len(self._tracks)
        if removed > 0:
            logger.debug(f"Removed {removed} dead tracks")
    
    def _get_confirmed_tracks(self) -> List[Track]:
        """
        Get tracks that are CONFIRMED or RECOGNIZED.
        
        Only confirmed tracks should be:
        - Displayed in UI
        - Used for gate decisions
        - Passed to recognition
        """
        return [
            track for track in self._tracks
            if track.phase in (TrackPhase.CONFIRMED, TrackPhase.RECOGNIZED)
        ]
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    # ==========================================
    # PUBLIC API FOR RECOGNITION
    # ==========================================
    
    def get_tracks_for_recognition(self) -> List[Track]:
        """
        Get tracks that need recognition.
        
        Returns tracks that are:
        - CONFIRMED phase
        - NOT yet recognized
        
        CRITICAL: This is how you get tracks to recognize.
        Do NOT run recognition per frame - run it per track returned here.
        """
        return [
            track for track in self._tracks
            if track.is_ready_for_recognition()
        ]
    
    def update_track_recognition(
        self,
        track_id: int,
        face_id: Optional[str],
        user_id: Optional[str],
        name: Optional[str],
        status: str,
        confidence: float
    ) -> bool:
        """
        Update track with recognition result.
        
        This marks the track as RECOGNIZED and prevents future recognition.
        
        Args:
            track_id: Track to update
            face_id: Database face ID (or None if not found)
            user_id: User ID (or None if not found)
            name: Person name (or None if not found)
            status: "AUTHORIZED", "UNKNOWN", or "WANTED"
            confidence: Match confidence (0-1)
        
        Returns:
            True if track was updated, False if track not found
        """
        track = self.get_track(track_id)
        if not track:
            logger.warning(f"Cannot update recognition: track {track_id} not found")
            return False
        
        if track.recognized:
            logger.warning(f"Track {track_id} already recognized, ignoring update")
            return False
        
        # Mark as recognized (FINAL)
        track.mark_recognized(face_id, user_id, name, status, confidence)
        
        # Update statistics
        self._stats.tracks_recognized += 1
        
        if status == "AUTHORIZED":
            self._stats.authorized_count += 1
        elif status == "WANTED":
            self._stats.wanted_count += 1
        else:  # UNKNOWN
            self._stats.unknown_count += 1
        
        return True
    
    def record_recognition_attempt(self, track_id: int):
        """Record a recognition attempt for a track."""
        track = self.get_track(track_id)
        if track:
            track.recognition_attempts += 1
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (including tentative)."""
        return self._tracks.copy()
    
    def get_all_active_tracks(self) -> List[Track]:
        """
        Get all active tracks (CONFIRMED or RECOGNIZED).
        
        Used for skip-detection optimization:
        - If all active tracks are recognized and stable, skip detection
        """
        return [
            track for track in self._tracks
            if track.phase in (TrackPhase.CONFIRMED, TrackPhase.RECOGNIZED)
        ]
    
    def get_statistics(self) -> TrackerStatistics:
        """Get tracker statistics."""
        self._stats.active_tracks = len(self._tracks)
        return self._stats
    
    def clear(self):
        """Clear all tracks and reset statistics."""
        self._tracks.clear()
        self._next_id = 1
        self._stats = TrackerStatistics()


# ==========================================
# BACKWARDS COMPATIBILITY ALIAS
# ==========================================
SimpleTracker = DeepSORTLiteTracker

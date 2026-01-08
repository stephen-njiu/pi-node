"""
Simple Face Tracker using IoU-based assignment.
Tracks faces across frames to maintain identity consistency.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked face."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    
    # Recognition state
    embedding: Optional[np.ndarray] = None
    face_id: Optional[str] = None
    user_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None  # AUTHORIZED, UNKNOWN, WANTED
    confidence: float = 0.0
    
    # Tracking state
    age: int = 0  # Frames since creation
    hits: int = 1  # Consecutive detections
    time_since_update: int = 0  # Frames since last match
    
    # Recognition attempts
    recognition_attempts: int = 0
    recognized: bool = False
    
    # Embedding history for averaging
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=5))


class SimpleTracker:
    """
    Simple IoU-based multi-object tracker.
    Maintains track IDs across frames for consistent face identity.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,  # Max frames to keep lost track
        min_hits: int = 3,  # Min hits before track is confirmed
        embedding_weight: float = 0.3  # Weight for embedding similarity in assignment
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.embedding_weight = embedding_weight
        
        self._tracks: list[Track] = []
        self._next_id = 1
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
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
    
    def _compute_cost_matrix(
        self,
        detections: list[tuple],  # (bbox, score, embedding)
        tracks: list[Track]
    ) -> np.ndarray:
        """
        Compute cost matrix for Hungarian assignment.
        Uses IoU and optionally embedding similarity.
        """
        n_det = len(detections)
        n_trk = len(tracks)
        
        if n_det == 0 or n_trk == 0:
            return np.zeros((n_det, n_trk))
        
        cost_matrix = np.zeros((n_det, n_trk))
        
        for d, (det_bbox, _, det_emb) in enumerate(detections):
            for t, track in enumerate(tracks):
                # IoU cost (lower is better, so use 1 - IoU)
                iou = self._compute_iou(det_bbox, track.bbox)
                iou_cost = 1 - iou
                
                # Embedding similarity cost (if available)
                emb_cost = 1.0
                if det_emb is not None and track.embedding is not None:
                    similarity = np.dot(det_emb, track.embedding)
                    emb_cost = 1 - similarity
                
                # Combined cost
                if det_emb is not None and track.embedding is not None:
                    cost_matrix[d, t] = (
                        (1 - self.embedding_weight) * iou_cost +
                        self.embedding_weight * emb_cost
                    )
                else:
                    cost_matrix[d, t] = iou_cost
        
        return cost_matrix
    
    def update(
        self,
        detections: list[tuple]  # List of (bbox, score, embedding)
    ) -> list[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, score, embedding) tuples
                       bbox: [x1, y1, x2, y2]
                       score: detection confidence
                       embedding: face embedding or None
        
        Returns:
            List of active tracks (confirmed only)
        """
        # Predict (simple: just age tracks)
        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1
        
        if not detections:
            # Remove dead tracks
            self._tracks = [t for t in self._tracks if t.time_since_update < self.max_age]
            return self._get_confirmed_tracks()
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(detections, self._tracks)
        
        # Hungarian assignment
        if linear_sum_assignment is not None and len(self._tracks) > 0:
            det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        else:
            # Fallback: greedy assignment
            det_indices, trk_indices = self._greedy_assignment(cost_matrix)
        
        # Process matches
        matched_dets = set()
        matched_trks = set()
        
        for d, t in zip(det_indices, trk_indices):
            if len(self._tracks) > 0:
                # Check if match is valid (IoU above threshold)
                det_bbox = detections[d][0]
                iou = self._compute_iou(det_bbox, self._tracks[t].bbox)
                
                if iou >= self.iou_threshold:
                    # Update track
                    track = self._tracks[t]
                    track.bbox = det_bbox
                    track.score = detections[d][1]
                    track.hits += 1
                    track.time_since_update = 0
                    
                    # Update embedding (exponential moving average)
                    det_emb = detections[d][2]
                    if det_emb is not None:
                        track.embedding_history.append(det_emb)
                        if track.embedding is None:
                            track.embedding = det_emb
                        else:
                            # Average recent embeddings
                            track.embedding = np.mean(
                                list(track.embedding_history), axis=0
                            )
                            # Re-normalize
                            norm = np.linalg.norm(track.embedding)
                            if norm > 0:
                                track.embedding = track.embedding / norm
                    
                    matched_dets.add(d)
                    matched_trks.add(t)
        
        # Create new tracks for unmatched detections
        for d, (bbox, score, embedding) in enumerate(detections):
            if d not in matched_dets:
                track = Track(
                    track_id=self._next_id,
                    bbox=bbox,
                    score=score,
                    embedding=embedding
                )
                if embedding is not None:
                    track.embedding_history.append(embedding)
                
                self._tracks.append(track)
                self._next_id += 1
        
        # Remove dead tracks
        self._tracks = [
            t for i, t in enumerate(self._tracks)
            if i in matched_trks or t.time_since_update < self.max_age
        ]
        
        return self._get_confirmed_tracks()
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> tuple:
        """Fallback greedy assignment when scipy unavailable."""
        if cost_matrix.size == 0:
            return np.array([]), np.array([])
        
        det_indices = []
        trk_indices = []
        
        n_det, n_trk = cost_matrix.shape
        used_trks = set()
        
        # Sort detections by minimum cost
        for d in range(n_det):
            min_cost = float('inf')
            best_t = -1
            
            for t in range(n_trk):
                if t not in used_trks and cost_matrix[d, t] < min_cost:
                    min_cost = cost_matrix[d, t]
                    best_t = t
            
            if best_t >= 0 and min_cost < 1.0:  # Reasonable threshold
                det_indices.append(d)
                trk_indices.append(best_t)
                used_trks.add(best_t)
        
        return np.array(det_indices), np.array(trk_indices)
    
    def _get_confirmed_tracks(self) -> list[Track]:
        """Get tracks that have been confirmed (enough hits)."""
        return [t for t in self._tracks if t.hits >= self.min_hits or t.recognized]
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def update_track_recognition(
        self,
        track_id: int,
        face_id: Optional[str],
        user_id: Optional[str],
        name: Optional[str],
        status: str,
        confidence: float
    ):
        """Update track with recognition result."""
        track = self.get_track(track_id)
        if track:
            track.face_id = face_id
            track.user_id = user_id
            track.name = name
            track.status = status
            track.confidence = confidence
            track.recognition_attempts += 1
            track.recognized = True
    
    def get_all_tracks(self) -> list[Track]:
        """Get all tracks including unconfirmed."""
        return self._tracks.copy()
    
    def clear(self):
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1

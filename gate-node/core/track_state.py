"""
Track State Manager - Prevents repeated actions for the same tracked face.
Implements cooldown logic so we don't keep opening/alerting for same person.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set
import logging


logger = logging.getLogger(__name__)


class TrackStatus(Enum):
    """Status of a tracked face."""
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    UNKNOWN = "UNKNOWN"
    WANTED = "WANTED"


@dataclass
class TrackState:
    """State information for a tracked face."""
    track_id: int
    status: TrackStatus = TrackStatus.PENDING
    person_id: Optional[str] = None
    metadata: Optional[dict] = None
    confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_attempt_time: float = 0.0
    attempt_count: int = 0
    recognized: bool = False
    cooldown_until: float = 0.0


class TrackStateManager:
    """
    Manages state for tracked faces to prevent repeated actions.
    
    Key responsibilities:
    - Track when we last took action for a face
    - Implement cooldown periods between actions
    - Prevent spam of gate opens/alerts for same person
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        attempt_interval: float = 0.5,
        cooldown: float = 30.0,
        cooldown_seconds: float = None,  # Alias for cooldown
    ):
        self.max_attempts = max_attempts
        self.attempt_interval = attempt_interval
        self.cooldown = cooldown_seconds if cooldown_seconds is not None else cooldown
        
        self._states: dict[int, TrackState] = {}  # track_id -> TrackState
        self._lock = threading.Lock()
    
    def should_attempt_recognition(self, track_id: int) -> bool:
        """
        Check if we should attempt recognition for this track.
        Returns False if track is recognized, in cooldown, or max attempts reached.
        """
        with self._lock:
            if track_id not in self._states:
                # New track - create state
                self._states[track_id] = TrackState(track_id=track_id)
                return True
            
            state = self._states[track_id]
            now = time.time()
            
            # Already recognized successfully
            if state.recognized:
                return False
            
            # In cooldown
            if now < state.cooldown_until:
                return False
            
            # Max attempts reached
            if state.attempt_count >= self.max_attempts:
                return False
            
            # Check attempt interval
            if now - state.last_attempt_time < self.attempt_interval:
                return False
            
            return True
    
    def record_attempt(self, track_id: int):
        """Record a recognition attempt for a track."""
        with self._lock:
            if track_id not in self._states:
                self._states[track_id] = TrackState(track_id=track_id)
            
            state = self._states[track_id]
            state.attempt_count += 1
            state.last_attempt_time = time.time()
    
    def record_success(
        self,
        track_id: int,
        status: TrackStatus,
        person_id: str,
        metadata: dict,
        confidence: float
    ):
        """Record successful recognition."""
        with self._lock:
            if track_id not in self._states:
                self._states[track_id] = TrackState(track_id=track_id)
            
            state = self._states[track_id]
            state.status = status
            state.person_id = person_id
            state.metadata = metadata
            state.confidence = confidence
            state.recognized = True
            state.cooldown_until = time.time() + self.cooldown
            
            logger.debug(f"Track {track_id}: Recognized as {status.value} ({person_id})")
    
    def record_failure(self, track_id: int):
        """Record failed recognition attempt."""
        with self._lock:
            if track_id not in self._states:
                self._states[track_id] = TrackState(track_id=track_id)
            
            state = self._states[track_id]
            
            # If max attempts reached, mark as unknown
            if state.attempt_count >= self.max_attempts:
                state.status = TrackStatus.UNKNOWN
                state.recognized = True
                state.cooldown_until = time.time() + self.cooldown
                logger.debug(f"Track {track_id}: Max attempts reached, marking as UNKNOWN")
    
    def get_state(self, track_id: int) -> Optional[TrackState]:
        """Get current state for a track."""
        with self._lock:
            return self._states.get(track_id)
    
    def cleanup_stale(self, active_track_ids: Set[int]):
        """Remove states for tracks that are no longer active."""
        with self._lock:
            stale_ids = set(self._states.keys()) - active_track_ids
            for track_id in stale_ids:
                del self._states[track_id]
            
            if stale_ids:
                logger.debug(f"Cleaned up {len(stale_ids)} stale track states")
    
    def cleanup_old_states(self, max_age_seconds: float = 300.0):
        """Remove states older than max_age_seconds."""
        with self._lock:
            now = time.time()
            to_remove = []
            
            for track_id, state in self._states.items():
                if now - state.first_seen > max_age_seconds:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self._states[track_id]
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} old track states")
    
    # Legacy methods for backwards compatibility
    def should_process(self, track_id: int) -> bool:
        """Alias for should_attempt_recognition."""
        return self.should_attempt_recognition(track_id)
    
    def record_decision(
        self,
        track_id: int,
        status: str,
        face_id: Optional[str] = None,
        name: Optional[str] = None,
        confidence: float = 0.0
    ) -> bool:
        """Legacy method - record a decision."""
        try:
            track_status = TrackStatus(status)
        except ValueError:
            track_status = TrackStatus.UNKNOWN
        
        metadata = {"full_name": name} if name else None
        self.record_success(track_id, track_status, face_id, metadata, confidence)
        return True
    
    def get_active_count(self) -> int:
        """Get number of active track states."""
        with self._lock:
            return len(self._states)
    
    def get_stats(self) -> dict:
        """Get state manager statistics."""
        with self._lock:
            status_counts = {}
            for state in self._states.values():
                status_counts[state.status.value] = status_counts.get(state.status.value, 0) + 1
            
            return {
                "active_tracks": len(self._states),
                "status_counts": status_counts
            }
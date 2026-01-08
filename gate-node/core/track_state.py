"""
Track State Manager - Prevents repeated actions for the same tracked face.
Implements cooldown logic so we don't keep opening/alerting for same person.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional
import logging


logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State information for a tracked face."""
    track_id: int
    status: str  # AUTHORIZED, UNKNOWN, WANTED
    face_id: Optional[str]
    name: Optional[str]
    confidence: float
    first_seen: float
    last_action_time: float
    action_count: int
    cooldown_until: float


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
        cooldown_seconds: float = 30.0,
        max_attempts: int = 3
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_attempts = max_attempts
        
        self._states: dict[int, TrackState] = {}  # track_id -> TrackState
        self._lock = threading.Lock()
    
    def should_process(self, track_id: int) -> bool:
        """
        Check if we should process recognition for this track.
        Returns False if track is in cooldown or max attempts reached.
        """
        with self._lock:
            if track_id not in self._states:
                return True
            
            state = self._states[track_id]
            now = time.time()
            
            # Check cooldown
            if now < state.cooldown_until:
                return False
            
            # Check max attempts (only for UNKNOWN)
            if state.status == "UNKNOWN" and state.action_count >= self.max_attempts:
                return False
            
            return True
    
    def record_decision(
        self,
        track_id: int,
        status: str,
        face_id: Optional[str] = None,
        name: Optional[str] = None,
        confidence: float = 0.0
    ) -> bool:
        """
        Record a decision for a track.
        Returns True if this is a new decision (action should be taken).
        Returns False if duplicate (already handled).
        """
        with self._lock:
            now = time.time()
            
            if track_id in self._states:
                state = self._states[track_id]
                
                # Check if in cooldown
                if now < state.cooldown_until:
                    return False
                
                # Check if same status (no change)
                if state.status == status and status == "AUTHORIZED":
                    # Don't repeat gate opens
                    return False
                
                # Update state
                state.status = status
                state.face_id = face_id
                state.name = name
                state.confidence = confidence
                state.last_action_time = now
                state.action_count += 1
                state.cooldown_until = now + self.cooldown_seconds
                
                logger.debug(f"Track {track_id}: Updated to {status} (count: {state.action_count})")
                return True
            
            else:
                # New track
                self._states[track_id] = TrackState(
                    track_id=track_id,
                    status=status,
                    face_id=face_id,
                    name=name,
                    confidence=confidence,
                    first_seen=now,
                    last_action_time=now,
                    action_count=1,
                    cooldown_until=now + self.cooldown_seconds
                )
                
                logger.debug(f"Track {track_id}: New track with {status}")
                return True
    
    def get_state(self, track_id: int) -> Optional[TrackState]:
        """Get current state for a track."""
        with self._lock:
            return self._states.get(track_id)
    
    def clear_track(self, track_id: int):
        """Remove state for a track (when track is lost)."""
        with self._lock:
            if track_id in self._states:
                del self._states[track_id]
    
    def cleanup_old_states(self, max_age_seconds: float = 300.0):
        """Remove states older than max_age_seconds."""
        with self._lock:
            now = time.time()
            to_remove = []
            
            for track_id, state in self._states.items():
                if now - state.last_action_time > max_age_seconds:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self._states[track_id]
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} old track states")
    
    def get_active_count(self) -> int:
        """Get number of active track states."""
        with self._lock:
            return len(self._states)
    
    def get_stats(self) -> dict:
        """Get state manager statistics."""
        with self._lock:
            status_counts = {}
            for state in self._states.values():
                status_counts[state.status] = status_counts.get(state.status, 0) + 1
            
            return {
                "active_tracks": len(self._states),
                "status_counts": status_counts
            }

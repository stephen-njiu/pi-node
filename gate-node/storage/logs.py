"""
Access Logger - SQLite-based local logging for access events.
Stores all gate access attempts with face crops for audit trail.
"""

import sqlite3
import os
import threading
import base64
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import cv2


logger = logging.getLogger(__name__)


@dataclass
class AccessEvent:
    """Represents an access event."""
    id: Optional[int]
    timestamp: str
    gate_id: str
    track_id: int
    face_id: Optional[str]
    user_id: Optional[str]
    name: Optional[str]
    status: str  # AUTHORIZED, UNKNOWN, WANTED
    decision: str  # OPEN, CLOSE
    confidence: float
    face_crop_b64: Optional[str]  # Base64 encoded JPEG
    synced: bool


class AccessLogger:
    """
    SQLite-based access logger.
    Stores all access events locally for:
    - Audit trail
    - Offline operation
    - Sync to backend when online
    """
    
    def __init__(self, db_path: str = "data/logs.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    gate_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    face_id TEXT,
                    user_id TEXT,
                    name TEXT,
                    status TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    face_crop_b64 TEXT,
                    synced INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON access_events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_synced ON access_events(synced)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON access_events(status)
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized access log database at {self.db_path}")
    
    def _encode_face_crop(self, frame: np.ndarray, bbox: tuple) -> Optional[str]:
        """Extract and encode face crop as base64 JPEG."""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Add some margin
            h, w = frame.shape[:2]
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
        
        except Exception as e:
            logger.error(f"Failed to encode face crop: {e}")
            return None
    
    def log_access(
        self,
        gate_id: str,
        track_id: int,
        status: str,
        decision: str,
        confidence: float,
        face_id: Optional[str] = None,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
        bbox: Optional[tuple] = None
    ) -> int:
        """
        Log an access event.
        
        Args:
            gate_id: Gate identifier
            track_id: Track ID from tracker
            status: AUTHORIZED, UNKNOWN, or WANTED
            decision: OPEN or CLOSE
            confidence: Recognition confidence (0-1)
            face_id: Matched face ID if any
            user_id: User ID if matched
            name: User name if matched
            frame: Video frame for face crop
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Event ID
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Encode face crop if provided
        face_crop_b64 = None
        if frame is not None and bbox is not None:
            face_crop_b64 = self._encode_face_crop(frame, bbox)
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO access_events 
                (timestamp, gate_id, track_id, face_id, user_id, name, status, decision, confidence, face_crop_b64, synced)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                timestamp, gate_id, track_id, face_id, user_id, name,
                status, decision, confidence, face_crop_b64
            ))
            
            event_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Logged access event #{event_id}: {status} -> {decision}")
            return event_id
    
    def get_unsynced_events(self, limit: int = 100) -> list[AccessEvent]:
        """Get events that haven't been synced to backend."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, timestamp, gate_id, track_id, face_id, user_id, name,
                       status, decision, confidence, face_crop_b64, synced
                FROM access_events
                WHERE synced = 0
                ORDER BY timestamp ASC
                LIMIT ?
            """, (limit,))
            
            events = []
            for row in cursor.fetchall():
                events.append(AccessEvent(
                    id=row[0],
                    timestamp=row[1],
                    gate_id=row[2],
                    track_id=row[3],
                    face_id=row[4],
                    user_id=row[5],
                    name=row[6],
                    status=row[7],
                    decision=row[8],
                    confidence=row[9],
                    face_crop_b64=row[10],
                    synced=bool(row[11])
                ))
            
            conn.close()
            return events
    
    def mark_synced(self, event_ids: list[int]):
        """Mark events as synced."""
        if not event_ids:
            return
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ",".join("?" * len(event_ids))
            cursor.execute(f"""
                UPDATE access_events
                SET synced = 1
                WHERE id IN ({placeholders})
            """, event_ids)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Marked {len(event_ids)} events as synced")
    
    def get_recent_events(
        self,
        limit: int = 50,
        status_filter: Optional[str] = None
    ) -> list[AccessEvent]:
        """Get recent access events for display."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if status_filter:
                cursor.execute("""
                    SELECT id, timestamp, gate_id, track_id, face_id, user_id, name,
                           status, decision, confidence, face_crop_b64, synced
                    FROM access_events
                    WHERE status = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (status_filter, limit))
            else:
                cursor.execute("""
                    SELECT id, timestamp, gate_id, track_id, face_id, user_id, name,
                           status, decision, confidence, face_crop_b64, synced
                    FROM access_events
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            events = []
            for row in cursor.fetchall():
                events.append(AccessEvent(
                    id=row[0],
                    timestamp=row[1],
                    gate_id=row[2],
                    track_id=row[3],
                    face_id=row[4],
                    user_id=row[5],
                    name=row[6],
                    status=row[7],
                    decision=row[8],
                    confidence=row[9],
                    face_crop_b64=row[10],
                    synced=bool(row[11])
                ))
            
            conn.close()
            return events
    
    def get_stats(self) -> dict:
        """Get logging statistics."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total events
            cursor.execute("SELECT COUNT(*) FROM access_events")
            total = cursor.fetchone()[0]
            
            # Unsynced events
            cursor.execute("SELECT COUNT(*) FROM access_events WHERE synced = 0")
            unsynced = cursor.fetchone()[0]
            
            # Events by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM access_events 
                GROUP BY status
            """)
            by_status = dict(cursor.fetchall())
            
            # Today's events
            cursor.execute("""
                SELECT COUNT(*) FROM access_events
                WHERE date(timestamp) = date('now')
            """)
            today = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_events": total,
                "unsynced_events": unsynced,
                "events_by_status": by_status,
                "today_events": today
            }
    
    def cleanup_old_events(self, days: int = 30):
        """Delete events older than specified days (keeps DB small)."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM access_events
                WHERE synced = 1
                AND datetime(timestamp) < datetime('now', ?)
            """, (f"-{days} days",))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old events")
            
            return deleted

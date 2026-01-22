"""
Sync Thread - Periodically syncs face database with backend.
Handles offline resilience and log upload.
"""

import threading
import time
import requests
import logging
from typing import Optional

import numpy as np

from storage import FaceDatabase


logger = logging.getLogger(__name__)


class SyncThread(threading.Thread):
    """
    Background thread for syncing with backend.
    
    Responsibilities:
    - Periodically fetch updated faces from backend
    - Upload unsynced access logs
    - Handle network failures gracefully
    """
    
    def __init__(
        self,
        face_db: FaceDatabase,
        backend_url: str,
        org_id: str,
        interval_seconds: float = 10,
        version_file: str = "data/sync_version.txt",
    ):
        super().__init__(name="SyncThread", daemon=True)
        
        self.face_db = face_db
        self.backend_url = backend_url.rstrip("/")
        self.org_id = org_id
        self.sync_interval = interval_seconds
        self.version_file = version_file
        
        self._stop_event = threading.Event()
        self._last_face_sync = 0.0
        
        # Stats
        self.last_sync_success = False
        self.last_sync_time: Optional[float] = None
        self.sync_error: Optional[str] = None
    
    def run(self):
        """Main sync loop."""
        logger.info("Sync thread started")
        
        # Print initial database status
        initial_count = self.face_db.count()
        initial_version = self.face_db.get_version()
        logger.info(f"[INIT] HNSW database: {initial_count} faces, version: {initial_version}")
        print(f"[INIT] HNSW database: {initial_count} faces, version: {initial_version}")
        
        # Initial sync on startup
        self._sync_faces()
        
        while not self._stop_event.is_set():
            now = time.time()
            
            # Check if face sync needed
            if now - self._last_face_sync >= self.sync_interval:
                self._sync_faces()
            
            # Sleep briefly
            self._stop_event.wait(timeout=5.0)
        
        logger.info("Sync thread stopped")
    
    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()
    
    def _sync_faces(self):
        """Sync faces from backend."""
        self._last_face_sync = time.time()
        
        try:
            current_version = self.face_db.get_version()
            
            url = f"{self.backend_url}/api/v1/faces/sync"
            params = {
                "org_id": self.org_id,
            }
            
            # Only include 'since' for delta sync (not first sync)
            if current_version and current_version != "0":
                params["since"] = current_version
            
            logger.info(f"Syncing faces from {url} (current version: {current_version})")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Debug: log raw response structure
            logger.debug(f"Sync response keys: {data.keys()}")
            
            # API returns: { version, upserts, deletes, count }
            upserts = data.get("upserts", [])
            deletes = data.get("deletes", [])
            new_version = data.get("version", current_version)
            
            logger.info(f"API returned: {len(upserts)} upserts, {len(deletes)} deletes, version: {new_version}")
            
            if upserts or deletes:
                # Debug: verify embedding format
                if upserts:
                    first_emb = upserts[0].get("embedding", [])
                    logger.debug(f"First embedding type: {type(first_emb)}, len: {len(first_emb) if first_emb else 0}")
                    if first_emb:
                        logger.debug(f"First embedding sample: {first_emb[:3]}... (first 3 values)")
                
                # Handle deletes first
                for face_id in deletes:
                    self.face_db.remove_face(face_id)
                
                # Handle upserts - add/update each face individually (delta sync)
                # Don't use sync_from_backend() as that clears all data (full sync only)
                added_count = 0
                for item in upserts:
                    try:
                        embedding = np.array(item["embedding"], dtype=np.float32)
                        success = self.face_db.add_face(
                            face_id=item["id"],
                            user_id=item.get("person_id"),
                            name=item["full_name"],
                            status=item["status"],
                            embedding=embedding
                        )
                        if success:
                            added_count += 1
                    except Exception as e:
                        logger.error(f"Failed to add face {item.get('id')}: {e}")
                
                logger.info(f"Added/updated {added_count}/{len(upserts)} faces")
                
                # Save to disk after batch add
                self.face_db.save()
                
                # Update version (this also saves to disk)
                self.face_db.set_version(new_version)
                
                # Print final count
                final_count = self.face_db.count()
                logger.info(f"[SYNC OK] {len(upserts)} upserts, {len(deletes)} deletes -> {final_count} faces in DB, version {new_version}")
                print(f"[SYNC OK] {len(upserts)} upserts, {len(deletes)} deletes -> {final_count} faces in DB, version {new_version}")
            else:
                logger.info("No updates from backend (database up to date)")
            
            self.last_sync_success = True
            self.last_sync_time = time.time()
            self.sync_error = None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Face sync failed (network): {e}")
            self.last_sync_success = False
            self.sync_error = str(e)
            
        except Exception as e:
            logger.error(f"Face sync failed: {e}")
            self.last_sync_success = False
            self.sync_error = str(e)
    
    def _upload_logs(self):
        """Upload unsynced access logs to backend."""
        self._last_log_upload = time.time()
        
        try:
            events = self.access_logger.get_unsynced_events(limit=50)
            
            if not events:
                return
            
            url = f"{self.backend_url}/api/v1/access-logs"
            
            # Prepare payload
            logs = []
            for event in events:
                logs.append({
                    "timestamp": event.timestamp,
                    "gate_id": event.gate_id,
                    "track_id": event.track_id,
                    "face_id": event.face_id,
                    "user_id": event.user_id,
                    "name": event.name,
                    "status": event.status,
                    "decision": event.decision,
                    "confidence": event.confidence,
                    "face_crop_b64": event.face_crop_b64
                })
            
            response = requests.post(
                url,
                json={"logs": logs},
                timeout=30
            )
            response.raise_for_status()
            
            # Mark as synced
            event_ids = [e.id for e in events if e.id]
            self.access_logger.mark_synced(event_ids)
            
            logger.info(f"Uploaded {len(events)} access logs")
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Log upload failed (network): {e}")
            
        except Exception as e:
            logger.error(f"Log upload failed: {e}")
    
    def force_sync(self):
        """Force immediate face sync."""
        self._last_face_sync = 0  # Reset timer
        logger.info("Force sync triggered")
    
    def get_status(self) -> dict:
        """Get sync status."""
        return {
            "last_sync_success": self.last_sync_success,
            "last_sync_time": self.last_sync_time,
            "sync_error": self.sync_error,
            "face_db_version": self.face_db.get_version(),
            "face_db_stats": self.face_db.get_stats(),
            "log_stats": self.access_logger.get_stats()
        }

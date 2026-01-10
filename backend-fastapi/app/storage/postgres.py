"""
Postgres storage for face embeddings (Pi sync).

Uses asyncpg for async database operations.
Stores embeddings as binary (bytea) for efficient storage and retrieval.
"""
from __future__ import annotations

import asyncpg
import struct
from typing import Optional, List, Dict, Any
from datetime import datetime


class PostgresStore:
    """Async Postgres connection pool for face embeddings."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Initialize connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    @staticmethod
    def embedding_to_bytes(embedding: List[float]) -> bytes:
        """
        Convert 512-float embedding to binary.
        Uses little-endian float32 format (2048 bytes total).
        """
        return struct.pack(f'<{len(embedding)}f', *embedding)
    
    @staticmethod
    def bytes_to_embedding(data: bytes) -> List[float]:
        """Convert binary back to float list."""
        count = len(data) // 4  # 4 bytes per float32
        return list(struct.unpack(f'<{count}f', data))
    
    async def upsert_face(
        self,
        face_id: str,
        org_id: str,
        full_name: str,
        embedding: List[float],
        email: Optional[str] = None,
        role: Optional[str] = None,
        status: str = "AUTHORIZED",
        image_url: Optional[str] = None,
        notes: Optional[str] = None,
        person_id: Optional[str] = None,
    ) -> None:
        """
        Insert or update a face record.
        Uses ON CONFLICT to handle duplicates.
        """
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        embedding_bytes = self.embedding_to_bytes(embedding)
        
        query = """
            INSERT INTO face (
                id, "personId", "orgId", "fullName", email, role, 
                status, embedding, "imageUrl", notes, 
                "createdAt", "updatedAt", "deletedAt"
            ) VALUES (
                $1, $2, $3, $4, $5, $6, 
                $7::\"FaceStatus\", $8, $9, $10, 
                NOW(), NOW(), NULL
            )
            ON CONFLICT (id) DO UPDATE SET
                "personId" = EXCLUDED."personId",
                "fullName" = EXCLUDED."fullName",
                email = EXCLUDED.email,
                role = EXCLUDED.role,
                status = EXCLUDED.status,
                embedding = EXCLUDED.embedding,
                "imageUrl" = EXCLUDED."imageUrl",
                notes = EXCLUDED.notes,
                "updatedAt" = NOW(),
                "deletedAt" = NULL
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                face_id,
                person_id,
                org_id,
                full_name,
                email,
                role,
                status,
                embedding_bytes,
                image_url,
                notes,
            )
    
    async def upsert_faces_batch(
        self,
        faces: List[Dict[str, Any]],
    ) -> int:
        """
        Batch insert/update faces.
        Each face dict should have: id, org_id, full_name, embedding, and optional fields.
        Returns count of upserted records.
        """
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        if not faces:
            return 0
        
        # Prepare data for executemany
        records = []
        for face in faces:
            records.append((
                face['id'],
                face.get('person_id'),
                face['org_id'],
                face['full_name'],
                face.get('email'),
                face.get('role'),
                face.get('status', 'AUTHORIZED'),
                self.embedding_to_bytes(face['embedding']),
                face.get('image_url'),
                face.get('notes'),
            ))
        
        query = """
            INSERT INTO face (
                id, "personId", "orgId", "fullName", email, role, 
                status, embedding, "imageUrl", notes, 
                "createdAt", "updatedAt", "deletedAt"
            ) VALUES (
                $1, $2, $3, $4, $5, $6, 
                $7::\"FaceStatus\", $8, $9, $10, 
                NOW(), NOW(), NULL
            )
            ON CONFLICT (id) DO UPDATE SET
                "personId" = EXCLUDED."personId",
                "fullName" = EXCLUDED."fullName",
                email = EXCLUDED.email,
                role = EXCLUDED.role,
                status = EXCLUDED.status,
                embedding = EXCLUDED.embedding,
                "imageUrl" = EXCLUDED."imageUrl",
                notes = EXCLUDED.notes,
                "updatedAt" = NOW(),
                "deletedAt" = NULL
        """
        
        async with self.pool.acquire() as conn:
            await conn.executemany(query, records)
        
        return len(records)
    
    async def soft_delete_face(self, face_id: str) -> bool:
        """Soft delete a face (sets deletedAt timestamp)."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        query = """
            UPDATE face 
            SET "deletedAt" = NOW(), "updatedAt" = NOW()
            WHERE id = $1 AND "deletedAt" IS NULL
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, face_id)
            return result == "UPDATE 1"
    
    async def get_faces_for_sync(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        include_deleted: bool = True,
    ) -> Dict[str, Any]:
        """
        Get faces for Pi sync.
        Returns faces updated since the given timestamp.
        
        Returns:
            {
                "version": "2026-01-08T12:34:56Z",
                "upserts": [...],
                "deletes": [...]
            }
        """
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            # Get current timestamp as version
            version = await conn.fetchval("SELECT NOW()")
            
            # Build query
            if since:
                # Delta sync - only changed records
                # Use >= instead of > to catch records updated at exact same timestamp
                # This may return some duplicates but ensures we don't miss any
                query = """
                    SELECT id, "personId", "orgId", "fullName", email, role, 
                           status, embedding, "imageUrl", notes, "deletedAt"
                    FROM face
                    WHERE "orgId" = $1 AND "updatedAt" >= $2
                    ORDER BY "updatedAt" ASC
                """
                try:
                    rows = await conn.fetch(query, org_id, since)
                    print(f"📊 Delta sync: org={org_id}, since={since}, found {len(rows)} records")
                except Exception as e:
                    print(f"❌ Delta sync query failed: {e}")
                    print(f"   org_id: {org_id}, since: {since} (type: {type(since)})")
                    raise
            else:
                # Full sync - all active records
                query = """
                    SELECT id, "personId", "orgId", "fullName", email, role, 
                           status, embedding, "imageUrl", notes, "deletedAt"
                    FROM face
                    WHERE "orgId" = $1 AND "deletedAt" IS NULL
                    ORDER BY "createdAt" ASC
                """
                rows = await conn.fetch(query, org_id)
        
        upserts = []
        deletes = []
        
        for row in rows:
            if row['deletedAt'] is not None:
                # This is a deletion
                deletes.append(row['id'])
            else:
                # This is an upsert
                upserts.append({
                    "id": row['id'],
                    "person_id": row['personId'],
                    "full_name": row['fullName'],
                    "email": row['email'],
                    "role": row['role'],
                    "status": row['status'],
                    "embedding": self.bytes_to_embedding(row['embedding']),
                    "image_url": row['imageUrl'],
                    "notes": row['notes'],
                })
        
        return {
            "version": version.isoformat() if version else datetime.utcnow().isoformat(),
            "upserts": upserts,
            "deletes": deletes,
        }
    
    async def get_face_count(self, org_id: str) -> int:
        """Get count of active faces for an organization."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        query = """
            SELECT COUNT(*) FROM face
            WHERE "orgId" = $1 AND "deletedAt" IS NULL
        """
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, org_id)

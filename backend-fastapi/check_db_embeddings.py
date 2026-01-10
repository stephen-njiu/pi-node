"""Check embeddings directly from PostgreSQL."""
import asyncio
import asyncpg
import numpy as np
import os
import struct
from dotenv import load_dotenv

load_dotenv()

async def check_embeddings():
    db_url = os.getenv('DATABASE_URL') or os.getenv('FG_DATABASE_URL')
    print(f"Connecting to database...")
    conn = await asyncpg.connect(db_url, ssl='require')
    
    # Get embeddings from different people  
    rows = await conn.fetch("""
        SELECT id, "fullName", embedding
        FROM face
        WHERE embedding IS NOT NULL
        ORDER BY "fullName"
        LIMIT 50
    """)
    
    print(f"Found {len(rows)} faces with embeddings")
    print()
    
    embeddings_by_name = {}
    for r in rows:
        name = r["fullName"]
        emb_bytes = r["embedding"]
        
        # Embeddings are stored as binary bytes (512 floats * 4 bytes = 2048 bytes)
        if isinstance(emb_bytes, (bytes, memoryview)):
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
        else:
            emb = np.array(emb_bytes, dtype=np.float32)
        
        if name not in embeddings_by_name:
            embeddings_by_name[name] = []
        embeddings_by_name[name].append(emb)
        
        # Print sample
        print(f'{name}: dim={len(emb)}, norm={np.linalg.norm(emb):.4f}, sample={emb[:5]}')
    
    print()
    print("="*60)
    print("Cross-person cosine distances (should be HIGH ~0.3-0.8 for different people):")
    print("="*60)
    
    names = list(embeddings_by_name.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            emb1 = embeddings_by_name[name1][0]
            emb2 = embeddings_by_name[name2][0]
            
            # Normalize
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            
            # Cosine distance
            similarity = np.dot(emb1, emb2)
            distance = 1 - similarity
            
            print(f"  {name1} <-> {name2}: distance={distance:.4f}, similarity={similarity:.2%}")
    
    print()
    print("="*60)
    print("Same-person cosine distances (should be LOW ~0.0-0.3):")
    print("="*60)
    
    for name, embs in embeddings_by_name.items():
        if len(embs) >= 2:
            emb1 = embs[0] / np.linalg.norm(embs[0])
            emb2 = embs[1] / np.linalg.norm(embs[1])
            similarity = np.dot(emb1, emb2)
            distance = 1 - similarity
            print(f"  {name} (self): distance={distance:.4f}, similarity={similarity:.2%}")
    
    await conn.close()

asyncio.run(check_embeddings())

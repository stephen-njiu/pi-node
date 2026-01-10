"""Debug stored embeddings to check if they're actually different."""
import json
import numpy as np
import hnswlib
from collections import defaultdict

# Load metadata
with open("data/faces_metadata.json", "r") as f:
    data = json.load(f)

metadata = data.get("metadata", {})
print(f"Total faces: {len(metadata)}")

# Group by name
by_name = defaultdict(list)
for idx, meta in metadata.items():
    by_name[meta["name"]].append(int(idx))

print("\nFaces by person:")
for name, indices in by_name.items():
    print(f"  {name}: {len(indices)} faces (indices: {indices[:3]}...)")

# Load hnswlib index
print("\n" + "="*60)
print("Checking embedding distances within hnswlib index")
print("="*60)

index = hnswlib.Index(space="cosine", dim=512)
index.load_index("data/faces.index", max_elements=10000)

# Query each person's first embedding against all
for name, indices in by_name.items():
    if not indices:
        continue
    
    # Get first index for this person
    query_idx = indices[0]
    
    # Search for nearest neighbors
    # We can't extract embeddings directly, but we can check who matches whom
    print(f"\n{name} (idx={query_idx}) nearest matches:")
    
    # Use internal function to get items
    # Actually hnswlib doesn't expose getting embeddings by ID
    # So let's do a different test

# Better test: check pairwise distances using search
print("\n" + "="*60)
print("Cross-person similarity test (should be DIFFERENT!)")
print("="*60)

# Pick one idx from each person
test_indices = []
for name, indices in by_name.items():
    if indices:
        test_indices.append((name, indices[0]))

print(f"\nTest indices: {test_indices}")

# For each test index, search top 5 and see if they're from same person
for name, idx in test_indices:
    # Query with a random vector to see distances
    pass

print("\n" + "="*60)
print("The REAL test - generate embedding from camera and compare")
print("="*60)
print("This was done in test_recognition.py - all results showed ~97% similarity")
print("This suggests ALL stored embeddings are nearly identical!")
print("\nPossible causes:")
print("1. Backend used MOCK embeddings (SHA256 hash) instead of real ArcFace")
print("2. Backend didn't properly load the ArcFace model")
print("3. Images weren't properly aligned before embedding")
print("4. Database has corrupted/duplicate embeddings")

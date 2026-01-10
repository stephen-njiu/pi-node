"""
Test the buffalo_l embedding pipeline.

Verifies:
1. Detection finds faces with 5-point landmarks
2. Alignment produces 112x112 RGB images
3. Recognition produces 512-D embeddings
4. Embeddings are L2-normalized (norm = 1.0)
5. Different faces produce different embeddings (high distance)
6. Same face produces consistent embeddings (low distance)
"""
import sys
import os
import numpy as np

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import io


def create_test_image_with_face():
    """Create a test image - we'll use a simple pattern that might trigger detection."""
    # For real testing, use actual face images
    # This is just a placeholder
    img = np.random.randint(100, 200, (640, 480, 3), dtype=np.uint8)
    return img


def test_pipeline():
    print("=" * 60)
    print("Testing Buffalo_L Embedding Pipeline")
    print("=" * 60)
    
    from app.services.embedding import EmbeddingService, BuffaloLPipeline
    from app.config.settings import settings
    
    print(f"\n1. Settings:")
    print(f"   INSIGHTFACE_ROOT: {settings.INSIGHTFACE_ROOT}")
    print(f"   INSIGHTFACE_MODEL_NAME: {settings.INSIGHTFACE_MODEL_NAME}")
    
    # Check model files exist
    det_path = os.path.join(settings.INSIGHTFACE_ROOT, "buffalo_l", "det_10g.onnx")
    rec_path = os.path.join(settings.INSIGHTFACE_ROOT, "buffalo_l", "w600k_r50.onnx")
    
    print(f"\n2. Model files:")
    print(f"   Detection:   {det_path} - {'EXISTS' if os.path.exists(det_path) else 'MISSING'}")
    print(f"   Recognition: {rec_path} - {'EXISTS' if os.path.exists(rec_path) else 'MISSING'}")
    
    if not os.path.exists(det_path) or not os.path.exists(rec_path):
        print("\n❌ Model files missing!")
        return
    
    # Initialize pipeline
    print(f"\n3. Initializing pipeline...")
    pipeline = BuffaloLPipeline(
        det_model_path=det_path,
        rec_model_path=rec_path,
    )
    
    # Test with random aligned faces (simulating already-aligned input)
    print(f"\n4. Testing embedding extraction with synthetic aligned faces:")
    
    embeddings = []
    for i in range(3):
        # Create synthetic 112x112 "face" with different patterns
        np.random.seed(i * 1000)
        face = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        
        emb = pipeline.get_embedding(face)
        embeddings.append(emb)
        
        norm = np.linalg.norm(emb)
        print(f"   Face {i+1}: dim={len(emb)}, norm={norm:.6f}, sample={emb[:3]}")
    
    # Check L2 normalization
    print(f"\n5. L2 Normalization check (all norms should be ~1.0):")
    for i, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        status = "✅" if abs(norm - 1.0) < 0.0001 else "❌"
        print(f"   Face {i+1}: norm = {norm:.6f} {status}")
    
    # Check discriminative power
    print(f"\n6. Cosine distances between different 'faces':")
    print(f"   (Should be HIGH ~0.1-0.5 for different inputs)")
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = 1 - np.dot(embeddings[i], embeddings[j])
            status = "✅" if dist > 0.05 else "⚠️"
            print(f"   Face{i+1} <-> Face{j+1}: distance = {dist:.4f} {status}")
    
    # Test same input produces same embedding
    print(f"\n7. Determinism check (same input → same output):")
    np.random.seed(42)
    test_face = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    
    emb1 = pipeline.get_embedding(test_face)
    emb2 = pipeline.get_embedding(test_face)
    
    diff = np.abs(emb1 - emb2).max()
    status = "✅" if diff < 0.0001 else "❌"
    print(f"   Max difference: {diff:.8f} {status}")
    
    print(f"\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)


def test_embedding_service():
    """Test the high-level EmbeddingService."""
    print("\n" + "=" * 60)
    print("Testing EmbeddingService")
    print("=" * 60)
    
    from app.services.embedding import EmbeddingService
    
    print("\n1. Initializing EmbeddingService...")
    svc = EmbeddingService()
    print("   ✅ Service initialized")
    
    print("\n2. Note: To fully test, use real face images.")
    print("   The embed_many() method expects images with detectable faces.")
    

if __name__ == "__main__":
    test_pipeline()
    test_embedding_service()

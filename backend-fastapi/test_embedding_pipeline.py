"""Test the embedding pipeline to understand why all embeddings are similar."""
import numpy as np
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.image import load_image_to_rgb, resize_for_arcface
from app.utils.face import detect_and_align
from app.services.embedding import EmbeddingService

# Create a test with random images to see if the embeddings are discriminative
def test_with_random_noise():
    """Generate random noise images and check if embeddings are different."""
    print("=" * 60)
    print("Test 1: Random noise images (should have VERY different embeddings)")
    print("=" * 60)
    
    svc = EmbeddingService()
    print(f"Backend type: {type(svc.backend).__name__}")
    
    from PIL import Image
    import io
    
    embeddings = []
    for i in range(3):
        # Create random noise image (112x112 RGB)
        noise = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
        
        emb = svc.embed_many([img_bytes])[0].vector
        embeddings.append(emb)
        print(f"Random image {i+1}: norm={np.linalg.norm(emb):.4f}, sample={emb[:5]}")
    
    # Check distances between random images
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = 1 - np.dot(embeddings[i], embeddings[j])
            print(f"  Random{i+1} <-> Random{j+1}: distance={dist:.4f}")
    
    print()


def test_with_solid_colors():
    """Generate solid color images and check if embeddings are different."""
    print("=" * 60)
    print("Test 2: Solid color images (should have different embeddings)")
    print("=" * 60)
    
    svc = EmbeddingService()
    
    from PIL import Image
    import io
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    embeddings = []
    
    for i, color in enumerate(colors):
        # Create solid color image (112x112 RGB)
        arr = np.full((112, 112, 3), color, dtype=np.uint8)
        img = Image.fromarray(arr)
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
        
        emb = svc.embed_many([img_bytes])[0].vector
        embeddings.append(emb)
        print(f"Color {color}: norm={np.linalg.norm(emb):.4f}, sample={emb[:5]}")
    
    # Check distances between solid colors
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = 1 - np.dot(embeddings[i], embeddings[j])
            print(f"  Color{i} <-> Color{j}: distance={dist:.4f}")
    
    print()


def test_with_same_image():
    """Check if the same image always produces the same embedding."""
    print("=" * 60)
    print("Test 3: Same image multiple times (should be IDENTICAL)")
    print("=" * 60)
    
    svc = EmbeddingService()
    
    from PIL import Image
    import io
    
    # Create one random image
    noise = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    img_bytes = buf.getvalue()
    
    embeddings = []
    for i in range(3):
        emb = svc.embed_many([img_bytes])[0].vector
        embeddings.append(emb)
        print(f"Same image run {i+1}: norm={np.linalg.norm(emb):.4f}, sample={emb[:5]}")
    
    # Check distances between same image runs
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = 1 - np.dot(embeddings[i], embeddings[j])
            print(f"  Run{i+1} <-> Run{j+1}: distance={dist:.6f}")
    
    print()


def test_model_output_shape():
    """Check what the model actually outputs."""
    print("=" * 60)
    print("Test 4: Model output shape and values")
    print("=" * 60)
    
    svc = EmbeddingService()
    
    from PIL import Image
    import io
    
    # Create a test image
    noise = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    img_bytes = buf.getvalue()
    
    # Get raw output from model
    if hasattr(svc.backend, 'session'):
        inp = svc.backend._preprocess(img_bytes)
        print(f"Input shape: {inp.shape}")
        print(f"Input dtype: {inp.dtype}")
        print(f"Input range: [{inp.min():.4f}, {inp.max():.4f}]")
        
        outputs = svc.backend.session.run(None, {svc.backend.input_name: inp})
        print(f"Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out.shape}")
            print(f"Output {i} range: [{out.min():.4f}, {out.max():.4f}]")
            print(f"Output {i} norm (raw): {np.linalg.norm(out):.4f}")
    
    print()


if __name__ == "__main__":
    test_model_output_shape()
    test_with_same_image()
    test_with_random_noise()
    test_with_solid_colors()

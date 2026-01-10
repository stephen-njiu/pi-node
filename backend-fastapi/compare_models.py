"""Compare different ONNX models to understand their expected inputs/outputs."""
import numpy as np
import onnxruntime as ort

models = [
    ("arcface.onnx", "models/arcface.onnx"),
    ("w600k_r50.onnx", "models/buffalo_l/w600k_r50.onnx"),
]

def test_model(name, path):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        
        print(f"\nInputs:")
        for inp in session.get_inputs():
            print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")
        
        print(f"\nOutputs:")
        for out in session.get_outputs():
            print(f"  {out.name}: shape={out.shape}, dtype={out.type}")
        
        # Determine input shape
        inp_shape = session.get_inputs()[0].shape
        batch = 1
        channels = inp_shape[1] if len(inp_shape) == 4 else 3
        height = inp_shape[2] if len(inp_shape) == 4 else inp_shape[1]
        width = inp_shape[3] if len(inp_shape) == 4 else inp_shape[2]
        
        # Handle symbolic dimensions
        if isinstance(height, str) or height is None:
            height = 112
        if isinstance(width, str) or width is None:
            width = 112
        
        print(f"\nTesting with input shape: ({batch}, {channels}, {height}, {width})")
        
        # Create random test inputs
        test_inputs = [
            ("Random noise 1", np.random.randn(batch, channels, height, width).astype(np.float32)),
            ("Random noise 2", np.random.randn(batch, channels, height, width).astype(np.float32)),
            ("All zeros", np.zeros((batch, channels, height, width), dtype=np.float32)),
            ("All ones", np.ones((batch, channels, height, width), dtype=np.float32)),
        ]
        
        embeddings = []
        for test_name, inp_data in test_inputs:
            # Normalize input to [-1, 1] range like ArcFace expects
            inp_normalized = inp_data * 0.9961  # Approximate [-1, 1] range
            
            outputs = session.run(None, {session.get_inputs()[0].name: inp_normalized})
            out = outputs[0].flatten()
            
            # Normalize output
            norm = np.linalg.norm(out) or 1.0
            out_normalized = out / norm
            
            embeddings.append((test_name, out_normalized))
            print(f"\n  {test_name}:")
            print(f"    Raw output shape: {outputs[0].shape}")
            print(f"    Raw norm: {np.linalg.norm(outputs[0]):.4f}")
            print(f"    Sample values: {out_normalized[:5]}")
        
        # Calculate distances
        print(f"\nCosine distances (should be HIGH ~0.3-0.8 for different inputs):")
        for i, (name1, emb1) in enumerate(embeddings):
            for j, (name2, emb2) in enumerate(embeddings[i+1:], i+1):
                dist = 1 - np.dot(emb1, emb2)
                print(f"  {name1} <-> {name2}: distance={dist:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    for name, path in models:
        test_model(name, path)

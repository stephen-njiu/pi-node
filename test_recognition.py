"""
Simple test script to verify face recognition pipeline.
Bypasses tracker to test detection → alignment → embedding → matching directly.
"""

import cv2
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from config import Config
from vision.detector import SCRFDDetector
from vision.recognizer import ArcFaceRecognizer
from vision.alignment import align_face
from storage.face_db import FaceDatabase

def main():
    config = Config()
    
    print("="*60)
    print("FACE RECOGNITION PIPELINE TEST")
    print("="*60)
    
    # Load models
    print(f"\n1. Loading SCRFD detector: {config.SCRFD_MODEL_PATH}")
    detector = SCRFDDetector(model_path=config.SCRFD_MODEL_PATH)
    
    print(f"\n2. Loading ArcFace recognizer: {config.ARCFACE_MODEL_PATH}")
    recognizer = ArcFaceRecognizer(model_path=config.ARCFACE_MODEL_PATH)
    
    print(f"\n3. Loading face database...")
    face_db = FaceDatabase()
    db_stats = face_db.get_stats()
    print(f"   Database stats: {db_stats}")
    
    if db_stats["total_faces"] == 0:
        print("\n   ⚠️ DATABASE IS EMPTY! Run sync first.")
        return
    
    # Open camera
    print(f"\n4. Opening camera {config.CAMERA_INDEX}...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("   ❌ Failed to open camera!")
        return
    
    print("   ✅ Camera opened")
    print("\n5. Starting recognition loop (press 'q' to quit)...")
    print("-"*60)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Detection
        detections = detector.detect(frame)
        
        # Draw all detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Show detection confidence
            cv2.putText(frame, f"det:{det.score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Process first detection with landmarks
        if detections and detections[0].landmarks is not None:
            det = detections[0]
            landmarks = det.landmarks
            
            # Draw landmarks
            for i, (lx, ly) in enumerate(landmarks):
                cv2.circle(frame, (int(lx), int(ly)), 3, (0, 0, 255), -1)
            
            # Align face using landmarks
            aligned = align_face(frame, landmarks)
            
            if aligned is not None:
                # Get embedding
                embedding = recognizer.get_embedding(aligned)
                
                if embedding is not None:
                    # Search database - get top 3 matches
                    # Note: hnswlib cosine returns DISTANCE (0=same, 2=opposite)
                    results = face_db.search(embedding, threshold=1.0, k=3)
                    
                    print(f"\nFrame {frame_count}: Found {len(detections)} face(s)")
                    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
                    
                    if results:
                        for i, (face_id, distance, meta) in enumerate(results):
                            similarity = 1.0 - distance  # Convert to similarity
                            name = meta.get("full_name", "?")
                            status = meta.get("status", "?")
                            print(f"  Match {i+1}: {name} ({status})")
                            print(f"           distance={distance:.4f}, similarity={similarity:.2%}")
                        
                        # Display best match on frame
                        best_name = results[0][2].get("full_name", "?")
                        best_dist = results[0][1]
                        best_sim = 1.0 - best_dist
                        
                        # Color based on similarity
                        if best_sim > 0.5:  # Good match
                            color = (0, 255, 0)  # Green
                        elif best_sim > 0.3:  # Marginal
                            color = (0, 255, 255)  # Yellow
                        else:  # Poor match
                            color = (0, 0, 255)  # Red
                        
                        x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                        cv2.putText(frame, f"{best_name}", (x1, y1-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"sim:{best_sim:.1%}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        print(f"  No matches found in database")
                        x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                        cv2.putText(frame, "UNKNOWN", (x1, y1-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show aligned face in corner
                aligned_display = cv2.resize(aligned, (112, 112))
                frame[10:122, 10:122] = aligned_display
        
        # Display
        cv2.imshow("Face Recognition Test (press 'q' to quit)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("Test complete")

if __name__ == "__main__":
    main()

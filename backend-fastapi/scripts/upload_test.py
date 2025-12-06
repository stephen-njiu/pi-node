import os
import glob
import requests

"""
Uploads up to 5 images found in the same directory as this script
to the FastAPI endpoint and prints the embeddings.

Usage: python scripts/upload_test.py
"""

API_URL = "http://localhost:8000/api/v1/embeddings"

def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Find images in this directory
    candidates = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        candidates.extend(glob.glob(os.path.join(script_dir, pattern)))

    images = [p for p in candidates if is_image(p)]
    if not images:
        print("No images found in:", script_dir)
        return

    # Limit to 5 as per API contract
    images = images[:5]
    print("Uploading:", " , ".join(os.path.basename(p) for p in images))

    files = []
    open_files = []
    try:
        for p in images:
            f = open(p, "rb")
            open_files.append(f)
            mime = "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"
            files.append(("files", (os.path.basename(p), f, mime)))

        resp = requests.post(API_URL, files=files)
        print("Status:", resp.status_code)
        if resp.ok:
            data = resp.json()
            print("Count:", data.get("count"))
            print("Store:", data.get("store"), "Stored:", data.get("stored"))
            items = data.get("items", [])
            for i, item in enumerate(items):
                vec = item.get("vector", [])
                # Print small preview of the vector
                preview = vec[:10]
                print(f"Embedding {i+1}: dim={len(vec)} preview={preview}")
        else:
            print(resp.text)
    finally:
        for f in open_files:
            try:
                f.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()

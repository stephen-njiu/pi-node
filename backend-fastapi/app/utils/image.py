from typing import Tuple
import io
import numpy as np
from PIL import Image


def load_image_to_rgb(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    return img.convert("RGB")


def resize_for_arcface(img: Image.Image, size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """Resize and normalize image for ArcFace-like models (if already aligned)."""
    img_resized = img.resize(size)
    arr = np.asarray(img_resized).astype(np.float32)
    # Normalize to [-1, 1]
    arr = (arr - 127.5) / 128.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    return arr

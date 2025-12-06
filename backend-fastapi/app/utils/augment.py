from __future__ import annotations
import io
import random
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Helper to clamp values

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False)


def translate(img: Image.Image, tx: int, ty: int) -> Image.Image:
    w, h = img.size
    return img.transform((w, h), Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BICUBIC)


def scale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * factor), int(h * factor)
    resized = img.resize((nw, nh), resample=Image.BICUBIC)
    # center-crop or pad back to original size
    if factor >= 1.0:
        # crop center
        left = (nw - w) // 2
        top = (nh - h) // 2
        return resized.crop((left, top, left + w, top + h))
    else:
        # paste on canvas
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        left = (w - nw) // 2
        top = (h - nh) // 2
        canvas.paste(resized, (left, top))
        return canvas


def hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def jitter_brightness_contrast(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def color_shift(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def slight_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def generate_variants(img: Image.Image, max_variants: int = 3, allow_flip: bool = False) -> List[Tuple[str, Image.Image]]:
    """
    Generate up to max_variants identity-preserving variants.
    Returns list of (name, image).
    """
    variants: List[Tuple[str, Image.Image]] = []

    ops = []
    # Geometric
    ops.append(lambda i: ("rot", rotate(i, random.uniform(-10, 10))))
    ops.append(lambda i: ("trans", translate(i, random.randint(-4, 4), random.randint(-4, 4))))
    ops.append(lambda i: ("scale", scale(i, _clamp(random.uniform(0.95, 1.05), 0.9, 1.1))))
    if allow_flip:
        ops.append(lambda i: ("flip", hflip(i)))

    # Photometric
    ops.append(lambda i: ("bc", jitter_brightness_contrast(i, _clamp(random.uniform(0.8, 1.2), 0.7, 1.3), _clamp(random.uniform(0.8, 1.2), 0.7, 1.3))))
    ops.append(lambda i: ("color", color_shift(i, _clamp(random.uniform(0.9, 1.1), 0.8, 1.2))))

    # Noise / blur
    ops.append(lambda i: ("noise", gaussian_noise(i, sigma=5.0)))  # low sigma
    ops.append(lambda i: ("blur", slight_blur(i, radius=1.0)))

    random.shuffle(ops)
    for fn in ops:
        if len(variants) >= max_variants:
            break
        name, aug = fn(img)
        variants.append((name, aug))

    return variants

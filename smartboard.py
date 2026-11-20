"""Detection-first whiteboard pipeline. All coordinates use image pixel centers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_photo(path: Path, max_side: int = 1600) -> tuple[np.ndarray, list[int], np.ndarray]:
    with Image.open(path) as photo:
        if photo.width * photo.height > 30_000_000:
            raise ValueError("Input exceeds the 30 megapixel limit.")
        photo = ImageOps.exif_transpose(photo).convert("RGB")
        width, height = photo.size
        scale = min(1, max_side / max(width, height))
        size = (round(width * scale), round(height * scale))
        if min(size) < 32:
            raise ValueError("Image must be at least 32 pixels on each side after resizing.")
        rgb = np.asarray(photo.resize(size, Image.Resampling.LANCZOS) if size != photo.size else photo)
    sx, sy = size[0] / width, size[1] / height
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return rgb, [width, height], resize

"""Detection-first whiteboard pipeline. All coordinates use image pixel centers."""

from __future__ import annotations

from pathlib import Path

import cv2
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


def rectify(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Warp a large, inset bright quadrilateral, or retain the whole frame."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if not 0.35 <= cv2.contourArea(contour) / (w * h) <= 0.95:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        xy = approx.reshape(4, 2).astype(np.float32)
        if np.any(xy.min(axis=0) <= 2) or np.any(xy.max(axis=0) >= [w - 3, h - 3]):
            continue
        # Accept only upright-ish quadrilaterals with four distinct corners.
        sums, diffs = xy.sum(axis=1), xy[:, 0] - xy[:, 1]
        indices = [sums.argmin(), diffs.argmax(), sums.argmax(), diffs.argmin()]
        if len(set(indices)) != 4:
            continue
        tl, tr, br, bl = corners = xy[indices]
        width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if min(width, height) < 64:
            continue
        target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        transform = cv2.getPerspectiveTransform(corners, target)
        return cv2.warpPerspective(rgb, transform, (width, height)), transform, True
    return rgb, np.eye(3), False


def enhance(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    window = max(31, round(min(gray.shape) * 0.08)) | 1
    illumination = cv2.GaussianBlur(gray, (window, window), 0)
    normalized = cv2.divide(gray, np.maximum(illumination, 1), scale=245)
    blended = cv2.addWeighted(normalized, 0.55, gray, 0.45, 0)
    return cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(blended)

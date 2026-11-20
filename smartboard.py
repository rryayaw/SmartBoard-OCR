"""Detection-first whiteboard pipeline. All coordinates use image pixel centers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class Detection:
    corrected: np.ndarray
    overlay: np.ndarray
    report: dict


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


def locate(gray: np.ndarray) -> list[list[int]]:
    """Group nearby dark strokes into candidate text regions, not verified text."""
    h, w = gray.shape
    window = max(15, round(min(h, w) * 0.035)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    otsu, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ink = (response >= max(12, round(otsu * 0.8))).astype(np.uint8)
    for shape in ((1, max(35, w // 9)), (max(35, h // 9), 1)):
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones(shape, np.uint8))
        ink[lines != 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    for i, (_, _, bw, bh, area) in enumerate(stats[1:], 1):
        if max(4, w * h // 350000) <= area and bw < w * 0.3 and bh < h * 0.22:
            keep[i] = 1
    ink = keep[labels]
    grouped = cv2.dilate(ink, np.ones((max(2, h // 230), max(5, w // 105)), np.uint8))
    count, groups, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    boxes = []
    for i, (x, y, bw, bh, _) in enumerate(stats[1:], 1):
        if bw < 9 or bh < 6 or bh > h * 0.24 or bw / bh > 28:
            continue
        ys, xs = np.nonzero((groups[y:y+bh, x:x+bw] == i) & ink[y:y+bh, x:x+bw])
        if len(xs) < 10 or not 0.04 <= len(xs) / (bw * bh) <= 0.82:
            continue
        x0 = max(0, x + int(xs.min()) - 2)
        y0 = max(0, y + int(ys.min()) - 2)
        x1 = min(w, x + int(xs.max()) + 3)
        y1 = min(h, y + int(ys.max()) + 3)
        boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def detect(path: Path, *, auto_rectify: bool = True) -> Detection:
    started = perf_counter()
    rgb, source_size, resize = load_photo(path)
    board, warp, rectified = rectify(rgb) if auto_rectify else (rgb, np.eye(3), False)
    corrected = enhance(board)
    boxes = locate(corrected)
    overlay = cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)
    warnings = []
    if auto_rectify and not rectified:
        warnings.append("Board outline not found; retained the full frame.")
    if not boxes:
        warnings.append("No candidate text regions found; this does not establish that the board is blank.")
    regions = []
    for number, (x, y, w, h) in enumerate(boxes, 1):
        regions.append({"id": f"r{number:03}", "bbox_xywh": [x, y, w, h]})
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (0, 135, 68), 2)
        cv2.putText(overlay, str(number), (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 105, 45), 1, cv2.LINE_AA)
    h, w = corrected.shape
    return Detection(corrected, overlay, {
        "schema_version": 1,
        "source_size_wh": source_size,
        "source_orientation": "EXIF-normalized",
        "coordinate_image": "corrected.png",
        "coordinate_size_wh": [w, h],
        "source_to_corrected": (warp @ resize).tolist(),
        "rectification": "applied" if rectified else ("not_found" if auto_rectify else "disabled"),
        "detector": "opencv_dark_stroke_candidates_v0",
        "regions": regions,
        "warnings": warnings,
        "detection_seconds": round(perf_counter() - started, 4),
    })

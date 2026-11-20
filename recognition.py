"""Optional Tesseract adapter; never modifies the detection results."""

from __future__ import annotations

from shutil import which
import subprocess
from time import monotonic

import cv2


def recognize(image, regions, executable: str, language: str, budget: float) -> dict:
    result = {"engine": "tesseract", "language": language, "by_region": {}}
    binary = which(executable)
    if binary is None:
        return {**result, "status": "unavailable", "warning": "Tesseract executable not found."}
    if not regions:
        return {**result, "status": "no_regions"}
    deadline = monotonic() + budget
    height, width = image.shape
    failure = None
    for region in regions:
        x, y, w, h = region["bbox_xywh"]
        crop = image[max(0, y - 4):min(height, y+h+4), max(0, x-4):min(width, x+w+4)]
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        ok, png = cv2.imencode(".png", crop)
        if not ok:
            failure = ("failed", "Could not encode a region crop.")
            break
        remaining = deadline - monotonic()
        if remaining <= 0:
            failure = ("timed_out", "Recognition exceeded its time budget.")
            break
        try:
            completed = subprocess.run(
                [binary, "stdin", "stdout", "-l", language, "--psm", "7"],
                input=png.tobytes(), capture_output=True, timeout=remaining, check=False,
            )
        except subprocess.TimeoutExpired:
            failure = ("timed_out", "Recognition exceeded its time budget.")
            break
        except OSError:
            failure = ("failed", "Could not start Tesseract.")
            break
        if completed.returncode:
            failure = ("failed", f"Tesseract exited with code {completed.returncode}; check the executable and language data.")
            break
        text = completed.stdout.decode("utf-8", errors="replace").strip()
        result["by_region"][region["id"]] = {"status": "recognized" if text else "no_text", "text": text}
    if failure:
        status, warning = failure
        return {**result, "status": "partial" if result["by_region"] else status, "failure": status, "warning": warning}
    return {**result, "status": "completed"}

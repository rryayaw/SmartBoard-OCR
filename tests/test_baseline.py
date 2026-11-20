import contextlib
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from cli import main
from recognition import recognize
from smartboard import detect, load_photo


class BaselineTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.photo = self.root / "board.png"
        rgb = np.full((480, 720, 3), 38, dtype=np.uint8)
        cv2.fillConvexPoly(rgb, np.array([[50, 48], [670, 60], [680, 438], [40, 420]], np.int32), (219, 219, 219))
        cv2.putText(rgb, "CLIENT", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
        cv2.putText(rgb, "SERVER", (430, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
        Image.fromarray(rgb).save(self.photo)

    def test_warp_boxes_and_coordinate_transform(self):
        result = detect(self.photo)
        report = result.report
        self.assertEqual(report["rectification"], "applied")
        self.assertGreaterEqual(len(report["regions"]), 2)
        self.assertEqual(result.overlay.shape[:2], result.corrected.shape)
        width, height = report["coordinate_size_wh"]
        for region in report["regions"]:
            x, y, w, h = region["bbox_xywh"]
            self.assertTrue(0 <= x < x+w <= width and 0 <= y < y+h <= height)
        mapped = cv2.perspectiveTransform(
            np.float32([[[130, 160], [430, 310]]]), np.asarray(report["source_to_corrected"])
        )[0]
        for mx, my in mapped:
            self.assertTrue(any(
                x-8 <= mx <= x+w+8 and y-8 <= my <= y+h+8
                for x, y, w, h in (r["bbox_xywh"] for r in report["regions"])
            ))

    def test_blank_board_full_frame_fallback(self):
        Image.new("RGB", (300, 200), (247, 247, 247)).save(self.photo)
        result = detect(self.photo)
        self.assertEqual(result.report["rectification"], "not_found")
        self.assertEqual(result.report["regions"], [])
        np.testing.assert_array_equal(result.report["source_to_corrected"], np.eye(3))

    def test_exif_orientation(self):
        image = Image.new("RGB", (100, 200))
        exif = image.getexif()
        exif[274] = 6
        photo = self.root / "phone.jpg"
        image.save(photo, exif=exif)
        rgb, size, _ = load_photo(photo)
        self.assertEqual(size, [200, 100])
        self.assertEqual(rgb.shape, (100, 200, 3))

    def run_cli(self, name, *flags):
        out = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.photo), str(out), *flags]), 0)
        return out, json.loads((out / "results.json").read_text(encoding="utf-8"))

    def test_missing_recognizer_keeps_byte_identical_detection(self):
        with patch("recognition.recognize") as hook:
            out, plain = self.run_cli("plain")
            hook.assert_not_called()
        other, extended = self.run_cli("extended", "--recognize", "--tesseract", str(self.root / "missing.exe"))
        self.assertEqual(plain["recognition"]["status"], "disabled")
        self.assertEqual(extended["recognition"]["status"], "unavailable")
        self.assertEqual(plain["regions"], extended["regions"])
        for name in ("corrected.png", "overlay.png"):
            self.assertEqual((out / name).read_bytes(), (other / name).read_bytes())

    def test_recognizer_timeout_retains_completed_region(self):
        regions = [{"id": "r001", "bbox_xywh": [0, 0, 20, 20]}, {"id": "r002", "bbox_xywh": [20, 20, 20, 20]}]
        good = subprocess.CompletedProcess([], 0, b"CLIENT\n", b"")
        with patch("recognition.which", return_value="tesseract"), patch(
            "recognition.subprocess.run", side_effect=[good, subprocess.TimeoutExpired("tesseract", 1)]
        ):
            report = recognize(np.full((40, 40), 200, np.uint8), regions, "tesseract", "eng", 10)
        self.assertEqual((report["status"], report["failure"]), ("partial", "timed_out"))
        self.assertEqual(report["by_region"], {"r001": {"status": "recognized", "text": "CLIENT"}})

    def test_recognizer_success_uses_same_region_ids(self):
        good = subprocess.CompletedProcess([], 0, b"CLIENT\n", b"")
        with patch("recognition.which", return_value="tesseract"), patch("recognition.subprocess.run", return_value=good):
            _, report = self.run_cli("success", "--recognize")
        self.assertEqual(report["recognition"]["status"], "completed")
        self.assertEqual(set(report["recognition"]["by_region"]), {r["id"] for r in report["regions"]})

    def test_recognizer_exception_still_writes_report(self):
        with patch("recognition.recognize", side_effect=RuntimeError("internal error")):
            _, report = self.run_cli("error", "--recognize")
        self.assertEqual(report["recognition"]["status"], "failed")
        self.assertTrue(report["regions"])


if __name__ == "__main__":
    unittest.main()

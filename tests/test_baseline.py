import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
from PIL import Image

from cli import main
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


    def test_cli_exports_detection_only(self):
        out, report = self.run_cli("detected")
        self.assertTrue((out / "corrected.png").is_file())
        self.assertTrue((out / "overlay.png").is_file())
        self.assertTrue(report["regions"])
        self.assertNotIn("recognition", report)

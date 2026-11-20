from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
from PIL import Image

from cli import main
from smartboard import enhance, load_photo, rectify


class BaselineTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_exif_orientation(self):
        image = Image.new("RGB", (100, 200))
        exif = image.getexif()
        exif[274] = 6
        photo = self.root / "phone.jpg"
        image.save(photo, exif=exif)
        rgb, size, _ = load_photo(photo)
        self.assertEqual(size, [200, 100])
        self.assertEqual(rgb.shape, (100, 200, 3))

    def test_resize_and_export(self):
        photo = self.root / "wide.png"
        Image.new("RGB", (2000, 400), "white").save(photo)
        rgb, size, transform = load_photo(photo)
        self.assertEqual(size, [2000, 400])
        self.assertEqual(rgb.shape, (320, 1600, 3))
        np.testing.assert_allclose(transform @ [0, 0, 1], [-0.1, -0.1, 1])
        output = self.root / "result"
        self.assertEqual(main([str(photo), str(output), "--no-rectify"]), 0)
        with Image.open(output / "corrected.png") as saved:
            self.assertEqual(saved.size, (1600, 320))

    def test_rectifies_bright_board(self):
        rgb = np.full((360, 600, 3), 32, dtype=np.uint8)
        cv2.fillConvexPoly(rgb, np.array([[45, 40], [550, 55], [530, 320], [55, 310]]), (220, 220, 220))
        corrected, transform, applied = rectify(rgb)
        self.assertTrue(applied)
        self.assertLess(corrected.shape[1], rgb.shape[1])
        mapped = cv2.perspectiveTransform(np.float32([[[45, 40]]]), transform)
        np.testing.assert_allclose(mapped, 0, atol=4)

    def test_fallback_retains_full_frame(self):
        rgb = np.full((200, 300, 3), 245, dtype=np.uint8)
        corrected, transform, applied = rectify(rgb)
        self.assertFalse(applied)
        np.testing.assert_array_equal(corrected, rgb)
        np.testing.assert_array_equal(transform, np.eye(3))
        self.assertEqual(enhance(corrected).shape, rgb.shape[:2])

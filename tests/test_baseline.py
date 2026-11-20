from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from cli import main
from smartboard import load_photo


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
        self.assertEqual(main([str(photo), str(output)]), 0)
        with Image.open(output / "oriented.png") as saved:
            self.assertEqual(saved.size, (1600, 320))

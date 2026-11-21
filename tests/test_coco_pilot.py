import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from PIL import Image

from coco_pilot import import_cvat


class ImportTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        Image.new("RGB", (80, 60)).save(self.root / "board.jpg")
        self.archive = self.root / "labels.zip"
        self.data = {
            "images": [{"id": 1, "file_name": "board.jpg", "width": 80, "height": 60}],
            "categories": [{"id": 1, "name": "text"}, {"id": 2, "name": "ignore"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 20, 10]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [40, 20, 20, 10]},
            ],
        }

    def load(self):
        with ZipFile(self.archive, "w") as export:
            export.writestr("annotations/instances_default.json", json.dumps(self.data))
        return import_cvat(self.root, [self.archive])

    def test_separates_text_and_ignore(self):
        record, = self.load()["images"]
        self.assertEqual(record["text"], [[1, 2, 20, 10]])
        self.assertEqual(record["ignore"], [[40, 20, 20, 10]])

    def test_rejects_wrong_dimensions(self):
        self.data["images"][0]["width"] = 100
        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            self.load()

    def test_rejects_invalid_bounds(self):
        self.data["annotations"][0]["bbox"][2] = 85
        with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
            self.load()

    def test_rejects_traversal(self):
        self.data["images"][0]["file_name"] = "../board.jpg"
        with self.assertRaisesRegex(ValueError, "Unsafe image path"):
            self.load()

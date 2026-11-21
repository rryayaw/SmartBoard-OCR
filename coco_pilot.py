"""Validate CVAT COCO exports and produce a local SmartBoard pilot manifest."""

import argparse
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from PIL import Image


def indexed(items):
    result = {item["id"]: item for item in items}
    if len(result) != len(items):
        raise ValueError("Duplicate COCO IDs")
    return result


def import_cvat(image_root, archives):
    image_root = Path(image_root).resolve()
    results = {}
    for archive in archives:
        archive = Path(archive)
        with ZipFile(archive) as export:
            members = [m for m in export.infolist() if m.filename.startswith("annotations/instances_")
                       and m.filename.endswith(".json")]
            if not members:
                raise ValueError(f"No COCO instance files in {archive.name}")
            for member in members:
                if member.file_size > 10_000_000:
                    raise ValueError(f"COCO JSON exceeds 10 MB: {member.filename}")
                data = json.loads(export.read(member))
                categories = indexed(data["categories"])
                images = indexed(data["images"])
                annotations = indexed(data["annotations"])
                records = {}
                for image_id, image in images.items():
                    name = image["file_name"]
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                        raise ValueError(f"Unsafe image path: {name}")
                    path = (image_root / Path(*relative.parts)).resolve()
                    if not path.is_relative_to(image_root) or path in results:
                        raise ValueError(f"External or duplicate image: {name}")
                    with Image.open(path) as source:
                        size = source.size[::-1] if source.getexif().get(274) in (5, 6, 7, 8) else source.size
                    if size != (image["width"], image["height"]):
                        raise ValueError(f"Dimension mismatch for {name}: file {size}, COCO {(image['width'], image['height'])}")
                    records[image_id] = results[path] = dict(
                        image=name, size_wh=list(size), export=archive.name, text=[], ignore=[]
                    )
                for annotation in annotations.values():
                    record = records[annotation["image_id"]]
                    label = categories[annotation["category_id"]]["name"]
                    if label not in ("text", "ignore"):
                        raise ValueError(f"Unsupported category: {label}")
                    bbox = annotation["bbox"]
                    if len(bbox) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in bbox):
                        raise ValueError(f"Invalid box: {bbox}")
                    x, y, w, h = bbox
                    width, height = record["size_wh"]
                    if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > width+.01 or y+h > height+.01:
                        raise ValueError(f"Out-of-bounds box for {record['image']}: {bbox}")
                    target = "ignore" if label == "ignore" or annotation.get("iscrowd") else "text"
                    if bbox in record[target]:
                        raise ValueError(f"Duplicate {target} box in {record['image']}: {bbox}")
                    record[target].append(bbox)
    if not results:
        raise ValueError("No images imported")
    return {"schema_version": 1, "coordinate_space": "exif_oriented_xywh", "images": list(results.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        manifest = import_cvat(args.image_root, args.archives)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        parser.exit(1, f"COCO import: {exc}\n")
    records = manifest["images"]
    print(f"Validated {len(records)} images, {sum(len(r['text']) for r in records)} text boxes, "
          f"{sum(len(r['ignore']) for r in records)} ignore boxes.")


if __name__ == "__main__":
    main()

"""Compare OpenCV and pretrained CRAFT on the same reviewed pilot manifest."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

from smartboard import detect, load_photo


def rectangle(box):
    x, y, w, h = box
    return np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])


def overlap(a, b):
    if np.any(a.max(axis=0) <= b.min(axis=0)) or np.any(b.max(axis=0) <= a.min(axis=0)):
        return 0.0
    area, _ = cv2.intersectConvexConvex(a, b)
    return max(0.0, min(float(area), cv2.contourArea(a), cv2.contourArea(b)))


def score(labels, predictions, ignore, threshold):
    """Maximum-count one-to-one IoU matching, then suppress unmatched ignored predictions."""
    edges = []
    for label in labels:
        candidates = []
        for i, pred in enumerate(predictions):
            intersection = overlap(label, pred)
            union = cv2.contourArea(label) + cv2.contourArea(pred) - intersection
            if union > 0 and intersection / union >= threshold:
                candidates.append((intersection / union, i))
        edges.append([i for _, i in sorted(candidates, reverse=True)])
    matches = {}

    def connect(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matches or connect(matches[j], seen):
                matches[j] = i
                return True
        return False

    for i in sorted(range(len(edges)), key=lambda i: len(edges[i])):
        connect(i, set())
    ignored = sum(
        any(overlap(pred, region) >= 0.5 * cv2.contourArea(pred) for region in ignore)
        for j, pred in enumerate(predictions) if j not in matches and cv2.contourArea(pred) > 0
    )
    return {"tp": len(matches), "fp": len(predictions) - len(matches) - ignored,
            "fn": len(labels) - len(matches), "ignored": ignored}


def metrics(row):
    tp, fp, fn = (row[key] for key in ("tp", "fp", "fn"))
    return dict(precision=tp / (tp+fp) if tp+fp else 0, recall=tp / (tp+fn) if tp+fn else 0,
                f1=2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0)


def project_regions(regions, transform):
    inverse = np.linalg.inv(transform)
    return [cv2.perspectiveTransform(np.float32([q]), inverse)[0] for q in regions]


def craft_regions(reader, rgb):
    horizontal, free = reader.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), canvas_size=1600,
                                     mag_ratio=1.0, min_size=8)
    regions = [rectangle((x0, y0, x1-x0, y1-y0)) for x0,x1,y0,y1 in horizontal[0]]
    regions.extend(cv2.convexHull(np.float32(q)).reshape(-1, 2) for q in free[0])
    return [q for q in regions if cv2.contourArea(q) > 0]


def run(manifest, image_root, output, reader):
    output.mkdir(parents=True, exist_ok=False)
    image_root = image_root.resolve()
    rows, artifacts = [], []
    for number, record in enumerate(manifest["images"], 1):
        source = (image_root / record["image"]).resolve()
        if not source.is_relative_to(image_root):
            raise ValueError("Manifest image escaped the input folder")
        w, h = record["size_wh"]
        shared = source
        resize = np.eye(3)
        if w*h > 30_000_000:
            shared = output / f"{number:02d}-analysis-input.png"
            with Image.open(source) as photo:
                photo = ImageOps.exif_transpose(photo).convert("RGB")
                photo.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
                sx, sy = photo.width/w, photo.height/h
                resize = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
                photo.save(shared)
        labels = [rectangle(box) for box in record["text"]]
        ignore = [rectangle(box) for box in record["ignore"]]
        start = perf_counter()
        baseline = detect(shared)
        predictions = project_regions(
            [rectangle(box["bbox_xywh"]) for box in baseline.report["regions"]],
            np.asarray(baseline.report["source_to_corrected"]) @ resize,
        )
        baseline_seconds = perf_counter()-start
        model_runs = [("opencv", predictions, baseline_seconds)]
        start = perf_counter()
        rgb, _, input_transform = load_photo(shared)
        regions = project_regions(craft_regions(reader, rgb), input_transform @ resize)
        model_runs.append(("craft", regions, perf_counter()-start))
        for model, predictions, seconds in model_runs:
            artifacts.append({"image": record["image"], "detector": model,
                              "regions": [q.tolist() for q in predictions]})
            for threshold in (.50, .75):
                counts = score(labels, predictions, ignore, threshold)
                rows.append(dict(image=record["image"], detector=model, iou=threshold,
                                 shared_resize=shared != source, seconds=round(seconds, 4),
                                 **counts, **metrics(counts)))
        print(f"Scored {record['image']}: OpenCV {len(model_runs[0][1])}, CRAFT {len(regions)}", flush=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "predictions.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    for model in ("opencv", "craft"):
        for threshold in (.50, .75):
            subset = [r for r in rows if r["detector"] == model and r["iou"] == threshold]
            counts = {k: sum(r[k] for r in subset) for k in ("tp", "fp", "fn", "ignored")}
            print(f"{model:6} IoU {threshold:.2f}: {counts}; "
                  + ", ".join(f"{k} {v:.3f}" for k, v in metrics(counts).items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/craft"))
    parser.add_argument("--download", action="store_true", help="Allow EasyOCR to download pretrained weights")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("coordinate_space") != "exif_oriented_xywh" or not manifest.get("images"):
        parser.error("Expected a non-empty CVAT pilot manifest")
    try:
        import easyocr
        import torch
    except ImportError as exc:
        parser.error(f"CRAFT dependencies unavailable ({exc}); see README")
    torch.set_num_threads(4)
    reader = easyocr.Reader(["en"], gpu=False, recognizer=False, detect_network="craft",
                            model_storage_directory=str(args.weights), download_enabled=args.download,
                            verbose=False)
    run(manifest, args.image_root, args.output, reader)


if __name__ == "__main__":
    main()

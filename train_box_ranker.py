"""Train a box ranker from cached OpenCV/CRAFT proposals and CVAT annotations."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_detectors import metrics, rectangle, score


DETECTORS = ("opencv", "craft")
THRESHOLDS = (.2, .3, .4, .5, .6, .7)


def xyxy(points):
    a = np.asarray(points)
    return np.r_[a.min(axis=0), a.max(axis=0)]


def iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2])).prod(axis=1)
    return intersection / ((a[2:] - a[:2]).prod() + (b[:, 2:] - b[:, :2]).prod(axis=1) - intersection)


def box_xyxy(boxes):
    a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    a[:, 2:] += a[:, :2]
    return a


def polygons(boxes):
    return [rectangle([*b[:2], *(b[2:] - b[:2])]) for b in boxes]


def close_on_line(a, b):
    ah, bh = a[3]-a[1], b[3]-b[1]
    gap = b[0]-a[2]
    shared_height = min(a[3], b[3]) - max(a[1], b[1])
    return (min(ah, bh) >= .55 * max(ah, bh)
            and shared_height >= .65 * min(ah, bh)
            and -.20 * min(ah, bh) <= gap <= .85 * max(ah, bh))


def candidates(record, proposals):
    w, h = record["size_wh"]
    sources = {}
    for detector in DETECTORS:
        clipped = [np.clip(xyxy(p), 0, [w, h, w, h]) for p in proposals[detector]]
        sources[detector] = sorted((b for b in clipped if np.all(b[2:] > b[:2])), key=lambda b: b[0])
    boxes, vectors = [], []
    for detector in DETECTORS:
        raw = sources[detector]
        other_name = "craft" if detector == "opencv" else "opencv"
        other = np.asarray(sources[other_name]).reshape(-1, 4)
        for i, first in enumerate(raw):
            box = first
            last = i
            for count in range(1, 5):
                bw, bh = box[2:] - box[:2]
                agreement = float(iou(box, other).max(initial=0))
                vectors.append([np.log(bw/w), np.log(bh/h), np.log(bw/bh),
                                detector == "craft", count-1, agreement,
                                min(box[0], w-box[2])/w, min(box[1], h-box[3])/h])
                boxes.append(box)
                neighbors = [j for j in range(last+1, len(raw)) if close_on_line(raw[last], raw[j])
                             and close_on_line(box, raw[j])]
                if not neighbors:
                    break
                last = neighbors[0]
                next_box = raw[last]
                box = np.r_[np.minimum(box[:2], next_box[:2]), np.maximum(box[2:], next_box[2:])]
    return np.asarray(boxes).reshape(-1, 4), np.asarray(vectors).reshape(-1, 8)


def labeled(boxes, record):
    labels, ignored = (box_xyxy(record[k]) for k in ("text", "ignore"))
    keep, targets = [], []
    for box in boxes:
        positive = iou(box, labels).max(initial=0) >= .5
        inter = np.maximum(0, np.minimum(box[2:], ignored[:, 2:])
                           - np.maximum(box[:2], ignored[:, :2])).prod(axis=1)
        keep.append(positive or not np.any(inter >= .5 * (box[2:] - box[:2]).prod()))
        targets.append(positive)
    return np.asarray(keep, dtype=bool), np.asarray(targets, dtype=float)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit(examples):
    features, truth = [], []
    for record, boxes, vectors in examples:
        keep, target = labeled(boxes, record)
        features.extend(vectors[keep])
        truth.extend(target[keep])
    x, y = np.asarray(features), np.asarray(truth)
    if len(np.unique(y)) != 2:
        raise ValueError("Training needs both matching and nonmatching proposals")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1.e-8] = 1
    x = np.c_[np.ones(len(x)), (x-mean)/scale]
    weights = np.zeros(x.shape[1])
    penalty = np.diag([0, *([3.] * (x.shape[1]-1))])
    for _ in range(16):
        p = sigmoid(x @ weights)
        weights -= np.linalg.solve((x.T * (p*(1-p))) @ x + penalty,
                                   x.T @ (p-y) + penalty @ weights)
    model = {"version": 1, "mean": mean.tolist(), "scale": scale.tolist(),
             "weights": weights.tolist(), "positive_candidates": int(y.sum()), "candidates": len(y)}
    model["threshold"] = max(THRESHOLDS, key=lambda t: evaluate(model, examples, t, .5)["f1"])
    return model


def select(model, boxes, vectors, threshold):
    normalized = (vectors - model["mean"]) / model["scale"]
    probability = sigmoid(np.c_[np.ones(len(vectors)), normalized] @ model["weights"])
    pending = np.argsort(-probability, kind="stable")
    pending = pending[probability[pending] >= threshold]
    chosen = []
    while len(pending):
        index = pending[0]
        chosen.append(index)
        pending = pending[1:]
        pending = pending[iou(boxes[index], boxes[pending]) < .45]
    return boxes[chosen], probability[chosen]


def score_record(record, predicted, threshold):
    labels, ignored = (polygons(box_xyxy(record[k])) for k in ("text", "ignore"))
    return score(labels, predicted, ignored, threshold)


def evaluate(model, examples, confidence, threshold):
    counts = {"tp": 0, "fp": 0, "fn": 0, "ignored": 0}
    for record, boxes, vectors in examples:
        predicted, _ = select(model, boxes, vectors, confidence)
        row = score_record(record, polygons(predicted), threshold)
        for key in counts:
            counts[key] += row[key]
    return {**counts, **metrics(counts)}


def baseline_scores(record, raw, threshold):
    result = {}
    for detector in DETECTORS:
        counts = score_record(record, [np.float32(p) for p in raw[detector]], threshold)
        result[detector] = {**counts, **metrics(counts)}
    return result


def make_report(records, raw):
    examples = [(r, *candidates(r, raw[r["image"]])) for r in records]
    folds = {}
    for omitted in examples:
        name = omitted[0]["image"]
        model = fit([e for e in examples if e is not omitted])
        folds[name] = {
            str(iou_level): {
                **baseline_scores(omitted[0], raw[name], iou_level),
                "ranker": evaluate(model, [omitted], model["threshold"], iou_level)
            } for iou_level in (.5, .75)}
    final = fit(examples)
    final["training_images"] = [r["image"] for r in records]
    totals = {}
    for iou_level in ("0.5", "0.75"):
        totals[iou_level] = {}
        for detector in (*DETECTORS, "ranker"):
            c = {k: sum(row[iou_level][detector][k] for row in folds.values())
                 for k in ("tp", "fp", "fn", "ignored")}
            totals[iou_level][detector] = {**c, **metrics(c)}
    return final, {"leave_one_photo_out": folds, "pooled_leave_one_photo_out": totals}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    train = actions.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("proposals", type=Path, help="predictions.json from benchmark_detectors.py")
    train.add_argument("model", type=Path)
    train.add_argument("report", type=Path)
    infer = actions.add_parser("infer")
    infer.add_argument("model", type=Path)
    infer.add_argument("proposals", type=Path, help="predictions.json from benchmark_detectors.py")
    infer.add_argument("manifest", type=Path, help="Image names/sizes, labels not used")
    infer.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    if manifest["coordinate_space"] != "exif_oriented_xywh":
        parser.error("Expected an EXIF-oriented manifest")
    records = manifest["images"]
    raw = {}
    for row in load_json(args.proposals):
        image = raw.setdefault(row["image"], {})
        if row["detector"] in image:
            parser.error("Duplicate image/detector proposals")
        image[row["detector"]] = row["regions"]
    names = {r["image"] for r in records}
    if not names or len(names) != len(records) or names.difference(raw):
        parser.error("Missing proposals or duplicate manifest images")
    if any(not set(DETECTORS).issubset(raw[n]) for n in names):
        parser.error("Each image needs OpenCV and CRAFT proposals")
    if args.action == "train":
        if len(records) < 2 or args.model.exists() or args.report.exists():
            parser.error("Training requires multiple images and fresh output paths")
        model, report = make_report(records, raw)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.model, model)
        save_json(args.report, report)
        print(json.dumps(report["pooled_leave_one_photo_out"], indent=2))
    else:
        model = load_json(args.model)
        if model.get("version") != 1:
            parser.error("Unsupported box ranker version")
        result = []
        for r in records:
            boxes, confidence = select(model, *candidates(r, raw[r["image"]]), model["threshold"])
            result.append({"image": r["image"], "regions": [
                {"bbox_xywh": [*b[:2], *(b[2:]-b[:2])], "score": float(p)}
                for b, p in zip(boxes, confidence)]})
        save_json(args.output, {"coordinate_space": manifest["coordinate_space"], "images": result})


if __name__ == "__main__":
    main()

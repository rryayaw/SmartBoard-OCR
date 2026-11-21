"""Suggest ranker-supported CRAFT merges without changing detections."""

import numpy as np

from benchmark_detectors import rectangle
from smartboard import locate
from train_box_ranker import box_xyxy, candidates, close_on_line, iou, select


def suggest_merges(board, regions, model):
    if model.get("version") != 1:
        raise ValueError("Unsupported ranker version")
    if len(regions) < 2:
        return []
    h, w = board.enhanced.shape
    ranked, confidence = select(
        model,
        *candidates({"size_wh": [w, h]}, {
            "craft": [r["polygon"] for r in regions],
            "opencv": [rectangle(b) for b in locate(board.enhanced)],
        }),
        model["threshold"],
    )
    boxes = box_xyxy([r["bbox_xywh"] for r in regions])
    reviewed = []
    seen = set()
    for candidate, p in zip(ranked, confidence):
        inside = np.flatnonzero(np.all(boxes[:, :2] >= candidate[:2] - 2, axis=1)
                                & np.all(boxes[:, 2:] <= candidate[2:] + 2, axis=1))
        if len(inside) != 2:
            continue
        i, j = sorted(inside, key=lambda k: boxes[k, 0])
        a, b = boxes[i], boxes[j]
        union = np.r_[np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])]
        if (i, j) not in seen and close_on_line(a, b) and iou(candidate, union[None])[0] >= .85:
            seen.add((i, j))
            reviewed.append({"region_ids": [regions[i]["id"], regions[j]["id"]],
                             "score": float(p), "status": "suggestion_only"})
    return reviewed

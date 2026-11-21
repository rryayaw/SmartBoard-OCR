"""Compare original and enhanced inputs with fixed detector settings."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark_detectors import metrics, rectangle, score
from board_ocr import detect_board, detector_backend, save_review, write_json
from smartboard import prepare_board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--detector", choices=("craft", "pp-ocrv6"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("coordinate_space") != "exif_oriented_xywh":
        parser.error("Expected EXIF-oriented source coordinates")
    predict = detector_backend(args.detector, args.weights)
    args.output.mkdir(parents=True)
    root = args.image_root.resolve()
    rows = []
    for index, record in enumerate(manifest["images"], 1):
        path = (root / record["image"]).resolve()
        if not path.is_relative_to(root):
            parser.error("Manifest entry escaped image root")
        board = prepare_board(path, auto_rectify=False)
        if board.source_size != record["size_wh"]:
            parser.error(f"Size mismatch: {path.name}")
        variants = {}
        for variant in ("original", "enhanced"):
            started = perf_counter()
            regions = detect_board(board, predict, variant)
            row = dict(image=path.name, variant=variant, candidates=len(regions),
                       seconds=round(perf_counter()-started, 4))
            variants[variant] = regions
            if record["text"]:
                labels, ignore = ([rectangle(b) for b in record[k]] for k in ("text", "ignore"))
                polygons = [np.float32(r["source_polygon"]) for r in regions]
                row["scores"] = {str(t): score(labels, polygons, ignore, t) for t in (.5, .75)}
            rows.append(row)
        output = args.output / f"{index:02d}-{path.stem}"
        result = save_review(output, path, board, variants, "original", args.detector)
        write_json(output / "results.json", result)
        print(path.name, *(len(v) for v in variants.values()), flush=True)
    pooled = {}
    for variant in ("original", "enhanced"):
        group = [r["scores"] for r in rows if r["variant"] == variant and "scores" in r]
        if group:
            pooled[variant] = {}
            for t in ("0.5", "0.75"):
                counts = {k: sum(s[t][k] for s in group) for k in ("tp", "fp", "fn", "ignored")}
                pooled[variant][t] = {**counts, **metrics(counts)}
    write_json(args.output / "comparison.json", {"detector": args.detector, "images": rows, "pooled": pooled})


if __name__ == "__main__":
    main()

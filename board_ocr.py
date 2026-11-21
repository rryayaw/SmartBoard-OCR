"""Run CRAFT on a prepared whiteboard and export reviewable results."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmark_detectors import craft_regions, project_regions
from smartboard import prepare_board


def load_craft(weights, download=False):
    import easyocr
    import torch

    torch.set_num_threads(4)
    return easyocr.Reader(
        ["en"], gpu=False, recognizer=False, detect_network="craft",
        model_storage_directory=str(weights), download_enabled=download, verbose=False,
    )


def region_list(polygons, board):
    height, width = board.enhanced.shape
    output = []
    for polygon in polygons:
        corners = np.clip(polygon, (0, 0), (width - 1, height - 1)).astype(np.float32)
        x, y, w, h = cv2.boundingRect(corners)
        if cv2.contourArea(corners) <= 0:
            continue
        projected = project_regions([corners], board.transform)[0]
        output.append({"bbox_xywh": [x, y, w, h], "polygon": corners.tolist(),
                       "source_polygon": projected.tolist()})
    output.sort(key=lambda r: (r["bbox_xywh"][1], r["bbox_xywh"][0]))
    return [{"id": f"r{i:03d}", **r} for i, r in enumerate(output, 1)]


def detect_board(board, reader, variant):
    rgb = board.color if variant == "original" else cv2.cvtColor(board.enhanced, cv2.COLOR_GRAY2RGB)
    return region_list(craft_regions(reader, rgb), board)


def draw_regions(image, regions):
    overlay = image.copy()
    for region in regions:
        points = np.rint(region["polygon"]).astype(np.int32)
        cv2.polylines(overlay, [points], True, (0, 170, 95), 2, cv2.LINE_AA)
        x, y, _, _ = region["bbox_xywh"]
        cv2.putText(overlay, region["id"], (x, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 125, 60), 1, cv2.LINE_AA)
    return overlay


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_review(output, image, board, detected, selected):
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(board.color).save(output / "board.png")
    Image.fromarray(board.enhanced).save(output / "corrected.png")
    for variant, regions in detected.items():
        image_data = (board.color if variant == "original" else
                      cv2.cvtColor(board.enhanced, cv2.COLOR_GRAY2RGB))
        Image.fromarray(draw_regions(image_data, regions)).save(output / f"overlay-{variant}.png")
    return {"source": image.name, "source_size_wh": board.source_size,
            "source_to_board": board.transform.tolist(), "geometry": board.geometry,
            "detector": "pretrained_craft", "selected_variant": selected,
            "regions": detected[selected], "variants": detected,
            "reading_order": "approximate_top_to_bottom",
            "warnings": (["Board outline not found; full frame retained. Use --crop to exclude surrounding content."]
                         if board.geometry == "full_frame_fallback" else [])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/craft"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--no-rectify", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Run both original color and enhanced grayscale")
    parser.add_argument("--variant", choices=("original", "enhanced"), default="original")
    parser.add_argument("--recognize", action="store_true", help="Export local TrOCR guesses and crops")
    parser.add_argument("--trocr-model", type=Path, default=Path("models/trocr-small-handwritten"))
    parser.add_argument("--ocr-limit", type=int, default=10, help="Maximum regions to transcribe")
    args = parser.parse_args(argv)
    if args.ocr_limit < 0:
        parser.error("--ocr-limit cannot be negative")
    if args.output.exists():
        parser.error("Output directory already exists")
    try:
        board = prepare_board(args.image, crop=args.crop, auto_rectify=not args.no_rectify)
        reader = load_craft(args.weights, args.download)
        variants = ("original", "enhanced") if args.compare else (args.variant,)
        detected = {variant: detect_board(board, reader, variant) for variant in variants}
        report = save_review(args.output, args.image, board, detected, args.variant)
        if args.recognize:
            from handwritten import export_transcript

            oc_image = board.color if args.variant == "original" else board.enhanced
            report["recognition"] = export_transcript(args.output, oc_image, report["regions"],
                                                      args.trocr_model, args.ocr_limit)
        write_json(args.output / "results.json", report)
    except (OSError, ValueError, ImportError, Image.DecompressionBombError) as exc:
        parser.exit(1, f"Board OCR: {exc}\n")
    print(", ".join(f"{variant}: {len(regions)} candidates" for variant, regions in detected.items()))


if __name__ == "__main__":
    main()

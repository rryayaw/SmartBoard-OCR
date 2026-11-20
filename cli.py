"""Run SmartBoard OCR on one local image."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image

from smartboard import detect


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New directory for results (not overwritten)")
    parser.add_argument("--no-rectify", action="store_true", help="Retain the entire image")
    parser.add_argument("--recognize", action="store_true", help="Enable optional Tesseract recognition")
    parser.add_argument("--tesseract", default="tesseract", help="Tesseract executable or its full path")
    parser.add_argument("--language", default="eng", help="Installed Tesseract language code")
    parser.add_argument("--ocr-budget", type=float, default=15, help="Total recognition time budget in seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.ocr_budget) or args.ocr_budget <= 0:
        parser.error("--ocr-budget must be a finite positive number")
    try:
        result = detect(args.image, auto_rectify=not args.no_rectify)
        args.output.mkdir(parents=True, exist_ok=False)
        Image.fromarray(result.corrected).save(args.output / "corrected.png")
        Image.fromarray(result.overlay).save(args.output / "overlay.png")
        report = result.report
        report["recognition"] = {"status": "disabled"}
        if args.recognize:
            try:
                from recognition import recognize

                report["recognition"] = recognize(
                    result.corrected, report["regions"], args.tesseract, args.language, args.ocr_budget
                )
            except Exception as exc:
                # This extension must not discard already-successful detection.
                report["recognition"] = {"status": "failed", "warning": f"Recognition extension error: {type(exc).__name__}."}
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        parser.exit(1, f"SmartBoard: {exc}\n")
    print(f"{len(report['regions'])} candidate region(s); recognition: {report['recognition']['status']}")
    for warning in [*report["warnings"], report["recognition"].get("warning")]:
        if warning:
            print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

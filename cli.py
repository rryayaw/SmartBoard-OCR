"""Run SmartBoard OCR on one local image."""

import argparse
import json
from pathlib import Path

from PIL import Image

from smartboard import detect


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New directory for results (not overwritten)")
    parser.add_argument("--no-rectify", action="store_true", help="Retain the entire image")
    args = parser.parse_args(argv)
    try:
        result = detect(args.image, auto_rectify=not args.no_rectify)
        args.output.mkdir(parents=True, exist_ok=False)
        Image.fromarray(result.corrected).save(args.output / "corrected.png")
        Image.fromarray(result.overlay).save(args.output / "overlay.png")
        report = result.report
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        parser.exit(1, f"SmartBoard: {exc}\n")
    print(f"{len(report['regions'])} candidate region(s)")
    for warning in report["warnings"]:
        if warning:
            print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

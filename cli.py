"""Process one local whiteboard photo."""

import argparse
from pathlib import Path

from PIL import Image

import numpy as np
from smartboard import enhance, load_photo, rectify


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New directory for results (not overwritten)")
    parser.add_argument("--no-rectify", action="store_true", help="Retain the entire image")
    args = parser.parse_args(argv)
    try:
        rgb, _, _ = load_photo(args.image)
        board, _, applied = rectify(rgb) if not args.no_rectify else (rgb, np.eye(3), False)
        processed = enhance(board)
        if not args.no_rectify and not applied:
            print("Warning: board outline not found; retained the full frame.")
        args.output.mkdir(parents=True, exist_ok=False)
        Image.fromarray(processed).save(args.output / "corrected.png")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        parser.exit(1, f"SmartBoard: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

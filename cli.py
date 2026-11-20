"""Process one local whiteboard photo."""

import argparse
from pathlib import Path

from PIL import Image

from smartboard import load_photo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New directory for results (not overwritten)")
    args = parser.parse_args(argv)
    try:
        rgb, _, _ = load_photo(args.image)
        processed = rgb
        args.output.mkdir(parents=True, exist_ok=False)
        Image.fromarray(processed).save(args.output / "oriented.png")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        parser.exit(1, f"SmartBoard: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

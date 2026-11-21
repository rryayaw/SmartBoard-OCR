"""Local, review-first handwriting transcription for detected board regions."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def make_crop(image, box):
    x, y, w, h = box
    pad = max(4, round(.12 * h))
    crop = image.crop((max(0, x-pad), max(0, y-pad),
                       min(image.width, x+w+pad), min(image.height, y+h+pad)))
    return ImageOps.expand(crop, border=6, fill="white").convert("RGB")


def load_trocr(model_dir):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    torch.set_num_threads(4)
    processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True).eval()

    def transcribe(crop):
        inputs = processor(images=crop, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(inputs.pixel_values, max_new_tokens=48, num_beams=2)
        return " ".join(processor.batch_decode(generated, skip_special_tokens=True)[0].split())

    return transcribe


def export_transcript(output, image, regions, model_dir, limit):
    source = Image.fromarray(image)
    crop_dir = output / "crops"
    crop_dir.mkdir()
    recognize = None
    warning = None
    if limit and regions:
        try:
            recognize = load_trocr(model_dir)
        except (OSError, ImportError, ValueError, RuntimeError) as exc:
            warning = f"TrOCR unavailable: {type(exc).__name__}: {exc}"
    entries = {}
    lines = []
    for number, region in enumerate(regions):
        key = region["id"]
        crop = make_crop(source, region["bbox_xywh"])
        path = f"crops/{key}.png"
        crop.save(output / path)
        text, status = "", "not_attempted"
        if recognize is not None and number < limit:
            try:
                text = recognize(crop)
                status = "needs_review" if text else "unreadable"
            except (ValueError, RuntimeError) as exc:
                status = "recognition_error"
                warning = f"TrOCR could not read {key}: {type(exc).__name__}: {exc}"
        entries[key] = {"status": status, "text": text, "crop": path}
        lines.append(f"{key}\t{text or '[' + status + ']'}")
    (output / "transcript.txt").write_text("UNVERIFIED OCR - review against crops\n\n" +
                                          "\n".join(lines) + "\n", encoding="utf-8")
    return {"engine": "trocr_small_handwritten", "by_region": entries,
            "status": "review_required" if recognize else ("unavailable" if warning else "not_attempted"),
            "warning": warning}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path, help="Existing board_ocr.py output directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/trocr-small-handwritten"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    report = json.loads((args.review / "results.json").read_text(encoding="utf-8"))
    image_name = "corrected.png" if report["selected_variant"] == "enhanced" else "board.png"
    args.output.mkdir(parents=True)
    with Image.open(args.review / image_name) as image:
        recognition = export_transcript(args.output, np.asarray(image), report["regions"], args.model, args.limit)
    (args.output / "recognition.json").write_text(
        json.dumps(recognition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if recognition["warning"]:
        print(recognition["warning"])


if __name__ == "__main__":
    main()

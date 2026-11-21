"""Train or run the experimental SmartBoard handwriting-box detector."""

import argparse
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from benchmark_detectors import metrics, rectangle, score


STRIDE = 4
VERSION = 1


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU())


class LineDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail = nn.Sequential(block(3, 16, 2), block(16, 24, 2))
        self.context = nn.Sequential(block(24, 32, 2), block(32, 48, 2),
                                     block(48, 48), block(48, 32))
        self.head = nn.Sequential(block(56, 32), block(32, 24), nn.Conv2d(24, 5, 1))
        with torch.no_grad():
            self.head[-1].bias[0] = -2.19

    def forward(self, image):
        detail = self.detail(image)
        context = F.interpolate(self.context(detail), size=detail.shape[-2:], mode="nearest")
        return self.head(torch.cat((detail, context), dim=1))


def read_rgb(path):
    with Image.open(path) as photo:
        return np.asarray(ImageOps.exif_transpose(photo).convert("RGB"))


def prepare(rgb, size, rng=None):
    h, w = rgb.shape[:2]
    factor = size / max(w, h) * (rng.uniform(.82, 1) if rng else 1)
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    x = rng.randint(0, size - nw) if rng else (size - nw) // 2
    y = rng.randint(0, size - nh) if rng else (size - nh) // 2
    frame = np.full((size, size, 3), 128, np.uint8)
    frame[y:y+nh, x:x+nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255
    if rng:
        frame = np.clip(frame * rng.uniform(.85, 1.15) + rng.uniform(-.07, .07), 0, 1)
    return torch.from_numpy(frame).permute(2, 0, 1), (nw / w, nh / h, x, y, w, h)


def targets(record, size, geometry):
    sx, sy, ox, oy, _, _ = geometry
    n = size // STRIDE
    grid_x, grid_y = np.meshgrid(np.arange(n) + .5, np.arange(n) + .5)
    gx = (grid_x * STRIDE - ox) / sx
    gy = (grid_y * STRIDE - oy) / sy
    heat = np.zeros((n, n), np.float32)
    valid = np.ones((n, n), bool)
    loc = np.zeros((4, n, n), np.float32)
    centers = np.zeros((n, n), bool)
    for x, y, w, h in record["ignore"]:
        valid &= ~((gx >= x) & (gx < x+w) & (gy >= y) & (gy < y+h))
    for x, y, w, h in record["text"]:
        cx = ((x + w/2) * sx + ox) / STRIDE
        cy = ((y + h/2) * sy + oy) / STRIDE
        bw, bh = w * sx / STRIDE, h * sy / STRIDE
        ix, iy = min(n-1, int(cx)), min(n-1, int(cy))
        # Nearby cells are softer negatives, not extra positive centers.
        bump = np.exp(-.5 * (((grid_x - cx) / max(.6, min(2, bw/6)))**2
                             + ((grid_y - cy) / max(.6, min(2, bh/6)))**2))
        heat = np.maximum(heat, np.minimum(bump, .999).astype(np.float32))
        if not centers[iy, ix]:
            loc[:, iy, ix] = (cx-ix, cy-iy, math.log(bw), math.log(bh))
            centers[iy, ix] = True
    heat[centers] = 1
    valid[centers] = True
    return tuple(torch.from_numpy(v) for v in (heat, valid, loc, centers))


def criterion(output, heat, valid, loc, centers):
    logits = output[:, 0]
    p = logits.sigmoid()
    positives = centers.sum().clamp_min(1)
    focal = ((1-p).square() * F.softplus(-logits) * centers
             + p.square() * (1-heat).pow(4) * F.softplus(logits) * (valid & ~centers))
    predicted = torch.cat((output[:, 1:3].sigmoid(), output[:, 3:]), dim=1)
    regression = F.smooth_l1_loss(predicted, loc, reduction="none") * centers[:, None]
    return focal.sum() / positives + 2 * regression.sum() / positives


@torch.inference_mode()
def predict(model, rgb, size, threshold):
    image, (sx, sy, ox, oy, w, h) = prepare(rgb, size)
    output = model(image[None])[0]
    heat = output[0].sigmoid()
    ys, xs = torch.where((heat >= threshold) & (heat == F.max_pool2d(heat[None], 3, 1, 1)[0]))
    confidence = heat[ys, xs]
    top = confidence.argsort(descending=True)[:500]
    ys, xs, confidence = ys[top], xs[top], confidence[top]
    dx, dy = output[1:3, ys, xs].sigmoid()
    bw, bh = output[3:5, ys, xs].clamp(-5, 8).exp() * STRIDE
    cx, cy = (xs + dx) * STRIDE, (ys + dy) * STRIDE
    boxes = torch.stack((((cx-bw/2-ox) / sx).clamp(0, w), ((cy-bh/2-oy) / sy).clamp(0, h),
                         ((cx+bw/2-ox) / sx).clamp(0, w), ((cy+bh/2-oy) / sy).clamp(0, h)), dim=1)
    keep = (cx >= ox) & (cx < ox+sx*w) & (cy >= oy) & (cy < oy+sy*h)
    keep &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, confidence = boxes[keep], confidence[keep]
    return [
        {"bbox_xywh": [x0, y0, x1-x0, y1-y0], "score": float(confidence[i])}
        for i in nms(boxes, confidence, .6).tolist()
        for x0, y0, x1, y1 in [boxes[i].tolist()]
    ]


def fit(examples, args, name):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = LineDetector().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0002)
    history = []
    for epoch in range(1, args.epochs+1):
        items = []
        for record, rgb in examples:
            image, geometry = prepare(rgb, args.size, rng)
            items.append((image, *targets(record, args.size, geometry)))
        image, heat, valid, loc, centers = map(torch.stack, zip(*items))
        optimizer.zero_grad()
        loss = criterion(model(image.contiguous(memory_format=torch.channels_last)), heat, valid, loc, centers)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in {name} at epoch {epoch}")
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append(float(loss.detach()))
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"{name} epoch {epoch}/{args.epochs}: loss {history[-1]:.3f}", flush=True)
    return model.eval(), history


def save_predictions(output, rgb, predictions, source):
    overlay = Image.fromarray(rgb)
    overlay.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    sx, sy = overlay.width/rgb.shape[1], overlay.height/rgb.shape[0]
    draw = ImageDraw.Draw(overlay)
    for region in predictions:
        x, y, w, h = region["bbox_xywh"]
        draw.rectangle((x*sx, y*sy, (x+w)*sx, (y+h)*sy), outline="#00e577", width=2)
    overlay.save(output / "overlay.png")
    write_json(output / "results.json", {"image": source, "coordinate_space": "exif_oriented_xywh",
                                         "regions": predictions})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("coordinate_space") != "exif_oriented_xywh":
        raise ValueError("Expected EXIF-oriented CVAT manifest")
    root = args.image_root.resolve()
    examples = []
    for record in manifest["images"]:
        path = (root / record["image"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image outside source folder: {path}")
        rgb = read_rgb(path)
        if list(rgb.shape[1::-1]) != record["size_wh"]:
            raise ValueError(f"Image size mismatch: {path}")
        examples.append((record, rgb))
    sanity = [ex for ex in examples if ex[0]["image"] == args.holdout]
    training = [ex for ex in examples if ex[0]["image"] != args.holdout]
    if len(sanity) != 1 or not training or len({r["image"] for r, _ in examples}) != len(examples):
        raise ValueError("Expected unique images and exactly one sanity-check photo")
    args.output.mkdir(parents=True, exist_ok=False)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    record, rgb = sanity[0]
    model, sanity_history = fit(training, args, f"{len(training)}-photo sanity run")
    predictions = predict(model, rgb, args.size, args.threshold)
    labels, ignored = ([rectangle(b) for b in record[k]] for k in ("text", "ignore"))
    polygons = [rectangle(p["bbox_xywh"]) for p in predictions]
    evaluation = {}
    for iou in (.50, .75):
        counts = score(labels, polygons, ignored, iou)
        evaluation[str(iou)] = {**counts, **metrics(counts)}
    save_predictions(args.output, rgb, predictions, record["image"])
    write_json(args.output / "sanity-metrics.json", {"image": record["image"],
               "not_in_sanity_training": True, "included_in_final_model": True,
               "threshold": args.threshold, "metrics_by_iou": evaluation})
    final, final_history = fit(examples, args, "all-photo final run")
    with args.checkpoint.open("xb") as stream:
        torch.save({"version": VERSION, "size": args.size, "threshold": args.threshold,
                    "state_dict": final.state_dict()}, stream)
    write_json(args.output / "training.json", {
        "manifest": str(args.manifest), "checkpoint": str(args.checkpoint),
        "seed": args.seed, "size": args.size, "epochs_each": args.epochs,
        "sanity_training_images": [r["image"] for r, _ in training],
        "all_training_images": [r["image"] for r, _ in examples],
        "sanity_loss": sanity_history, "final_loss": final_history})
    print(f"Saved {args.checkpoint}; sanity result: {evaluation}", flush=True)


def infer(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["version"] != VERSION:
        raise ValueError("Unsupported model version")
    model = LineDetector()
    model.load_state_dict(checkpoint["state_dict"])
    rgb = read_rgb(args.image)
    threshold = checkpoint["threshold"] if args.threshold is None else args.threshold
    predictions = predict(model.eval(), rgb, checkpoint["size"], threshold)
    args.output.mkdir(parents=True, exist_ok=False)
    save_predictions(args.output, rgb, predictions, args.image.name)
    print(f"Saved {len(predictions)} predicted text regions in {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    t = modes.add_parser("train")
    t.add_argument("manifest", type=Path)
    t.add_argument("image_root", type=Path)
    t.add_argument("checkpoint", type=Path)
    t.add_argument("output", type=Path)
    t.add_argument("--holdout", default="angled.jpg")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--size", type=int, default=448)
    t.add_argument("--seed", type=int, default=14)
    t.add_argument("--threshold", type=float, default=.15)
    i = modes.add_parser("infer")
    i.add_argument("checkpoint", type=Path)
    i.add_argument("image", type=Path)
    i.add_argument("output", type=Path)
    i.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    if args.mode == "train" and (args.epochs < 1 or args.size < 128 or args.size % 16):
        parser.error("epochs must be positive and size a multiple of 16, at least 128")
    torch.set_num_threads(4)
    (train if args.mode == "train" else infer)(args)


if __name__ == "__main__":
    main()

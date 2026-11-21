# SmartBoard OCR

Automated Whiteboard Diagram and Text Extraction System.

A computer vision course project for turning photographs of handwritten
whiteboard content into a deskewed image and extracted, searchable text.

## Run Locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe cli.py "C:\path\to\board.jpg" outputs\board-01
```

Use a new output directory for each run. The CLI writes:

- `corrected.png`: enhanced grayscale, rectified if a board outline is found.
- `overlay.png`: numbered boxes for **candidate** text regions.
- `results.json`: boxes in `[x, y, width, height]` pixels on `corrected.png`,
  the transform from the EXIF-oriented source, warnings, and detection time.

Use `--no-rectify` if automatic cropping removes content. Region IDs identify
boxes in a run, not a reading order.

### Optional Recognition

Install Tesseract separately with the desired language data; it is not required
for detection. Add `--recognize` and optionally `--tesseract` with the full
executable path:

```powershell
.\.venv\Scripts\python.exe cli.py "C:\path\to\board.jpg" outputs\board-01-ocr --recognize
```

The adapter reads each grayscale crop as a single line and stores text by
region ID under `recognition` in the JSON. Missing assets, errors, or timeouts
do not invalidate the detection output. Omit `--recognize` to return to the
detection-only flow.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic images and simulated OCR responses. Real-board localization
accuracy and handwriting recognition quality are not established. Evaluate on
reviewed boxes first; add checked transcriptions only when testing recognition.

## Import CVAT Labels

Export annotations as CVAT COCO 1.0 ZIPs, then pass the ZIPs and the local
directory containing the matching original images:

```powershell
.\.venv\Scripts\python.exe coco_pilot.py data\dev\variated outputs\pilot-manifest.json $zip1 $zip2
```

Set `$zip1`, `$zip2`, etc. to the paths of your CVAT export files. The importer checks
EXIF-oriented dimensions, box bounds, and duplicate images/boxes. It preserves
explicit `ignore` boxes separately from `text` and does not infer glare masks
or change your source files. Keep the manifest local. These pilot examples are
for development, not a held-out test split.

### Compare OpenCV With Pretrained CRAFT

CRAFT is an optional CPU benchmark and does not change the default CLI.
In your virtual environment, install CPU PyTorch and the optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-craft.txt
.\.venv\Scripts\python.exe benchmark_detectors.py outputs\pilot-manifest.json data\dev\variated outputs\craft-benchmark --download
```

Use `--download` on the first run to allow EasyOCR to fetch its pretrained
detector weights into `models/craft`. No recognition model is loaded. Both
pipelines are scored against the same boxes using one-to-one matching at
IoU 0.50 and 0.75, with source-space polygons and per-image scores in the run
folder. An unmatched prediction is ignored only if an explicit ignore box
covers at least half its area. A word box is not a match for a full line unless
it meets the same overlap rule. This compares pipeline defaults, not a
controlled ablation of preprocessing or grouping.

## Train the Five-Photo Prototype

The optional training script uses CPU PyTorch and torchvision (same install
command as above; EasyOCR is not needed). It trains a small box detector from
scratch. Use the five-image CVAT manifest and matching originals:

```powershell
.\.venv\Scripts\python.exe train_detector.py train outputs\pilot-label-review\pilot-manifest-review-v2.json data\dev\variated models\pilot-line\pilot-five.pt outputs\pilot-line-training
.\.venv\Scripts\python.exe train_detector.py infer models\pilot-line\pilot-five.pt data\dev\variated\clean.jpg outputs\pilot-line-example
```

If using the local `outputs\craft-deps` directory instead of installing PyTorch
in the venv, set `$env:PYTHONPATH = (Resolve-Path outputs\craft-deps).Path` first.
Use fresh output and checkpoint paths for reruns.

By default the script trains on four photos and saves a one-photo sanity check
on `angled.jpg`, then trains a fresh checkpoint on all five. The sanity check
is not an independent test of the final weights. Inference exports boxes and
confidence scores in EXIF-oriented source coordinates, plus a PNG overlay. It
does not transcribe handwriting or replace the OpenCV CLI.

### Train the Proposal Ranker

An alternative to training a detector from scratch: generate OpenCV and
pretrained CRAFT proposals with `benchmark_detectors.py`, then train the
ranker from the same CVAT manifest and the benchmark's `predictions.json`.
The ranker considers short groups of nearby boxes alongside the original
proposals. It learns which candidates resemble the annotated regions;
the OpenCV and CRAFT weights and settings are unchanged.

```powershell
.\.venv\Scripts\python.exe train_box_ranker.py train outputs\pilot-label-review\pilot-manifest-review-v2.json outputs\pilot-label-review\benchmark-draft\predictions.json models\pilot-line\box-ranker.json outputs\pilot-line-ranker\metrics.json
.\.venv\Scripts\python.exe train_box_ranker.py infer models\pilot-line\box-ranker.json outputs\pilot-label-review\benchmark-draft\predictions.json outputs\pilot-label-review\pilot-manifest-review-v2.json outputs\pilot-line-ranker\predictions.json
```

The metrics omit one photo at a time and fit both the ranker and its score
cutoff on the remaining photos. The saved model is separately fit on all five.
For new images, generate proposals using a manifest with image names, oriented
sizes, and empty `text` and `ignore` lists, then run the ranker `infer` command
with that manifest. Inference never uses labels. Leave-one-photo-out scores on
the development pilot do not establish accuracy on independent boards.

## Status

The default CLI still uses OpenCV heuristics, with no training or CRAFT
dependency. The optional trained detector is a five-photo experiment; there
is no web interface or deployment yet.

Custom datasets, model weights, credentials, and local development logs are not
included in this repository.

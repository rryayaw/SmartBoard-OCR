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

## Status

Local CLI baseline only. The detector uses OpenCV heuristics, with no custom
training or CRAFT dependency. There is no web interface or deployment yet.

Custom datasets, model weights, credentials, and local development logs are not
included in this repository.

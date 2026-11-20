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

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This stage uses synthetic tests; real-board localization accuracy has not been measured.

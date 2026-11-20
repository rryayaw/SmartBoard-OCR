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

This stage corrects EXIF orientation, resizes large photos, and writes `oriented.png`.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

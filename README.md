# SmartBoard OCR

Automated Whiteboard Diagram and Text Extraction System.

A computer vision course project for turning photographs of handwritten
whiteboard content into a deskewed image and extracted, searchable text.

## Planned Approach

- OpenCV preprocessing and perspective correction.
- PyTesseract/Tesseract as the initial OCR baseline.
- Optional CRAFT text detection before recognition, subject to baseline results.
- A Flask or Streamlit interface for image uploads and results.

## Evaluation

Planned evaluation covers text localization (IoU), recognition (CER/WER), and
end-to-end processing time using scene-text benchmarks and custom whiteboard
photographs.

## Status

Project planning and repository setup. No application pipeline or benchmark
results are implemented yet.

Custom datasets, model weights, credentials, and local development logs are not
included in this repository.

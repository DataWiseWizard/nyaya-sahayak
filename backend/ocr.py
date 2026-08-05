"""
ocr.py
------
Extracts text from an uploaded photo of a document (e.g. a scanned notice,
court order, or judgment page) using Tesseract OCR, so that content can be
fed into the same text-based chat/RAG pipeline as a typed question.

Scope note (agreed with the user): this does TEXT extraction only, not
visual/image understanding. A photo of a document gets read as text; this
is NOT a vision-capable model that can interpret photos of scenes,
objects, injuries, etc. — that would need a genuinely multimodal local
model, which is a much heavier addition for a narrower use case here.

Requires the Tesseract OCR binary to be installed separately — pip alone
is not enough, pytesseract is just a Python wrapper around the binary:
  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
    If it's not automatically on PATH afterwards, set the TESSERACT_CMD
    environment variable to the full path of tesseract.exe.
  Mac: brew install tesseract
  Linux: sudo apt install tesseract-ocr
"""

from __future__ import annotations

import os
from io import BytesIO

import pytesseract
from PIL import Image

_custom_cmd = os.environ.get("TESSERACT_CMD")
if _custom_cmd:
    pytesseract.pytesseract.tesseract_cmd = _custom_cmd


class OCRNotAvailableError(RuntimeError):
    """Raised when the Tesseract binary itself can't be found — distinct
    from a normal OCR failure, so the UI can show clear install guidance."""


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Runs OCR on raw image bytes (e.g. from Streamlit's file_uploader) and
    returns the extracted text, cleaned of excessive blank lines.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Could not read this as an image: {e}") from e

    try:
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as e:
        raise OCRNotAvailableError(
            "Tesseract OCR isn't installed or isn't on PATH. Install it "
            "from https://github.com/UB-Mannheim/tesseract/wiki (Windows), "
            "'brew install tesseract' (Mac), or 'apt install tesseract-ocr' "
            "(Linux), then restart the app. If it's installed but still not "
            "found, set the TESSERACT_CMD environment variable to the full "
            "path of the tesseract executable."
        ) from e

    # Collapse noisy OCR blank lines while preserving paragraph structure
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return cleaned.strip()
"""OCR interface with availability probing. Graceful when no engine exists."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class OcrAvailability:
    available: bool
    engine: str = ""
    reason: str = ""


def probe() -> OcrAvailability:
    """Detect any usable OCR path without importing heavy optional deps blindly."""
    try:
        import pytesseract  # noqa: F401

        if shutil.which("tesseract"):
            return OcrAvailability(True, engine="tesseract+pytesseract")
        return OcrAvailability(
            False, reason="pytesseract installed but tesseract binary not found"
        )
    except ImportError:
        pass
    for module in ("easyocr", "paddleocr"):
        try:
            __import__(module)
            return OcrAvailability(True, engine=module)
        except ImportError:
            continue
    return OcrAvailability(
        False,
        reason="no OCR engine installed (tesseract/pytesseract, easyocr, paddleocr all missing)",
    )


def run_on_image(image_path: str, engine: Optional[str] = None) -> str:
    availability = probe()
    if not availability.available:
        raise RuntimeError(availability.reason)
    engine = engine or availability.engine
    if engine == "tesseract+pytesseract":
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as im:
            return pytesseract.image_to_string(im)
    if engine == "easyocr":
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        return "\n".join(reader.readtext(image_path, detail=0))
    raise RuntimeError(f"OCR engine '{engine}' has no runner in this harness")

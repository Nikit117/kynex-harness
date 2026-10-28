"""Image backend (pillow). Metadata only — pixels need OCR/vision escalation."""

from __future__ import annotations

from .base import ParsedDocument, ParsedPage
from ..workspace.corpus import DocumentRef


class ImageParser:
    name = "image"

    def parse(self, ref: DocumentRef) -> ParsedDocument:
        try:
            from PIL import Image
        except ImportError:
            return ParsedDocument(
                ref.doc_id,
                ref.fingerprint,
                ref.file_type,
                errors=["pillow is not installed"],
            )
        try:
            with Image.open(ref.path) as im:
                width, height, mode, fmt = im.width, im.height, im.mode, im.format
        except Exception as exc:
            return ParsedDocument(
                ref.doc_id,
                ref.fingerprint,
                ref.file_type,
                errors=[f"image unreadable: {type(exc).__name__}: {exc}"],
            )
        page = ParsedPage(page_no=1, text="", char_count=0, image_coverage=1.0)
        meta = {
            "paginated": False,
            "needs_ocr": True,
            "width": width,
            "height": height,
            "mode": mode,
            "format": str(fmt),
        }
        return ParsedDocument(
            ref.doc_id,
            ref.fingerprint,
            ref.file_type,
            pages=[page],
            meta=meta,
            warnings=["image-only source; OCR/vision required for content"],
        )

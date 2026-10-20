"""Builtin parser registration. Importing this package wires all formats."""

from .base import ParserRegistry
from .docx import DocxParser
from .image import ImageParser
from .pdf import PdfParser
from .spreadsheet import SpreadsheetParser
from .text import TextParser

__all__ = [
    "ParserRegistry",
    "PdfParser",
    "DocxParser",
    "SpreadsheetParser",
    "ImageParser",
    "TextParser",
    "build_default_registry",
]


def build_default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    pdf, docx, sheet, image, text = (
        PdfParser(),
        DocxParser(),
        SpreadsheetParser(),
        ImageParser(),
        TextParser(),
    )
    for ext in (".pdf",):
        registry.register(ext, pdf)
    for ext in (".docx",):
        registry.register(ext, docx)
    for ext in (".xlsx", ".xlsm"):
        registry.register(ext, sheet)
    for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        registry.register(ext, image)
    for ext in (".txt", ".md", ".csv"):
        registry.register(ext, text)
    return registry

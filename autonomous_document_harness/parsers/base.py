"""Parser interfaces. Parser-specific behavior stays behind this boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..workspace.corpus import DocumentRef


@dataclass
class TextBlock:
    page: Optional[int]
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class TableData:
    rows: List[List[str]]
    page: Optional[int]
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)


@dataclass
class ParsedPage:
    page_no: int  # 1-based; non-paginated formats use a single page_no=1
    text: str = ""
    blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    char_count: int = 0
    image_coverage: float = 0.0
    image_bboxes: List[Tuple[float, float, float, float]] = field(
        default_factory=list
    )
    table_detection_failed: bool = False


@dataclass
class SheetData:
    name: str
    max_row: int = 0
    max_col: int = 0
    values: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    merged: List[str] = field(default_factory=list)
    has_formulas: bool = False

    def cell_text(self, row: int, col: int) -> str:
        value = self.values.get((row, col), "")
        return "" if value is None else str(value)


@dataclass
class ParsedDocument:
    doc_id: str
    fingerprint: str
    file_type: str
    pages: List[ParsedPage] = field(default_factory=list)
    sheets: List[SheetData] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages if p.text)


class DocumentParser(Protocol):
    name: str

    def parse(self, ref: DocumentRef) -> ParsedDocument: ...


class ParserRegistry:
    """Extension-dispatched parsers. Unknown formats degrade gracefully."""

    def __init__(self) -> None:
        self._parsers: Dict[str, Any] = {}

    def register(self, ext: str, parser: Any) -> None:
        self._parsers[ext.lower()] = parser

    def parse(self, ref: DocumentRef) -> ParsedDocument:
        parser = self._parsers.get(ref.file_type.lower())
        if parser is None:
            return ParsedDocument(
                doc_id=ref.doc_id,
                fingerprint=ref.fingerprint,
                file_type=ref.file_type,
                meta={"supported": False},
                errors=[f"no parser for '{ref.file_type}' ({ref.filename})"],
            )
        try:
            return parser.parse(ref)
        except Exception as exc:  # parsers must never take down a run
            return ParsedDocument(
                doc_id=ref.doc_id,
                fingerprint=ref.fingerprint,
                file_type=ref.file_type,
                errors=[
                    f"parser crashed on {ref.filename}: {type(exc).__name__}: {exc}"
                ],
            )

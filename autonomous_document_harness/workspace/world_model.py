"""Document world model: cheap reconnaissance before expensive reasoning.

Progressive disclosure starts here — full parses are cached once and every
later tool (search/read/tables) reuses them instead of re-reading files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..parsers import build_default_registry
from ..parsers.base import ParsedDocument, ParserRegistry
from ..retrieval import lexical as lexical_mod
from ..retrieval.hybrid import hybrid_search
from ..retrieval.semantic import TfIdfIndex
from ..retrieval.structural import HeadingIndex
from .corpus import Corpus, DocumentRef
from .indexing import Hit, LexicalIndex

TEXT_PAGE_MIN_CHARS = 30


@dataclass
class PageProfile:
    page_no: int
    char_count: int
    has_text: bool
    table_count: int
    heading_count: int
    image_coverage: float


@dataclass
class DocumentProfile:
    doc_id: str
    filename: str
    file_type: str
    fingerprint: str
    page_count: int
    classification: str  # selectable|scanned|hybrid|spreadsheet|text|image|unsupported
    text_chars: int
    avg_text_density: float
    image_coverage_avg: float
    table_candidates: int
    headings: List[str]
    sheets: List[Dict[str, Any]]
    needs_ocr: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class WorldModel:
    profiles: Dict[str, DocumentProfile] = field(default_factory=dict)
    lexical: LexicalIndex = field(default_factory=LexicalIndex)
    tfidf: TfIdfIndex = field(default_factory=TfIdfIndex)
    headings: HeadingIndex = field(default_factory=HeadingIndex)
    corpus_stats: Dict[str, Any] = field(default_factory=dict)
    parsed: Dict[str, ParsedDocument] = field(default_factory=dict)
    refs: Dict[str, DocumentRef] = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(
        cls, corpus: Corpus, registry: Optional[ParserRegistry] = None
    ) -> "WorldModel":
        wm = cls()
        registry = registry or build_default_registry()
        for ref in corpus:
            wm.refs[ref.doc_id] = ref
            parsed = registry.parse(ref)
            wm.parsed[ref.doc_id] = parsed
            profile = wm._profile(ref, parsed)
            wm.profiles[ref.doc_id] = profile
            wm._index(ref, parsed)
        wm.corpus_stats = wm._stats()
        return wm

    def _profile(self, ref: DocumentRef, parsed: ParsedDocument) -> DocumentProfile:
        pages = parsed.pages
        page_profiles = [
            PageProfile(
                p.page_no,
                p.char_count,
                p.char_count >= TEXT_PAGE_MIN_CHARS,
                len(p.tables),
                len(p.headings),
                p.image_coverage,
            )
            for p in pages
        ]
        text_chars = sum(p.char_count for p in page_profiles)
        sheets = [
            {
                "name": s.name,
                "rows": s.max_row,
                "cols": s.max_col,
                "merged": list(s.merged),
                "has_formulas": s.has_formulas,
                "cells": len(s.values),
            }
            for s in parsed.sheets
        ]
        headings = [h for p in pages for h in p.headings]
        table_candidates = sum(p.table_count for p in page_profiles) + len(
            [s for s in parsed.sheets if s.max_row > 1]
        )
        classification, needs_ocr = self._classify(
            ref, parsed, page_profiles, text_chars
        )
        return DocumentProfile(
            doc_id=ref.doc_id,
            filename=ref.filename,
            file_type=ref.file_type,
            fingerprint=ref.fingerprint,
            page_count=len(pages) or (1 if sheets else 0),
            classification=classification,
            text_chars=text_chars,
            avg_text_density=(text_chars / max(len(pages), 1)),
            image_coverage_avg=(
                sum(p.image_coverage for p in page_profiles)
                / max(len(page_profiles), 1)
            ),
            table_candidates=table_candidates,
            headings=headings[:50],
            sheets=sheets,
            needs_ocr=needs_ocr,
            errors=list(parsed.errors),
        )

    @staticmethod
    def _classify(
        ref: DocumentRef,
        parsed: ParsedDocument,
        pages: List[PageProfile],
        text_chars: int,
    ) -> tuple:
        if parsed.errors and not parsed.pages and not parsed.sheets:
            if ref.file_type == ".pptx":
                return "unsupported", False
            return "unsupported", False
        if ref.file_type in (".xlsx", ".xlsm"):
            return "spreadsheet", False
        if ref.file_type in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
            return "image", True
        if ref.file_type in (".txt", ".md", ".csv"):
            return "text", False
        if ref.file_type == ".docx":
            return (
                ("selectable", False)
                if text_chars >= TEXT_PAGE_MIN_CHARS
                else ("scanned", True)
            )
        # PDF: fraction of pages carrying a usable text layer
        if pages:
            frac = sum(1 for p in pages if p.has_text) / len(pages)
            if frac >= 0.8:
                return "selectable", False
            if frac <= 0.2:
                return "scanned", True
            return "hybrid", any(not p.has_text for p in pages)
        return "unsupported", False

    def _index(self, ref: DocumentRef, parsed: ParsedDocument) -> None:
        for page in parsed.pages:
            if page.text and page.text.strip():
                self.lexical.add(ref.doc_id, page.page_no, None, page.text)
                self.tfidf.add(ref.doc_id, page.page_no, None, page.text)
            for heading in page.headings:
                self.headings.add(ref.doc_id, page.page_no, heading)
            for table in page.tables:
                for r_idx, row in enumerate(table.rows, start=1):
                    row_text = " | ".join(c for c in row if c)
                    if row_text.strip():
                        cell = f"R{r_idx}"
                        self.lexical.add(ref.doc_id, page.page_no, cell, row_text)
                        self.tfidf.add(ref.doc_id, page.page_no, cell, row_text)
        for sheet in parsed.sheets:
            by_row: dict = {}
            for (r, c), v in sheet.values.items():
                by_row.setdefault(r, []).append("" if v is None else str(v))
            for r_idx, cells in by_row.items():
                row_text = " | ".join(x for x in cells if x)
                if row_text.strip():
                    cell = f"{sheet.name}!A{r_idx}"
                    self.lexical.add(
                        ref.doc_id, None, cell, f"[{sheet.name}] {row_text}"
                    )
                    self.tfidf.add(ref.doc_id, None, cell, f"[{sheet.name}] {row_text}")

    def _stats(self) -> Dict[str, Any]:
        by_class: Dict[str, int] = {}
        by_fingerprint: Dict[str, List[str]] = {}
        for p in self.profiles.values():
            by_class[p.classification] = by_class.get(p.classification, 0) + 1
            by_fingerprint.setdefault(p.fingerprint, []).append(p.doc_id)
        return {
            "total_documents": len(self.profiles),
            "by_classification": by_class,
            "total_pages": sum(p.page_count for p in self.profiles.values()),
            "total_text_chars": sum(p.text_chars for p in self.profiles.values()),
            "needs_ocr_count": sum(1 for p in self.profiles.values() if p.needs_ocr),
            "total_table_candidates": sum(
                p.table_candidates for p in self.profiles.values()
            ),
            "duplicate_groups": [
                sorted(ids) for ids in by_fingerprint.values() if len(ids) > 1
            ],
            "lexical_items": len(self.lexical),
            "semantic_items": len(self.tfidf),
            "headings": len(self.headings),
        }

    # -- agent-facing convenience ------------------------------------------
    def search_lexical(self, query: str, top_k: int = 10) -> List[Hit]:
        return lexical_mod.search(self.lexical, query, top_k=top_k)

    def search_semantic(self, query: str, top_k: int = 10) -> List[Hit]:
        return self.tfidf.search(query, top_k=top_k)

    def search_hybrid(self, query: str, top_k: int = 10) -> List[Hit]:
        return hybrid_search(self.lexical, self.tfidf, query, top_k=top_k)

    def search_headings(self, query: str, top_k: int = 10):
        return self.headings.search(query, top_k=top_k)

    def summary(self) -> Dict[str, Any]:
        return {
            "corpus": self.corpus_stats,
            "documents": [
                {
                    "doc_id": p.doc_id,
                    "filename": p.filename,
                    "file_type": p.file_type,
                    "classification": p.classification,
                    "pages": p.page_count,
                    "text_chars": p.text_chars,
                    "tables": p.table_candidates,
                    "sheets": [s["name"] for s in p.sheets],
                    "needs_ocr": p.needs_ocr,
                    "errors": p.errors,
                }
                for p in self.profiles.values()
            ],
        }

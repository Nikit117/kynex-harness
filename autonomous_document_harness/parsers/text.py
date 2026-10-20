"""Plain-text + CSV backend. CSV rows also surface as a table + sheet-like index."""

from __future__ import annotations

import csv
from pathlib import Path

from .base import ParsedDocument, ParsedPage, TableData, TextBlock
from ..workspace.corpus import DocumentRef


class TextParser:
    name = "text"

    def parse(self, ref: DocumentRef) -> ParsedDocument:
        try:
            raw = Path(ref.path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ParsedDocument(
                ref.doc_id,
                ref.fingerprint,
                ref.file_type,
                errors=[f"text unreadable: {type(exc).__name__}: {exc}"],
            )
        tables = []
        headings = []
        if ref.file_type == ".csv":
            try:
                rows = [
                    r for r in csv.reader(raw.splitlines()) if any(c.strip() for c in r)
                ]
                if rows:
                    tables.append(TableData(rows=rows, page=1, source="csv"))
            except Exception:
                pass
        elif ref.file_type == ".md":
            headings = [
                ln.lstrip("#").strip()
                for ln in raw.splitlines()
                if ln.strip().startswith("#")
            ][:30]
        blocks = [TextBlock(page=1, text=ln) for ln in raw.splitlines() if ln.strip()][
            :500
        ]
        meta = {"paginated": False, "lines": len(raw.splitlines())}
        page = ParsedPage(
            page_no=1,
            text=raw,
            blocks=blocks,
            tables=tables,
            headings=headings,
            char_count=len(raw),
        )
        return ParsedDocument(
            ref.doc_id, ref.fingerprint, ref.file_type, pages=[page], meta=meta
        )

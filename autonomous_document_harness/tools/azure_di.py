"""Azure Document Intelligence perception: layout, OCR, tables with geometry.

First-class agent tool input (not a pipeline stage): the investigator calls it
for targeted pages only. Results preserve provenance (doc, page, polygon/bbox,
text/cells, confidence, method). Offline-safe: SDK import is lazy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def di_configured(config) -> bool:
    """True only when SDK + endpoint + key are all present. No network touched."""
    try:
        import azure.ai.documentintelligence  # noqa: F401
    except ImportError:
        return False
    model = getattr(config, "model", config)
    return bool(getattr(model, "di_endpoint", "") and getattr(model, "di_key", ""))


def polygon_bbox(polygon: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """Quad/octagon point list -> (x0, y0, x1, y1). DI units preserved as-is."""
    try:
        xs = [float(polygon[i]) for i in range(0, len(polygon), 2)]
        ys = [float(polygon[i]) for i in range(1, len(polygon), 2)]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("azure-ai-documentintelligence")
    except Exception:
        return "unknown"


CACHE_VERSION = f"di-v1:sdk-{_sdk_version()}"


@dataclass
class DiCell:
    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    kind: str = ""
    polygon: List[float] = field(default_factory=list)
    confidence: Optional[float] = None

    @property
    def bbox(self):
        return polygon_bbox(self.polygon)


@dataclass
class DiTable:
    page: int
    rows: int
    cols: int
    cells: List[DiCell] = field(default_factory=list)

    def grid(self) -> List[List[str]]:
        table = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        for cell in self.cells:
            for r in range(cell.row, min(cell.row + cell.row_span, self.rows)):
                for c in range(cell.col, min(cell.col + cell.col_span, self.cols)):
                    if not table[r][c]:
                        table[r][c] = cell.text
        return table

    def header(self) -> List[str]:
        heads = [
            c.text
            for c in sorted(self.cells, key=lambda c: (c.row, c.col))
            if c.row == 0
        ]
        if heads:
            return heads
        grid = self.grid()
        return grid[0] if grid else []

    def avg_confidence(self) -> Optional[float]:
        confs = [c.confidence for c in self.cells if c.confidence is not None]
        return sum(confs) / len(confs) if confs else None


@dataclass
class DiPage:
    page_no: int
    text: str = ""
    line_count: int = 0


@dataclass
class MappedLayout:
    pages: List[DiPage] = field(default_factory=list)
    tables: List[DiTable] = field(default_factory=list)
    model_id: str = "prebuilt-layout"
    cached: bool = False

    def to_items(self, doc_id: str) -> List[Dict[str, Any]]:
        """Miner-ready snippets with azure_di provenance."""
        items = []
        for page in self.pages:
            if page.text.strip():
                items.append(
                    {
                        "snippet": page.text[:2000],
                        "provenance": {
                            "document_id": doc_id,
                            "page": page.page_no,
                            "source_type": "text",
                            "method": "azure_di",
                        },
                    }
                )
        for table in self.tables:
            header, grid = table.header(), table.grid()
            bbox = None
            polys = [c.polygon for c in table.cells if c.polygon]
            if polys:
                flat = [v for poly in polys for v in poly]
                bbox = polygon_bbox(flat)
            for row in grid[1:]:
                line = " | ".join(header) + "  //  " + " | ".join(row)
                if line.strip(" |/"):
                    items.append(
                        {
                            "snippet": line,
                            "provenance": {
                                "document_id": doc_id,
                                "page": table.page,
                                "bbox": list(bbox) if bbox else None,
                                "source_type": "table",
                                "method": "azure_di",
                            },
                        }
                    )
        return items

    def to_data(self) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "cached": self.cached,
            "pages": [
                {"page": p.page_no, "chars": len(p.text), "lines": p.line_count}
                for p in self.pages
            ],
            "tables": [
                {
                    "page": t.page,
                    "rows": t.rows,
                    "cols": t.cols,
                    "header": t.header()[:8],
                    "avg_confidence": t.avg_confidence(),
                }
                for t in self.tables
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization for the versioned disk cache (§24)."""
        return {
            "cache_version": CACHE_VERSION,
            "model_id": self.model_id,
            "pages": [
                {
                    "page_no": p.page_no,
                    "text": p.text,
                    "line_count": p.line_count,
                }
                for p in self.pages
            ],
            "tables": [
                {
                    "page": t.page,
                    "rows": t.rows,
                    "cols": t.cols,
                    "cells": [
                        {
                            "text": c.text,
                            "row": c.row,
                            "col": c.col,
                            "row_span": c.row_span,
                            "col_span": c.col_span,
                            "kind": c.kind,
                            "polygon": c.polygon,
                            "confidence": c.confidence,
                        }
                        for c in t.cells
                    ],
                }
                for t in self.tables
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MappedLayout":
        if payload.get("cache_version") != CACHE_VERSION:
            raise ValueError("stale DI cache version")
        mapped = cls(model_id=payload.get("model_id", "prebuilt-layout"))
        for page in payload.get("pages", []):
            mapped.pages.append(
                DiPage(
                    page_no=page["page_no"],
                    text=page.get("text", ""),
                    line_count=page.get("line_count", 0),
                )
            )
        for table in payload.get("tables", []):
            mapped.tables.append(
                DiTable(
                    page=table["page"],
                    rows=table["rows"],
                    cols=table["cols"],
                    cells=[
                        DiCell(
                            text=c["text"],
                            row=c["row"],
                            col=c["col"],
                            row_span=c.get("row_span", 1),
                            col_span=c.get("col_span", 1),
                            kind=c.get("kind", ""),
                            polygon=c.get("polygon", []),
                            confidence=c.get("confidence"),
                        )
                        for c in table.get("cells", [])
                    ],
                )
            )
        mapped.cached = True
        return mapped


def map_result(sdk_result: Any) -> MappedLayout:
    """Pure mapping from an AnalyzeResult (or test stub) to MappedLayout."""
    mapped = MappedLayout()
    for sdk_page in getattr(sdk_result, "pages", []) or []:
        lines = getattr(sdk_page, "lines", []) or []
        text = "\n".join(getattr(line, "content", "") or "" for line in lines)
        mapped.pages.append(
            DiPage(
                page_no=int(getattr(sdk_page, "page_number", 0)),
                text=text,
                line_count=len(lines),
            )
        )
    for sdk_table in getattr(sdk_result, "tables", []) or []:
        cells = []
        for sdk_cell in getattr(sdk_table, "cells", []) or []:
            cells.append(
                DiCell(
                    text=str(getattr(sdk_cell, "content", "") or ""),
                    row=int(getattr(sdk_cell, "row_index", 0)),
                    col=int(getattr(sdk_cell, "column_index", 0)),
                    row_span=int(getattr(sdk_cell, "row_span", 1) or 1),
                    col_span=int(getattr(sdk_cell, "column_span", 1) or 1),
                    kind=str(getattr(sdk_cell, "kind", "") or ""),
                    polygon=list(getattr(sdk_cell, "polygon", []) or []),
                    confidence=getattr(sdk_cell, "confidence", None),
                )
            )
        regions = getattr(sdk_table, "bounding_regions", []) or []
        page = int(getattr(regions[0], "page_number", 0)) if regions else 0
        mapped.tables.append(
            DiTable(
                page=page,
                rows=int(getattr(sdk_table, "row_count", 0) or 0),
                cols=int(getattr(sdk_table, "column_count", 0) or 0),
                cells=cells,
            )
        )
    return mapped


class AzureDIClient:
    """Thin wrapper: targeted prebuilt-layout analysis with result caching.

    Memory cache is always active. A disk cache (JSON per versioned key)
    persists across runs when cache_dir is given; keys embed content
    fingerprint + pages + model + SDK version, so ground truth or benchmark
    outputs can never contaminate extraction caches (they are never inputs).
    """

    def __init__(
        self,
        endpoint: str,
        key: str,
        cache: Optional[Dict[str, MappedLayout]] = None,
        cache_dir: str = "",
    ) -> None:
        self.endpoint = endpoint
        self.key = key
        self.cache = cache if cache is not None else {}
        self.cache_dir = cache_dir

    def _versioned_key(self, pdf_path: str, pages: List[int], cache_key: str) -> str:
        base = cache_key or f"{pdf_path}:{sorted(set(pages))}"
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]
        return f"{digest}:{CACHE_VERSION.replace(':', '-')}"

    def analyze_layout(
        self, pdf_path: str, pages: List[int], cache_key: str = ""
    ) -> MappedLayout:
        if not pages:
            raise ValueError("analyze_layout needs explicit target pages")
        key = self._versioned_key(pdf_path, pages, cache_key)
        if key in self.cache:
            hit = self.cache[key]
            hit.cached = True
            return hit
        if self.cache_dir:
            disk_hit = self._read_disk(key)
            if disk_hit is not None:
                self.cache[key] = disk_hit
                return disk_hit
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise RuntimeError(
                "azure-ai-documentintelligence is not installed"
            ) from exc
        pages_arg = ",".join(str(p) for p in sorted(set(pages)))
        client = DocumentIntelligenceClient(self.endpoint, AzureKeyCredential(self.key))
        with open(pdf_path, "rb") as fh:
            poller = client.begin_analyze_document(
                "prebuilt-layout", body=fh, pages=pages_arg
            )
            sdk_result = poller.result(timeout=300)
        mapped = map_result(sdk_result)
        self.cache[key] = mapped
        self._write_disk(key, mapped)
        return mapped

    def _disk_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
        return Path(self.cache_dir) / f"di_{safe}.json"

    def _read_disk(self, key: str) -> Optional[MappedLayout]:
        path = self._disk_path(key)
        if path is None or not path.exists():
            return None
        try:
            return MappedLayout.from_dict(json.loads(path.read_text("utf-8")))
        except (ValueError, KeyError):
            return None  # stale/corrupt cache never poisons results

    def _write_disk(self, key: str, mapped: MappedLayout) -> None:
        path = self._disk_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(mapped.to_dict()), encoding="utf-8")
        except OSError:
            pass  # cache is best-effort; extraction never depends on it

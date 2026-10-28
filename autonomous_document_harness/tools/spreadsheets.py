"""Spreadsheet tools: workbook overview + addressed range reads."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Tuple

from ..core.schemas import ExtractionMethod, SourceType
from ..workspace.world_model import WorldModel
from .registry import ToolError, provenance
from .text import _require_doc


def read_spreadsheet(
    world: WorldModel, doc_id: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _, parsed = _require_doc(world, doc_id)
    if not parsed.sheets:
        raise ToolError(f"document '{doc_id}' has no spreadsheets")
    sheets = [
        {
            "name": s.name,
            "rows": s.max_row,
            "cols": s.max_col,
            "merged": list(s.merged),
            "has_formulas": s.has_formulas,
            "nonempty_cells": len(s.values),
        }
        for s in parsed.sheets
    ]
    prov = [
        provenance(
            doc_id,
            cell=f"{s['name']}!A1",
            source_type=SourceType.SHEET_CELL,
            method=ExtractionMethod.SPREADSHEET,
        )
        for s in sheets
    ]
    return ({"doc_id": doc_id, "sheets": sheets}, prov)


def _jsonable(value: Any) -> Any:
    """Preserve JSON scalar types (ints stay ints); stringify the rest."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


def read_sheet_range(
    world: WorldModel, doc_id: str, sheet: str, cell_range: str = "A1:Z50"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _, parsed = _require_doc(world, doc_id)
    target = next(
        (s for s in parsed.sheets if s.name.lower() == (sheet or "").lower()), None
    )
    if target is None:
        raise ToolError(f"sheet '{sheet}' not found in '{doc_id}'")
    try:
        from openpyxl.utils import range_boundaries

        min_col, min_row, max_col, max_row = range_boundaries(cell_range)
    except Exception:
        raise ToolError(f"invalid cell range '{cell_range}' (expected e.g. 'A1:E10')")
    max_row = min(max_row, target.max_row)
    max_col = min(max_col, target.max_col)
    values = [
        [_jsonable(target.values.get((r, c), "")) for c in range(min_col, max_col + 1)]
        for r in range(min_row, max_row + 1)
    ]
    cell_ref = f"{target.name}!{cell_range}"
    prov = [
        provenance(
            doc_id,
            cell=cell_ref,
            source_type=SourceType.SHEET_CELL,
            method=ExtractionMethod.SPREADSHEET,
            snippet=" | ".join(str(v) for v in values[0]) if values else "",
        )
    ]
    return (
        {
            "doc_id": doc_id,
            "sheet": target.name,
            "range": cell_range,
            "origin": [min_row, min_col],
            "values": values,
        },
        prov,
    )

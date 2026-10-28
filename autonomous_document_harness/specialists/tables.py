"""Table specialist: bounded deep-dive into one table/sheet region."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..tools.computer import DocumentComputer

MAX_ROWS_SAMPLE = 25


def investigate_table(
    computer: DocumentComputer,
    doc_id: str,
    sheet: Optional[str] = None,
    page: Optional[int] = None,
    table_index: int = 0,
    question: str = "",
) -> Dict[str, Any]:
    """Return structured table facts. Raises ToolError only via observations."""
    detected = computer.registry.call("detect_tables", doc_id=doc_id, page=page)
    if detected.status != "ok":
        return {
            "doc_id": doc_id,
            "status": detected.status,
            "message": detected.message,
            "rows": [],
        }
    extracted = computer.registry.call(
        "extract_table", doc_id=doc_id, table_index=table_index, page=page, sheet=sheet
    )
    if extracted.status != "ok":
        return {
            "doc_id": doc_id,
            "status": extracted.status,
            "message": extracted.message,
            "rows": [],
        }
    rows: List[List[Any]] = extracted.data.get("rows", [])
    header = rows[0] if rows else []
    logical = computer.registry.call(
        "reconstruct_table", doc_id=doc_id, page=page, sheet=sheet
    )
    logical_tables = (
        logical.data.get("logical_tables", []) if logical.status == "ok" else []
    )
    return {
        "doc_id": doc_id,
        "status": "ok",
        "question": question,
        "sheet": extracted.data.get("sheet"),
        "page": extracted.data.get("page"),
        "n_rows": extracted.data.get("n_rows"),
        "n_cols": extracted.data.get("n_cols"),
        "header": header,
        "rows_sample": rows[:MAX_ROWS_SAMPLE],
        "logical_tables": logical_tables,
        "provenance": extracted.provenance,
    }

"""Structured (layout-derived) array extraction.

The agent transcribes table cells by eye, and the transcription loses exactly
what the official scorer compares: currency symbols, thousands separators, long
text and the column alignment. Measured over the 370-document run, **133
documents have the right row COUNT and not one row that matches the ground
truth exactly** — the largest single loss class (14.89 doc-equivalents).

Document Intelligence reads the same tables structurally, so the rows can be
built from the table cells instead of from the model's rendering. On
``real_franklin_nov_2025`` the table row is

    ['117649', 'ACEA', '$0.00', '$2,800.00', '$0.00', 'STAFF ED SERVICES']

and the ground truth is that row verbatim.

The mapping from a table column to a schema field is a name normalisation
("Check Number" -> ``check_number``). A table is only used when its header maps
onto the schema's item fields, so a summary table that does not describe the
array is ignored rather than merged.

Kill switch: ``ADH_STRUCTURED_ARRAYS=0``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

_DI_PAGES_PER_CALL = 25


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokens(value: Any) -> set:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def match_field(header_cell: Any, props: List[str]) -> Optional[str]:
    """Map a table header cell onto a schema field name (None if it does not)."""
    head = _norm(header_cell)
    if not head:
        return None
    for name in props:
        if _norm(name) == head:
            return name
    head_tokens = _tokens(header_cell)
    if not head_tokens:
        return None
    best, best_score = None, 0.0
    for name in props:
        name_tokens = _tokens(name)
        if not name_tokens:
            continue
        score = len(head_tokens & name_tokens) / len(head_tokens | name_tokens)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 0.6 else None


def _resolve(node: Any, root: Dict[str, Any]) -> Any:
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 8:
        target: Any = root
        for part in node["$ref"].lstrip("#/").split("/"):
            target = target[part]
        node = target
        seen += 1
    return node


def _page_field(item_schema: Dict[str, Any]) -> Optional[str]:
    props = list((item_schema.get("properties") or {}).keys())
    for name in ("source_page", "page", "page_number", "page_no"):
        if name in props:
            return name
    return None


def _tables_for(pdf_path: str, pages: int) -> List[Any]:
    """Read the document's tables with Document Intelligence."""
    from autonomous_document_harness.deep.model import ensure_azure_env
    from autonomous_document_harness.tools.azure_di import AzureDIClient

    ensure_azure_env()
    endpoint = os.environ.get("ADH_DI_ENDPOINT", "") or os.environ.get(
        "AZURE_DI_ENDPOINT", ""
    )
    key = os.environ.get("ADH_DI_API_KEY", "") or os.environ.get("AZURE_DI_API_KEY", "")
    if not (endpoint and key):
        return []
    client = AzureDIClient(endpoint, key, cache={})
    out: List[Any] = []
    target = list(range(1, min(pages, 200) + 1))
    for i in range(0, len(target), _DI_PAGES_PER_CALL):
        chunk = target[i : i + _DI_PAGES_PER_CALL]
        try:
            layout = client.analyze_layout(
                str(pdf_path),
                pages=chunk,
                cache_key=f"structured:{pdf_path}:{chunk[0]}",
            )
        except Exception:  # noqa: BLE001 -- best effort
            continue
        out.extend(layout.tables)
    return out


def structured_arrays(
    pdf_path: str,
    json_schema: Dict[str, Any],
    *,
    max_pages: int = 200,
) -> Dict[str, List[Dict[str, Any]]]:
    """Rows per array field, built from the document's tables.

    Returns ``{field_name: [row, ...]}`` for the fields whose table header maps
    onto the schema's item fields.
    """
    if os.environ.get("ADH_STRUCTURED_ARRAYS", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return {}
    props = json_schema.get("properties") or {}
    array_fields: Dict[str, Dict[str, Any]] = {}
    for name, spec in props.items():
        resolved = _resolve(spec, json_schema)
        items = resolved.get("items") if isinstance(resolved, dict) else None
        if not isinstance(items, dict):
            continue
        item_schema = _resolve(items, json_schema)
        fields = list((item_schema.get("properties") or {}).keys())
        if len(fields) >= 2:
            array_fields[name] = {
                "item_schema": item_schema,
                "fields": fields,
                "page_field": _page_field(item_schema),
            }
    if not array_fields:
        return {}

    try:
        from fitz import open as pdoc

        doc = pdoc(str(pdf_path))
        total_pages = doc.page_count
        doc.close()
    except Exception:  # noqa: BLE001
        return {}

    tables = _tables_for(pdf_path, total_pages)
    if not tables:
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for table in tables:
        try:
            grid = table.grid()
        except Exception:  # noqa: BLE001
            continue
        if len(grid) < 2:
            continue
        header = grid[0]
        for name, spec in array_fields.items():
            fields = spec["fields"]
            mapping: Dict[int, str] = {}
            for idx, cell in enumerate(header):
                matched = match_field(cell, fields)
                if matched and matched not in mapping.values():
                    mapping[idx] = matched
            if len(mapping) < 2 or len(mapping) < 0.5 * len(fields):
                continue
            rows = out.setdefault(name, [])
            for raw in grid[1:]:
                item: Dict[str, Any] = {}
                for idx, field in mapping.items():
                    if idx < len(raw):
                        value = raw[idx]
                        if value is not None and str(value).strip():
                            item[field] = str(value).strip()
                if not item:
                    continue
                page_field = spec["page_field"]
                if page_field and page_field not in item:
                    item[page_field] = table.page
                rows.append(item)
    return {k: v for k, v in out.items() if v}

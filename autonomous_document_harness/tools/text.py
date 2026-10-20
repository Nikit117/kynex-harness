"""Text-reading tools: page / full-text / section / surrounding context."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..core.schemas import ExtractionMethod, SourceType
from ..workspace.world_model import WorldModel
from .registry import ToolError, provenance


def _require_doc(world: WorldModel, doc_id: str):
    ref = world.refs.get(doc_id)
    if ref is None:
        raise ToolError(f"unknown document '{doc_id}'")
    parsed = world.parsed.get(doc_id)
    if parsed is None:
        raise ToolError(f"document '{doc_id}' was never parsed")
    if parsed.errors and not parsed.pages:
        raise ToolError(f"document '{doc_id}' unreadable: {parsed.errors[0]}")
    return ref, parsed


def read_text(
    world: WorldModel, doc_id: str, max_chars: int = 4000
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _, parsed = _require_doc(world, doc_id)
    full = parsed.full_text()
    truncated = len(full) > max_chars
    prov = [
        provenance(
            doc_id,
            page=(parsed.pages[0].page_no if parsed.pages else None),
            source_type=SourceType.TEXT,
            method=ExtractionMethod.NATIVE_TEXT,
            snippet=full[:300],
        )
    ]
    return (
        {
            "doc_id": doc_id,
            "text": full[:max_chars],
            "truncated": truncated,
            "char_count": len(full),
        },
        prov,
    )


def read_page(
    world: WorldModel, doc_id: str, page: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _, parsed = _require_doc(world, doc_id)
    for p in parsed.pages:
        if p.page_no == page:
            prov = [
                provenance(
                    doc_id,
                    page=page,
                    source_type=SourceType.PAGE,
                    method=ExtractionMethod.NATIVE_TEXT,
                    snippet=p.text[:300],
                )
            ]
            return (
                {
                    "doc_id": doc_id,
                    "page": page,
                    "text": p.text,
                    "char_count": p.char_count,
                    "tables": len(p.tables),
                    "headings": list(p.headings),
                    "image_coverage": p.image_coverage,
                },
                prov,
            )
    raise ToolError(f"document '{doc_id}' has no page {page}")


def read_section(
    world: WorldModel, doc_id: str, heading: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _, parsed = _require_doc(world, doc_id)
    needle = (heading or "").strip().lower()
    if not needle:
        raise ToolError("heading must be a non-empty string")
    match_page = None
    for p in parsed.pages:
        for h in p.headings:
            if needle in h.lower():
                match_page = p
                break
        if match_page:
            break
    if match_page is None:
        # fall back to first text occurrence
        for p in parsed.pages:
            if needle in p.text.lower():
                match_page = p
                break
    if match_page is None:
        raise ToolError(f"heading/section '{heading}' not found in '{doc_id}'")
    others = [h for p in parsed.pages for h in p.headings if needle not in h.lower()]
    start = match_page.text.lower().find(needle)
    section = match_page.text[start : start + 3000]
    pages_covered = [match_page.page_no]
    for p in parsed.pages:
        if p.page_no <= match_page.page_no:
            continue
        cut = len(p.text)
        for h in others:
            pos = p.text.find(h)
            if 0 <= pos < cut:
                cut = pos
        section += "\n" + p.text[:cut]
        pages_covered.append(p.page_no)
        if cut < len(p.text):
            break
        if len(section) > 6000:
            break
    prov = [
        provenance(
            doc_id,
            page=pg,
            source_type=SourceType.TEXT,
            method=ExtractionMethod.NATIVE_TEXT,
        )
        for pg in pages_covered
    ]
    return (
        {
            "doc_id": doc_id,
            "heading": heading,
            "text": section[:6000],
            "pages_covered": pages_covered,
        },
        prov,
    )


def read_surrounding_context(
    world: WorldModel, doc_id: str, page: int, window: int = 1
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _, parsed = _require_doc(world, doc_id)
    lo, hi = page - max(window, 0), page + max(window, 0)
    pages = [
        {"page": p.page_no, "text": p.text[:2000], "char_count": p.char_count}
        for p in parsed.pages
        if lo <= p.page_no <= hi
    ]
    if not pages:
        raise ToolError(f"no pages in range [{lo}, {hi}] for '{doc_id}'")
    prov = [
        provenance(
            doc_id,
            page=p["page"],
            source_type=SourceType.PAGE,
            method=ExtractionMethod.NATIVE_TEXT,
        )
        for p in pages
    ]
    return ({"doc_id": doc_id, "page": page, "window": window, "pages": pages}, prov)

"""Search tools: lexical / semantic / hybrid / headings over the world model."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..workspace.indexing import Hit
from ..workspace.world_model import WorldModel
from .registry import ToolError, provenance


def _hit_to_dict(hit: Hit) -> Dict[str, Any]:
    return {
        "provenance": provenance(
            hit.provenance_doc,
            page=hit.page,
            cell=hit.cell,
            source_type="text",
            method=hit.method,
            snippet=hit.snippet,
        ),
        "snippet": hit.snippet,
        "score": hit.score,
        "method": hit.method,
    }


def text_search(
    world: WorldModel, query: str, mode: str = "hybrid", top_k: int = 10
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    query = (query or "").strip()
    if not query:
        raise ToolError("query must be a non-empty string")
    mode = (mode or "hybrid").lower()
    if mode == "lexical":
        hits = world.search_lexical(query, top_k=top_k)
    elif mode == "semantic":
        hits = world.search_semantic(query, top_k=top_k)
    elif mode == "hybrid":
        hits = world.search_hybrid(query, top_k=top_k)
    else:
        raise ToolError(f"unknown retrieval mode '{mode}' (lexical|semantic|hybrid)")
    dicts = [_hit_to_dict(h) for h in hits]
    return (
        {"query": query, "mode": mode, "count": len(dicts), "hits": dicts},
        [d["provenance"] for d in dicts],
    )


def heading_search(
    world: WorldModel, query: str, top_k: int = 10
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    query = (query or "").strip()
    if not query:
        raise ToolError("query must be a non-empty string")
    hits = world.search_headings(query, top_k=top_k)
    dicts = [
        {
            "provenance": provenance(
                h.doc_id,
                page=h.page,
                source_type="heading",
                method="structural",
                snippet=h.heading,
            ),
            "heading": h.heading,
            "level": h.level,
            "score": h.score,
        }
        for h in hits
    ]
    return (
        {"query": query, "count": len(dicts), "headings": dicts},
        [d["provenance"] for d in dicts],
    )

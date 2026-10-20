"""Cross-document research specialist: group evidence by source document."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..tools.computer import DocumentComputer

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def cross_document_research(
    computer: DocumentComputer, query: str, top_k: int = 12
) -> Dict[str, Any]:
    supporting = computer.registry.call(
        "find_supporting_evidence", query=query, top_k=top_k
    )
    contradicting = computer.registry.call(
        "find_contradicting_evidence", query=query, top_k=top_k
    )
    groups: Dict[str, List[Dict[str, Any]]] = {}
    numbers: Dict[str, List[float]] = {}
    if supporting.status == "ok":
        for hit in supporting.data.get("hits", []):
            doc = hit["provenance"]["document_id"]
            groups.setdefault(doc, []).append(hit)
            numbers.setdefault(doc, []).extend(_numbers(hit.get("snippet", "")))
    return {
        "query": query,
        "documents": sorted(groups),
        "hits_per_document": {d: len(h) for d, h in groups.items()},
        "numbers_per_document": {d: nums[:10] for d, nums in numbers.items()},
        "contradiction_candidates": contradicting.data.get("candidates", [])
        if contradicting.status == "ok"
        else [],
        "status": supporting.status,
    }


def _numbers(text: str) -> List[float]:
    out = []
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out

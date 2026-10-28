"""Provenance helpers: tamper-evident snippet hashes + evidence cards."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from ..core.schemas import Evidence


def snippet_hash(snippet: str) -> str:
    return hashlib.sha256(snippet.encode("utf-8", errors="replace")).hexdigest()[:16]


def region_key(document_id: str, page: Any, cell: Any) -> str:
    return f"{document_id}:{page if page is not None else '-'}:{cell or '-'}"


def evidence_card(evidence: Evidence, filename: str = "") -> Dict[str, Any]:
    region = evidence.region
    return {
        "evidence_id": evidence.evidence_id,
        "document_id": region.document_id,
        "filename": filename,
        "page": region.page,
        "cell": region.cell,
        "bbox": list(region.bbox) if region.bbox else None,
        "source_type": region.source_type,
        "extraction_method": region.extraction_method,
        "method": evidence.method,
        "confidence": evidence.confidence,
        "snippet": evidence.snippet,
        "snippet_hash": snippet_hash(evidence.snippet),
        "produced_by": evidence.produced_by,
    }

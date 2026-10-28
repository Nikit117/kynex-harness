"""Structure-aware retrieval: headings index + section scoping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..workspace.indexing import tokenize


@dataclass
class HeadingHit:
    doc_id: str
    page: Optional[int]
    heading: str
    level: int
    score: float


class HeadingIndex:
    def __init__(self) -> None:
        self._entries: List[Tuple[str, Optional[int], str, int]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self, doc_id: str, page: Optional[int], heading: str, level: int = 1
    ) -> None:
        if heading and heading.strip():
            self._entries.append((doc_id, page, heading.strip(), level))

    def search(self, query: str, top_k: int = 10) -> List[HeadingHit]:
        terms = set(tokenize(query))
        if not terms:
            return []
        scored = []
        for doc_id, page, heading, level in self._entries:
            htoks = set(tokenize(heading))
            overlap = len(terms & htoks) / max(len(terms), 1)
            if query.strip().lower() in heading.lower():
                overlap += 1.0
            if overlap > 0:
                scored.append(
                    HeadingHit(doc_id, page, heading, level, round(overlap, 4))
                )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def context_for(self, doc_id: str, page: Optional[int]) -> List[str]:
        """Nearest headings at or before the given page (section scope)."""
        cands = [
            (p if p is not None else 1, h)
            for d, p, h, _ in self._entries
            if d == doc_id
        ]
        target = page if page is not None else 1
        cands.sort()
        return [h for p, h in cands if p <= target][-3:]

    def by_document(self, doc_id: str) -> List[Dict[str, object]]:
        return [
            {"page": p, "heading": h, "level": lv}
            for d, p, h, lv in self._entries
            if d == doc_id
        ]

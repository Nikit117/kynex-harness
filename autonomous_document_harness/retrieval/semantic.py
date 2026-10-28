"""Local TF-IDF semantic index. No embedding model required; swappable later."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..workspace.indexing import Hit, tokenize


@dataclass
class _Doc:
    doc_id: str
    page: Optional[int]
    cell: Optional[str]
    text: str
    terms: Counter


class TfIdfIndex:
    def __init__(self) -> None:
        self._docs: List[_Doc] = []
        self._df: Counter = Counter()

    def __len__(self) -> int:
        return len(self._docs)

    def add(
        self, doc_id: str, page: Optional[int], cell: Optional[str], text: str
    ) -> None:
        if not text or not text.strip():
            return
        terms = Counter(tokenize(text))
        self._docs.append(_Doc(doc_id, page, cell, text, terms))
        for tok in terms:
            self._df[tok] += 1

    def _vector(self, terms: Counter) -> Dict[str, float]:
        n = max(len(self._docs), 1)
        total = sum(terms.values()) or 1
        return {
            t: (c / total) * (1.0 + math.log(n / (1 + self._df.get(t, 0))))
            for t, c in terms.items()
        }

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b[t] for t in a if t in b)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        if not query.strip() or not self._docs:
            return []
        qv = self._vector(Counter(tokenize(query)))
        scored = []
        for doc in self._docs:
            score = self._cosine(qv, self._vector(doc.terms))
            if score > 0:
                snippet = doc.text[:240]
                scored.append(
                    Hit(
                        doc.doc_id,
                        doc.page,
                        doc.cell,
                        snippet,
                        round(score, 4),
                        method="semantic",
                    )
                )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

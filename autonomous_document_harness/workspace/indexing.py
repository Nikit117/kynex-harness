"""Lexical inverted index. Exact identifiers (AI-801, 10-P-101A) are first-class."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/\-]*[a-z0-9]|[a-z0-9]")
_SPLIT_RE = re.compile(r"[._/\-]+")


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if len(raw) < 2 and not raw.isdigit():
            continue
        tokens.append(raw)
        if len(raw) > 2 and _SPLIT_RE.search(raw):
            tokens.extend(p for p in _SPLIT_RE.split(raw) if len(p) >= 2 or p.isdigit())
    return tokens


@dataclass
class Hit:
    provenance_doc: str
    page: Optional[int]
    cell: Optional[str]
    snippet: str
    score: float
    method: str = "lexical"


@dataclass
class _Item:
    doc_id: str
    page: Optional[int]
    cell: Optional[str]
    text: str
    lowered: str = field(init=False)

    def __post_init__(self) -> None:
        self.lowered = self.text.lower()


class LexicalIndex:
    def __init__(self) -> None:
        self._items: List[_Item] = []
        self._postings: Dict[str, Dict[int, int]] = defaultdict(dict)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self, doc_id: str, page: Optional[int], cell: Optional[str], text: str
    ) -> None:
        if not text or not text.strip():
            return
        idx = len(self._items)
        self._items.append(_Item(doc_id, page, cell, text))
        seen = set()
        for tok in tokenize(text):
            if tok in seen:
                continue
            seen.add(tok)
            self._postings[tok][idx] = self._postings[tok].get(idx, 0) + 1

    def _idf(self, term: str) -> float:
        return 1.0 + math.log(
            (1 + len(self._items)) / (1 + len(self._postings.get(term, {})))
        )

    @staticmethod
    def _snippet(text: str, query: str, width: int = 240) -> str:
        lowered = text.lower()
        pos = lowered.find(query.lower().strip()[:24])
        if pos < 0:
            return text[:width]
        start = max(0, pos - width // 3)
        prefix = "…" if start > 0 else ""
        return prefix + text[start : start + width]

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        query = query.strip()
        if not query or not self._items:
            return []
        terms = tokenize(query)
        scores: Dict[int, float] = defaultdict(float)
        for tok in terms:
            for idx in self._postings.get(tok, {}):
                scores[idx] += self._idf(tok)
        lowered_q = query.lower()
        for idx, item in enumerate(self._items):
            if lowered_q in item.lowered:
                scores[idx] += 2.0 + min(len(query) / 20.0, 2.0)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            Hit(
                provenance_doc=self._items[i].doc_id,
                page=self._items[i].page,
                cell=self._items[i].cell,
                snippet=self._snippet(self._items[i].text, query),
                score=round(s, 4),
            )
            for i, s in ranked
            if s > 0
        ]

    def term_postings(
        self, term: str
    ) -> List[Tuple[str, Optional[int], Optional[str]]]:
        return [
            (self._items[i].doc_id, self._items[i].page, self._items[i].cell)
            for i in self._postings.get(term.lower(), {})
        ]

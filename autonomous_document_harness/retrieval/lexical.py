"""Lexical search with exact-identifier preference (tag/equipment codes)."""

from __future__ import annotations

import re
from typing import List

from ..workspace.indexing import Hit, LexicalIndex

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{2,}$")


def looks_like_identifier(query: str) -> bool:
    q = query.strip()
    return bool(_IDENTIFIER_RE.match(q)) and any(c.isdigit() or c in "-_/" for c in q)


def search(index: LexicalIndex, query: str, top_k: int = 10) -> List[Hit]:
    hits = index.search(
        query, top_k=top_k * 3 if looks_like_identifier(query) else top_k
    )
    if looks_like_identifier(query):
        needle = query.strip().lower()
        exact = [h for h in hits if needle in h.snippet.lower()]
        rest = [h for h in hits if h not in exact]
        hits = exact + rest
    for h in hits:
        h.method = "lexical"
    return hits[:top_k]

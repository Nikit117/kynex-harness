"""Hybrid retrieval: reciprocal-rank fusion over lexical + semantic rankings."""

from __future__ import annotations

from typing import List, Tuple

from ..workspace.indexing import Hit, LexicalIndex
from . import lexical as lexical_mod
from .semantic import TfIdfIndex

RRF_K = 60


def _key(h: Hit) -> Tuple[str, str, str]:
    return (h.provenance_doc, str(h.page), h.cell or "")


def rrf_fuse(rankings: List[List[Hit]], k: int = RRF_K) -> List[Hit]:
    fused: dict = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = _key(hit)
            if key not in fused:
                fused[key] = Hit(
                    hit.provenance_doc,
                    hit.page,
                    hit.cell,
                    hit.snippet,
                    0.0,
                    method="hybrid",
                )
            fused[key].score += 1.0 / (k + rank)
            if len(hit.snippet) > len(fused[key].snippet):
                fused[key].snippet = hit.snippet
    out = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    for h in out:
        h.score = round(h.score, 4)
    return out


def hybrid_search(
    lexical: LexicalIndex, tfidf: TfIdfIndex, query: str, top_k: int = 10
) -> List[Hit]:
    lex_hits = lexical_mod.search(lexical, query, top_k=top_k * 2)
    sem_hits = tfidf.search(query, top_k=top_k * 2)
    if not lex_hits and not sem_hits:
        return []
    return rrf_fuse([lex_hits, sem_hits])[:top_k]

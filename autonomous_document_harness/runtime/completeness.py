"""Completeness sweep (§19): multi-channel coverage report, not a guarantee.

"Every discovered claim is verified" does not imply complete extraction. This
module cross-checks every discovery channel and exposes what was — and was
not — investigated, so stopping decisions are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class ChannelCoverage:
    channel: str
    queries_or_targets: int = 0
    hits_total: int = 0
    examined: int = 0
    unexamined: List[Dict[str, Any]] = field(default_factory=list)


def build_sweep_report(
    world,
    needs,
    searched: Set[str],
    visited_pages: Set[Tuple[str, int]],
    visited_sheets: Set[Tuple[str, str]],
    tables_inspected: Set[str],
    claims,
) -> Dict[str, Any]:
    """Assemble investigated/uninvestigated evidence across channels."""
    from ..models.heuristic import need_search_queries

    channels: List[ChannelCoverage] = []
    uninvestigated: List[Dict[str, Any]] = []
    for mode in ("lexical", "semantic", "hybrid"):
        hits_total, examined = 0, 0
        queries = 0
        for need in needs:
            for query in need_search_queries(need):
                queries += 1
                try:
                    hits = {
                        "lexical": world.search_lexical,
                        "semantic": world.search_semantic,
                        "hybrid": world.search_hybrid,
                    }[mode](query, top_k=20)
                except Exception:
                    hits = []
                hits_total += len(hits)
                for hit in hits:
                    loc = (
                        hit.provenance_doc,
                        hit.page if hit.page is not None else (hit.cell or ""),
                    )
                    if hit.page is None:
                        done = (hit.provenance_doc, hit.cell or "") in visited_sheets
                    else:
                        done = (hit.provenance_doc, hit.page) in visited_pages
                    if done:
                        examined += 1
                    elif len(uninvestigated) < 12:
                        uninvestigated.append(
                            {
                                "channel": mode,
                                "doc_id": hit.provenance_doc,
                                "page": hit.page,
                                "cell": hit.cell,
                                "query": query,
                            }
                        )
        channels.append(ChannelCoverage(mode, queries, hits_total, examined))
    # Structural channel: headings indexed vs heading searches executed.
    heading_searches = sum(1 for q in searched if q.startswith("headings:"))
    channels.append(
        ChannelCoverage(
            "structural",
            heading_searches,
            len(world.headings),
            min(heading_searches, len(world.headings)),
        )
    )
    # Table inventory channel.
    inventoried = sum(p.table_candidates for p in world.profiles.values())
    inspected_docs = sorted(tables_inspected)
    channels.append(
        ChannelCoverage(
            "tables",
            len(inspected_docs),
            inventoried,
            len(inspected_docs),
            unexamined=[
                {
                    "channel": "tables",
                    "doc_id": doc_id,
                    "reason": "table candidates never inspected",
                }
                for doc_id, profile in world.profiles.items()
                if profile.table_candidates > 0 and doc_id not in tables_inspected
            ][:6],
        )
    )
    uninvestigated.extend(channels[-1].unexamined)
    # Entity coverage per need attribute.
    covered_attrs = {c.attribute.strip().lower() for c in claims}
    unresolved_categories = [
        {"subject": n.subject, "attribute": n.attribute}
        for n in needs
        if n.required
        and n.attribute.strip().lower() not in covered_attrs
        and not (n.subject == "*" and covered_attrs)
    ]
    examined_total = sum(c.examined for c in channels)
    hits_total_all = sum(c.hits_total for c in channels)
    if not unresolved_categories and not uninvestigated:
        assessment, rationale = "complete", "no uninvestigated candidates remain"
    elif not unresolved_categories and uninvestigated:
        assessment, rationale = (
            "partial",
            f"{len(uninvestigated)} candidate "
            "location(s) unexamined but all needs addressed",
        )
    else:
        assessment, rationale = (
            "weak",
            f"{len(unresolved_categories)} required categor(ies) without claims",
        )
    return {
        "channels": [
            {
                "channel": c.channel,
                "queries_or_targets": c.queries_or_targets,
                "hits_total": c.hits_total,
                "examined": c.examined,
            }
            for c in channels
        ],
        "investigated": examined_total,
        "candidates": hits_total_all,
        "uninvestigated": uninvestigated,
        "unresolved_categories": unresolved_categories,
        "assessment": assessment,
        "rationale": rationale,
    }

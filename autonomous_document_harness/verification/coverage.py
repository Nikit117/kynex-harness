"""Coverage gate: every corpus location that could bear on the goal was examined.

Needs addressed by the first retrieved document must not blind the agent to
contradictory evidence sitting in never-opened documents. Unscanned documents
are flagged for perception (DI/vision) while required needs stay unaddressed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.schemas import GateResult, RecoveryTask, make_id
from ..models.heuristic import need_addressed, need_search_queries
from .base import Gate, VerificationContext

READABLE = frozenset({"selectable", "text", "spreadsheet", "hybrid"})
UNSCANNED = frozenset({"scanned", "image"})
MAX_READS_PER_ROUND = 6


def _sheet_visited(doc_id: str, cell: str, visited_sheets: set) -> bool:
    """Sheets are visited wholesale: any read of the sheet covers its rows."""
    if (doc_id, cell) in visited_sheets:
        return True
    sheet = cell.split("!")[0] if "!" in cell else ""
    return bool(sheet) and (doc_id, sheet) in visited_sheets


class CoverageGate:
    name = "coverage"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        world = getattr(ctx, "world", None)
        if world is None:
            return GateResult(
                gate=self.name,
                passed=True,
                failures=[],
                details={"note": "no world model attached"},
            )
        visited_pages = set(getattr(ctx, "visited_pages", set()) or set())
        visited_sheets = set(getattr(ctx, "visited_sheets", set()) or set())
        pending: List[Dict[str, Any]] = []
        seen = set()
        for need in ctx.goal.needs:
            if not need.required:
                continue
            for query in need_search_queries(need):
                try:
                    hits = world.lexical.search(query, top_k=30)
                except AttributeError:
                    continue
                for hit in hits:
                    profile = world.profiles.get(hit.provenance_doc)
                    if profile is None or profile.classification not in READABLE:
                        continue
                    if profile.errors and not profile.page_count:
                        continue
                    if hit.page is None:
                        key = (hit.provenance_doc, hit.cell or "")
                        if not hit.cell or key in seen:
                            continue
                        seen.add(key)
                        if _sheet_visited(hit.provenance_doc, hit.cell, visited_sheets):
                            continue
                        pending.append(
                            {
                                "doc_id": hit.provenance_doc,
                                "page": None,
                                "cell": hit.cell,
                            }
                        )
                    else:
                        key = (hit.provenance_doc, hit.page)
                        if key in visited_pages or key in seen:
                            continue
                        seen.add(key)
                        pending.append(
                            {
                                "doc_id": hit.provenance_doc,
                                "page": hit.page,
                                "cell": None,
                            }
                        )
                    if len(pending) >= MAX_READS_PER_ROUND:
                        break
                if len(pending) >= MAX_READS_PER_ROUND:
                    break
        # Unscanned documents are invisible to lexical screening: while required
        # needs stay unaddressed, their first pages are perception candidates.
        unaddressed = [
            n
            for n in ctx.goal.needs
            if n.required and not need_addressed(n, ctx.claims)
        ]
        if unaddressed:
            for doc_id, profile in world.profiles.items():
                if profile.classification not in UNSCANNED:
                    continue
                if not profile.page_count or (doc_id, 1) in visited_pages:
                    continue
                pending.append(
                    {"doc_id": doc_id, "page": 1, "cell": None, "via": "perception"}
                )
                if len(pending) >= MAX_READS_PER_ROUND:
                    break
        if not pending:
            return GateResult(
                gate=self.name, passed=True, failures=[], details={"unexamined": 0}
            )
        failures = [
            f"{len(pending)} unexamined evidence location(s), e.g. "
            f"{pending[0]['doc_id']} p.{pending[0]['page'] or pending[0]['cell']}"
        ]
        recovery = [
            RecoveryTask(
                task_id=make_id("rec"),
                claim_ids=[],
                reason="corpus locations matching the goal were never examined",
                suggested_actions=["read_candidate_pages"],
                context={"reads": pending},
            )
        ]
        return GateResult(
            gate=self.name,
            passed=False,
            failures=failures,
            recovery=recovery,
            details={"unexamined": len(pending)},
        )

"""Contradiction gate: no open contradiction may survive into output silently."""

from __future__ import annotations

from typing import List

from ..core.schemas import GateResult, RecoveryTask, make_id
from .base import VerificationContext


class ContradictionGate:
    name = "contradiction"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        claim_ids = {c.claim_id for c in ctx.claims}
        for contra in ctx.contradictions:
            if getattr(contra, "status", "open") != "open":
                continue  # resolved / explicitly-unresolved contradictions pass
            involved = [cid for cid in contra.claim_ids if cid in claim_ids]
            if not involved:
                continue
            failures.append(
                f"{contra.contradiction_id} ({contra.subject}/{contra.attribute}): "
                f"values {contra.values} disagree across {len(involved)} claims"
            )
            recovery.append(
                RecoveryTask(
                    task_id=make_id("rec"),
                    claim_ids=involved,
                    reason=f"conflicting values {contra.values}: revision? scope? spare treatment?",
                    suggested_actions=[
                        "compare_sources",
                        "inspect_revisions",
                        "search_related_schedule",
                        "resolve_contradiction",
                    ],
                )
            )
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={"open": len(recovery)},
        )

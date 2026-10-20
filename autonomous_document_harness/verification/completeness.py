"""Completeness gate: every required need was investigated (claimed or explicit)."""

from __future__ import annotations

from typing import List

from ..core.schemas import GateResult, RecoveryTask, make_id
from ..models.heuristic import need_addressed
from .base import VerificationContext


class CompletenessGate:
    name = "completeness"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        warnings: List[str] = []
        for need in ctx.goal.needs:
            if need_addressed(need, ctx.claims):
                continue
            if need.required:
                failures.append(
                    f"required need {need.need_id} ({need.subject}/{need.attribute}) "
                    "has no claim and no explicit unresolved marker"
                )
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[],
                        reason=f"uninvestigated required need {need.subject}/{need.attribute}",
                        suggested_actions=[
                            "search_subject_candidates",
                            "expand_retrieval",
                            "read_candidate_pages",
                        ],
                    )
                )
            else:
                warnings.append(f"optional need {need.need_id} unaddressed")
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={"needs": len(ctx.goal.needs), "warnings": warnings},
        )

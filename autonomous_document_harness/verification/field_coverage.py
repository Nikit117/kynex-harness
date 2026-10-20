"""Field-coverage gate (§8): requested fields must reach a terminal state.

COMPLETE or explicitly NOT_FOUND/BLOCKED pass. UNSEEN/DISCOVERING/PARTIAL/
VERIFIED_PARTIAL fail with targeted investigation recovery. Trivia cannot
satisfy this gate: only saturation or explicit absence counts.
"""

from __future__ import annotations

from typing import List

from ..core.schemas import GateResult, RecoveryTask, make_id
from .base import Gate, VerificationContext

OPEN_STATES = frozenset({"UNSEEN", "DISCOVERING", "PARTIAL", "VERIFIED_PARTIAL"})


class FieldCoverageGate:
    name = "field_coverage"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        stated = 0
        for need in ctx.goal.needs:
            state = getattr(need, "state", "")
            if not need.required:
                continue
            if not getattr(need, "field_ref", None):
                continue  # prompt-parsed needs use the completeness gate
            stated += 1
            if state in OPEN_STATES or not state:
                failures.append(
                    f"need {need.need_id} ({need.attribute}): state={state or 'UNSET'} "
                    "— requested field not yet saturated"
                )
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[],
                        reason=f"requested field '{need.attribute}' is {state or 'UNSET'}",
                        suggested_actions=[
                            "search_subject_candidates",
                            "expand_retrieval",
                            "read_candidate_pages",
                            "inspect_tables",
                        ],
                    )
                )
        details = {"field_needs": stated}
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details=details,
        )

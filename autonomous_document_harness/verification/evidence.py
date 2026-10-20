"""Evidence gate: every reported claim resolves to real source evidence."""

from __future__ import annotations

from typing import List

from ..core.schemas import GateResult, RecoveryTask, make_id
from .base import Gate, VerificationContext


class EvidenceGate:
    name = "evidence"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        for claim in ctx.claims:
            if claim.verification.value == "unresolved":
                if not claim.unresolved_reason:
                    failures.append(f"{claim.claim_id}: unresolved without a reason")
                continue
            resolved = [eid for eid in claim.evidence_ids if eid in ctx.evidence]
            if not claim.evidence_ids or not resolved:
                failures.append(
                    f"{claim.claim_id} ({claim.subject}/{claim.attribute}): "
                    "no resolvable source evidence"
                )
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[claim.claim_id],
                        reason="claim has no source evidence",
                        suggested_actions=[
                            "search_corroboration",
                            "read_candidate_pages",
                        ],
                    )
                )
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={"claims": len(ctx.claims)},
        )

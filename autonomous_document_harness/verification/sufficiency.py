"""Hallucination gate + evidence-sufficiency gate (finish responsibly?)."""

from __future__ import annotations

from typing import List

from ..core.schemas import GateResult, RecoveryTask, make_id
from .base import Gate, VerificationContext


def _reported(ctx: VerificationContext):
    return [c for c in ctx.claims if c.verification.value != "unresolved"]


class HallucinationGate:
    name = "hallucination"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        for claim in _reported(ctx):
            if not [eid for eid in claim.evidence_ids if eid in ctx.evidence]:
                failures.append(
                    f"{claim.claim_id} ({claim.subject}/{claim.attribute}): "
                    "reported without any evidence"
                )
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[claim.claim_id],
                        reason="output claim has no evidence (hallucination risk)",
                        suggested_actions=[
                            "find_supporting_evidence",
                            "drop_or_mark_unresolved",
                        ],
                    )
                )
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={"reported": len(_reported(ctx))},
        )


class EvidenceSufficiencyGate:
    name = "sufficiency"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        reported = _reported(ctx)
        verified = [c for c in reported if c.verification.value == "verified"]
        ratio = (len(verified) / len(reported)) if reported else 0.0
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        if not reported:
            failures.append("no reported claims: nothing verified")
        elif ratio < ctx.min_verified_ratio:
            failures.append(
                f"verified ratio {ratio:.2f} below threshold "
                f"{ctx.min_verified_ratio:.2f}"
            )
            weak = [c.claim_id for c in reported if c.verification.value != "verified"]
            recovery.append(
                RecoveryTask(
                    task_id=make_id("rec"),
                    claim_ids=weak,
                    reason="insufficient verified evidence to finish responsibly",
                    suggested_actions=[
                        "targeted_recovery",
                        "search_corroboration",
                        "mark_unresolved_with_reason",
                    ],
                )
            )
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={
                "verified": len(verified),
                "reported": len(reported),
                "ratio": round(ratio, 3),
            },
        )

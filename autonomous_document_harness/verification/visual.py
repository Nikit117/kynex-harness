"""Visual source gate: critical claims need pixel-grounded evidence.

Without a vision model the gate fails open-critical claims with a targeted
recovery task — it never passes them blindly.
"""

from __future__ import annotations

from typing import List

from ..core.schemas import GateResult, RecoveryTask, make_id
from .base import Gate, VerificationContext

_GROUNDED_METHODS = frozenset({"vision", "ocr", "azure_di", "gpt54_vision"})


class VisualGate:
    name = "visual"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        for claim in ctx.claims:
            if not (claim.qualifiers or {}).get("critical"):
                continue
            if claim.verification.value == "unresolved":
                continue
            grounded = any(
                (
                    ctx.evidence[eid].method in _GROUNDED_METHODS
                    or ctx.evidence[eid].region.extraction_method in _GROUNDED_METHODS
                )
                for eid in claim.evidence_ids
                if eid in ctx.evidence
            )
            if not grounded:
                failures.append(
                    f"{claim.claim_id} ({claim.subject}/{claim.attribute}): critical "
                    "value lacks visual/OCR grounding"
                )
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[claim.claim_id],
                        reason="critical value not checked against rendered source",
                        suggested_actions=[
                            "render_region",
                            "inspect_region_visually",
                            "ocr_region",
                        ],
                    )
                )
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={
                "critical_claims": sum(
                    1 for c in ctx.claims if (c.qualifiers or {}).get("critical")
                )
            },
        )

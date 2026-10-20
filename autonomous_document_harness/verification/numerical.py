"""Numerical gate: generic arithmetic sanity (totals, signs, units).

Domain packs may append validators via ctx.domain_validators; the core owns
only domain-free rules.
"""

from __future__ import annotations

from typing import List

from ..core.schemas import GateResult, RecoveryTask, make_id
from .base import VerificationContext

_COUNT_SUFFIXES = ("count", "total", "number", "qty", "quantity")


class NumericalGate:
    name = "numerical"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        for claim in ctx.claims:
            if claim.verification.value == "unresolved":
                continue
            attr = claim.attribute.strip().lower()
            value = claim.value
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if attr.endswith(_COUNT_SUFFIXES) and value < 0:
                    failures.append(f"{claim.claim_id}: negative count {value}")
                    recovery.append(self._rec(claim.claim_id, "negative count value"))
                if (claim.units == "%" or "percent" in attr) and not 0 <= value <= 100:
                    failures.append(
                        f"{claim.claim_id}: percentage out of range: {value}"
                    )
                    recovery.append(
                        self._rec(claim.claim_id, "percentage out of [0,100]")
                    )
            quals = claim.qualifiers or {}
            if "expected_total" in quals and "components" in quals:
                try:
                    total = float(quals["expected_total"])
                    parts = [float(x) for x in quals["components"]]
                    if abs(sum(parts) - total) > max(1e-6, abs(total) * 1e-6):
                        failures.append(
                            f"{claim.claim_id}: components sum to {sum(parts)}, "
                            f"expected {total}"
                        )
                        recovery.append(
                            self._rec(
                                claim.claim_id,
                                "component sum disagrees with stated total",
                                ["recompute_total", "inspect_component_rows"],
                            )
                        )
                except (TypeError, ValueError):
                    pass
        for validator in ctx.domain_validators:
            for failure in validator(ctx) or []:
                failures.append(str(failure))
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={"checked": len(ctx.claims)},
        )

    @staticmethod
    def _rec(
        claim_id: str, reason: str, actions: List[str] | None = None
    ) -> RecoveryTask:
        return RecoveryTask(
            task_id=make_id("rec"),
            claim_ids=[claim_id],
            reason=reason,
            suggested_actions=actions or ["inspect_original_region"],
        )

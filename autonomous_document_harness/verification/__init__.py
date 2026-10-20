"""Verification harness: run all gates, aggregate recovery tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..core.schemas import GateResult, RecoveryTask
from .base import Gate, VerificationContext
from .completeness import CompletenessGate
from .contradiction import ContradictionGate
from .coverage import CoverageGate
from .entailment import EntailmentGate
from .evidence import EvidenceGate
from .field_coverage import FieldCoverageGate
from .numerical import NumericalGate
from .schema import SchemaGate
from .sufficiency import EvidenceSufficiencyGate, HallucinationGate
from .visual import VisualGate

__all__ = [
    "Gate",
    "VerificationContext",
    "VerificationReport",
    "VerificationHarness",
    "build_default_harness",
]


@dataclass
class VerificationReport:
    passed: bool
    results: List[GateResult] = field(default_factory=list)
    recovery: List[RecoveryTask] = field(default_factory=list)

    def failed_gates(self) -> List[str]:
        return [r.gate for r in self.results if not r.passed]

    def to_dict(self):
        return {
            "passed": self.passed,
            "gates": [
                {
                    "gate": r.gate,
                    "passed": r.passed,
                    "failures": list(r.failures),
                    "recovery": [t.task_id for t in r.recovery],
                    "details": dict(r.details),
                }
                for r in self.results
            ],
            "recovery": [
                {
                    "task_id": t.task_id,
                    "claim_ids": list(t.claim_ids),
                    "reason": t.reason,
                    "suggested_actions": list(t.suggested_actions),
                    "status": t.status,
                    "attempts": t.attempts,
                    "outcome": t.outcome,
                    "cost": t.cost,
                }
                for t in self.recovery
            ],
        }


class VerificationHarness:
    def __init__(self, gates: List[Gate]) -> None:
        self.gates = list(gates)

    def evaluate(self, ctx: VerificationContext) -> VerificationReport:
        results = [gate.evaluate(ctx) for gate in self.gates]
        recovery: List[RecoveryTask] = []
        seen = set()
        for result in results:
            for task in result.recovery:
                key = (tuple(sorted(task.claim_ids)), task.reason)
                if key not in seen:
                    seen.add(key)
                    recovery.append(task)
        return VerificationReport(
            passed=all(r.passed for r in results),
            results=results,
            recovery=recovery[:25],
        )


def build_default_harness(
    min_overlap: float = 0.34, min_verified_ratio: float = 0.8
) -> VerificationHarness:
    gates: List[Gate] = [
        SchemaGate(),
        EvidenceGate(),
        EntailmentGate(),
        ContradictionGate(),
        NumericalGate(),
        VisualGate(),
        CompletenessGate(),
        HallucinationGate(),
        EvidenceSufficiencyGate(),
        CoverageGate(),
        FieldCoverageGate(),
    ]
    harness = VerificationHarness(gates)
    _ = (min_overlap, min_verified_ratio)
    return harness

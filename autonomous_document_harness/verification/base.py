"""Gate protocol + shared verification context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol

from ..core.schemas import Claim, Evidence, GateResult, InvestigationGoal


@dataclass
class VerificationContext:
    goal: InvestigationGoal
    claims: List[Claim] = field(default_factory=list)
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    contradictions: List[Any] = field(default_factory=list)  # evidence.contradictions
    min_overlap: float = 0.34
    min_verified_ratio: float = 0.8
    domain_validators: List[Callable[["VerificationContext"], List[str]]] = field(
        default_factory=list
    )
    world: Any = None  # workspace.WorldModel (coverage gate; optional)
    visited_pages: set = field(default_factory=set)
    visited_sheets: set = field(default_factory=set)


class Gate(Protocol):
    name: str

    def evaluate(self, ctx: VerificationContext) -> GateResult: ...

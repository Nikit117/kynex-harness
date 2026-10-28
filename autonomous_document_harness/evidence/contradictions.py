"""Contradictions are first-class investigation tasks — never silently resolved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core.schemas import make_id
from .claims import ClaimStore


@dataclass
class Contradiction:
    contradiction_id: str
    subject: str
    attribute: str
    claim_ids: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    status: str = "open"  # open | resolved | unresolved
    resolution: str = ""


def detect_contradictions(store: ClaimStore) -> List[Contradiction]:
    """Group same subject/attribute claims with differing values."""
    groups: Dict[str, List] = {}
    for claim in store.all_claims():
        if claim.verification.value == "unresolved":
            continue
        key = f"{claim.subject.strip().lower()}|{claim.attribute.strip().lower()}"
        groups.setdefault(key, []).append(claim)
    out = []
    for key, claims in groups.items():
        distinct: List[Any] = []
        for claim in claims:
            if not any(_loose_equal(claim.value, seen) for seen in distinct):
                distinct.append(claim.value)
        if len(distinct) > 1:
            subject, _, attribute = key.partition("|")
            # recover original casing from the first claim
            first = claims[0]
            out.append(
                Contradiction(
                    contradiction_id=make_id("contra"),
                    subject=first.subject,
                    attribute=first.attribute,
                    claim_ids=[c.claim_id for c in claims],
                    values=distinct,
                )
            )
    return out


def _loose_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return str(a).strip().lower() == str(b).strip().lower()


def resolve_contradiction(
    contra: Contradiction, winning_claim_id: str, note: str
) -> Contradiction:
    if winning_claim_id not in contra.claim_ids:
        raise ValueError(f"{winning_claim_id} is not part of {contra.contradiction_id}")
    contra.status = "resolved"
    contra.resolution = note
    return contra


def mark_unresolvable(contra: Contradiction, note: str) -> Contradiction:
    contra.status = "unresolved"
    contra.resolution = note
    return contra

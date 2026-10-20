"""Canonical knowledge model: normalized facts referencing (never replacing) evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core.schemas import Claim, Evidence, VerificationState


@dataclass
class CanonicalFact:
    subject: str
    attribute: str
    value: Any
    units: str | None
    claim_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    facets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalStore:
    facts: List[CanonicalFact] = field(default_factory=list)

    @classmethod
    def from_claims(
        cls, claims: List[Claim], evidence_by_id: Dict[str, Evidence]
    ) -> "CanonicalStore":
        """Only VERIFIED claims become canonical facts. Raw evidence untouched."""
        store = cls()
        for claim in claims:
            if claim.verification != VerificationState.VERIFIED:
                continue
            known = [eid for eid in claim.evidence_ids if eid in evidence_by_id]
            facets = dict(claim.qualifiers or {})
            facets.setdefault("entity", claim.subject.strip())
            store.facts.append(
                CanonicalFact(
                    subject=claim.subject.strip(),
                    attribute=claim.attribute.strip(),
                    value=claim.value,
                    units=claim.units,
                    claim_ids=[claim.claim_id],
                    evidence_ids=known,
                    facets=facets,
                )
            )
        return store

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": [
                {
                    "subject": f.subject,
                    "attribute": f.attribute,
                    "value": f.value,
                    "units": f.units,
                    "claim_ids": list(f.claim_ids),
                    "evidence_ids": list(f.evidence_ids),
                    "facets": dict(f.facets),
                }
                for f in self.facts
            ]
        }

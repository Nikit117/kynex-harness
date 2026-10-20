"""Evidence memory: agent-facing index over the claim store.

Facts live in evidence/claims.py (ClaimStore). This is a query view only —
it holds IDs and tallies, never competing copies of values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core.schemas import VerificationState


@dataclass
class EvidenceMemory:
    by_subject_attribute: Dict[str, List[str]] = field(default_factory=dict)
    verification_tally: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def key(subject: str, attribute: str) -> str:
        return f"{subject.strip().lower()}|{attribute.strip().lower()}"

    def index_claim(self, claim) -> None:  # Claim (duck-typed to avoid cycles)
        self.by_subject_attribute.setdefault(
            self.key(claim.subject, claim.attribute), []
        ).append(claim.claim_id)
        state = (
            claim.verification.value
            if isinstance(claim.verification, VerificationState)
            else str(claim.verification)
        )
        self.verification_tally[state] = self.verification_tally.get(state, 0) + 1

    def claim_ids_for(self, subject: str, attribute: str) -> List[str]:
        return list(self.by_subject_attribute.get(self.key(subject, attribute), []))

    def has_verified(
        self, subject: str, attribute: str, store
    ) -> bool:  # ClaimStore (duck-typed)
        return any(
            store.get_claim(cid) is not None
            and store.get_claim(cid).verification == VerificationState.VERIFIED
            for cid in self.claim_ids_for(subject, attribute)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indexed_pairs": len(self.by_subject_attribute),
            "verification_tally": dict(self.verification_tally),
        }

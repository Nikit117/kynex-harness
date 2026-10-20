"""Atomic claim store. Claims are the unit of output; evidence is never overwritten."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.schemas import Claim, Evidence, SourceRegion, VerificationState, make_id


@dataclass
class ClaimStore:
    claims: Dict[str, Claim] = field(default_factory=dict)
    evidence: Dict[str, Evidence] = field(default_factory=dict)

    # -- evidence ---------------------------------------------------------
    def add_evidence(
        self,
        region: SourceRegion,
        snippet: str,
        method: str,
        confidence: float,
        produced_by: str = "investigator",
    ) -> Evidence:
        ev = Evidence(
            evidence_id=make_id("ev"),
            region=region,
            snippet=snippet,
            method=method,
            confidence=confidence,
            produced_by=produced_by,
        )
        self.evidence[ev.evidence_id] = ev
        return ev

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        return self.evidence.get(evidence_id)

    # -- claims ------------------------------------------------------------
    def create_claim(
        self,
        subject: str,
        attribute: str,
        value: Any,
        evidence_ids: Optional[List[str]] = None,
        units: Optional[str] = None,
        qualifiers: Optional[Dict[str, Any]] = None,
        tier: str = "observation",
        field_ref: Optional[Dict[str, Any]] = None,
    ) -> Claim:
        claim = Claim(
            claim_id=make_id("claim"),
            subject=subject.strip(),
            attribute=attribute.strip(),
            value=value,
            evidence_ids=list(evidence_ids or []),
            verification=VerificationState.UNVERIFIED,
            units=units,
            qualifiers=dict(qualifiers or {}),
            tier=tier,
            field_ref=dict(field_ref or {}),
        )
        self.claims[claim.claim_id] = claim
        return claim

    def add_link(self, claim_id: str, evidence_id: str) -> None:
        claim = self.claims[claim_id]
        if evidence_id not in claim.evidence_ids:
            claim.evidence_ids.append(evidence_id)

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self.claims.get(claim_id)

    def claims_for(self, subject: str, attribute: str) -> List[Claim]:
        return [
            c
            for c in self.claims.values()
            if c.subject.strip().lower() == subject.strip().lower()
            and c.attribute.strip().lower() == attribute.strip().lower()
        ]

    def find_duplicate(
        self,
        subject: str,
        attribute: str,
        value: Any,
        doc_id: str,
        page: Optional[int],
        cell: Optional[str],
    ) -> Optional[Claim]:
        for claim in self.claims_for(subject, attribute):
            if not _values_equal(claim.value, value):
                continue
            for eid in claim.evidence_ids:
                ev = self.evidence.get(eid)
                if ev is None:
                    continue
                if (
                    ev.region.document_id == doc_id
                    and ev.region.page == page
                    and ev.region.cell == cell
                ):
                    return claim
        return None

    def all_claims(self) -> List[Claim]:
        return list(self.claims.values())

    # -- verification transitions -------------------------------------------
    def mark_verified(self, claim_id: str) -> None:
        self.claims[claim_id].verification = VerificationState.VERIFIED

    def mark_uncertain(self, claim_id: str, reason: str = "") -> None:
        claim = self.claims[claim_id]
        claim.verification = VerificationState.UNCERTAIN
        claim.unresolved_reason = reason

    def mark_contradicted(self, claim_id: str, reason: str = "") -> None:
        claim = self.claims[claim_id]
        claim.verification = VerificationState.CONTRADICTED
        claim.unresolved_reason = reason

    def mark_unresolved(self, claim_id: str, reason: str) -> None:
        claim = self.claims[claim_id]
        claim.verification = VerificationState.UNRESOLVED
        claim.unresolved_reason = reason

    def evidence_for(self, claim: Claim) -> List[Evidence]:
        return [
            self.evidence[eid] for eid in claim.evidence_ids if eid in self.evidence
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "subject": c.subject,
                    "attribute": c.attribute,
                    "value": c.value,
                    "units": c.units,
                    "evidence_ids": list(c.evidence_ids),
                    "verification": c.verification.value,
                    "qualifiers": dict(c.qualifiers),
                    "unresolved_reason": c.unresolved_reason,
                    "derived_from": list(c.derived_from),
                    "tier": c.tier,
                    "field_ref": dict(c.field_ref),
                }
                for c in self.claims.values()
            ],
            "evidence": [
                {
                    "evidence_id": e.evidence_id,
                    "snippet": e.snippet,
                    "method": e.method,
                    "confidence": e.confidence,
                    "region": {
                        "document_id": e.region.document_id,
                        "page": e.region.page,
                        "cell": e.region.cell,
                        "bbox": list(e.region.bbox) if e.region.bbox else None,
                        "source_type": e.region.source_type,
                        "extraction_method": e.region.extraction_method,
                    },
                }
                for e in self.evidence.values()
            ],
        }


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return str(a).strip().lower() == str(b).strip().lower()

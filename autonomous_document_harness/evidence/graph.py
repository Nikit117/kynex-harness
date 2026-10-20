"""Live evidence graph. Built DURING reasoning — contradictions steer the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..core.schemas import Claim, Evidence, EvidenceRelation
from .claims import _values_equal
from .provenance import region_key


@dataclass
class Edge:
    src_kind: str  # claim | evidence | region
    src_id: str
    relation: str
    dst_kind: str
    dst_id: str


@dataclass
class EvidenceGraph:
    nodes_claim: Dict[str, Claim] = field(default_factory=dict)
    nodes_evidence: Dict[str, Evidence] = field(default_factory=dict)
    regions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    # -- construction ------------------------------------------------------
    def add_claim(self, claim: Claim) -> None:
        self.nodes_claim[claim.claim_id] = claim

    def add_evidence(self, evidence: Evidence) -> None:
        self.nodes_evidence[evidence.evidence_id] = evidence
        key = region_key(
            evidence.region.document_id, evidence.region.page, evidence.region.cell
        )
        self.regions.setdefault(
            key,
            {
                "document_id": evidence.region.document_id,
                "page": evidence.region.page,
                "cell": evidence.region.cell,
                "source_type": evidence.region.source_type,
            },
        )
        self.edges.append(
            Edge(
                "evidence",
                evidence.evidence_id,
                EvidenceRelation.LOCATED_IN.value,
                "region",
                key,
            )
        )

    def support(self, evidence_id: str, claim_id: str) -> None:
        self.edges.append(
            Edge(
                "evidence",
                evidence_id,
                EvidenceRelation.SUPPORTS.value,
                "claim",
                claim_id,
            )
        )

    def contradict(self, evidence_id: str, claim_id: str) -> None:
        self.edges.append(
            Edge(
                "evidence",
                evidence_id,
                EvidenceRelation.CONTRADICTS.value,
                "claim",
                claim_id,
            )
        )

    def derive(self, child_id: str, parent_id: str) -> None:
        self.edges.append(
            Edge(
                "claim",
                child_id,
                EvidenceRelation.DERIVED_FROM.value,
                "claim",
                parent_id,
            )
        )

    def corroborate(self, evidence_a: str, evidence_b: str) -> None:
        self.edges.append(
            Edge(
                "evidence",
                evidence_a,
                EvidenceRelation.CORROBORATES.value,
                "evidence",
                evidence_b,
            )
        )

    def conflict(self, evidence_a: str, evidence_b: str) -> None:
        self.edges.append(
            Edge(
                "evidence",
                evidence_a,
                EvidenceRelation.CONFLICTS_WITH.value,
                "evidence",
                evidence_b,
            )
        )

    # -- queries --------------------------------------------------------------
    def neighbors(
        self, kind: str, node_id: str, relation: str | None = None
    ) -> List[Edge]:
        return [
            e
            for e in self.edges
            if e.src_kind == kind
            and e.src_id == node_id
            and (relation is None or e.relation == relation)
        ]

    def supporting_evidence(self, claim_id: str) -> List[str]:
        return [
            e.src_id
            for e in self.edges
            if e.dst_kind == "claim"
            and e.dst_id == claim_id
            and e.relation == EvidenceRelation.SUPPORTS.value
        ]

    def contradictions(self) -> List[Tuple[str, str]]:
        """Claim pairs asserting different values for the same subject/attribute."""
        pairs = []
        claims = list(self.nodes_claim.values())
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                a, b = claims[i], claims[j]
                if (
                    a.subject.strip().lower() == b.subject.strip().lower()
                    and a.attribute.strip().lower() == b.attribute.strip().lower()
                    and not _values_equal(a.value, b.value)
                    and a.verification.value != "unresolved"
                    and b.verification.value != "unresolved"
                ):
                    pairs.append((a.claim_id, b.claim_id))
        return pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": list(self.nodes_claim),
            "evidence": list(self.nodes_evidence),
            "regions": dict(self.regions),
            "edges": [
                {
                    "src": f"{e.src_kind}:{e.src_id}",
                    "relation": e.relation,
                    "dst": f"{e.dst_kind}:{e.dst_id}",
                }
                for e in self.edges
            ],
        }

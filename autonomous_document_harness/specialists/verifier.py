"""Verifier specialist: batch entailment over unverified claims."""

from __future__ import annotations

from typing import Any, Dict, List

from ..evidence.claims import ClaimStore
from ..verification.entailment import entails


def verify_batch(
    store: ClaimStore, min_overlap: float = 0.34, max_claims: int = 50, gateway=None
) -> Dict[str, Any]:
    """Mark each UNVERIFIED claim verified/uncertain. Returns per-claim summary.

    With a model gateway: independent verdicts SUPPORTED | CONTRADICTED |
    AMBIGUOUS (verifier sees only claim + snippets, never extractor reasoning).
    Without: deterministic heuristic entailment.
    """
    checked: List[Dict[str, Any]] = []
    for claim in store.all_claims()[:max_claims]:
        if claim.verification.value != "unverified":
            continue
        snippets = [
            store.evidence[eid].snippet
            for eid in claim.evidence_ids
            if eid in store.evidence
        ]
        if not snippets:
            store.mark_uncertain(claim.claim_id, "no evidence attached yet")
            checked.append(
                {
                    "claim_id": claim.claim_id,
                    "supported": False,
                    "score": 0.0,
                    "reason": "no-evidence",
                }
            )
            continue
        if gateway is not None:
            try:
                result = gateway.verify_claim_structured(
                    claim.value, snippets, context=f"{claim.subject}/{claim.attribute}"
                )
            except Exception:
                result = None
            if result is not None:
                verdict = result.get("verdict", "AMBIGUOUS")
                if verdict == "SUPPORTED":
                    store.mark_verified(claim.claim_id)
                    checked.append(
                        {
                            "claim_id": claim.claim_id,
                            "supported": True,
                            "score": 1.0,
                            "via": "model",
                            "rationale": result.get("rationale", ""),
                        }
                    )
                elif verdict == "CONTRADICTED":
                    store.mark_uncertain(
                        claim.claim_id,
                        "independent verifier contradicts: "
                        f"{result.get('rationale', '')[:200]}",
                    )
                    checked.append(
                        {
                            "claim_id": claim.claim_id,
                            "supported": False,
                            "score": 0.0,
                            "via": "model",
                            "rationale": result.get("rationale", ""),
                        }
                    )
                else:
                    store.mark_uncertain(
                        claim.claim_id,
                        "independent verifier ambiguous: "
                        f"{result.get('rationale', '')[:200]}",
                    )
                    checked.append(
                        {
                            "claim_id": claim.claim_id,
                            "supported": False,
                            "score": 0.5,
                            "via": "model",
                            "rationale": result.get("rationale", ""),
                        }
                    )
                continue
        supported, score, _ = entails(claim.value, snippets, min_overlap)
        if supported:
            store.mark_verified(claim.claim_id)
        else:
            store.mark_uncertain(claim.claim_id, "cited snippets do not entail value")
        checked.append(
            {"claim_id": claim.claim_id, "supported": supported, "score": score}
        )
    return {
        "checked": len(checked),
        "verified": sum(1 for c in checked if c["supported"]),
        "details": checked,
    }

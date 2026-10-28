"""Semantic admission gate for the Deep Agents runtime (§11).

Wraps the general heuristic gate (models/admission.py) so only ANSWER
verdicts may contribute toward contract completion. The gate works against
the Pydantic ExtractionContract via duck typing (target_entity,
target_attribute, answer_type, exclusions).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..models.admission import (
    AMBIGUOUS,
    ANSWER,
    IRRELEVANT,
    SUPPORTING_CONTEXT,
    evaluate,
)


def admit_observation(
    snippet: str,
    contract: Any,
    exclusions: List[str] | None = None,
) -> Tuple[str, str]:
    """Return (verdict, reason) for one observation against one contract.

    ANSWER observations may become candidate answers; SUPPORTING_CONTEXT is
    retained in the observation ledger only and never advances completion.
    """
    return evaluate(snippet, contract, exclusions)


def classify_batch(
    snippets: List[str], contract: Any, exclusions: List[str] | None = None
) -> List[Dict[str, str]]:
    """Classify many snippets; returns [{snippet, verdict, reason}]."""
    out = []
    for snippet in snippets:
        verdict, reason = admit_observation(snippet, contract, exclusions)
        out.append({"snippet": snippet, "verdict": verdict, "reason": reason})
    return out


__all__ = [
    "admit_observation",
    "classify_batch",
    "ANSWER",
    "SUPPORTING_CONTEXT",
    "IRRELEVANT",
    "AMBIGUOUS",
]

"""Heuristic entailment: does the cited snippet actually state the value?

Pure function shared by the entailment gate AND the verify_claim tool, so both
judge support identically. Numbers must appear literally; text needs overlap.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from ..core.schemas import GateResult, RecoveryTask, make_id
from ..workspace.indexing import tokenize
from .base import Gate, VerificationContext

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def snippet_numbers(text: str) -> List[float]:
    out = []
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def entails(
    value: Any, snippets: List[str], min_overlap: float = 0.34
) -> Tuple[bool, float, int]:
    """Return (supported, score, matched_snippet_index)."""
    matched_idx, score = -1, 0.0
    if isinstance(value, bool):
        tokens: List[str] = [str(value).lower()]
        numeric = None
    elif isinstance(value, (int, float)):
        numeric, tokens = float(value), [str(value)]
    else:
        numeric, tokens = None, [t for t in tokenize(str(value)) if len(t) > 1]
    for idx, snippet in enumerate(snippets):
        if numeric is not None:
            if any(abs(n - numeric) < 1e-9 for n in snippet_numbers(snippet)):
                return True, 1.0, idx
            continue
        overlap = len(set(tokens) & set(tokenize(snippet))) / max(len(set(tokens)), 1)
        if overlap > score:
            score, matched_idx = overlap, idx
    return (score >= float(min_overlap)), round(score, 4), matched_idx


class EntailmentGate:
    name = "entailment"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        for claim in ctx.claims:
            if claim.verification.value == "unresolved":
                continue
            snippets = [
                ctx.evidence[eid].snippet
                for eid in claim.evidence_ids
                if eid in ctx.evidence
            ]
            if not snippets:
                continue  # evidence gate owns the no-evidence case
            supported, score, _ = entails(claim.value, snippets, ctx.min_overlap)
            if not supported:
                failures.append(
                    f"{claim.claim_id} ({claim.subject}/{claim.attribute}={claim.value!r}): "
                    f"no cited snippet entails the value (score {score})"
                )
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[claim.claim_id],
                        reason="cited evidence does not state the claimed value",
                        suggested_actions=[
                            "inspect_original_region",
                            "search_corroboration",
                        ],
                    )
                )
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details={"checked": len(ctx.claims)},
        )

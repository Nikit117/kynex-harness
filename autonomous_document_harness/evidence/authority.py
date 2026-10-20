"""Generic source-authority reasoning: revisions, dates, supersession, class.

No domain lists. Signals come from evidence text and filenames only. When no
signal establishes precedence, authority stays uncertain — never a silent pick.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DATE_RE = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
REV_RE = re.compile(r"\b[Rr]ev(?:ision)?\.?\s*([A-Z]|\d+)\b")
ADDENDUM_RE = re.compile(r"\baddendum\b", re.IGNORECASE)
SUPERSEDED_SELF_RE = re.compile(r"\bsuperseded\b", re.IGNORECASE)
SUPERSEDES_OTHER_RE = re.compile(r"\bsupersedes\b", re.IGNORECASE)
CLASS_HINTS = (
    "addendum",
    "appendix",
    "schedule",
    "summary",
    "datasheet",
    "specification",
    "revision",
)


@dataclass
class AuthoritySignal:
    claim_id: str
    kind: str  # revision | date | supersession | doc_class
    value: str
    evidence_id: str = ""


def _rev_key(token: str) -> tuple:
    token = token.strip()
    if token.isdigit():
        return (0, int(token))
    if len(token) == 1 and token.isalpha():
        return (1, ord(token.upper()))
    return (2, token)


def extract_signals(claim, evidence_list, filename: str = "") -> List[AuthoritySignal]:
    """Collect precedence signals for one claim from its evidence + filename."""
    signals: List[AuthoritySignal] = []
    texts = [filename] + [e.snippet for e in evidence_list]
    for ev in evidence_list:
        for match in REV_RE.finditer(ev.snippet):
            signals.append(
                AuthoritySignal(
                    claim.claim_id, "revision", match.group(1), ev.evidence_id
                )
            )
        for match in DATE_RE.finditer(ev.snippet):
            signals.append(
                AuthoritySignal(claim.claim_id, "date", match.group(0), ev.evidence_id)
            )
        if SUPERSEDED_SELF_RE.search(ev.snippet):
            # "Rev B, superseded": the statement describes ITSELF as obsolete.
            signals.append(
                AuthoritySignal(
                    claim.claim_id,
                    "superseded_self",
                    ev.snippet[:120],
                    ev.evidence_id,
                )
            )
        elif SUPERSEDES_OTHER_RE.search(ev.snippet):
            signals.append(
                AuthoritySignal(
                    claim.claim_id,
                    "supersedes_other",
                    ev.snippet[:120],
                    ev.evidence_id,
                )
            )
        lowered_snippet = ev.snippet.lower()
        for hint in CLASS_HINTS:
            if hint in lowered_snippet:
                signals.append(
                    AuthoritySignal(claim.claim_id, "doc_class", hint, ev.evidence_id)
                )
                break
    lowered = filename.lower()
    for hint in CLASS_HINTS:
        if hint in lowered:
            signals.append(AuthoritySignal(claim.claim_id, "doc_class", hint))
            break
    for match in REV_RE.finditer(filename):
        signals.append(AuthoritySignal(claim.claim_id, "revision", match.group(1)))
    # Deduplicate, keep order.
    seen, unique = set(), []
    for signal in signals:
        key = (signal.kind, signal.value)
        if key not in seen:
            seen.add(key)
            unique.append(signal)
    return unique


def propose_authority(contra, store, filenames: Dict[str, str]) -> Dict[str, Any]:
    """Return {winning_claim_id|None, rationale, signals}.

    Precedence: explicit supersession > higher revision > later date >
    addendum-class over base for overlapping scope. Ties/absence -> uncertain.
    """
    per_claim: Dict[str, List[AuthoritySignal]] = {}
    for claim_id in contra.claim_ids:
        claim = store.get_claim(claim_id)
        if claim is None:
            continue
        evidence = store.evidence_for(claim)
        doc = evidence[0].region.document_id if evidence else ""
        per_claim[claim_id] = extract_signals(claim, evidence, filenames.get(doc, ""))
    # 1a. A claim marked superseded in its own evidence loses.
    obsolete = [
        cid
        for cid, sigs in per_claim.items()
        if any(s.kind == "superseded_self" for s in sigs)
    ]
    if len(obsolete) == 1 and len(per_claim) > 1:
        winner = next(cid for cid in per_claim if cid not in obsolete)
        return {
            "winning_claim_id": winner,
            "rationale": ("supersession: rival marked superseded in its own evidence"),
            "signals": {
                k: [(s.kind, s.value) for s in v] for k, v in per_claim.items()
            },
        }
    # 1b. An active "supersedes" statement wins for its claim.
    for claim_id, signals in per_claim.items():
        if any(s.kind == "supersedes_other" for s in signals):
            return {
                "winning_claim_id": claim_id,
                "rationale": (
                    "supersession: explicit superseding statement in evidence"
                ),
                "signals": {
                    k: [(s.kind, s.value) for s in v] for k, v in per_claim.items()
                },
            }
    # 2. Highest revision token wins (letters and numbers compared separately).
    revs = {
        cid: [s.value for s in sigs if s.kind == "revision"]
        for cid, sigs in per_claim.items()
    }
    if any(revs.values()):
        ranked = sorted(
            ((cid, max(vals, key=_rev_key)) for cid, vals in revs.items() if vals),
            key=lambda pair: _rev_key(pair[1]),
        )
        if len(ranked) >= 2 and _rev_key(ranked[-1][1]) != _rev_key(ranked[-2][1]):
            return {
                "winning_claim_id": ranked[-1][0],
                "rationale": f"higher revision ({ranked[-1][1]} over {ranked[-2][1]})",
                "signals": {
                    k: [(s.kind, s.value) for s in v] for k, v in per_claim.items()
                },
            }
    # 3. Later date wins.
    dates = {
        cid: sorted({s.value for s in sigs if s.kind == "date"})
        for cid, sigs in per_claim.items()
    }
    dated = {cid: vals[-1] for cid, vals in dates.items() if vals}
    if len(set(dated.values())) > 1:
        winner = max(dated, key=lambda cid: dated[cid])
        return {
            "winning_claim_id": winner,
            "rationale": f"later evidence date ({dated[winner]})",
            "signals": {
                k: [(s.kind, s.value) for s in v] for k, v in per_claim.items()
            },
        }
    # 4. Addendum class outranks base documents for overlapping scope.
    classes = {
        cid: [s.value for s in sigs if s.kind == "doc_class"]
        for cid, sigs in per_claim.items()
    }
    if any("addendum" in vals for vals in classes.values()) and any(
        "addendum" not in vals for vals in classes.values()
    ):
        winner = next(cid for cid, vals in classes.items() if "addendum" in vals)
        return {
            "winning_claim_id": winner,
            "rationale": "addendum supersedes base document for overlapping scope",
            "signals": {
                k: [(s.kind, s.value) for s in v] for k, v in per_claim.items()
            },
        }
    return {
        "winning_claim_id": None,
        "rationale": "no precedence signal; authority uncertain",
        "signals": {k: [(s.kind, s.value) for s in v] for k, v in per_claim.items()},
    }

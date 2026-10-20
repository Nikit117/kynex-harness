"""Semantic admission gate (§REAL03-4): observation -> contract verdict.

ANSWER | SUPPORTING_CONTEXT | IRRELEVANT | AMBIGUOUS. Only ANSWER may
contribute toward contract completion. Heuristic by default (offline,
deterministic); model verdicts for heuristic-AMBIGUOUS cases when a gateway
is available. No domain lists; local token normalization only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

ANSWER = "ANSWER"
SUPPORTING_CONTEXT = "SUPPORTING_CONTEXT"
IRRELEVANT = "IRRELEVANT"
AMBIGUOUS = "AMBIGUOUS"


def _norm_tokens(text: str) -> set:
    from ..workspace.indexing import tokenize

    tokens = set(tokenize(text or ""))
    variants = set(tokens)
    for token in tokens:  # light singular/plural folding, local to admission
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            variants.add(token[:-1])
        variants.add(token + "s")
    return variants


# Explicit negative verdicts ("absent content" answers). A negative value is
# admissible only with a search-scope snippet: the gate enforces the shape
# (record linkage + scope words) so negatives can never be sprayed blindly.
NEGATIVE_VALUE_RE = re.compile(
    r"^\s*(not specified|not applicable|n/a|should be omitted|no .*"
    r"(found|stated|specified|required)|none|absent)\b",
    re.IGNORECASE,
)
SCOPE_WORD_RE = re.compile(
    r"search|inspect|read|page|section|table|sheet|scope|found|absent|"
    r"omitted|checked|reviewed|no .*stat",
    re.IGNORECASE,
)


def negative_verdict_ok(value: str, snippet: str, subject: str) -> bool:
    """True when an explicit negative verdict is adequately evidenced.

    Requires: (1) the value itself is a negative verdict, (2) the snippet
    describes the search scope (scope words present), (3) the scope names
    a REAL record: shared substantive tokens with the subject, IGNORING
    negative filler words ("not specified" as a subject is not a record key
    and can never anchor admission by itself).
    """
    from ..workspace.indexing import tokenize

    if not NEGATIVE_VALUE_RE.search(value or ""):
        return False
    if not SCOPE_WORD_RE.search(snippet or ""):
        return False
    filler = {
        "not",
        "specified",
        "applicable",
        "na",
        "n/a",
        "none",
        "absent",
        "omitted",
        "unknown",
        "specifieds",
    }
    key_tokens = {t for t in tokenize(subject or "") if len(t) > 2} - filler
    snip_tokens = set(tokenize(snippet or ""))
    shared = key_tokens & snip_tokens
    if len(shared) < 2:
        return False

    # At least one shared token must be specific (an identifier, code, or
    # long field word) — bare category words ("remote", "unit", "type") name
    # no actual record and would admit vacuous negatives.
    def _specific(token: str) -> bool:
        return (
            any(ch.isdigit() for ch in token)
            or len(token) >= 7
            or (token.isupper() and len(token) >= 2)
        )

    return any(_specific(t) for t in shared)


def _has_concrete_value(
    snippet: str, answer_type: str, exclude: set | None = None
) -> bool:
    lowered = snippet.lower()
    if answer_type == "numeric":
        return bool(re.search(r"-?\d[\d,]*", snippet))
    if answer_type == "boolean":
        return bool(
            re.search(
                r"\b(yes|no|true|false|required|not specified|applicable|specified)\b",
                lowered,
            )
        )
    exclude = exclude or set()
    if re.search(r"\d", snippet):
        return True
    for match in re.finditer(r"\b[A-Z]{2,}(?:[-/][A-Z0-9]+)*\b", snippet):
        if match.group(0).lower() not in exclude:
            return True
    if '"' in snippet:
        return True
    titled = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", snippet)
    return (
        any(all(w.lower() not in exclude for w in phrase.split()) for phrase in titled)
        and len(snippet) > 60
    )


def _attribute_hit(
    snippet: str,
    snippet_tokens: set,
    attribute_terms: List[str],
    entity_terms: List[str],
) -> bool:
    """Keyword hit, or a number directly quantifying the entity (<n> <entity>)."""
    if any(t in snippet_tokens for t in attribute_terms):
        return True
    words = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", snippet) if w]
    entity_positions = {
        i
        for i, w in enumerate(words)
        if w in entity_terms
        or w + "s" in entity_terms
        or (w.endswith("s") and w[:-1] in entity_terms)
    }
    for i, word in enumerate(words):
        if re.fullmatch(r"-?\d[\d,]*", word):
            if any(abs(i - pos) <= 3 for pos in entity_positions):
                return True
    return False


POINTER_WORDS = frozenset(
    {
        "attach",
        "attached",
        "below",
        "table",
        "section",
        "schedule",
        "figure",
        "page",
        "appendix",
        "drawing",
        "list",
        "refer",
        "see",
        "detail",
    }
)


def _entity_score(
    entity_terms: List[str],
    qualifier: str,
    snippet_tokens: set,
    snippet_raw: str = "",
    entity_text: str = "",
) -> float:
    if not entity_terms:
        return 0.5
    hits = sum(
        1
        for term in entity_terms
        if term in snippet_tokens
        or term + "s" in snippet_tokens
        or (term.endswith("s") and term[:-1] in snippet_tokens)
    )
    score = hits / len(entity_terms)
    if score < 0.75 and snippet_raw:
        # Generic initialism: snippet "FAT" for entity "factory acceptance test".
        # Word order matters — derive initials from the entity text, not the set.
        ordered = [w.lower() for w in re.split(r"[^A-Za-z]+", entity_text) if w]
        core = [w for w in ordered if len(w) > 2]
        initials = "".join(w[0] for w in core[:5])
        for token in re.findall(r"\b[A-Za-z]{2,5}\b", snippet_raw):
            if token.lower() == initials and len(initials) >= 2:
                return 0.75
    return score


def alignment_scores(snippet: str, contract) -> Dict[str, Any]:
    """Numeric alignment signals used by evaluate() (exposed for escalation)."""
    snippet_tokens = _norm_tokens(snippet)
    entity_terms = [
        t
        for t in _norm_tokens(f"{contract.target_entity} {contract.entity_qualifier}")
        if len(t) > 2
    ]
    attribute_terms = [t for t in _norm_tokens(contract.target_attribute) if len(t) > 2]
    entity = _entity_score(
        entity_terms,
        contract.entity_qualifier,
        snippet_tokens,
        snippet or "",
        f"{contract.target_entity} {contract.entity_qualifier}",
    )
    attribute_hit = _attribute_hit(
        snippet or "", snippet_tokens, attribute_terms, entity_terms
    )
    entity_lower = {t.lower() for t in entity_terms}
    value_bearing = _has_concrete_value(
        snippet or "", contract.answer_type, exclude=entity_lower
    )
    return {
        "entity": entity,
        "attribute_hit": attribute_hit,
        "value_bearing": value_bearing,
    }


def evaluate(
    snippet: str, contract, exclusions: List[str] | None = None
) -> Tuple[str, str]:
    """Heuristic admission verdict + concise reason. No model needed."""
    from ..models.heuristic import is_excluded

    if is_excluded(snippet or "", exclusions or []):
        return IRRELEVANT, "matches request exclusions"
    scores = alignment_scores(snippet or "", contract)
    entity = scores["entity"]
    attribute_hit = scores["attribute_hit"]
    value_bearing = scores["value_bearing"]
    needs_value = contract.answer_type in ("numeric", "boolean")
    if entity < 0.5 and not attribute_hit:
        return IRRELEVANT, "no entity or attribute alignment"
    if entity >= 0.5 and attribute_hit and (value_bearing or not needs_value):
        return ANSWER, "entity+attribute/value alignment"
    if entity >= 0.5 and value_bearing and entity >= 0.66:
        if contract.answer_type == "boolean":
            # A Yes/No verdict stated on-entity IS the boolean answer.
            return ANSWER, "entity+verdict alignment"
        if contract.answer_type not in ("numeric", "boolean"):
            return ANSWER, "entity+value alignment"
        return AMBIGUOUS, "typed value present without attribute confirmation"
    if entity >= 0.5 and needs_value and not value_bearing:
        snippet_words = {w.lower() for w in re.split(r"[^A-Za-z]+", snippet or "") if w}
        if snippet_words & POINTER_WORDS:
            return AMBIGUOUS, "entity present; value likely in referenced region"
        return SUPPORTING_CONTEXT, "entity topical but no value stated"
    if entity >= 0.5:
        return SUPPORTING_CONTEXT, "entity topical, attribute unanswered"
    return AMBIGUOUS, "weak alignment either way"


def admit(
    snippet: str,
    contract,
    gateway=None,
    exclusions: List[str] | None = None,
    escalate_gray_zone: bool = False,
) -> Tuple[str, str]:
    """Heuristic verdict, escalated to the model when AMBIGUOUS.

    With escalate_gray_zone=True, borderline rejections (heuristic
    IRRELEVANT/SUPPORTING_CONTEXT but entity alignment >= 0.25 — e.g.
    vocabulary mismatch like cubicle/cabinet that keyword scoring cannot see)
    are also judged by the model, which may infer synonym semantics. Clear
    ANSWERs are always trusted; clear misses (< 0.25) never escalate.
    """
    verdict, reason = evaluate(snippet, contract, exclusions)
    if verdict == ANSWER or gateway is None:
        return verdict, reason
    if verdict != AMBIGUOUS and not escalate_gray_zone:
        return verdict, reason
    if verdict != AMBIGUOUS:
        try:
            entity = alignment_scores(snippet or "", contract)["entity"]
        except Exception:
            return verdict, reason
        if entity < 0.25:
            return verdict, reason
    try:
        from ..models.schemas import ModelRequest
        from .json_utils import parse_json_response

        response = gateway.complete(
            ModelRequest(
                kind="chat",
                max_tokens=300,
                response_format="json_object",
                prompt=(
                    "Decide whether this observation answers the requested contract. "
                    'Reply in JSON like {"verdict": "ANSWER|SUPPORTING_CONTEXT|'
                    'IRRELEVANT|AMBIGUOUS", "reason": "<concise>"}. '
                    "Synonyms are allowed (e.g. cubicle/cabinet, panel/enclosure). "
                    "ANSWER only if the observation states the requested attribute "
                    "for the requested entity. Glossary definitions and generic "
                    "descriptions are SUPPORTING_CONTEXT or IRRELEVANT, never ANSWER.\n"
                    f"Contract entity: {contract.target_entity}\n"
                    f"Contract attribute: {contract.target_attribute}\n"
                    f"Observation: {snippet[:800]}"
                ),
            )
        )
        payload, _ = parse_json_response(response.text)
        model_verdict = str(payload.get("verdict", "AMBIGUOUS")).upper()
        if model_verdict in (ANSWER, SUPPORTING_CONTEXT, IRRELEVANT, AMBIGUOUS):
            return model_verdict, str(payload.get("reason", ""))[:200]
    except Exception:
        pass
    return verdict, reason

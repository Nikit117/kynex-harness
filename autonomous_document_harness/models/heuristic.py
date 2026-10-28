"""Local heuristic reasoner: goal parsing + generic value-with-context mining.

Deterministic, offline, dependency-free. It powers the autonomous loop when no
model provider is configured, and documents the exact baseline that any future
learned component must beat (see eval/ablations).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..core.schemas import InformationNeed, InvestigationGoal, make_id
from .schemas import CandidateClaim

_SENTENCE_SPLIT = re.compile(r"[.\n;]+")
_EXCLUSION_CLAUSE = re.compile(
    r"(?:exclud(?:e|es|ing)|omit(?:ting)?|ignoring?|without|except(?:ing)?)\s+([^.;]+)",
    re.IGNORECASE,
)
_VERB_START = re.compile(
    r"^\s*(extract|find|list|get|identify|compare|determine|return|show|"
    r"what|which|how many|capture|collect)\b",
    re.IGNORECASE,
)
_SUBJECT_PHRASE = re.compile(
    r"\b(?:for|of|across|per|by|system[\s-]?wise)\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)?)"
)
_CODE_VALUE = re.compile(r"\b([A-Z]{1,6})\s+(-?\d[\d,]*(?:\.\d+)?)(%?)([A-Za-z]?)\b")
_CODE_EQ_VALUE = re.compile(
    r"\b([A-Z]{1,6})\s*=\s*(-?\d[\d,]*(?:\.\d+)?)(%?)([A-Za-z]?)\b"
)
_SUBJECT_COLON = re.compile(
    r"\b[Ss]ystem\s+([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\s*:\s*(.+)"
)
_SUBJECT_EQ = re.compile(
    r"\b([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*?)\s+([A-Z]{1,6})\s*[:=]\s*"
    r"(-?\d[\d,]*(?:\.\d+)?)(%?)([A-Za-z]?)\b"
)

GENERIC_ATTRIBUTES = frozenset(
    {
        "count",
        "counts",
        "fact",
        "facts",
        "value",
        "values",
        "number",
        "numbers",
        "data",
        "info",
        "io_count",
        "io",
        "total",
        "totals",
        "requirement",
        "requirements",
        "clause",
        "record",
        "records",
        "item",
        "items",
        "detail",
        "details",
        "obligation",
        "obligations",
        "table",
        "specification",
        "description",
    }
)


def parse_goal(prompt: str) -> InvestigationGoal:
    """Turn an arbitrary natural-language objective into needs + exclusions."""
    prompt = (prompt or "").strip()
    exclusions = _parse_exclusions(prompt)
    needs: List[InformationNeed] = []
    for sentence in _SENTENCE_SPLIT.split(prompt):
        sentence = sentence.strip(" ,")
        if not sentence or not _VERB_START.search(sentence):
            continue
        lowered = sentence.lower()
        if any(k in lowered for k in ("count", "number", "total", "how many")):
            attribute = "count"
        elif "clause" in lowered or "termination" in lowered:
            attribute = "clause"
        elif "date" in lowered:
            attribute = "date"
        elif "table" in lowered:
            attribute = "table"
        elif "requirement" in lowered:
            attribute = "requirement"
        else:
            attribute = "fact"
        subjects = _SUBJECT_PHRASE.findall(sentence)
        subject = subjects[0].strip() if subjects else "*"
        needs.append(
            InformationNeed(
                need_id=make_id("need"),
                subject=subject,
                attribute=attribute,
                exclusions=list(exclusions),
            )
        )
    if not needs:
        needs.append(
            InformationNeed(
                need_id=make_id("need"),
                subject="*",
                attribute="fact",
                exclusions=list(exclusions),
            )
        )
    constraints = [
        s.strip()
        for s in _SENTENCE_SPLIT.split(prompt)
        if re.search(
            r"\bmust\b|\brequired\b|\bevery\b|\bnever\b|\bexclud", s, re.IGNORECASE
        )
    ]
    return InvestigationGoal(
        raw_prompt=prompt,
        needs=needs,
        output_schema={"required": [], "types": {}},
        constraints=constraints,
    )


def _parse_exclusions(prompt: str) -> List[str]:
    exclusions: List[str] = []
    for match in _EXCLUSION_CLAUSE.finditer(prompt):
        clause = match.group(1)
        clause = re.split(r"\bfrom\b|\bwith\b", clause, maxsplit=1)[0]
        for part in re.split(r",|\band\b", clause):
            phrase = part.strip(" .").lower()
            phrase = re.sub(r"^(the|all|any)\s+", "", phrase)
            if phrase and phrase not in exclusions:
                exclusions.append(phrase)
    return exclusions


def _parse_number(raw: str) -> Any:
    cleaned = raw.replace(",", "")
    try:
        as_int = int(cleaned)
        return as_int
    except ValueError:
        return float(cleaned)


def _excluded(text: str, exclusions: List[str]) -> bool:
    lowered = text.lower()
    for exc in exclusions:
        if not exc:
            continue
        if exc in lowered:
            return True
        # Word-level: "spare quantities" also excludes a row labeled "Spare".
        # Exact words only (no stemming): header "Total" must NOT match "totals".
        words = [w for w in re.split(r"[^a-z0-9]+", exc) if len(w) > 3]
        if any(re.search(r"\b" + re.escape(w) + r"\b", lowered) for w in words):
            return True
    return False


def _header_excluded(cell: str, exclusions: List[str]) -> bool:
    """Column-level exclusion with plural tolerance ("Total" vs "totals").

    Applies ONLY to header cells choosing columns — never to whole lines, so a
    "Total" column is dropped while base-count columns in the same row survive.
    """
    lowered = str(cell or "").strip().lower()
    if not lowered:
        return False
    for exc in exclusions:
        for word in re.split(r"[^a-z0-9]+", exc):
            word = word.strip().lower()
            if len(word) <= 3:
                continue
            if word == lowered or word.rstrip("s") == lowered or word == lowered + "s":
                return True
    return False


def is_excluded(text: str, exclusions: List[str]) -> bool:
    """Public exclusion check shared by heuristic and model mining paths."""
    return _excluded(text or "", exclusions or [])


def extract_candidates(
    items: List[Dict[str, Any]], needs: List[InformationNeed], exclusions: List[str]
) -> Tuple[List[CandidateClaim], Dict[str, int]]:
    """Mine generic subject/attribute/value triples from observation snippets."""
    candidates: List[CandidateClaim] = []
    stats = {"items": len(items), "lines": 0, "skipped_excluded": 0, "candidates": 0}
    need_by_attr = {n.attribute.strip().lower(): n for n in needs}
    for item in items:
        snippet = str(item.get("snippet", ""))
        prov = dict(item.get("provenance", {}))
        for line in snippet.splitlines():
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            if _excluded(line, exclusions):
                stats["skipped_excluded"] += 1
                continue
            for cand in _mine_line(line, prov, need_by_attr, exclusions):
                candidates.append(cand)
                stats["candidates"] += 1
    return candidates, stats


def _need_for(
    attribute: str, need_by_attr: Dict[str, InformationNeed]
) -> InformationNeed | None:
    key = attribute.strip().lower()
    if key in need_by_attr:
        return need_by_attr[key]
    for generic in GENERIC_ATTRIBUTES:
        if generic in need_by_attr:
            return need_by_attr[generic]
    return None


def _emit(
    subject: str,
    attribute: str,
    value: Any,
    units: str | None,
    line: str,
    prov: Dict[str, Any],
    need_by_attr: Dict[str, InformationNeed],
    out: List[CandidateClaim],
    qualifiers: Dict[str, Any] | None = None,
) -> None:
    need = _need_for(attribute, need_by_attr)
    if need is None:
        return
    if need.subject != "*" and subject.strip().lower() != need.subject.strip().lower():
        return
    out.append(
        CandidateClaim(
            subject=subject.strip(),
            attribute=attribute.strip(),
            value=value,
            units=units or None,
            snippet=line,
            provenance=dict(prov),
            need_id=need.need_id,
            qualifiers=dict(qualifiers or {}),
        )
    )


def _ocr_qualifiers(line: str, end: int) -> Dict[str, Any]:
    """Digit-letter adjacency (e.g. '124B') signals OCR substitution risk."""
    if 0 <= end < len(line) and line[end].isalpha():
        return {"ocr_suspect": True}
    return {}


def _trail_qualifiers(base: Dict[str, Any], trail: str) -> Dict[str, Any]:
    """A letter captured immediately after the number is an OCR suspect."""
    quals = dict(base)
    if trail and trail.isalpha():
        quals["ocr_suspect"] = True
    return quals


def qualify_subject(subject: str, head: str) -> str:
    """Banner/group roots join the entity key ("Boiler" -> "Plant Boiler").

    Only when the header path root is absent from the subject already; plain
    single-segment headers leave the subject untouched.
    """
    parts = [p.strip() for p in str(head or "").split(">") if p.strip()]
    if len(parts) < 2:
        return subject
    root = parts[0]
    tokens = set(re.split(r"[^a-z0-9]+", subject.lower()))
    if root.lower() in tokens:
        return subject
    return f"{root} {subject}".strip()


def _split_header_path(head: str) -> tuple[str, Dict[str, Any]]:
    """Hierarchical header segment: attribute = leaf, path preserved.

    "Plant > AI" -> ("AI", {"header_path": "Plant > AI"}); plain "AI" unchanged.
    """
    parts = [p.strip() for p in str(head or "").split(">") if p.strip()]
    if len(parts) > 1:
        return parts[-1], {"header_path": " > ".join(parts)}
    return str(head or "").strip(), {}


def _mine_line(
    line: str,
    prov: Dict[str, Any],
    need_by_attr: Dict[str, InformationNeed],
    exclusions: List[str] | None = None,
) -> List[CandidateClaim]:
    out: List[CandidateClaim] = []
    vision_fact = re.match(
        r"^\s*CLAIM:\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$", line
    )
    if vision_fact:
        # GPT-5.4 Vision structured fact emission (see tools/vision_llm.py).
        subject, attribute, raw = (
            vision_fact.group(1),
            vision_fact.group(2),
            vision_fact.group(3),
        )
        try:
            value: Any = _parse_number(raw.replace(",", ""))
        except (ValueError, AttributeError):
            value = raw.strip().strip("\"'")
        _emit(subject, attribute, value, None, line, prov, need_by_attr, out)
        return out
    colon = _SUBJECT_COLON.search(line)
    if colon:
        subject, rest = colon.group(1), colon.group(2)
        for match in _CODE_VALUE.finditer(rest):
            code, raw, pct, trail = (
                match.group(1),
                match.group(2),
                match.group(3),
                match.group(4),
            )
            _emit(
                subject,
                code,
                _parse_number(raw),
                "%" if pct else None,
                line,
                prov,
                need_by_attr,
                out,
                qualifiers=_trail_qualifiers(_ocr_qualifiers(line, match.end()), trail),
            )
        for match in _CODE_EQ_VALUE.finditer(rest):
            code, raw, pct, trail = (
                match.group(1),
                match.group(2),
                match.group(3),
                match.group(4),
            )
            _emit(
                subject,
                code,
                _parse_number(raw),
                "%" if pct else None,
                line,
                prov,
                need_by_attr,
                out,
                qualifiers=_trail_qualifiers(_ocr_qualifiers(line, match.end()), trail),
            )
        if out:
            return out
    eq = _SUBJECT_EQ.search(line)
    if eq:
        subject, code, raw, pct, trail = (
            eq.group(1),
            eq.group(2),
            eq.group(3),
            eq.group(4),
            eq.group(5),
        )
        _emit(
            subject,
            code,
            _parse_number(raw),
            "%" if pct else None,
            line,
            prov,
            need_by_attr,
            out,
            qualifiers=_trail_qualifiers(_ocr_qualifiers(line, eq.end()), trail),
        )
        if out:
            return out
    if "//" in line:
        # Header-contextualized row built by the investigator, e.g.
        # "System | AI | AO  //  Plant | 1248 | 312".
        head_part, _, val_part = line.partition("//")
        headers = [c.strip() for c in head_part.split("|")]
        values = [c.strip() for c in val_part.split("|")]
        if len(headers) == len(values) and len(headers) >= 2:
            # Entity key = leading non-numeric cells ("Plant Boiler", "Plant", ...).
            key_len = 0
            for cell in values[:-1]:
                if re.match(r"^[A-Za-z]", (cell or "").strip()) and not re.fullmatch(
                    r"-?\d[\d,]*(?:\.\d+)?%?", (cell or "").strip()
                ):
                    key_len += 1
                else:
                    break
            key_len = max(1, min(key_len, len(values) - 1))
            subject = " ".join(values[:key_len]).strip()
            if subject and re.match(r"^[A-Za-z]", subject):
                for head, cell in zip(headers[key_len:], values[key_len:]):
                    attr, quals = _split_header_path(head)
                    if not re.fullmatch(r"[A-Za-z%]{1,8}", attr or ""):
                        continue
                    if _header_excluded(head, exclusions or []):
                        continue  # excluded column (totals/spare/...), not the row
                    num = re.fullmatch(
                        r"(?P<number>-?\d[\d,]*(?:\.\d+)?)(?P<pct>%?)"
                        r"(?P<trail>[A-Za-z]?)",
                        (cell or "").strip(),
                    )
                    if num:
                        quals = _trail_qualifiers(quals, num.group("trail"))
                        _emit(
                            qualify_subject(subject, head),
                            attr.upper().rstrip("%"),
                            _parse_number(num.group("number")),
                            "%" if num.group("pct") else None,
                            line,
                            prov,
                            need_by_attr,
                            out,
                            qualifiers=quals,
                        )
    return out


def need_search_queries(need: InformationNeed) -> List[str]:
    """Free world-model queries most likely to locate a need's evidence."""
    if getattr(need, "queries", None):
        seen, out = set(), []
        for query in need.queries:
            if query not in seen:
                seen.add(query)
                out.append(query)
        return out
    attr = need.attribute.strip()
    base = [f"{need.subject} {attr}" if need.subject != "*" else attr]
    if attr.lower() in ("count", "counts", "io_count"):
        base += ["I/O", "schedule", "AI AO DI DO"]
    if attr.lower() in ("fact", "facts"):
        base += ["summary", "schedule"]
    if attr.lower() == "clause":
        base += ["termination", "clause"]
    seen, out = set(), []
    for query in base:
        if query not in seen:
            seen.add(query)
            out.append(query)
    return out


def need_addressed(need: InformationNeed, claims) -> bool:
    """Discovery needs (subject '*' + generic attribute) need only *some* claim.

    Field-driven needs (field_ref present) are addressed ONLY by COMPLETE or
    explicitly NOT_FOUND states — claims alone never satisfy them (§5).
    """
    if getattr(need, "field_ref", None) and getattr(need, "state", "") in (
        "COMPLETE",
        "NOT_FOUND",
    ):
        return True
    if need.attribute.strip().lower() in GENERIC_ATTRIBUTES and need.subject == "*":
        return len(claims) > 0
    return any(
        (
            need.subject == "*"
            or c.subject.strip().lower() == need.subject.strip().lower()
        )
        and c.attribute.strip().lower() == need.attribute.strip().lower()
        for c in claims
    )

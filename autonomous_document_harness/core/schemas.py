"""Core typed schemas. Stdlib dataclasses only — no domain logic, no IO."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SourceType:
    TEXT = "text"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    IMAGE = "image"
    FIGURE = "figure"
    SHEET_CELL = "sheet_cell"
    HEADING = "heading"
    PAGE = "page"
    REGION = "region"
    ALL = {TEXT, TABLE, TABLE_CELL, IMAGE, FIGURE, SHEET_CELL, HEADING, PAGE, REGION}


class ExtractionMethod:
    NATIVE_TEXT = "native_text"
    NATIVE_TABLE = "native_table"
    SPREADSHEET = "spreadsheet"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    OCR = "ocr"
    VISION = "vision"
    AZURE_DI = "azure_di"
    GPT54_VISION = "gpt54_vision"
    HEURISTIC = "heuristic"
    MANUAL = "manual"


class VerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class TaskStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    LOCATED_IN = "located_in"
    CORROBORATES = "corroborates"
    CONFLICTS_WITH = "conflicts_with"


BBox = Tuple[float, float, float, float]


def validate_bbox(value: Any) -> BBox:
    """Validate (x0, y0, x1, y1) with strictly positive area."""
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"bbox must be a 4-tuple (x0, y0, x1, y1), got {value!r}")
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"bbox coordinates must be numeric, got {value!r}")
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"bbox must have positive area, got {value!r}")
    return (x0, y0, x1, y1)


_counters: Dict[str, int] = {}


def reset_id_counters() -> None:
    _counters.clear()


def make_id(prefix: str) -> str:
    n = _counters.get(prefix, 0) + 1
    _counters[prefix] = n
    return f"{prefix}_{n:04d}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceRegion:
    document_id: str
    page: Optional[int]
    bbox: Optional[BBox]
    source_type: str
    extraction_method: str
    snippet: str = ""
    cell: Optional[str] = None  # e.g. "B4" or "B4:D9" for spreadsheets

    def __post_init__(self) -> None:
        if self.source_type not in SourceType.ALL:
            raise ValueError(f"unknown source_type {self.source_type!r}")
        if self.bbox is not None:
            self.bbox = validate_bbox(self.bbox)


@dataclass
class Evidence:
    evidence_id: str
    region: SourceRegion
    snippet: str
    method: str
    confidence: float
    produced_by: str = "investigator"
    created_at: str = field(default_factory=utcnow_iso)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence!r}")


@dataclass
class Claim:
    claim_id: str
    subject: str
    attribute: str
    value: Any
    evidence_ids: List[str] = field(default_factory=list)
    verification: VerificationState = VerificationState.UNVERIFIED
    units: Optional[str] = None
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    unresolved_reason: str = ""
    derived_from: List[str] = field(default_factory=list)
    # Claim tiers (§4): observations are mined facts; answer claims are
    # linked to a requested field (field_ref: {sheet, need_id}).
    tier: str = "observation"
    field_ref: Dict[str, Any] = field(default_factory=dict)

    def is_verified(self) -> bool:
        return self.verification == VerificationState.VERIFIED

    def is_unresolved(self) -> bool:
        return self.verification == VerificationState.UNRESOLVED


@dataclass
class InformationNeed:
    need_id: str
    subject: str  # entity of interest, "*" = discover from corpus
    attribute: str
    filters: Dict[str, Any] = field(default_factory=dict)
    exclusions: List[str] = field(default_factory=list)
    required: bool = True
    # Field-driven needs (sanitized request schema, §4-§5). All optional.
    state: str = "UNSEEN"  # UNSEEN|DISCOVERING|PARTIAL|VERIFIED_PARTIAL|COMPLETE|NOT_FOUND|BLOCKED
    field_ref: Dict[str, Any] = field(default_factory=dict)
    queries: List[str] = field(default_factory=list)
    saturation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationGoal:
    raw_prompt: str
    needs: List[InformationNeed] = field(default_factory=list)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    spec: Optional[Dict[str, Any]] = None  # ObjectiveSpec.to_dict() when compiled


@dataclass
class RecoveryTask:
    task_id: str
    claim_ids: List[str]
    reason: str
    suggested_actions: List[str] = field(default_factory=list)
    status: str = TaskStatus.OPEN.value
    attempts: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    cost: float = 0.0


@dataclass
class GateResult:
    gate: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    recovery: List[RecoveryTask] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    seq: int
    kind: str  # plan | tool | evidence | claim | gate | recovery | decision | done
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    t: float = 0.0


@dataclass
class BudgetLimits:
    max_iterations: int = 24
    max_tool_calls: int = 200
    max_model_calls: int = 60
    max_visual_calls: int = 40
    token_budget: int = 200000
    timeout_seconds: float = 1800.0


class BudgetExhausted(Exception):
    pass


_BUDGET_KINDS = ("iteration", "tool", "model", "visual", "token")


@dataclass
class BudgetUsage:
    limits: BudgetLimits
    used: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in _BUDGET_KINDS})

    def _cap(self, kind: str) -> int:
        return {
            "iteration": self.limits.max_iterations,
            "tool": self.limits.max_tool_calls,
            "model": self.limits.max_model_calls,
            "visual": self.limits.max_visual_calls,
            "token": self.limits.token_budget,
        }[kind]

    def consume(self, kind: str, amount: int = 1) -> bool:
        """Record usage. Returns False (never raises) when the cap is hit."""
        if kind not in self.used:
            raise ValueError(f"unknown budget kind {kind!r}")
        if self.used[kind] + amount > self._cap(kind):
            return False
        self.used[kind] += amount
        return True

    def exhausted(self) -> List[str]:
        return [k for k in self.used if self.used[k] >= self._cap(k)]

    def to_dict(self) -> Dict[str, Any]:
        return {"used": dict(self.used), "exhausted": self.exhausted()}

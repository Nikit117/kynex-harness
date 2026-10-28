"""Model-layer schemas (provider-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelRequest:
    kind: str  # parse_goal | extract | entail | vision | chat
    prompt: str = ""
    max_tokens: int = 2000
    images: List[str] = field(default_factory=list)  # data-URLs or https URLs
    response_format: str = ""  # "" | "json_object"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    backend: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


@dataclass
class CandidateClaim:
    subject: str
    attribute: str
    value: Any
    units: Optional[str]
    snippet: str
    provenance: Dict[str, Any]
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    need_id: str = ""


class ModelError(Exception):
    pass


class ModelUnavailable(ModelError):
    pass

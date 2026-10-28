"""Typed tool registry. Tools return Observations — never raw prose, never raises."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.schemas import BudgetUsage


class ToolError(Exception):
    """Expected tool failure (bad args, missing page, unsupported format)."""


class ToolUnavailable(ToolError):
    """Backend capability missing (no OCR engine, no vision model)."""


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, str] = field(default_factory=dict)
    cost: float = 1.0
    needs_visual: bool = False
    needs_model: bool = False


@dataclass
class Observation:
    tool: str
    args: Dict[str, Any]
    status: str  # ok | unavailable | error
    data: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    cost: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "data": self.data,
            "provenance": self.provenance,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "message": self.message,
        }


def provenance(
    doc_id: str,
    page: Optional[int] = None,
    cell: Optional[str] = None,
    bbox: Optional[tuple] = None,
    source_type: str = "text",
    method: str = "native_text",
    snippet: str = "",
    image_hash: Optional[str] = None,
    dpi: Optional[int] = None,
) -> Dict[str, Any]:
    record = {
        "document_id": doc_id,
        "page": page,
        "cell": cell,
        "bbox": list(bbox) if bbox else None,
        "source_type": source_type,
        "method": method,
        "snippet": snippet[:300],
    }
    if image_hash:
        record["image_hash"] = image_hash
    if dpi:
        record["dpi"] = dpi
    return record


class ToolRegistry:
    def __init__(self, budgets: Optional[BudgetUsage] = None) -> None:
        self.budgets = budgets
        self._specs: Dict[str, ToolSpec] = {}
        self._funcs: Dict[str, Callable[..., Any]] = {}

    def register(self, spec: ToolSpec, func: Callable[..., Any]) -> None:
        self._specs[spec.name] = spec
        self._funcs[spec.name] = func

    def specs(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def call(self, name: str, **kwargs: Any) -> Observation:
        started = time.perf_counter()
        if name not in self._funcs:
            return Observation(
                tool=name,
                args=dict(kwargs),
                status="error",
                message=f"unknown tool '{name}'",
            )
        spec = self._specs[name]
        if self.budgets is not None:
            if not self.budgets.consume("tool"):
                return Observation(
                    tool=name,
                    args=dict(kwargs),
                    status="error",
                    message="tool budget exhausted; stopping gracefully",
                )
            if spec.needs_visual and not self.budgets.consume("visual"):
                return Observation(
                    tool=name,
                    args=dict(kwargs),
                    status="error",
                    message="visual budget exhausted; stopping gracefully",
                )
            if spec.needs_model and not self.budgets.consume("model"):
                return Observation(
                    tool=name,
                    args=dict(kwargs),
                    status="error",
                    message="model budget exhausted; stopping gracefully",
                )
        try:
            data, prov = self._funcs[name](**kwargs)
            return Observation(
                tool=name,
                args=dict(kwargs),
                status="ok",
                data=data or {},
                provenance=prov or [],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cost=spec.cost,
            )
        except ToolUnavailable as exc:
            return Observation(
                tool=name,
                args=dict(kwargs),
                status="unavailable",
                message=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except ToolError as exc:
            return Observation(
                tool=name,
                args=dict(kwargs),
                status="error",
                message=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # tools must never take down the investigator
            return Observation(
                tool=name,
                args=dict(kwargs),
                status="error",
                message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    @staticmethod
    def unavailable(name: str, kwargs: Dict[str, Any], reason: str) -> Observation:
        return Observation(
            tool=name, args=dict(kwargs), status="unavailable", message=reason
        )

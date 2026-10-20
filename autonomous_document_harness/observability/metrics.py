"""Run metrics: calls, latency, tokens and cost estimates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

TOKEN_PER_CHAR = 0.25
COST_PER_1K_TOKENS_USD = 0.002


@dataclass
class RunMetrics:
    tool_calls: int = 0
    model_calls: int = 0
    visual_calls: int = 0
    pages_inspected: int = 0
    claims_created: int = 0
    contradictions_found: int = 0
    recovery_attempts: int = 0
    di_calls: int = 0
    total_latency_ms: float = 0.0
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def record_tool(
        self,
        tool: str,
        latency_ms: float,
        data_chars: int = 0,
        needs_model: bool = False,
        needs_visual: bool = False,
    ) -> None:
        self.tool_calls += 1
        self.total_latency_ms += latency_ms
        if needs_model:
            self.model_calls += 1
        if needs_visual:
            self.visual_calls += 1
        if tool == "di_analyze_document":
            self.di_calls += 1
        tokens = int(data_chars * TOKEN_PER_CHAR)
        self.estimated_tokens += tokens
        self.estimated_cost_usd += tokens / 1000.0 * COST_PER_1K_TOKENS_USD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "model_calls": self.model_calls,
            "visual_calls": self.visual_calls,
            "di_calls": self.di_calls,
            "pages_inspected": self.pages_inspected,
            "claims_created": self.claims_created,
            "contradictions_found": self.contradictions_found,
            "recovery_attempts": self.recovery_attempts,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "estimated_tokens": self.estimated_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }

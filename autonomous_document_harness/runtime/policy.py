"""Information-gain policy: score candidates, downrank repeats, log everything.

utility(action) = (gain * relevance * p_resolve) / (cost + latency + redundancy)
Heuristic today, learned tomorrow — every decision is logged for evaluation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..core.config import RuntimeConfig
from .planner import Action

_LATENCY_EST = {
    "search_text": 0.3,
    "semantic_search": 0.4,
    "hybrid_search": 0.4,
    "render_page": 1.0,
    "render_region": 1.2,
    "run_ocr": 1.5,
    "inspect_page_visually": 2.0,
    "calculate": 0.1,
}


@dataclass
class Decision:
    action_id: str
    tool: str
    score: float
    components: Dict[str, float]
    selected: bool


@dataclass
class PolicyHistory:
    seen: Counter = field(default_factory=Counter)  # (tool, target) -> runs
    failures: Counter = field(default_factory=Counter)  # (tool, target) -> fails
    successes: Counter = field(default_factory=Counter)

    def note(self, action: Action, status: str) -> None:
        key = (action.tool, action.target)
        self.seen[key] += 1
        if status == "ok":
            self.successes[key] += 1
        elif status == "error":
            self.failures[key] += 1


class ActionPolicy:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def score(
        self, action: Action, history: PolicyHistory
    ) -> Tuple[float, Dict[str, float]]:
        key = (action.tool, action.target)
        repeats = history.seen.get(key, 0)
        fails = history.failures.get(key, 0)
        gain = action.prior_gain + (0.2 if action.recovery else 0.0)
        relevance = 1.0 if action.need_ids or action.recovery else 0.6
        p_resolve = (1.0 - self.config.failure_penalty) ** fails
        cost = action.cost
        if action.tool in (
            "inspect_page_visually",
            "inspect_region_visually",
            "inspect_table_visually",
        ):
            cost *= self.config.visual_cost_weight
        redundancy = self.config.redundancy_penalty * repeats
        if repeats >= 1 and action.tool in (
            "search_text",
            "semantic_search",
            "hybrid_search",
            "list_documents",
        ):
            redundancy += 2.0  # identical searches are almost never useful twice
        latency = _LATENCY_EST.get(action.tool, 0.5)
        total = (gain * relevance * p_resolve) / max(cost + latency + redundancy, 1e-6)
        components = {
            "gain": round(gain, 3),
            "relevance": relevance,
            "p_resolve": round(p_resolve, 3),
            "cost": round(cost, 3),
            "latency": latency,
            "redundancy": round(redundancy, 3),
            "repeats": repeats,
            "failures": fails,
        }
        return round(total, 4), components

    def select(
        self, candidates: List[Action], history: PolicyHistory, max_actions: int
    ) -> Tuple[List[Action], List[Decision]]:
        scored = []
        for action in candidates:
            key = (action.tool, action.target)
            # Proven-barren repeats are excluded (logged, not silent): an action
            # run twice with no success never resolves anything a third time.
            if history.seen.get(key, 0) >= 2 and history.successes.get(key, 0) == 0:
                scored.append(
                    (action, 0.0, {"excluded": 1.0, "repeats": history.seen[key]})
                )
                continue
            scored.append((action, *self.score(action, history)))
        scored.sort(key=lambda triple: triple[1], reverse=True)
        selected = [action for action, total, _ in scored[:max_actions] if total > 0.01]
        chosen = {a.action_id for a in selected}
        decisions = [
            Decision(
                action_id=a.action_id,
                tool=a.tool,
                score=total,
                components=comp,
                selected=a.action_id in chosen,
            )
            for a, total, comp in scored
        ]
        return selected, decisions

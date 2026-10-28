"""Working memory: current hypotheses, recent observations, active pages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List


@dataclass
class WorkingMemory:
    hypotheses: List[str] = field(default_factory=list)
    recent_observations: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    active_pages: Dict[str, List[int]] = field(default_factory=dict)
    reasoning_state: str = "starting"

    def add_observation(self, tool: str, status: str, summary: str) -> None:
        self.recent_observations.append(
            {"tool": tool, "status": status, "summary": summary}
        )

    def set_hypothesis(self, hypothesis: str) -> None:
        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)

    def focus_page(self, doc_id: str, page: int) -> None:
        pages = self.active_pages.setdefault(doc_id, [])
        if page not in pages:
            pages.append(page)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "hypotheses": list(self.hypotheses),
            "recent": list(self.recent_observations),
            "active_pages": {k: list(v) for k, v in self.active_pages.items()},
            "reasoning_state": self.reasoning_state,
        }

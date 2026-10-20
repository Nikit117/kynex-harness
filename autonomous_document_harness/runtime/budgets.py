"""Hard budgets with honest exhaustion (never fabricated completion)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from ..core.schemas import BudgetLimits, BudgetUsage


@dataclass
class RunBudgets:
    usage: BudgetUsage = field(
        default_factory=lambda: BudgetUsage(limits=BudgetLimits())
    )
    deadline_mono: float = field(default_factory=lambda: time.monotonic() + 1800.0)

    @classmethod
    def with_limits(cls, limits: BudgetLimits) -> "RunBudgets":
        return cls(
            usage=BudgetUsage(limits=limits),
            deadline_mono=time.monotonic() + limits.timeout_seconds,
        )

    def time_left(self) -> float:
        return max(0.0, self.deadline_mono - time.monotonic())

    def exhausted_reasons(self) -> List[str]:
        reasons = [f"{kind} budget exhausted" for kind in self.usage.exhausted()]
        if self.time_left() <= 0:
            reasons.append("timeout exhausted")
        return reasons

    def ok(self) -> bool:
        return not self.exhausted_reasons()

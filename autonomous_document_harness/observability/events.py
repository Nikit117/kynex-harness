"""Trace event kinds (stable vocabulary for machine-readable traces)."""

PLAN = "plan"
TOOL = "tool"
EVIDENCE = "evidence"
CLAIM = "claim"
GATE = "gate"
RECOVERY = "recovery"
DECISION = "decision"
MEMORY = "memory"
SPECIALIST = "specialist"
DONE = "done"

ALL_KINDS = frozenset(
    {PLAN, TOOL, EVIDENCE, CLAIM, GATE, RECOVERY, DECISION, MEMORY, SPECIALIST, DONE}
)

"""RunTrace: ordered machine-readable execution record + chain summary."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..core.schemas import TraceEvent
from . import events


@dataclass
class RunTrace:
    run_id: str
    objective: str = ""
    started: float = field(default_factory=time.time)
    events: List[TraceEvent] = field(default_factory=list)

    def append(
        self, kind: str, summary: str, data: Dict[str, Any] | None = None
    ) -> TraceEvent:
        event = TraceEvent(
            seq=len(self.events) + 1,
            kind=kind,
            summary=summary,
            data=data or {},
            t=time.time() - self.started,
        )
        self.events.append(event)
        return event

    def chain(self) -> str:
        """One-line investigation story: goal -> actions -> gates -> outcome."""
        parts = [f"goal({self.objective[:60]})"]
        for event in self.events:
            if event.kind == events.TOOL:
                parts.append(
                    f"{event.data.get('tool', 'tool')}({event.data.get('status', '?')})"
                )
            elif event.kind == events.GATE:
                parts.append(
                    f"{event.data.get('gate', 'gate')}:"
                    f"{'PASS' if event.data.get('passed') else 'FAIL'}"
                )
            elif event.kind in (events.CLAIM, events.RECOVERY, events.DECISION):
                parts.append(f"{event.kind}({event.summary[:48]})")
        if self.events and self.events[-1].kind == events.DONE:
            parts.append(f"done({self.events[-1].summary[:48]})")
        return " -> ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "events": [
                {
                    "seq": e.seq,
                    "kind": e.kind,
                    "summary": e.summary,
                    "data": e.data,
                    "t": round(e.t, 3),
                }
                for e in self.events
            ],
        }

    def save_jsonl(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(
                    json.dumps(
                        {
                            "run_id": self.run_id,
                            "seq": event.seq,
                            "kind": event.kind,
                            "summary": event.summary,
                            "data": event.data,
                            "t": round(event.t, 3),
                        }
                    )
                    + "\n"
                )
        return path

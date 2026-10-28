"""Normalized investigation events (UI-independent).

The TUI (and any future web/API surface) subscribes to HarnessEvents instead
of coupling to LangGraph internals. Producers: tool hooks in the deep runtime
plus snapshot-diff state transitions computed here.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

MAX_QUEUE = 5000


@dataclass
class HarnessEvent:
    kind: str
    title: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    """Thread-safe fan-out bus with bounded drain."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[HarnessEvent]" = queue.Queue()
        self._subscribers: List[Callable[[HarnessEvent], None]] = []
        self.dropped = 0

    def post(
        self,
        kind: str,
        title: str,
        detail: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._queue.qsize() >= MAX_QUEUE:
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        event = HarnessEvent(
            kind=kind, title=title, detail=detail, data=dict(data or {})
        )
        self._queue.put(event)
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[HarnessEvent], None]) -> None:
        self._subscribers.append(callback)

    def drain(self, limit: int = 200) -> List[HarnessEvent]:
        events = []
        while len(events) < limit:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events


# Canonical event kinds (stable contract for future UIs).
RUN_STARTED = "run_started"
OBJECTIVE_COMPILED = "objective_compiled"
CONTRACT_CREATED = "contract_created"
SEARCH_STARTED = "search_started"
SEARCH_COMPLETED = "search_completed"
CANDIDATE_REGION = "candidate_region"
PAGE_READ = "page_read"
TABLE_DETECTED = "table_detected"
PERCEPTION_ESCALATED = "perception_escalated"
DI_CALLED = "di_called"
VISION_CALLED = "vision_called"
ANSWER_CREATED = "answer_created"
ANSWER_REJECTED = "answer_rejected"
ANSWER_VERIFIED = "answer_verified"
CONTRADICTION = "contradiction"
RECOVERY_STARTED = "recovery_started"
RECOVERY_DONE = "recovery_done"
CONTRACT_COMPLETED = "contract_completed"
COVERAGE_UPDATED = "coverage_updated"
RUN_COMPLETED = "run_completed"
RUN_FAILED = "run_failed"
ANSWER_STREAM_STARTED = "answer_stream_started"
ANSWER_STREAM_DELTA = "answer_stream_delta"
ANSWER_STREAM_COMPLETED = "answer_stream_completed"
ANSWER_STREAM_CANCELLED = "answer_stream_cancelled"


class StateWatcher:
    """Derives transition events from RunStore snapshots (poll-friendly)."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._last_states: Dict[str, str] = {}
        self._last_answers = 0

    def observe(self, store) -> None:
        try:
            summary = store.contract_summary()
        except Exception:
            return
        for entry in summary:
            previous = self._last_states.get(entry["contract_id"])
            current = entry["state"]
            self._last_states[entry["contract_id"]] = current
            if (
                previous is not None
                and previous != current
                and current
                in (
                    "COMPLETE",
                    "NOT_FOUND",
                    "BLOCKED",
                )
            ):
                self.bus.post(
                    CONTRACT_COMPLETED,
                    f"Contract {current.lower()}: {entry['sheet']} / {entry['field']}",
                    f"{entry['answers']} answers",
                    {"contract_id": entry["contract_id"], "state": current},
                )
        try:
            total = len(store.answers)
        except Exception:
            return
        if total != self._last_answers:
            self._last_answers = total
            self.bus.post(
                COVERAGE_UPDATED,
                f"{total} candidate answers",
                "",
                {"answers": total},
            )


def tool_event_to_kind(tool_name: str, phase: str, payload: Dict[str, Any]) -> tuple:
    """Map a registry tool event onto (canonical kind, title, detail)."""
    name = (tool_name or "").lower()
    status = str(payload.get("status", ""))
    verdict = str(payload.get("verdict", ""))
    summary = str(payload.get("summary", "") or payload.get("reason", ""))
    if phase == "verdict":
        if verdict == "ANSWER":
            subject = str(payload.get("subject", ""))
            value = str(payload.get("value", ""))
            return (ANSWER_CREATED, f"Candidate: {subject} -> {value}", summary)
        return (ANSWER_REJECTED, f"Rejected ({verdict})", summary)
    if phase != "finished":
        if "search" in name or "evidence" in name:
            return (SEARCH_STARTED, f"Searching: {name}", summary)
        return ("tool_started", f"{tool_name}", summary)
    if status != "ok":
        return (
            "tool_unavailable" if status == "unavailable" else "tool_error",
            f"{tool_name}: {status}",
            str(payload.get("error", summary)),
        )
    if "search" in name or "supporting_evidence" in name or "conflicting" in name:
        return (SEARCH_COMPLETED, f"Search: {name}", summary)
    if name in ("read_page", "read_section", "read_surrounding_context", "read_text"):
        return (PAGE_READ, f"Reading ({name})", summary)
    if name in ("detect_tables", "detect_layout"):
        return (TABLE_DETECTED, f"Tables ({name})", summary)
    if name == "di_analyze_document" or name == "run_azure_document_intelligence":
        return (DI_CALLED, "Azure Document Intelligence", summary)
    if "visual" in name or "vision" in name or name in ("render_page", "render_region"):
        return (VISION_CALLED, f"Vision ({name})", summary)
    if name == "verify_answer":
        return (ANSWER_VERIFIED, "Verification", summary)
    if name == "request_targeted_recovery":
        return (RECOVERY_STARTED, "Targeted recovery", summary)
    return ("tool_finished", f"{tool_name}", summary)

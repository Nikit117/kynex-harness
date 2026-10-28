"""Animated activity indicator with dynamic status verbs."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widget import Widget
from textual.app import RenderResult
from textual.timer import Timer
from rich.text import Text


_SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]

_EVENT_VERBS: dict[str, str] = {
    "run_started": "Understanding request",
    "objective_compiled": "Compiling objective",
    "search_started": "Searching workspace",
    "search_completed": "Search complete",
    "page_read": "Reading page",
    "table_detected": "Inspecting table",
    "vision_called": "Checking with Vision",
    "perception_escalated": "Rendering scanned page",
    "di_called": "Running Document Intelligence",
    "answer_created": "Recording finding",
    "answer_verified": "Verifying evidence",
    "recovery_started": "Recovering missing records",
    "recovery_done": "Recovery complete",
    "contradiction": "Resolving contradiction",
    "run_completed": "Complete",
    "run_failed": "Failed",
    "coverage_updated": "Checking completeness",
    "contract_completed": "Contract resolved",
    "contract_created": "Building contract",
    # Investigative QA events
    "classify_intent": "Classifying question",
    "toc_built": "Building Table of Contents",
    "page_relevance": "Scoring page relevance",
    "reading_pages": "Reading document pages",
    "tool_call": "",
    "generating_answer": "Generating answer",
    # V3-specific
    "WorkUnitsCompiled": "Compiling work units",
    "ContextSelected": "Planning context",
    "BatchExtracted": "Extracting records",
    "NormalizationCompleted": "Normalizing values",
    "LedgerUpdated": "Updating ledger",
    "FailureClassified": "Classifying failures",
    "VerificationStarted": "Verifying evidence",
    "VerificationCompleted": "Verification complete",
    "RecordsDiscovered": "Discovering records",
}


class ActivityIndicator(Widget):
    """Animated spinner with dynamic status text. Presentation-only."""

    DEFAULT_CSS = """
    ActivityIndicator {
        height: auto;
        min-height: 1;
        max-height: 6;
        padding: 0 1;
    }
    """

    active: reactive[bool] = reactive(False)
    verb: reactive[str] = reactive("")
    detail: reactive[str] = reactive("")
    _frame: int = 0
    _timer: Timer | None = None
    _steps: list[str] = []

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.15, self._tick, pause=True)

    def _tick(self) -> None:
        if self.active:
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            self.refresh()

    def render(self) -> RenderResult:
        if not self.active and not self._steps:
            return Text("")

        t = Text()
        if self.active:
            spinner = _SPINNER_FRAMES[self._frame]
            t.append(f" {spinner} ", style="bold cyan")
            t.append(self.verb or "Working", style="bold")
            if self.detail:
                t.append(f"\n   {self.detail}", style="dim")

        for step in self._steps[-4:]:
            t.append("\n   ├─ ", style="dim cyan")
            t.append(step, style="dim")

        return t

    def start(self, verb: str = "Working", detail: str = "") -> None:
        self.active = True
        self.verb = verb
        self.detail = detail
        self._steps = []
        self._frame = 0
        if self._timer:
            self._timer.resume()

    def update_verb(self, verb: str, detail: str = "") -> None:
        self.verb = verb
        if detail:
            self.detail = detail
        self.refresh()

    def add_step(self, text: str) -> None:
        self._steps.append(text)
        if len(self._steps) > 20:
            self._steps = self._steps[-20:]
        self.refresh()

    def stop(self, summary: str = "") -> None:
        self.active = False
        if self._timer:
            self._timer.pause()
        if summary:
            self._steps.append(summary)
        self.refresh()

    _TOOL_VERBS: dict[str, str] = {
        "search_document": "Searching documents",
        "search_text": "Searching text",
        "search_headings": "Searching headings",
        "read_page": "Reading page",
        "read_section": "Reading section",
        "read_nearby_pages": "Reading nearby pages",
        "read_surrounding_context": "Reading context",
        "get_table_of_contents": "Reading table of contents",
        "find_tables": "Detecting tables",
        "extract_table": "Extracting table",
        "reconstruct_table": "Reconstructing table",
        "analyze_with_document_intelligence": "Running Document Intelligence",
        "inspect_page_visually": "Inspecting page (Vision)",
        "inspect_region_visually": "Inspecting region (Vision)",
        "render_page": "Rendering page",
        "submit_candidate_answer": "Recording answer",
        "submit_answer_batch": "Submitting answers",
        "verify_answer": "Verifying evidence",
        "read_spreadsheet": "Reading spreadsheet",
        "read_sheet_range": "Reading sheet range",
    }

    def on_event_kind(self, kind: str, title: str = "", detail: str = "") -> None:
        if kind == "tool_call" and title:
            tool_name = title.split("(")[0] if "(" in title else title
            verb = self._TOOL_VERBS.get(tool_name, f"Calling {tool_name}")
            self.update_verb(verb, detail)
            self.add_step(title)
            return
        verb = _EVENT_VERBS.get(kind, "")
        if verb:
            self.update_verb(verb, detail)
        if title:
            self.add_step(title)

    def clear(self) -> None:
        self.active = False
        self._steps = []
        self.verb = ""
        self.detail = ""
        if self._timer:
            self._timer.pause()
        self.refresh()

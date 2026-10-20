"""Top status bar: engine, docs, pages, cost, session name."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widget import Widget
from textual.app import RenderResult
from rich.text import Text


class StatusBar(Widget):
    """Compact top status bar showing engine state at a glance."""

    DEFAULT_CSS = """
    StatusBar {
        dock: top;
        height: 1;
        background: $primary-background;
        color: $text-muted;
    }
    """

    engine: reactive[str] = reactive("HARNESS V3")
    doc_count: reactive[int] = reactive(0)
    page_count: reactive[int] = reactive(0)
    session_name: reactive[str] = reactive("")
    cost: reactive[str] = reactive("$0.00")
    model_name: reactive[str] = reactive("GPT-5.4")
    status_text: reactive[str] = reactive("Ready")

    def render(self) -> RenderResult:
        parts = []
        parts.append(("bold", self.engine))
        if self.doc_count:
            parts.append(("", f" {self.doc_count} docs"))
            if self.page_count:
                parts.append(("dim", f" {self.page_count}p"))
        parts.append(("dim", f" {self.model_name}"))
        parts.append(("dim", f" {self.cost}"))

        t = Text()
        t.append(" ")
        for style, content in parts:
            t.append(content, style=style)
            t.append("  ", style="dim")

        pad = self.size.width - t.cell_len
        if pad > 0:
            status_style = "green" if self.status_text == "Ready" else "yellow"
            status = Text(self.status_text, style=status_style)
            t.append_text(Text(" " * max(0, pad - len(self.status_text) - 1)))
            t.append_text(status)
        return t

    def update_stats(
        self,
        doc_count: int | None = None,
        page_count: int | None = None,
        session_name: str | None = None,
        cost: str | None = None,
        status_text: str | None = None,
    ) -> None:
        if doc_count is not None:
            self.doc_count = doc_count
        if page_count is not None:
            self.page_count = page_count
        if session_name is not None:
            self.session_name = session_name
        if cost is not None:
            self.cost = cost
        if status_text is not None:
            self.status_text = status_text

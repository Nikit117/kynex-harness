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
        color: $text;
    }
    """

    engine: reactive[str] = reactive("KYNEX")
    doc_count: reactive[int] = reactive(0)
    page_count: reactive[int] = reactive(0)
    session_name: reactive[str] = reactive("")
    cost: reactive[str] = reactive("$0.00")
    model_name: reactive[str] = reactive("GPT-5.4")
    status_text: reactive[str] = reactive("Ready")

    def render(self) -> RenderResult:
        t = Text()
        t.append(" KYNEX", style="bold cyan")

        if self.session_name:
            t.append("  ▪  ", style="dim")
            t.append(self.session_name, style="")

        t.append("  │  ", style="dim")
        t.append(self.model_name, style="dim")

        if self.doc_count:
            t.append("  │  ", style="dim")
            t.append(f"{self.doc_count} docs", style="")
            if self.page_count:
                t.append(f" ({self.page_count}p)", style="dim")

        t.append("  │  ", style="dim")
        t.append(self.cost, style="dim")

        status_style = "green" if self.status_text == "Ready" else "yellow"
        status_segment = Text()
        status_segment.append("● ", style=status_style)
        status_segment.append(self.status_text, style=status_style)

        pad = self.size.width - t.cell_len - status_segment.cell_len - 1
        if pad > 0:
            t.append(" " * pad)
        t.append_text(status_segment)

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

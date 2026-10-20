"""Scrollable conversation view: user messages, assistant responses, citations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.app import RenderResult
from textual.reactive import reactive
from textual.widgets import Static
from rich.text import Text
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.console import Group


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ACTIVITY = "activity"


class MessageBubble(Static):
    """Single conversation message rendered with role-appropriate styling."""

    DEFAULT_CSS = """
    MessageBubble {
        margin: 0 1 1 1;
        padding: 0 1;
    }
    MessageBubble.user-msg {
        background: $primary 10%;
        border-left: thick $primary;
        margin-right: 4;
    }
    MessageBubble.assistant-msg {
        margin-left: 0;
        margin-right: 0;
    }
    MessageBubble.system-msg {
        color: $text-muted;
        text-style: italic;
    }
    MessageBubble.activity-msg {
        color: $text-muted;
        margin: 0 1 0 3;
    }
    """

    def __init__(
        self,
        role: MessageRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.citations = citations or []
        self.add_class(f"{role.value}-msg")

    def render(self) -> RenderResult:
        parts = []
        if self.role == MessageRole.USER:
            t = Text()
            t.append("You\n", style="bold cyan")
            t.append(self.content)
            parts.append(t)
        elif self.role == MessageRole.ASSISTANT:
            t = Text()
            t.append("HARNESS\n", style="bold green")
            parts.append(t)
            try:
                parts.append(Markdown(self.content))
            except Exception:
                parts.append(Text(self.content))
        elif self.role == MessageRole.SYSTEM:
            parts.append(Text(self.content, style="dim italic"))
        elif self.role == MessageRole.ACTIVITY:
            parts.append(Text(self.content, style="dim"))

        if self.citations:
            t = Text()
            t.append("\n")
            for c in self.citations:
                cid = c.get("id", "?")
                doc = c.get("document", "?")
                page = c.get("page", "?")
                method = c.get("method", "")
                t.append(f"[{cid}]", style="bold")
                t.append(f" {doc} p.{page}", style="dim")
                if method:
                    t.append(f" [{method}]", style="dim italic")
                t.append("\n")
            parts.append(t)

        return Group(*parts) if len(parts) > 1 else parts[0]


class ConversationView(VerticalScroll):
    """Scrollable container for conversation messages."""

    DEFAULT_CSS = """
    ConversationView {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    """

    def add_message(
        self,
        role: MessageRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        scroll: bool = True,
    ) -> MessageBubble:
        bubble = MessageBubble(role, content, citations)
        self.mount(bubble)
        if scroll:
            self.call_after_refresh(self.scroll_end, animate=False)
        return bubble

    def add_user_message(self, content: str) -> MessageBubble:
        return self.add_message(MessageRole.USER, content)

    def add_assistant_message(
        self, content: str, citations: list[dict[str, Any]] | None = None
    ) -> MessageBubble:
        return self.add_message(MessageRole.ASSISTANT, content, citations)

    def add_system_message(self, content: str) -> MessageBubble:
        return self.add_message(MessageRole.SYSTEM, content)

    def add_activity(self, content: str) -> MessageBubble:
        return self.add_message(MessageRole.ACTIVITY, content)

    def add_collapsible_activity(
        self, summary: str, steps: list[str]
    ) -> MessageBubble:
        lines = [f"  {summary}"]
        if steps:
            for s in steps:
                lines.append(f"   {s}")
        return self.add_activity("\n".join(lines))

    def start_streaming_message(self) -> MessageBubble:
        """Create an empty assistant bubble for progressive streaming."""
        bubble = MessageBubble(MessageRole.ASSISTANT, "")
        self.mount(bubble)
        self.call_after_refresh(self.scroll_end, animate=False)
        return bubble

    def update_streaming_message(
        self, bubble: MessageBubble, content: str
    ) -> None:
        """Append delta text to a streaming bubble and re-render."""
        bubble.content = content
        bubble.refresh()
        self.call_after_refresh(self.scroll_end, animate=False)

    def finalize_streaming_message(
        self,
        bubble: MessageBubble,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        """Finalize a streaming bubble with citations."""
        if citations:
            bubble.citations = citations
            bubble.refresh()
        self.call_after_refresh(self.scroll_end, animate=False)

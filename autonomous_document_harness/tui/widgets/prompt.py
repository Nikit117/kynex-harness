"""Enhanced input widget with command history and slash-command autocomplete."""

from __future__ import annotations

from typing import Any

from textual.widgets import Input
from textual.suggester import SuggestFromList


COMMAND_LIST: list[dict[str, str]] = [
    {"cmd": "/add", "desc": "Stage files or folders"},
    {"cmd": "/remove", "desc": "Remove document from workspace"},
    {"cmd": "/docs", "desc": "List loaded documents"},
    {"cmd": "/open", "desc": "Open page or evidence"},
    {"cmd": "/status", "desc": "Engine and run status"},
    {"cmd": "/evidence", "desc": "View evidence findings"},
    {"cmd": "/trace", "desc": "View investigation trace"},
    {"cmd": "/coverage", "desc": "Resolution coverage"},
    {"cmd": "/contracts", "desc": "List contracts"},
    {"cmd": "/unresolved", "desc": "Unresolved contracts"},
    {"cmd": "/retry", "desc": "Retry unresolved"},
    {"cmd": "/session", "desc": "Session management"},
    {"cmd": "/session list", "desc": "List all sessions"},
    {"cmd": "/session rename", "desc": "Rename current session"},
    {"cmd": "/new", "desc": "Start new session"},
    {"cmd": "/save", "desc": "Save checkpoint"},
    {"cmd": "/history", "desc": "Command history"},
    {"cmd": "/model", "desc": "Model information"},
    {"cmd": "/provider", "desc": "Configure model credentials"},
    {"cmd": "/provider azure", "desc": "Set up Azure OpenAI"},
    {"cmd": "/provider openai", "desc": "Set up OpenAI-compatible"},
    {"cmd": "/provider di", "desc": "Set up Document Intelligence"},
    {"cmd": "/provider clear", "desc": "Remove saved credentials"},
    {"cmd": "/tools", "desc": "Available tools"},
    {"cmd": "/cost", "desc": "Session usage and cost"},
    {"cmd": "/health", "desc": "Service health check"},
    {"cmd": "/engine", "desc": "Engine info (V3/V2)"},
    {"cmd": "/copy", "desc": "Copy last answer to clipboard"},
    {"cmd": "/copy all", "desc": "Copy full conversation to clipboard"},
    {"cmd": "/txt", "desc": "Export full chat history to file"},
    {"cmd": "/export", "desc": "Export results"},
    {"cmd": "/clear", "desc": "Clear display"},
    {"cmd": "/help", "desc": "Show all commands"},
    {"cmd": "/quit", "desc": "Quit KYNEX"},
]


class CommandSuggester(SuggestFromList):
    """Suggests slash commands as the user types."""

    def __init__(self) -> None:
        super().__init__([c["cmd"] for c in COMMAND_LIST], case_sensitive=False)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        return await super().get_suggestion(value)


class HarnessInput(Input):
    """Input with command autocomplete and history navigation."""

    DEFAULT_CSS = """
    HarnessInput {
        dock: bottom;
        margin: 0 1;
        border: round $primary 40%;
    }
    HarnessInput:focus {
        border: round $primary;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            placeholder="Ask anything, or type / for commands",
            suggester=CommandSuggester(),
            **kwargs,
        )
        self._history: list[str] = []
        self._history_idx: int = -1
        self._draft: str = ""

    @property
    def history(self) -> list[str]:
        return self._history

    def push_history(self, line: str) -> None:
        line = line.strip()
        if line and (not self._history or self._history[-1] != line):
            self._history.append(line)
        self._history_idx = -1
        self._draft = ""

    def _on_key(self, event: Any) -> None:
        if event.key == "up" and self._history:
            if self._history_idx == -1:
                self._draft = self.value
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            self.value = self._history[self._history_idx]
            self.cursor_position = len(self.value)
            event.prevent_default()
        elif event.key == "down":
            if self._history_idx >= 0:
                self._history_idx += 1
                if self._history_idx >= len(self._history):
                    self._history_idx = -1
                    self.value = self._draft
                else:
                    self.value = self._history[self._history_idx]
                self.cursor_position = len(self.value)
                event.prevent_default()

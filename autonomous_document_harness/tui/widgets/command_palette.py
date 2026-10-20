"""Lightweight command palette for quick command access."""

from __future__ import annotations

from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option
from textual.app import ComposeResult

from .prompt import COMMAND_LIST


class CommandPalette(ModalScreen[str]):
    """Modal command palette triggered by Ctrl+P."""

    DEFAULT_CSS = """
    CommandPalette {
        align: center middle;
    }
    CommandPalette > OptionList {
        width: 60;
        max-height: 20;
        border: round $primary;
        background: $surface;
    }
    """

    BINDINGS = [("escape", "dismiss_palette", "Close")]

    def compose(self) -> ComposeResult:
        options = [
            Option(f"{c['cmd']:20s} {c['desc']}", id=c["cmd"].replace("/", "cmd_").replace(" ", "_"))
            for c in COMMAND_LIST
            if not c["cmd"].startswith("/cpq ") and not c["cmd"].startswith("/session ")
        ]
        yield OptionList(*options)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        label = str(event.option.prompt).strip()
        cmd = label.split()[0] if label else ""
        self.dismiss(cmd)

    def action_dismiss_palette(self) -> None:
        self.dismiss("")

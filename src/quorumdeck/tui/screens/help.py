"""Keyboard reference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Markdown

HELP = """\
# quorumdeck

| Key | Action |
| --- | --- |
| `enter` | Send prompt |
| `ctrl+l` | Clear all panels |
| `ctrl+s` | Save session |
| `f1` / `?` | This help |
| `ctrl+q` | Quit |

Agents, models and the orchestration pattern come from `agents.yaml`.
Run `deck config path` to see which file is in use.
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [("escape,q,question_mark,f1", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Markdown(HELP)

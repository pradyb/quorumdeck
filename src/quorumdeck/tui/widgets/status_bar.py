"""The one-line summary that makes cross-provider spend visible."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

from quorumdeck.core.costs import format_usd
from quorumdeck.core.events import Usage


class StatusBar(Static):
    state: reactive[str] = reactive("ready")
    usage: reactive[Usage] = reactive(Usage(), always_update=True)
    pattern: reactive[str] = reactive("single")
    agents: reactive[int] = reactive(1)

    def render(self) -> str:
        u = self.usage
        return (
            f"{self.state}  ·  {self.pattern} × {self.agents}  ·  "
            f"{u.total_tokens} tok  ·  {format_usd(u.cost_usd)}"
        )

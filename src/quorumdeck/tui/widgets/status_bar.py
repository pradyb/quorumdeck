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
    budget_usd: reactive[float | None] = reactive(None)

    def render(self) -> str:
        u = self.usage
        cost = format_usd(u.cost_usd)
        if self.budget_usd is not None:
            # Visible before the cap is hit, not just explained after: a hard
            # stop that shows no progress toward it is still a surprise.
            cost = f"{cost} / {format_usd(self.budget_usd)}"
        return (
            f"{self.state}  ·  {self.pattern} × {self.agents}  ·  "
            f"{u.total_tokens} tok  ·  {cost}"
        )

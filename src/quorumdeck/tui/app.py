"""The terminal application.

This module is a *view*: it turns core events into widgets and keystrokes into
core calls. It holds no knowledge of any provider, which is why adding a
backend never touches this file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input

from quorumdeck.config.loader import sessions_dir
from quorumdeck.config.schema import DeckFile
from quorumdeck.core.agent import Agent
from quorumdeck.core.events import (
    RunFailed,
    RunFinished,
    RunStarted,
    TextDelta,
)
from quorumdeck.core.orchestrator import Orchestrator
from quorumdeck.providers import Provider, default_provider
from quorumdeck.tui.screens.help import HelpScreen
from quorumdeck.tui.widgets.agent_panel import AgentPanel
from quorumdeck.tui.widgets.status_bar import StatusBar


class QuorumDeckApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "quorumdeck"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+s", "save", "Save"),
        ("f1,question_mark", "help", "Help"),
    ]

    def __init__(self, config: DeckFile, provider: Provider | None = None) -> None:
        super().__init__()
        self.config = config
        engine = provider or default_provider()
        specs = config.specs()
        self.agents = [
            Agent(spec, engine, timeout_s=config.defaults.timeout_s) for spec in specs
        ]
        self.orchestrator = Orchestrator(
            self.agents,
            pattern=config.deck.pattern,
            rounds=config.deck.rounds,
            judge_id=config.deck.judge,
        )
        self.session = self.orchestrator.new_session(max_messages=config.defaults.max_messages)
        if config.deck.title:
            self.sub_title = config.deck.title

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="deck"):
            for agent in self.agents:
                yield AgentPanel(agent.spec, id=f"panel-{agent.id}")
        yield StatusBar()
        yield Input(placeholder="Ask the deck…", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        status = self.query_one(StatusBar)
        status.pattern = str(self.config.deck.pattern)
        status.agents = len(self.agents)
        self.query_one("#prompt", Input).focus()

    def panel(self, agent_id: str) -> AgentPanel:
        return self.query_one(f"#panel-{agent_id}", AgentPanel)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""
        self.run_turn(prompt)

    @work(exclusive=True, group="turn")
    async def run_turn(self, prompt: str) -> None:
        status = self.query_one(StatusBar)
        prompt_input = self.query_one("#prompt", Input)
        prompt_input.disabled = True
        status.state = "thinking"

        for agent in self.agents:
            await self.panel(agent.id).add_user(prompt)

        try:
            async for event in self.orchestrator.run_turn(self.session, prompt):
                match event:
                    case RunStarted(agent_id=agent_id):
                        self.panel(agent_id).begin_assistant()
                    case TextDelta(agent_id=agent_id, text=text):
                        await self.panel(agent_id).append(text)
                    case RunFinished(agent_id=agent_id, usage=usage, elapsed_s=elapsed):
                        await self.panel(agent_id).end_assistant(usage, elapsed)
                        status.usage = self.session.usage
                    case RunFailed(agent_id=agent_id, error=error):
                        await self.panel(agent_id).fail(error)
                    case _:
                        pass
        except NotImplementedError as exc:
            self.notify(str(exc), severity="error", timeout=10)
        finally:
            status.state = "ready"
            prompt_input.disabled = False
            prompt_input.focus()

    async def action_clear(self) -> None:
        self.session = self.orchestrator.new_session(
            max_messages=self.config.defaults.max_messages
        )
        for agent in self.agents:
            await self.panel(agent.id).clear()
        self.query_one(StatusBar).usage = self.session.usage

    def action_save(self) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path: Path = sessions_dir() / f"{stamp}.json"
        try:
            self.session.save(path)
        except OSError as exc:
            self.notify(f"could not save: {exc}", severity="error")
            return
        self.notify(f"saved {path}")

    def action_help(self) -> None:
        self.push_screen(HelpScreen())


def run(config: DeckFile, provider: Provider | None = None) -> None:
    QuorumDeckApp(config, provider).run()

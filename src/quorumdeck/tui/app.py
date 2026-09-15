"""The terminal application.

This module is a *view*: it turns core events into widgets and keystrokes into
core calls. It holds no knowledge of any provider, which is why adding a
backend never touches this file.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input

from quorumdeck.config import loader
from quorumdeck.config.loader import sessions_dir
from quorumdeck.config.schema import DeckFile
from quorumdeck.core.agent import Agent
from quorumdeck.core.events import (
    PromptInjected,
    RunFailed,
    RunFinished,
    RunStarted,
    TextDelta,
    ToolCallFailed,
    ToolCallFinished,
    ToolCallStarted,
)
from quorumdeck.core.orchestrator import BudgetExceeded, Orchestrator
from quorumdeck.mcp_pool import ToolPool, open_tool_pool
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

    def __init__(
        self,
        config: DeckFile,
        provider: Provider | None = None,
        *,
        resume: Path | None = None,
        tool_pool: ToolPool | None = None,
    ) -> None:
        super().__init__()
        # A deck is read at a glance across several panels at once, so it
        # wants a theme with real contrast between panes rather than
        # Textual's flat default. ^p palette still switches it per session;
        # this only sets what a fresh deck opens with.
        self.theme = "tokyo-night"
        self.config = config
        engine = provider or default_provider()
        # Already-connected, not opened here: an MCP connection uses anyio
        # task groups, which must be entered and exited from the same task --
        # Textual does not guarantee on_mount and on_unmount share one, so the
        # pool's whole lifetime has to wrap App.run_async() from the outside
        # (see the module-level run() below), not live inside app methods.
        # None here (every test that builds a bare QuorumDeckApp does) means
        # no deck uses tools; that is exactly what an empty pool already says.
        pool = tool_pool or ToolPool(group=None, tools={})
        specs = config.specs()
        self.agents = [
            Agent(
                spec,
                engine,
                timeout_s=config.defaults.timeout_s,
                tools=pool.specs_for(spec.tool_allowlist),
                call_tool=pool.call,
            )
            for spec in specs
        ]
        self.orchestrator = Orchestrator(
            self.agents,
            pattern=config.deck.pattern,
            rounds=config.deck.rounds,
            judge_id=config.deck.judge,
            budget_usd=config.deck.budget_usd,
        )
        if resume is not None:
            # Raises ConfigError, same contract as a bad agents.yaml -- this
            # runs before Textual takes over the terminal, so it surfaces as
            # a plain stderr message, not a broken screen.
            self.session = loader.resume_session(
                resume,
                agent_ids=self.orchestrator.agent_ids,
                max_messages=config.defaults.max_messages,
            )
        else:
            self.session = self.orchestrator.new_session(
                max_messages=config.defaults.max_messages
            )
        self._resumed = resume is not None
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

    async def on_mount(self) -> None:
        status = self.query_one(StatusBar)
        status.pattern = str(self.config.deck.pattern)
        status.agents = len(self.agents)
        status.budget_usd = self.config.deck.budget_usd
        if self._resumed:
            for agent in self.agents:
                await self.panel(agent.id).replay(self.session.thread(agent.id))
            status.usage = self.session.usage
            self.notify(f"resumed {sum(len(t) for _, t in self.session)} prior messages")
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
        try:
            # Checked before anything is shown: run_turn() would refuse this
            # too, but only once the first event is pulled -- after a prompt
            # bubble is already sitting in every panel with no reply coming.
            self.orchestrator.check_budget(self.session)
        except BudgetExceeded as exc:
            self.notify(str(exc), severity="error", timeout=10)
            return

        status = self.query_one(StatusBar)
        prompt_input = self.query_one("#prompt", Input)
        prompt_input.disabled = True
        status.state = "thinking"

        for agent in self.agents:
            await self.panel(agent.id).add_user(prompt)

        try:
            async for event in self.orchestrator.run_turn(self.session, prompt):
                match event:
                    case PromptInjected(agent_id=agent_id, text=text):
                        # A judge shown candidate answers, a debate's author
                        # shown the critic's objection: rendered the same way
                        # as the human's own prompt, since from the agent's
                        # side it is indistinguishable from one.
                        await self.panel(agent_id).add_user(text)
                    case RunStarted(agent_id=agent_id):
                        self.panel(agent_id).begin_assistant()
                    case TextDelta(agent_id=agent_id, text=text):
                        await self.panel(agent_id).append(text)
                    case ToolCallStarted(agent_id=agent_id, name=name, arguments=arguments):
                        await self.panel(agent_id).tool_call_started(name, arguments)
                    case ToolCallFinished(agent_id=agent_id, result=result):
                        await self.panel(agent_id).tool_call_finished(result)
                    case ToolCallFailed(agent_id=agent_id, error=error):
                        await self.panel(agent_id).tool_call_failed(error)
                    case RunFinished(agent_id=agent_id, usage=usage, elapsed_s=elapsed):
                        await self.panel(agent_id).end_assistant(usage, elapsed)
                        status.usage = self.session.usage
                    case RunFailed(agent_id=agent_id, error=error):
                        await self.panel(agent_id).fail(error)
                    case _:
                        pass
        except (NotImplementedError, BudgetExceeded) as exc:
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


async def run_async(
    config: DeckFile, provider: Provider | None = None, *, resume: Path | None = None
) -> None:
    # The pool's async context manager and App.run_async() are awaited from
    # this one coroutine, so they share a task throughout -- the constraint
    # QuorumDeckApp's own docstring note above explains.
    async with open_tool_pool(config) as pool:
        await QuorumDeckApp(config, provider, resume=resume, tool_pool=pool).run_async()


def run(
    config: DeckFile, provider: Provider | None = None, *, resume: Path | None = None
) -> None:
    asyncio.run(run_async(config, provider, resume=resume))

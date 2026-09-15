"""Smoke tests for the view layer.

These drive the real app through Textual's pilot, so a broken CSS selector or a
renamed widget id fails here rather than in front of a user.
"""

from __future__ import annotations

import pytest

from quorumdeck.config.schema import DeckFile
from quorumdeck.core.events import Usage
from quorumdeck.tui.app import QuorumDeckApp
from quorumdeck.tui.widgets.agent_panel import AgentPanel
from quorumdeck.tui.widgets.status_bar import StatusBar
from tests.conftest import FakeProvider

FANOUT = {
    "version": 1,
    "deck": {"pattern": "fanout"},
    "agents": [
        {"id": "a", "name": "Alpha", "model": "fake/a"},
        {"id": "b", "name": "Beta", "model": "fake/b"},
    ],
}


@pytest.fixture
def app() -> QuorumDeckApp:
    provider = FakeProvider(["one ", "two"], usage=Usage(10, 4, 0.002))
    return QuorumDeckApp(DeckFile.from_mapping(FANOUT), provider)


async def test_a_panel_is_composed_for_every_agent(app):
    async with app.run_test() as pilot:
        panels = pilot.app.query(AgentPanel)
        assert {p.spec.id for p in panels} == {"a", "b"}
        assert pilot.app.query_one(StatusBar).agents == 2


async def test_submitting_a_prompt_streams_into_every_panel(app):
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"hello")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        for agent_id in ("a", "b"):
            assert [m.content for m in pilot.app.session.thread(agent_id)] == [
                "hello",
                "one two",
            ]
        # Two agents, 4 output tokens each.
        assert pilot.app.session.usage.output_tokens == 8
        assert pilot.app.query_one(StatusBar).usage.cost_usd == pytest.approx(0.004)


async def test_a_failed_turn_is_rendered_and_costs_nothing():
    config = DeckFile.from_mapping(FANOUT)
    app = QuorumDeckApp(config, FakeProvider(fail_with="nope"))
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert all(p.has_class("-failed") for p in pilot.app.query(AgentPanel))
        assert pilot.app.session.usage.total_tokens == 0


async def test_clear_resets_the_session(app):
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        await pilot.press("ctrl+l")
        await pilot.pause()

        assert pilot.app.session.thread("a") == []
        assert pilot.app.session.usage == Usage()


JUDGE = {
    "version": 1,
    "deck": {"pattern": "judge", "judge": "referee"},
    "agents": [
        {"id": "a", "name": "Alpha", "model": "fake/a"},
        {"id": "b", "name": "Beta", "model": "fake/b"},
        {"id": "referee", "name": "Referee", "model": "fake/r", "role": "judge"},
    ],
}


async def test_a_judge_deck_fills_every_panel_including_the_judges():
    """The judge runs a phase later than its peers; the view must not care."""
    app = QuorumDeckApp(DeckFile.from_mapping(JUDGE), FakeProvider(["ruled"]))
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert {p.spec.id for p in pilot.app.query(AgentPanel)} == {"a", "b", "referee"}
        for agent_id in ("a", "b", "referee"):
            assert pilot.app.session.thread(agent_id)[-1].content == "ruled"
        # The judge's own prompt carries the candidate answers, so its input
        # token count is not comparable with a candidate's -- but all three ran.
        assert pilot.app.session.usage.output_tokens == 15


async def test_the_prompt_and_the_footer_do_not_share_a_row(app):
    """Two widgets both `dock: bottom` overlap -- Textual docks each edge by
    the max extent of everything docked to it, not the sum, so a second
    bottom dock in the same screen as Footer swallows Footer's row instead of
    stacking above it. #prompt must stay in ordinary flow."""
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        prompt = pilot.app.query_one("#prompt")
        footer = pilot.app.query_one("Footer")

        prompt_rows = range(prompt.region.y, prompt.region.y + prompt.region.height)
        assert footer.region.y not in prompt_rows
        assert prompt.region.height == 3  # top border + content + bottom border, all present


async def test_the_deck_opens_in_tokyo_night(app):
    """Several panels are read at once; the default theme needs real contrast."""
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == "tokyo-night"
        assert pilot.app.current_theme.name == "tokyo-night"


DEBATE = {
    "version": 1,
    "deck": {"pattern": "debate", "rounds": 1},
    "agents": [
        {"id": "author", "name": "Author", "model": "fake/a"},
        {"id": "critic", "name": "Critic", "model": "fake/c", "role": "critic"},
    ],
}


class _SequencedProvider:
    """Returns a different scripted reply on each successive call, in order.

    Debate calls the same shared provider three times in a fixed sequence
    (draft, critique, revision); FakeProvider's fixed chunks cannot tell those
    calls apart.
    """

    name = "sequenced"

    def __init__(self, replies):
        self._replies = list(replies)
        self._call = 0

    async def stream(self, request):
        from quorumdeck.providers.base import Chunk, Completed

        text = self._replies[self._call]
        self._call += 1
        yield Chunk(text)
        yield Completed(usage=Usage(1, 1, 0.0), finish_reason="stop")


async def test_a_debate_deck_shows_the_critique_and_revision():
    app = QuorumDeckApp(
        DeckFile.from_mapping(DEBATE), _SequencedProvider(["draft", "critique", "revision"])
    )
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        # Two author replies -- the draft, then the revision -- land in the
        # author's own thread; the critic's single critique in its own.
        author_replies = [m.content for m in pilot.app.session.thread("author")][1::2]
        assert author_replies == ["draft", "revision"]
        assert pilot.app.session.thread("critic")[-1].content == "critique"

        # PromptInjected renders exactly like the human's own prompt: a
        # "user-turn" bubble in the panel it was shown to, not just data
        # sitting invisibly in the session.
        author_panel = pilot.app.query_one("#panel-author", AgentPanel)
        critic_panel = pilot.app.query_one("#panel-critic", AgentPanel)
        author_bubbles = list(author_panel.query(".user-turn"))
        critic_bubbles = list(critic_panel.query(".user-turn"))

        # One bubble for the human's own prompt, one for the injected round text.
        assert len(author_bubbles) == 2
        assert len(critic_bubbles) == 2
        assert "Critique" in str(author_bubbles[-1].content)
        assert "Candidate answer" in str(critic_bubbles[-1].content)


async def test_resuming_replays_history_into_every_panel(tmp_path):
    from quorumdeck.core.session import Session

    session = Session(["a", "b"])
    session.add_user("earlier question")
    session.add_assistant("a", "a's earlier answer")
    session.add_assistant("b", "b's earlier answer")
    saved = session.save(tmp_path / "s.json")

    app = QuorumDeckApp(DeckFile.from_mapping(FANOUT), FakeProvider(["ignored"]), resume=saved)
    async with app.run_test() as pilot:
        await pilot.pause()

        assert [m.content for m in pilot.app.session.thread("a")] == [
            "earlier question",
            "a's earlier answer",
        ]

        a_panel = pilot.app.query_one("#panel-a", AgentPanel)
        b_panel = pilot.app.query_one("#panel-b", AgentPanel)
        # One user bubble (the replayed question) and one reply per panel.
        assert len(list(a_panel.query(".user-turn"))) == 1
        assert len(list(a_panel.query("Markdown"))) == 1
        assert len(list(b_panel.query(".user-turn"))) == 1


async def test_resuming_restores_the_status_bars_running_total():
    from quorumdeck.core.session import Session

    session = Session(["a", "b"])
    session.record_usage(Usage(100, 50, 0.05))

    class NothingProvider:
        name = "nothing"

        async def stream(self, request):
            return
            yield

    app = QuorumDeckApp(DeckFile.from_mapping(FANOUT), NothingProvider())
    app.session = session  # swap in a pre-populated session without touching disk
    app._resumed = True

    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query_one(StatusBar).usage == Usage(100, 50, 0.05)


def test_a_resumed_session_must_match_the_active_decks_agents(tmp_path):
    from quorumdeck.config.loader import ConfigError
    from quorumdeck.core.session import Session

    saved = Session(["someone-else"]).save(tmp_path / "s.json")

    with pytest.raises(ConfigError, match="was saved with agents"):
        QuorumDeckApp(DeckFile.from_mapping(FANOUT), FakeProvider(), resume=saved)


BUDGETED = {
    "version": 1,
    "deck": {"budget_usd": 1.0},
    "agents": [{"id": "a", "name": "A", "model": "fake/a"}],
}


async def test_the_status_bar_shows_the_budget_when_one_is_set():
    app = QuorumDeckApp(DeckFile.from_mapping(BUDGETED), FakeProvider(["ok"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        rendered = pilot.app.query_one(StatusBar).render()
        assert "$1.00" in rendered


async def test_a_turn_is_refused_once_the_budget_is_already_spent():
    from quorumdeck.core.events import Usage

    app = QuorumDeckApp(DeckFile.from_mapping(BUDGETED), FakeProvider(["should not run"]))
    async with app.run_test() as pilot:
        pilot.app.session.record_usage(Usage(0, 0, 1.0))

        await pilot.click("#prompt")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        # Refused before anything happened: no bubble, no reply in the thread.
        assert pilot.app.session.thread("a") == []
        panel = pilot.app.query_one("#panel-a", AgentPanel)
        assert not list(panel.query(".user-turn"))


TOOL_DECK = {
    "version": 1,
    "mcp_servers": {"demo": {"command": "unused"}},
    "agents": [{"id": "a", "name": "A", "model": "fake/a", "tools": {"demo": "*"}}],
}


def make_fake_tool_pool(tools, call_tool):
    """Stands in for a live quorumdeck.mcp_pool.ToolPool: same interface, no
    real MCP connection -- passed straight to QuorumDeckApp(tool_pool=...),
    the same way a real one is once mcp_pool.open_tool_pool has connected it.
    """
    from quorumdeck.core.tools import ToolError

    class FakePool:
        def specs_for(self, allowlist):
            return list(tools) if allowlist else []

        async def call(self, name, arguments):
            try:
                return await call_tool(name, arguments)
            except Exception as exc:
                raise ToolError(str(exc)) from exc

    return FakePool()


class _ToolCallingProvider:
    """Round 1 requests a tool call; round 2 answers using its result."""

    name = "tool-calling"

    def __init__(self, *, fail=False):
        self._call = 0
        self._fail = fail

    async def stream(self, request):
        from quorumdeck.core.events import Usage
        from quorumdeck.providers.base import Chunk, Completed, ToolCallRequested

        call = self._call
        self._call += 1
        if call == 0:
            yield ToolCallRequested(id="1", name="demo.echo", arguments='{"x": 1}')
        else:
            yield Chunk("42")
        yield Completed(usage=Usage(1, 1, 0.0), finish_reason="stop")


async def test_a_tool_call_renders_in_the_panel():
    from quorumdeck.core.tools import ToolSpec

    async def call_tool(name, arguments):
        return "the answer"

    pool = make_fake_tool_pool(
        [ToolSpec(name="demo.echo", description="", parameters={})], call_tool
    )
    app = QuorumDeckApp(
        DeckFile.from_mapping(TOOL_DECK), _ToolCallingProvider(), tool_pool=pool
    )
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = pilot.app.query_one("#panel-a", AgentPanel)
        assert list(panel.query(".tool-call"))
        assert list(panel.query(".tool-result"))
        assert not panel.has_class("-failed")
        assert pilot.app.session.thread("a")[-1].content == "42"


async def test_a_failed_tool_call_renders_but_does_not_fail_the_panel():
    from quorumdeck.core.tools import ToolError, ToolSpec

    async def call_tool(name, arguments):
        raise ToolError("permission denied")

    pool = make_fake_tool_pool(
        [ToolSpec(name="demo.echo", description="", parameters={})], call_tool
    )
    app = QuorumDeckApp(
        DeckFile.from_mapping(TOOL_DECK), _ToolCallingProvider(), tool_pool=pool
    )
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = pilot.app.query_one("#panel-a", AgentPanel)
        assert list(panel.query(".tool-error"))
        # The tool call failed, not the turn -- the model still answered.
        assert not panel.has_class("-failed")

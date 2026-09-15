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

from __future__ import annotations

import asyncio

import pytest

from agentdeck.core.events import RunFinished, RunStarted, TextDelta
from agentdeck.core.orchestrator import Orchestrator, Pattern, merge
from tests.conftest import FakeProvider


async def collect(stream):
    return [event async for event in stream]


async def test_single_updates_session(make_agent):
    orch = Orchestrator([make_agent("a", FakeProvider(["ok"]))], pattern=Pattern.SINGLE)
    session = orch.new_session()

    await collect(orch.run_turn(session, "hi"))

    thread = session.thread("a")
    assert [m.content for m in thread] == ["hi", "ok"]
    assert session.usage.output_tokens == 5


async def test_fanout_runs_every_agent_and_keeps_threads_separate(make_agent):
    agents = [
        make_agent("a", FakeProvider(["A1", "A2"])),
        make_agent("b", FakeProvider(["B1"])),
    ]
    orch = Orchestrator(agents, pattern=Pattern.FANOUT)
    session = orch.new_session()

    events = await collect(orch.run_turn(session, "hi"))

    started = {e.agent_id for e in events if isinstance(e, RunStarted)}
    finished = {e.agent_id for e in events if isinstance(e, RunFinished)}
    assert started == finished == {"a", "b"}

    # An agent must never see a peer's answer -- that is what keeps the
    # comparison fair.
    assert [m.content for m in session.thread("a")] == ["hi", "A1A2"]
    assert [m.content for m in session.thread("b")] == ["hi", "B1"]
    assert session.usage.output_tokens == 10


async def test_fanout_isolates_one_agents_failure(make_agent):
    agents = [
        make_agent("good", FakeProvider(["fine"])),
        make_agent("bad", FakeProvider(fail_with="boom")),
    ]
    orch = Orchestrator(agents, pattern=Pattern.FANOUT)
    session = orch.new_session()

    events = await collect(orch.run_turn(session, "hi"))

    assert {e.agent_id for e in events if isinstance(e, RunFinished)} == {"good"}
    assert [m.content for m in session.thread("bad")] == ["hi"]


async def test_merge_preserves_per_stream_order_while_interleaving():
    async def gen(name: str, count: int, delay: float):
        for i in range(count):
            await asyncio.sleep(delay)
            yield TextDelta(agent_id=name, text=str(i))

    events = [e async for e in merge([gen("fast", 3, 0.001), gen("slow", 3, 0.01)])]

    assert len(events) == 6
    for name in ("fast", "slow"):
        assert [e.text for e in events if e.agent_id == name] == ["0", "1", "2"]
    # The fast stream should not have been blocked behind the slow one.
    assert events[0].agent_id == "fast"


@pytest.mark.parametrize("pattern", [Pattern.DEBATE, Pattern.PIPELINE, Pattern.JUDGE])
async def test_unimplemented_patterns_fail_loudly(make_agent, pattern):
    orch = Orchestrator([make_agent("a"), make_agent("b")], pattern=pattern)
    session = orch.new_session()

    with pytest.raises(NotImplementedError, match="ROADMAP"):
        await collect(orch.run_turn(session, "hi"))


def test_empty_deck_is_rejected():
    with pytest.raises(ValueError, match="at least one agent"):
        Orchestrator([])

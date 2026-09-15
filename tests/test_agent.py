from __future__ import annotations

from quorumdeck.core.agent import Agent, AgentSpec
from quorumdeck.core.events import RunFailed, RunFinished, RunStarted, TextDelta
from quorumdeck.core.messages import Role, user
from tests.conftest import FakeProvider


async def collect(stream):
    return [event async for event in stream]


async def test_run_emits_started_deltas_then_finished(make_agent):
    agent = make_agent("a", FakeProvider(["foo", "bar"]))
    events = await collect(agent.run([user("hi")]))

    assert isinstance(events[0], RunStarted)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["foo", "bar"]

    final = events[-1]
    assert isinstance(final, RunFinished)
    assert final.text == "foobar"
    assert final.usage.output_tokens == 5
    assert final.elapsed_s >= 0


async def test_system_prompt_is_prepended_once():
    provider = FakeProvider()
    spec = AgentSpec(id="a", model="fake/a", system_prompt="be terse")
    agent = Agent(spec, provider)

    await collect(agent.run([user("hi")]))

    sent = provider.requests[0].messages
    assert sent[0].role is Role.SYSTEM
    assert sent[0].content == "be terse"
    assert sum(1 for m in sent if m.role is Role.SYSTEM) == 1


async def test_provider_failure_becomes_run_failed(make_agent):
    agent = make_agent("a", FakeProvider(fail_with="401 unauthorized"))
    events = await collect(agent.run([user("hi")]))

    assert isinstance(events[-1], RunFailed)
    assert "401" in events[-1].error
    assert not any(isinstance(e, RunFinished) for e in events)

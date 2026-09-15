from __future__ import annotations

from quorumdeck.core.agent import MAX_TOOL_ROUNDS, Agent, AgentSpec
from quorumdeck.core.events import (
    RunFailed,
    RunFinished,
    RunStarted,
    TextDelta,
    ToolCallFailed,
    ToolCallFinished,
    ToolCallStarted,
)
from quorumdeck.core.events import Usage as EventUsage
from quorumdeck.core.messages import Role, user
from quorumdeck.core.tools import ToolError, ToolSpec
from quorumdeck.providers.base import Chunk, Completed, ToolCallRequested
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


class ScriptedToolProvider:
    """Each call to stream() plays the next scripted round: plain text, or
    one or more tool calls, always ending in Completed."""

    name = "scripted-tools"

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self._call = 0
        self.requests: list = []

    async def stream(self, request):
        self.requests.append(request)
        round_ = self._rounds[self._call]
        self._call += 1
        for chunk in round_.get("text", []):
            yield Chunk(chunk)
        for call in round_.get("tool_calls", []):
            yield ToolCallRequested(
                id=call["id"], name=call["name"], arguments=call["arguments"]
            )
        yield Completed(
            usage=EventUsage(1, 1, 0.0), finish_reason=round_.get("finish_reason", "stop")
        )


def make_call(name="echo", arguments='{"text": "hi"}', call_id="call_1"):
    return {"id": call_id, "name": name, "arguments": arguments}


async def test_a_tool_call_runs_and_its_result_reaches_the_final_answer(make_agent):
    calls = []

    async def call_tool(name, arguments):
        calls.append((name, arguments))
        return "42"

    provider = ScriptedToolProvider(
        [
            {"tool_calls": [make_call()]},
            {"text": ["the answer is 42"]},
        ]
    )
    tool = ToolSpec(name="echo", description="echoes", parameters={})
    agent = make_agent("a", provider, tools=[tool], call_tool=call_tool)

    events = await collect(agent.run([user("hi")]))

    assert calls == [("echo", {"text": "hi"})]
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    finished_calls = [e for e in events if isinstance(e, ToolCallFinished)]
    assert [e.name for e in started] == ["echo"]
    assert [e.result for e in finished_calls] == ["42"]

    final = events[-1]
    assert isinstance(final, RunFinished)
    assert final.text == "the answer is 42"
    # Both rounds cost tokens; both must be counted.
    assert final.usage.output_tokens == 2


async def test_the_tool_result_is_sent_back_as_context_for_the_next_round(make_agent):
    async def call_tool(name, arguments):
        return "the secret is 42"

    provider = ScriptedToolProvider(
        [
            {"tool_calls": [make_call(call_id="abc")]},
            {"text": ["done"]},
        ]
    )
    tool = ToolSpec(name="echo", description="echoes", parameters={})
    agent = make_agent("a", provider, tools=[tool], call_tool=call_tool)

    await collect(agent.run([user("hi")]))

    second_request_messages = provider.requests[1].messages
    tool_messages = [m for m in second_request_messages if m.role is Role.TOOL]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "the secret is 42"
    assert tool_messages[0].tool_call_id == "abc"

    assistant_messages = [m for m in second_request_messages if m.role is Role.ASSISTANT]
    assert assistant_messages[-1].tool_calls[0]["id"] == "abc"


async def test_a_failing_tool_call_is_reported_and_fed_back_not_fatal(make_agent):
    async def call_tool(name, arguments):
        raise ToolError("permission denied")

    provider = ScriptedToolProvider(
        [
            {"tool_calls": [make_call()]},
            {"text": ["sorry, I could not do that"]},
        ]
    )
    tool = ToolSpec(name="echo", description="echoes", parameters={})
    agent = make_agent("a", provider, tools=[tool], call_tool=call_tool)

    events = await collect(agent.run([user("hi")]))

    failed = [e for e in events if isinstance(e, ToolCallFailed)]
    assert [e.error for e in failed] == ["permission denied"]
    assert isinstance(events[-1], RunFinished)  # the model got to recover, not a hard failure

    tool_message = next(m for m in provider.requests[1].messages if m.role is Role.TOOL)
    assert "permission denied" in tool_message.content


async def test_unparseable_tool_arguments_fail_that_call_without_crashing(make_agent):
    async def call_tool(name, arguments):
        raise AssertionError("should never be reached with bad arguments")

    provider = ScriptedToolProvider(
        [
            {"tool_calls": [make_call(arguments="{not json")]},
            {"text": ["ok"]},
        ]
    )
    tool = ToolSpec(name="echo", description="echoes", parameters={})
    agent = make_agent("a", provider, tools=[tool], call_tool=call_tool)

    events = await collect(agent.run([user("hi")]))

    failed = [e for e in events if isinstance(e, ToolCallFailed)]
    assert len(failed) == 1
    assert "unparseable" in failed[0].error
    assert isinstance(events[-1], RunFinished)


async def test_a_tool_call_with_nothing_to_run_it_fails_cleanly(make_agent):
    """Defensive: a model can still name a tool that was never offered."""
    provider = ScriptedToolProvider(
        [
            {"tool_calls": [make_call()]},
            {"text": ["ok"]},
        ]
    )
    agent = make_agent("a", provider)  # no tools, no call_tool

    events = await collect(agent.run([user("hi")]))

    failed = [e for e in events if isinstance(e, ToolCallFailed)]
    assert "no tools configured" in failed[0].error
    assert isinstance(events[-1], RunFinished)


async def test_the_loop_gives_up_after_too_many_tool_rounds(make_agent):
    calls = []

    async def call_tool(name, arguments):
        calls.append(1)
        return "still not enough"

    # Every round asks for another tool call -- never a final answer.
    provider = ScriptedToolProvider([{"tool_calls": [make_call()]}] * 20)
    tool = ToolSpec(name="echo", description="echoes", parameters={})
    agent = make_agent("a", provider, tools=[tool], call_tool=call_tool)

    events = await collect(agent.run([user("hi")]))

    assert len(calls) == MAX_TOOL_ROUNDS
    assert not any(isinstance(e, RunFinished) for e in events)
    final = events[-1]
    assert isinstance(final, RunFailed)
    assert "tool-call rounds" in final.error


async def test_the_tools_offered_are_passed_to_the_provider(make_agent):
    tool = ToolSpec(name="filesystem.read_file", description="reads a file", parameters={})
    agent = make_agent("a", FakeProvider(["ok"]), tools=[tool])

    await collect(agent.run([user("hi")]))

    assert list(agent.provider.requests[0].tools) == [tool]


async def test_an_agent_with_no_tools_offers_none(make_agent):
    agent = make_agent("a", FakeProvider(["ok"]))

    await collect(agent.run([user("hi")]))

    assert agent.provider.requests[0].tools == ()

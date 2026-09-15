from __future__ import annotations

import asyncio

import pytest

from quorumdeck.core.events import RunFailed, RunFinished, RunStarted, TextDelta
from quorumdeck.core.orchestrator import Orchestrator, Pattern, merge
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


@pytest.mark.parametrize("pattern", [Pattern.DEBATE, Pattern.PIPELINE])
async def test_unimplemented_patterns_fail_loudly(make_agent, pattern):
    orch = Orchestrator([make_agent("a"), make_agent("b")], pattern=pattern)
    session = orch.new_session()

    with pytest.raises(NotImplementedError, match="ROADMAP"):
        await collect(orch.run_turn(session, "hi"))


def test_empty_deck_is_rejected():
    with pytest.raises(ValueError, match="at least one agent"):
        Orchestrator([])


def judge_deck(make_agent, *, candidates=("A", "B"), judge_reply="verdict"):
    agents = [
        make_agent(chr(ord("a") + i), FakeProvider([text])) for i, text in enumerate(candidates)
    ]
    agents.append(make_agent("referee", FakeProvider([judge_reply]), role="judge"))
    return Orchestrator(agents, pattern=Pattern.JUDGE, judge_id="referee")


async def test_judge_runs_the_candidates_then_the_judge(make_agent):
    orch = judge_deck(make_agent)
    session = orch.new_session()

    events = await collect(orch.run_turn(session, "hi"))
    finished = [e for e in events if isinstance(e, RunFinished)]

    # The judge speaks last, after both candidates are in.
    assert [e.agent_id for e in finished] == ["a", "b", "referee"]
    assert session.thread("referee")[-1].content == "verdict"
    assert session.usage.output_tokens == 15  # all three turns counted


async def test_the_judge_sees_the_candidate_answers_and_the_others_do_not(make_agent):
    orch = judge_deck(make_agent, candidates=("apples", "oranges"))
    session = orch.new_session()

    await collect(orch.run_turn(session, "hi"))

    referee_prompt = session.thread("referee")[1].content
    assert "apples" in referee_prompt and "oranges" in referee_prompt
    # The candidates must not be contaminated by their peers' answers.
    assert [m.content for m in session.thread("a")] == ["hi", "apples"]
    assert "oranges" not in " ".join(m.content for m in session.thread("a"))


async def test_the_judge_is_shown_numbered_candidates_not_model_names(make_agent):
    """A judge asked to rank 'Claude' and 'GPT' is partly ranking the brands."""
    orch = judge_deck(make_agent)
    session = orch.new_session()

    await collect(orch.run_turn(session, "hi"))

    referee_prompt = session.thread("referee")[1].content
    assert "Candidate 1" in referee_prompt and "Candidate 2" in referee_prompt
    assert "fake/a" not in referee_prompt


async def test_the_judge_still_rules_when_one_candidate_fails(make_agent):
    agents = [
        make_agent("a", FakeProvider(fail_with="boom")),
        make_agent("b", FakeProvider(["survived"])),
        make_agent("referee", FakeProvider(["verdict"]), role="judge"),
    ]
    orch = Orchestrator(agents, pattern=Pattern.JUDGE, judge_id="referee")
    session = orch.new_session()

    events = await collect(orch.run_turn(session, "hi"))

    assert [e.agent_id for e in events if isinstance(e, RunFailed)] == ["a"]
    referee_prompt = session.thread("referee")[1].content
    assert "survived" in referee_prompt
    assert "Candidate 2" not in referee_prompt  # only the surviving answer is offered


async def test_the_judge_fails_cleanly_when_every_candidate_fails(make_agent):
    agents = [
        make_agent("a", FakeProvider(fail_with="boom")),
        make_agent("b", FakeProvider(fail_with="boom")),
        make_agent("referee", FakeProvider(["never asked"]), role="judge"),
    ]
    orch = Orchestrator(agents, pattern=Pattern.JUDGE, judge_id="referee")
    session = orch.new_session()

    events = await collect(orch.run_turn(session, "hi"))

    failed = [e for e in events if isinstance(e, RunFailed)]
    assert [e.agent_id for e in failed] == ["a", "b", "referee"]
    assert "nothing to judge" in failed[-1].error
    assert not any(isinstance(e, RunFinished) for e in events)


def test_a_judge_deck_must_name_a_judge_that_exists(make_agent):
    with pytest.raises(ValueError, match="not in the deck"):
        Orchestrator(
            [make_agent("a"), make_agent("b")], pattern=Pattern.JUDGE, judge_id="nobody"
        )

    with pytest.raises(ValueError, match="needs judge_id"):
        Orchestrator([make_agent("a"), make_agent("b")], pattern=Pattern.JUDGE)

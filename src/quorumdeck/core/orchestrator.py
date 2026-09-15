"""How several agents share one turn.

The orchestrator is the reason this project exists. Every pattern is a
different answer to "what does one user prompt mean when there are N models in
the room", and each one emits the same flat event stream so the UI never has to
know which pattern is running.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from enum import StrEnum

from quorumdeck.core.agent import Agent
from quorumdeck.core.events import Event, PromptInjected, RunFailed, RunFinished
from quorumdeck.core.session import Session


class Pattern(StrEnum):
    """Orchestration shapes. See ``docs`` in the README for the intended flow."""

    SINGLE = "single"
    FANOUT = "fanout"
    DEBATE = "debate"
    PIPELINE = "pipeline"
    JUDGE = "judge"


async def merge(streams: Iterable[AsyncIterator[Event]]) -> AsyncIterator[Event]:
    """Interleave concurrent event streams, preserving each stream's own order.

    Events arrive as they are produced, so a fast model is not held back by a
    slow one -- the whole point of running them side by side.
    """
    queue: asyncio.Queue[Event | object] = asyncio.Queue()
    done = object()

    async def pump(stream: AsyncIterator[Event]) -> None:
        try:
            async for event in stream:
                await queue.put(event)
        finally:
            await queue.put(done)

    tasks = [asyncio.create_task(pump(s)) for s in streams]
    if not tasks:
        return

    remaining = len(tasks)
    try:
        while remaining:
            item = await queue.get()
            if item is done:
                remaining -= 1
            else:
                yield item  # type: ignore[misc]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class Orchestrator:
    """Runs a turn across a deck of agents under one :class:`Pattern`."""

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        pattern: Pattern = Pattern.SINGLE,
        rounds: int = 2,
        judge_id: str | None = None,
    ) -> None:
        if not agents:
            raise ValueError("a deck needs at least one agent")
        self.agents = list(agents)
        self.pattern = pattern
        self.rounds = rounds
        self.judge_id = judge_id
        self._by_id: Mapping[str, Agent] = {a.id: a for a in self.agents}

        if pattern is Pattern.JUDGE:
            if judge_id is None:
                raise ValueError("pattern 'judge' needs judge_id to name an agent")
            if judge_id not in self._by_id:
                raise ValueError(
                    f"judge_id '{judge_id}' is not in the deck: {', '.join(self.agent_ids)}"
                )

        if pattern is Pattern.DEBATE:
            critics = [a for a in self.agents if a.spec.role == "critic"]
            if len(self.agents) != 2 or len(critics) != 1:
                raise ValueError(
                    "pattern 'debate' needs exactly one author and one agent with "
                    f"role 'critic', got {len(self.agents)} agent(s) and "
                    f"{len(critics)} critic(s)"
                )

    @property
    def agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def new_session(self, *, max_messages: int = 200) -> Session:
        return Session(self.agent_ids, max_messages=max_messages)

    async def run_turn(self, session: Session, prompt: str) -> AsyncIterator[Event]:
        """Stream one user turn, folding results back into ``session``.

        The session is updated as RunFinished events pass through, so a caller
        that consumes the whole stream ends up with correct history and totals
        without doing any bookkeeping itself.
        """
        session.add_user(prompt)

        match self.pattern:
            case Pattern.SINGLE:
                stream = self._run_single(session)
            case Pattern.FANOUT:
                stream = self._run_fanout(session)
            case Pattern.JUDGE:
                stream = self._run_judge(session)
            case Pattern.DEBATE:
                stream = self._run_debate(session)
            case Pattern.PIPELINE:
                raise NotImplementedError(
                    f"the '{self.pattern}' pattern is defined in the config schema "
                    "but not implemented yet -- see ROADMAP.md"
                )
            case _:  # pragma: no cover - StrEnum is exhaustive
                raise ValueError(f"unknown pattern: {self.pattern}")

        async for event in stream:
            if isinstance(event, RunFinished):
                session.add_assistant(event.agent_id, event.text)
                session.record_usage(event.usage)
            elif isinstance(event, RunFailed):
                # Keep the thread consistent: a failed turn leaves no reply.
                pass
            yield event

    def _run_single(self, session: Session) -> AsyncIterator[Event]:
        agent = self.agents[0]
        return agent.run(session.thread(agent.id))

    def _run_fanout(self, session: Session) -> AsyncIterator[Event]:
        return merge(agent.run(session.thread(agent.id)) for agent in self.agents)

    async def _run_judge(self, session: Session) -> AsyncIterator[Event]:
        """Fan out to the candidates, then hand their answers to the judge.

        Two phases over one event stream: a consumer sees the candidates stream
        in parallel exactly as in ``fanout``, then the judge as an ordinary
        single turn. Nothing in the vocabulary marks the phase boundary, because
        a panel does not need to know.
        """
        judge = self._by_id[self.judge_id]  # type: ignore[index]
        candidates = [a for a in self.agents if a.id != judge.id]

        answers: dict[str, str] = {}
        async for event in merge(a.run(session.thread(a.id)) for a in candidates):
            if isinstance(event, RunFinished):
                answers[event.agent_id] = event.text
            yield event

        # Config order, so the same deck numbers its candidates the same way
        # twice running -- the judge is asked to be reproducible too.
        ordered = [answers[a.id] for a in candidates if a.id in answers]
        if not ordered:
            yield RunFailed(
                agent_id=judge.id,
                model=judge.spec.model,
                error="every candidate failed, so there was nothing to judge",
            )
            return

        judge_prompt = _judge_prompt(ordered)
        session.add_user_to(judge.id, judge_prompt)
        yield PromptInjected(agent_id=judge.id, text=judge_prompt)
        async for event in judge.run(session.thread(judge.id)):
            yield event

    async def _run_debate(self, session: Session) -> AsyncIterator[Event]:
        """author drafts, critic objects, author revises -- for ``self.rounds`` rounds.

        Each side sees only its own thread plus what this pattern deliberately
        shows it: the critic is shown the author's latest draft, the author is
        shown the critic's latest objection. Neither ever sees the other's full
        history -- the same isolation ``fanout`` and ``judge`` rely on.

        ``rounds`` counts critique/revision pairs *after* the author's first,
        unprompted answer. ``rounds: 2`` is draft, critique, revision, critique,
        revision -- five turns, ending on the author's final revision.
        """
        critic = next(a for a in self.agents if a.spec.role == "critic")
        author = next(a for a in self.agents if a.id != critic.id)

        draft = None
        async for event in author.run(session.thread(author.id)):
            if isinstance(event, RunFinished):
                draft = event.text
            yield event
        if draft is None:
            return  # the author's own RunFailed already explains why

        for _ in range(self.rounds):
            critique_prompt = _critique_request(draft)
            session.add_user_to(critic.id, critique_prompt)
            yield PromptInjected(agent_id=critic.id, text=critique_prompt)
            critique = None
            async for event in critic.run(session.thread(critic.id)):
                if isinstance(event, RunFinished):
                    critique = event.text
                yield event
            if critique is None:
                return

            revision_prompt = _revision_request(critique)
            session.add_user_to(author.id, revision_prompt)
            yield PromptInjected(agent_id=author.id, text=revision_prompt)
            draft = None
            async for event in author.run(session.thread(author.id)):
                if isinstance(event, RunFinished):
                    draft = event.text
                yield event
            if draft is None:
                return


def _judge_prompt(answers: Sequence[str]) -> str:
    """Render the candidate answers as one prompt for the judge.

    Candidates are numbered rather than named. An LLM asked to rank "Claude" and
    "GPT" is partly ranking the brands; this tool exists to compare answers, so
    the judge is shown only answers. The numbering follows config order, so the
    caller can still map a verdict back to an agent.
    """
    blocks = "\n\n".join(
        f"--- Candidate {index} ---\n{text.strip()}" for index, text in enumerate(answers, 1)
    )
    return (
        f"Below are {len(answers)} independent candidate answers to the question "
        "above.\n\nProduce the best single answer, drawing on whatever each one "
        "gets right. Where they disagree, say which reading you took and why. Do "
        "not reproduce a candidate verbatim unless it is genuinely the best "
        f"available answer.\n\n{blocks}"
    )


def _critique_request(draft: str) -> str:
    """Ask the critic for the strongest concrete objection to a draft."""
    return (
        "Here is a candidate answer to the question above. Find the strongest "
        "concrete objection to it -- a factual error, a missing case, an "
        "unjustified claim. Do not praise it, and do not soften the objection "
        f"to be polite.\n\n--- Candidate answer ---\n{draft.strip()}"
    )


def _revision_request(critique: str) -> str:
    """Ask the author to revise in light of one round of critique."""
    return (
        "A critic raised the following objection to your last answer. Revise "
        "your answer to address it. If the objection does not hold, say briefly "
        f"why not, but do not simply repeat your previous answer.\n\n"
        f"--- Critique ---\n{critique.strip()}"
    )

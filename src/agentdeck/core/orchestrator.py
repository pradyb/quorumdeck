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

from agentdeck.core.agent import Agent
from agentdeck.core.events import Event, RunFailed, RunFinished
from agentdeck.core.session import Session


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
            case Pattern.DEBATE | Pattern.PIPELINE | Pattern.JUDGE:
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

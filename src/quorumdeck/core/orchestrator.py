"""How several agents share one turn.

The orchestrator is the reason this project exists. Every pattern is a
different answer to "what does one user prompt mean when there are N models in
the room", and each one emits the same flat event stream so the UI never has to
know which pattern is running.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from enum import StrEnum

from quorumdeck.core.agent import Agent
from quorumdeck.core.costs import format_usd
from quorumdeck.core.events import Event, PromptInjected, RunFailed, RunFinished
from quorumdeck.core.messages import Message, assistant, user
from quorumdeck.core.session import Session


class Pattern(StrEnum):
    """Orchestration shapes. See ``docs`` in the README for the intended flow."""

    SINGLE = "single"
    FANOUT = "fanout"
    DEBATE = "debate"
    PIPELINE = "pipeline"
    JUDGE = "judge"


# A runaway planner could decompose a task into far more steps than anyone
# meant to pay for. Not a config knob yet -- nothing has asked for that, and
# every new schema field is a thing to maintain.
MAX_PIPELINE_STEPS = 20

# Best-effort only: never trusted. A backend that understands it uses it; one
# that does not silently drops it (litellm.drop_params), same as any other
# knob this project sends optimistically. _parse_plan is the real contract.
_JSON_RESPONSE_FORMAT: Mapping[str, object] = {"response_format": {"type": "json_object"}}


class PlanError(ValueError):
    """The planner's reply could not be turned into a runnable plan."""


class BudgetExceeded(RuntimeError):
    """The deck has already spent its budget; refused before running anything.

    Not a RunFailed -- it isn't any one agent's failure, and there is no real
    agent id to attribute it to. Raised before the turn's own prompt is even
    recorded, the same way a config error stops a deck before it starts,
    rather than after silently spending past the cap.
    """


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
        budget_usd: float | None = None,
    ) -> None:
        if not agents:
            raise ValueError("a deck needs at least one agent")
        self.agents = list(agents)
        self.pattern = pattern
        self.rounds = rounds
        self.judge_id = judge_id
        self.budget_usd = budget_usd
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

        if pattern is Pattern.PIPELINE:
            planners = [a for a in self.agents if a.spec.role == "planner"]
            workers = [a for a in self.agents if a.spec.role == "worker"]
            if len(planners) != 1:
                raise ValueError(
                    "pattern 'pipeline' needs exactly one agent with role "
                    f"'planner', found {len(planners)}"
                )
            if not workers:
                raise ValueError(
                    "pattern 'pipeline' needs at least one agent with role 'worker'"
                )

    @property
    def agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def new_session(self, *, max_messages: int = 200) -> Session:
        return Session(self.agent_ids, max_messages=max_messages)

    def check_budget(self, session: Session) -> None:
        """Raise :class:`BudgetExceeded` if this deck has already spent its cap.

        Public, and safe to call before doing anything UI-visible with a
        turn that ``run_turn`` will refuse anyway -- ``run_turn`` calls this
        itself, but a caller that shows a prompt in the UI *before* the first
        event arrives (the TUI mounts a bubble into every panel up front)
        needs to know it will be refused before doing that, not after.
        """
        if self.budget_usd is not None and session.usage.cost_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"this deck has spent {format_usd(session.usage.cost_usd)} of its "
                f"{format_usd(self.budget_usd)} budget -- raise deck.budget_usd, or "
                "start a fresh session"
            )

    async def run_turn(self, session: Session, prompt: str) -> AsyncIterator[Event]:
        """Stream one user turn, folding results back into ``session``.

        The session is updated as RunFinished events pass through, so a caller
        that consumes the whole stream ends up with correct history and totals
        without doing any bookkeeping itself.
        """
        self.check_budget(session)
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
                stream = self._run_pipeline(session)
            case _:  # pragma: no cover - StrEnum is exhaustive
                raise ValueError(f"unknown pattern: {self.pattern}")

        async for event in stream:
            # The single place session mutation happens, regardless of which
            # pattern produced the event. A _run_* method that is itself one
            # of merge()'s input streams (pipeline's workers) is driven by
            # merge()'s own pump task, not by this loop, so it cannot rely on
            # a mutation landing before its *own* next step needs to read it
            # back -- only this loop's consumption order is the guaranteed
            # one, because merge() promises to preserve each stream's order
            # by the time events reach here.
            if isinstance(event, RunFinished):
                session.add_assistant(event.agent_id, event.text)
                session.record_usage(event.usage)
            elif isinstance(event, PromptInjected):
                session.add_user_to(event.agent_id, event.text)
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
            yield PromptInjected(agent_id=critic.id, text=critique_prompt)
            critique = None
            async for event in critic.run(session.thread(critic.id)):
                if isinstance(event, RunFinished):
                    critique = event.text
                yield event
            if critique is None:
                return

            revision_prompt = _revision_request(critique)
            yield PromptInjected(agent_id=author.id, text=revision_prompt)
            draft = None
            async for event in author.run(session.thread(author.id)):
                if isinstance(event, RunFinished):
                    draft = event.text
                yield event
            if draft is None:
                return

    async def _run_pipeline(self, session: Session) -> AsyncIterator[Event]:
        """planner decomposes the task; worker agents execute the steps.

        "Execute" still means one more LLM call, not a tool call -- MCP
        support (ROADMAP 0.3) is what will let a worker actually do
        something. Steps route to a named worker; one worker's own steps run
        in order, so a later step can depend on an earlier one in the same
        queue (via the same ``Session.add_user_to`` judge and debate use),
        but workers never see each other's output. So this only suits
        independently-parallelizable steps, not a dependency graph that
        crosses workers.
        """
        planner = next(a for a in self.agents if a.spec.role == "planner")
        workers = {a.id: a for a in self.agents if a.spec.role == "worker"}

        planning_prompt = _planning_request(sorted(workers))
        yield PromptInjected(agent_id=planner.id, text=planning_prompt)

        plan_text = None
        async for event in planner.run(session.thread(planner.id), extra=_JSON_RESPONSE_FORMAT):
            if isinstance(event, RunFinished):
                plan_text = event.text
            yield event
        if plan_text is None:
            return  # the planner's own RunFailed already explains why

        try:
            steps = _parse_plan(plan_text, frozenset(workers))
        except PlanError as exc:
            # The planner's turn succeeded; the *plan* is what's unusable, so
            # this is a pipeline-level failure rather than a second run of the
            # same agent. Attributed to the planner anyway, the same way
            # judge attributes "nothing to judge" to the judge -- it names
            # the agent whose output caused nothing further to happen.
            yield RunFailed(agent_id=planner.id, model=planner.spec.model, error=str(exc))
            return

        grouped: dict[str, list[str]] = {}
        for assignee, task in steps:
            grouped.setdefault(assignee, []).append(task)

        async for event in merge(
            self._run_worker_steps(workers[worker_id], tasks, session)
            for worker_id, tasks in grouped.items()
        ):
            yield event

    async def _run_worker_steps(
        self, worker: Agent, tasks: Sequence[str], session: Session
    ) -> AsyncIterator[Event]:
        """Run one worker's assigned steps in order, stopping on the first failure.

        A later step may depend on an earlier one in the *same* worker's
        queue -- that's why they run in order instead of also being fanned
        out -- so there is nothing useful left to do here once one fails.

        This is one of several streams handed to ``merge()``, so it is driven
        by merge()'s own pump task rather than by ``run_turn``'s fold loop --
        which means it cannot assume ``session.add_assistant`` has landed for
        step *N* by the time it needs to build step *N+1*'s request; the fold
        loop that does that mutation is a separate, slower consumer draining
        the merged queue at its own pace. So it tracks its own copy of the
        thread instead of re-reading ``session.thread()`` between steps -- the
        same reason ``_run_debate`` keeps ``draft``/``critique`` in a local
        variable rather than trusting the session to already reflect them.
        """
        thread: list[Message] = list(session.thread(worker.id))
        for task in tasks:
            thread.append(user(task))
            yield PromptInjected(agent_id=worker.id, text=task)

            text = None
            async for event in worker.run(thread):
                if isinstance(event, RunFinished):
                    text = event.text
                yield event
            if text is None:
                return
            thread.append(assistant(text, name=worker.id))


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


def _planning_request(worker_ids: Sequence[str]) -> str:
    """Ask the planner for a JSON step list naming one of this deck's workers per step."""
    roster = ", ".join(worker_ids)
    example_assignee = worker_ids[0] if worker_ids else "worker-id"
    return (
        "Break the task above into concrete steps. Respond with ONLY a JSON "
        "array, no prose before or after it -- even if there is only one "
        "step, it must still be inside an array. Each element is an object "
        f'with "assignee" (one of: {roster}) and "task": a self-contained '
        "instruction -- the worker carrying it out will not see this plan or "
        f"any other step, only its own task text. At most {MAX_PIPELINE_STEPS} "
        "steps.\n\n"
        f'Example: [{{"assignee": "{example_assignee}", "task": "..."}}]'
    )


def _parse_plan(text: str, worker_ids: frozenset[str]) -> list[tuple[str, str]]:
    """Extract ``[(assignee, task), ...]`` from the planner's freeform reply.

    Small and local models rarely obey a strict JSON mode, and
    ``response_format`` is sent as a hint that some backends silently drop --
    never trusted -- so this pulls a fenced ```json block if there is one,
    else the first balanced top-level JSON value anywhere in the text.
    """
    raw = _extract_json(text)
    if raw is None:
        raise PlanError(
            "the planner's reply wasn't a parseable JSON step list "
            f"(got: {text.strip()[:200]!r}) -- try a stronger planner model"
        )
    try:
        steps = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlanError(f"the planner's plan is not valid JSON: {exc}") from exc

    # Asked for "a JSON array", a model that decomposes the task into exactly
    # one step will sometimes emit that step bare rather than wrapping it --
    # observed against a real local 7B model, not a hypothetical. Treated as
    # a one-step plan rather than rejected.
    if isinstance(steps, dict):
        steps = [steps]

    if not isinstance(steps, list) or not steps:
        raise PlanError("the planner's plan must be a JSON step, or an array of steps")
    if len(steps) > MAX_PIPELINE_STEPS:
        raise PlanError(
            f"the plan has {len(steps)} steps, more than the {MAX_PIPELINE_STEPS} "
            "this build allows"
        )

    default_worker = next(iter(worker_ids)) if len(worker_ids) == 1 else None
    parsed: list[tuple[str, str]] = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or "task" not in step:
            raise PlanError(f"step {index} is missing a 'task'")
        assignee = step.get("assignee") or default_worker
        if assignee is None:
            raise PlanError(
                f"step {index} has no 'assignee' and this deck has more than "
                "one worker, so it is ambiguous which one should run it"
            )
        if assignee not in worker_ids:
            raise PlanError(
                f"step {index} is assigned to '{assignee}', which is not a "
                f"worker in this deck: {', '.join(sorted(worker_ids))}"
            )
        task = str(step["task"]).strip()
        if not task:
            raise PlanError(f"step {index} has an empty task")
        parsed.append((assignee, task))
    return parsed


def _extract_json(text: str) -> str | None:
    """Pull a JSON array or object out of freeform text.

    Tries a fenced ```json block first, then the first balanced top-level
    array, then the first balanced top-level object -- the array is
    preferred when both are present, since that is what was actually asked
    for; the object fallback exists for the single-step case above.
    """
    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    for open_char, close_char in ("[]", "{}"):
        found = _extract_balanced(text, open_char, close_char)
        if found is not None:
            return found
    return None


def _extract_balanced(text: str, open_char: str, close_char: str) -> str | None:
    """The first top-level ``open_char ... close_char`` span in ``text``."""
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == open_char:
            depth += 1
        elif text[index] == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None

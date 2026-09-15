"""One agent: a model, a persona, and its own slice of the conversation."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from quorumdeck.core.events import (
    Event,
    ReasoningDelta,
    RunFailed,
    RunFinished,
    RunStarted,
    TextDelta,
    ToolCallFailed,
    ToolCallFinished,
    ToolCallStarted,
    Usage,
)
from quorumdeck.core.messages import Message, Role, assistant, system, tool_result
from quorumdeck.core.tools import ToolError, ToolSpec
from quorumdeck.providers.base import (
    Chunk,
    Completed,
    CompletionRequest,
    Provider,
    ProviderError,
    Reasoning,
    ToolCallRequested,
)

log = logging.getLogger(__name__)

# A tool-using turn is a loop: ask, maybe call tools, ask again with the
# results. Bounded so a model that keeps calling tools instead of answering
# cannot spin forever -- the same reasoning as pipeline's MAX_PIPELINE_STEPS.
MAX_TOOL_ROUNDS = 10


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Everything about an agent that a user can put in a config file."""

    id: str
    model: str
    name: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    api_base: str | None = None
    role: str = "peer"
    # server name -> allowed tool names on it, or "*" for all; resolved
    # against a live tool pool at Agent-construction time, not here -- this
    # is just the config's own request, before anything is connected.
    tool_allowlist: Mapping[str, list[str] | Literal["*"]] | None = None

    @property
    def label(self) -> str:
        return self.name or self.id


class Agent:
    """Turns a spec plus a message list into a stream of :mod:`~quorumdeck.core.events`.

    Holds no conversation state -- :class:`~quorumdeck.core.session.Session` owns
    that, so the same agent can be replayed or forked without surprises.
    """

    def __init__(
        self,
        spec: AgentSpec,
        provider: Provider,
        *,
        timeout_s: float | None = None,
        tools: Sequence[ToolSpec] = (),
        call_tool: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.timeout_s = timeout_s
        self.tools = tools
        # Injected rather than imported: core/ does not know MCP exists, only
        # that something can turn a tool name and arguments into a result --
        # the same reasoning as Provider itself.
        self.call_tool = call_tool

    @property
    def id(self) -> str:
        return self.spec.id

    def _request(
        self, messages: Sequence[Message], *, extra: Mapping[str, Any] | None = None
    ) -> CompletionRequest:
        prepared = list(messages)
        if self.spec.system_prompt and not any(m.role is Role.SYSTEM for m in prepared):
            prepared.insert(0, system(self.spec.system_prompt))
        return CompletionRequest(
            model=self.spec.model,
            messages=prepared,
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
            reasoning_effort=self.spec.reasoning_effort,
            api_base=self.spec.api_base,
            timeout_s=self.timeout_s,
            tools=self.tools,
            extra=extra or {},
        )

    async def run(
        self, messages: Sequence[Message], *, extra: Mapping[str, Any] | None = None
    ) -> AsyncIterator[Event]:
        """Stream one turn. Always terminates with RunFinished or RunFailed.

        With no tools offered this is one completion call, exactly as before
        tool support existed. With tools, it is a bounded loop: ask, and if
        the model asks for tools instead of answering, run them, feed the
        results back, and ask again -- up to MAX_TOOL_ROUNDS times, the same
        reasoning as pipeline's step cap, so a model that keeps calling tools
        instead of answering cannot spin forever.

        ``extra`` is an escape hatch for one call, not the agent's own
        persona -- pipeline uses it to hint ``response_format`` at the
        planner without teaching this module what that key means. It reaches
        the provider unchanged; a backend that does not understand it drops
        it (``litellm.drop_params``), same as any other unsupported knob.
        """
        spec = self.spec
        yield RunStarted(agent_id=spec.id, model=spec.model)

        started = perf_counter()
        thread = list(messages)
        all_text: list[str] = []
        total_usage = Usage()
        finish_reason: str | None = None

        for _ in range(MAX_TOOL_ROUNDS):
            round_text: list[str] = []
            tool_calls: list[ToolCallRequested] = []

            try:
                async for event in self.provider.stream(self._request(thread, extra=extra)):
                    match event:
                        case Chunk(text=text):
                            round_text.append(text)
                            all_text.append(text)
                            yield TextDelta(agent_id=spec.id, text=text)
                        case Reasoning(text=text):
                            yield ReasoningDelta(agent_id=spec.id, text=text)
                        case ToolCallRequested() as call:
                            tool_calls.append(call)
                        case Completed(usage=run_usage, finish_reason=reason):
                            total_usage = total_usage + run_usage
                            finish_reason = reason
            except ProviderError as exc:
                log.warning("agent %s failed: %s", spec.id, exc)
                yield RunFailed(agent_id=spec.id, model=spec.model, error=str(exc))
                return
            except Exception as exc:  # one agent must not kill the deck
                log.exception("unexpected failure in agent %s", spec.id)
                yield RunFailed(
                    agent_id=spec.id,
                    model=spec.model,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return

            if not tool_calls:
                yield RunFinished(
                    agent_id=spec.id,
                    model=spec.model,
                    text="".join(all_text),
                    usage=total_usage,
                    elapsed_s=perf_counter() - started,
                    finish_reason=finish_reason,
                )
                return

            thread.append(
                assistant(
                    "".join(round_text),
                    tool_calls=[
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in tool_calls
                    ],
                )
            )
            for call in tool_calls:
                yield ToolCallStarted(
                    agent_id=spec.id, name=call.name, arguments=call.arguments
                )
                call_started = perf_counter()

                try:
                    arguments = json.loads(call.arguments) if call.arguments.strip() else {}
                except json.JSONDecodeError as exc:
                    error = f"the model sent unparseable arguments: {exc}"
                    yield ToolCallFailed(agent_id=spec.id, name=call.name, error=error)
                    thread.append(tool_result(f"error: {error}", tool_call_id=call.id))
                    continue

                if self.call_tool is None:
                    # Should not happen -- a model is only offered tools this
                    # agent has some way to run -- but a model can still name
                    # a tool that was never offered, and that must not crash
                    # the turn any more than a bad argument does.
                    error = (
                        f"'{call.name}' was requested but this agent has no tools configured"
                    )
                    yield ToolCallFailed(agent_id=spec.id, name=call.name, error=error)
                    thread.append(tool_result(f"error: {error}", tool_call_id=call.id))
                    continue

                try:
                    result_text = await self.call_tool(call.name, arguments)
                except ToolError as exc:
                    yield ToolCallFailed(agent_id=spec.id, name=call.name, error=str(exc))
                    thread.append(tool_result(f"error: {exc}", tool_call_id=call.id))
                    continue

                yield ToolCallFinished(
                    agent_id=spec.id,
                    name=call.name,
                    result=result_text,
                    elapsed_s=perf_counter() - call_started,
                )
                thread.append(tool_result(result_text, tool_call_id=call.id))

        yield RunFailed(
            agent_id=spec.id,
            model=spec.model,
            error=f"exceeded {MAX_TOOL_ROUNDS} tool-call rounds without a final answer",
        )

"""One agent: a model, a persona, and its own slice of the conversation."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from quorumdeck.core.events import (
    Event,
    ReasoningDelta,
    RunFailed,
    RunFinished,
    RunStarted,
    TextDelta,
    Usage,
)
from quorumdeck.core.messages import Message, Role, system
from quorumdeck.providers.base import (
    Chunk,
    Completed,
    CompletionRequest,
    Provider,
    ProviderError,
    Reasoning,
)

log = logging.getLogger(__name__)


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
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.timeout_s = timeout_s

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
            extra=extra or {},
        )

    async def run(
        self, messages: Sequence[Message], *, extra: Mapping[str, Any] | None = None
    ) -> AsyncIterator[Event]:
        """Stream one turn. Always terminates with RunFinished or RunFailed.

        ``extra`` is an escape hatch for one call, not the agent's own
        persona -- pipeline uses it to hint ``response_format`` at the
        planner without teaching this module what that key means. It reaches
        the provider unchanged; a backend that does not understand it drops
        it (``litellm.drop_params``), same as any other unsupported knob.
        """
        spec = self.spec
        yield RunStarted(agent_id=spec.id, model=spec.model)

        started = perf_counter()
        parts: list[str] = []
        usage = Usage()
        finish_reason: str | None = None

        try:
            async for event in self.provider.stream(self._request(messages, extra=extra)):
                match event:
                    case Chunk(text=text):
                        parts.append(text)
                        yield TextDelta(agent_id=spec.id, text=text)
                    case Reasoning(text=text):
                        yield ReasoningDelta(agent_id=spec.id, text=text)
                    case Completed(usage=run_usage, finish_reason=reason):
                        usage = run_usage
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

        yield RunFinished(
            agent_id=spec.id,
            model=spec.model,
            text="".join(parts),
            usage=usage,
            elapsed_s=perf_counter() - started,
            finish_reason=finish_reason,
        )

"""The port every engine plugs into.

Keeping this protocol small is the whole point: if LiteLLM is ever the wrong
choice, replacing it means writing one new module in this package and changing
nothing in ``core/`` or ``tui/``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from quorumdeck.core.events import Usage
from quorumdeck.core.messages import Message


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    model: str
    messages: Sequence[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    timeout_s: float | None = None
    api_base: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str


@dataclass(frozen=True, slots=True)
class Reasoning:
    text: str


@dataclass(frozen=True, slots=True)
class Completed:
    usage: Usage
    finish_reason: str | None = None


ProviderEvent: TypeAlias = Chunk | Reasoning | Completed


class ProviderError(RuntimeError):
    """Raised for anything the caller can act on: auth, quota, bad model."""


@runtime_checkable
class Provider(Protocol):
    """Streams a completion as provider-neutral events."""

    name: str

    def stream(self, request: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        """Yield events until the response is complete.

        Implementations must emit exactly one :class:`Completed` last, and must
        raise :class:`ProviderError` rather than leaking SDK exceptions.
        """
        ...

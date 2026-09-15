"""Shared fixtures. No test in this suite is allowed to touch a network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from agentdeck.core.agent import Agent, AgentSpec
from agentdeck.core.events import Usage
from agentdeck.providers.base import (
    Chunk,
    Completed,
    CompletionRequest,
    ProviderError,
    ProviderEvent,
)


class FakeProvider:
    """A scripted provider: deterministic text, optional delay, optional failure."""

    name = "fake"

    def __init__(
        self,
        chunks: Sequence[str] = ("hello", " world"),
        *,
        delay: float = 0.0,
        fail_with: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.delay = delay
        self.fail_with = fail_with
        self.usage = usage or Usage(input_tokens=10, output_tokens=5, cost_usd=0.001)
        self.requests: list[CompletionRequest] = []

    async def stream(self, request: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        self.requests.append(request)
        if self.fail_with:
            raise ProviderError(self.fail_with)
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield Chunk(chunk)
        yield Completed(usage=self.usage, finish_reason="stop")


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def make_agent():
    def _make(agent_id: str, provider: FakeProvider | None = None, **kwargs) -> Agent:
        spec = AgentSpec(id=agent_id, model=f"fake/{agent_id}", **kwargs)
        return Agent(spec, provider or FakeProvider())

    return _make

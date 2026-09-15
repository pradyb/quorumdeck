"""Provider-agnostic streaming events.

Everything downstream -- TUI, headless CLI, tests -- consumes this vocabulary,
so swapping the engine underneath ``providers/`` never reaches the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Usage:
    """Token and cost accounting for one or more runs."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):  # pragma: no cover - defensive
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True, slots=True)
class RunStarted:
    agent_id: str
    model: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    agent_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """Extended-thinking text, when the model exposes it separately."""

    agent_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RunFinished:
    agent_id: str
    model: str
    text: str
    usage: Usage
    elapsed_s: float
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunFailed:
    agent_id: str
    model: str
    error: str


@dataclass(frozen=True, slots=True)
class PromptInjected:
    """One agent was shown something its peers were not -- a judge given the
    candidate answers, a debate's author given the critic's objection.

    Carries no cost and updates no session bookkeeping on its own (the reply
    it provokes still arrives as an ordinary RunFinished). It exists only so a
    UI can render what an agent was just asked, rather than showing a reply
    appear with no visible cause.
    """

    agent_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    agent_id: str
    name: str  # "server.tool", already prefixed by the tool pool
    arguments: str  # raw JSON text, shown as-is rather than re-serialized


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    agent_id: str
    name: str
    result: str
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class ToolCallFailed:
    agent_id: str
    name: str
    error: str


Event: TypeAlias = (
    RunStarted
    | TextDelta
    | ReasoningDelta
    | RunFinished
    | RunFailed
    | PromptInjected
    | ToolCallStarted
    | ToolCallFinished
    | ToolCallFailed
)

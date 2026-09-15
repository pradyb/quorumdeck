"""The shape of ``agents.yaml``.

Validation lives here rather than at the call site so a typo produces one clear
message at load time instead of an AttributeError three layers down.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quorumdeck.core.agent import AgentSpec
from quorumdeck.core.orchestrator import Pattern

SCHEMA_VERSION = 1

Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", max_length=64)]


class Defaults(BaseModel):
    """Applied to every agent that does not override them."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    timeout_s: float | None = Field(default=120.0, gt=0)
    max_messages: int = Field(default=200, gt=1)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    model: str = Field(min_length=1)
    name: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    api_base: str | None = None
    role: Literal["peer", "planner", "worker", "critic", "judge"] = "peer"

    def to_spec(self, defaults: Defaults) -> AgentSpec:
        return AgentSpec(
            id=self.id,
            model=self.model,
            name=self.name,
            system_prompt=self.system_prompt,
            temperature=(
                self.temperature if self.temperature is not None else defaults.temperature
            ),
            max_tokens=(
                self.max_tokens if self.max_tokens is not None else defaults.max_tokens
            ),
            reasoning_effort=(
                self.reasoning_effort
                if self.reasoning_effort is not None
                else defaults.reasoning_effort
            ),
            api_base=self.api_base,
            role=self.role,
        )


class DeckConfig(BaseModel):
    """How the agents relate to each other for a single user turn."""

    model_config = ConfigDict(extra="forbid")

    pattern: Pattern = Pattern.SINGLE
    rounds: int = Field(default=2, ge=1, le=10)
    judge: Identifier | None = None
    title: str | None = None


class DeckFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = SCHEMA_VERSION
    deck: DeckConfig = Field(default_factory=DeckConfig)
    defaults: Defaults = Field(default_factory=Defaults)
    agents: list[AgentConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> DeckFile:
        if self.version != SCHEMA_VERSION:
            raise ValueError(
                f"config version {self.version} is not supported "
                f"(this build reads version {SCHEMA_VERSION})"
            )

        ids = [a.id for a in self.agents]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate agent id(s): {', '.join(sorted(duplicates))}")

        pattern = self.deck.pattern
        count = len(self.agents)

        if pattern is Pattern.SINGLE and count > 1:
            raise ValueError(
                f"pattern 'single' takes exactly one agent, got {count}; "
                "use pattern 'fanout' to run them side by side"
            )
        if pattern in {Pattern.FANOUT, Pattern.DEBATE, Pattern.PIPELINE} and count < 2:
            raise ValueError(f"pattern '{pattern}' needs at least 2 agents, got {count}")

        if pattern is Pattern.JUDGE:
            if self.deck.judge is None:
                raise ValueError("pattern 'judge' requires deck.judge to name an agent")
            if self.deck.judge not in ids:
                raise ValueError(
                    f"deck.judge '{self.deck.judge}' is not one of: {', '.join(ids)}"
                )
            if count < 2:
                raise ValueError("pattern 'judge' needs a judge plus at least one answerer")

        if pattern is Pattern.DEBATE:
            critics = [a.id for a in self.agents if a.role == "critic"]
            if count != 2 or len(critics) != 1:
                raise ValueError(
                    "pattern 'debate' takes exactly one author and one agent with "
                    f"role 'critic' (got {count} agents, {len(critics)} critic(s))"
                )

        if pattern is Pattern.PIPELINE and not any(a.role == "planner" for a in self.agents):
            raise ValueError("pattern 'pipeline' needs one agent with role 'planner'")

        return self

    def specs(self) -> list[AgentSpec]:
        return [a.to_spec(self.defaults) for a in self.agents]

    def with_pattern(self, pattern: Pattern) -> DeckFile:
        """Return a copy running ``pattern``, re-checking the deck rules.

        ``model_copy`` would skip validation, which is how ``--pattern single``
        used to slip past the rule rejecting it on a multi-agent deck and get
        quietly downgraded to one agent.
        """
        data = self.model_dump()
        data["deck"]["pattern"] = pattern
        return DeckFile.model_validate(data)

    @classmethod
    def from_mapping(cls, data: Any) -> DeckFile:
        if not isinstance(data, dict):
            raise ValueError("config root must be a mapping")
        return cls.model_validate(data)

"""Conversation primitives shared by every layer.

Deliberately free of provider SDKs and UI imports: ``core`` must stay
importable in a headless test process with nothing but the stdlib.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    # Wire-shaped already ({"id", "type": "function", "function": {"name",
    # "arguments"}}) -- to_wire() is documented as OpenAI-shaped, so storing
    # these the same way avoids inventing a second shape for the one thing
    # that has to round-trip through it exactly.
    tool_calls: tuple[Mapping[str, Any], ...] | None = None
    tool_call_id: str | None = None  # set on a Role.TOOL message: which call this answers

    def to_wire(self) -> dict[str, Any]:
        """Render as an OpenAI-shaped message dict."""
        wire: dict[str, Any] = {"role": str(self.role), "content": self.content}
        if self.name is not None:
            wire["name"] = self.name
        if self.tool_calls is not None:
            wire["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.tool_call_id is not None:
            wire["tool_call_id"] = self.tool_call_id
        return wire

    def with_content(self, content: str) -> Message:
        return replace(self, content=content)


def system(content: str) -> Message:
    return Message(Role.SYSTEM, content)


def user(content: str) -> Message:
    return Message(Role.USER, content)


def assistant(
    content: str,
    name: str | None = None,
    *,
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
) -> Message:
    return Message(Role.ASSISTANT, content, name, tuple(tool_calls) if tool_calls else None)


def tool_result(content: str, *, tool_call_id: str) -> Message:
    """One tool's answer, addressed back to the call that asked for it."""
    return Message(Role.TOOL, content, tool_call_id=tool_call_id)

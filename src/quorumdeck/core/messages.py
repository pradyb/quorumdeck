"""Conversation primitives shared by every layer.

Deliberately free of provider SDKs and UI imports: ``core`` must stay
importable in a headless test process with nothing but the stdlib.
"""

from __future__ import annotations

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

    def to_wire(self) -> dict[str, Any]:
        """Render as an OpenAI-shaped message dict."""
        wire: dict[str, Any] = {"role": str(self.role), "content": self.content}
        if self.name is not None:
            wire["name"] = self.name
        return wire

    def with_content(self, content: str) -> Message:
        return replace(self, content=content)


def system(content: str) -> Message:
    return Message(Role.SYSTEM, content)


def user(content: str) -> Message:
    return Message(Role.USER, content)


def assistant(content: str, name: str | None = None) -> Message:
    return Message(Role.ASSISTANT, content, name)

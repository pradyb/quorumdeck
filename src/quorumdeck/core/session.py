"""Conversation state for a deck: one transcript per agent, plus totals."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quorumdeck.core.costs import format_usd
from quorumdeck.core.events import Usage
from quorumdeck.core.messages import Message, Role, assistant, user
from quorumdeck.core.tools import summarize

SCHEMA_VERSION = 1


class Session:
    """Per-agent transcripts sharing one user-visible conversation.

    Each agent sees only its own replies, never its peers' -- that is what makes
    a fan-out comparison fair.
    """

    def __init__(self, agent_ids: Sequence[str], *, max_messages: int = 200) -> None:
        self._agent_ids = list(agent_ids)
        self._threads: dict[str, list[Message]] = {a: [] for a in self._agent_ids}
        self.max_messages = max_messages
        self.usage = Usage()
        self.created_at = datetime.now(UTC)

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    def thread(self, agent_id: str) -> list[Message]:
        return list(self._threads[agent_id])

    def add_user(self, content: str) -> None:
        """Broadcast a prompt to every agent's thread."""
        for messages in self._threads.values():
            messages.append(user(content))
        self._trim()

    def add_user_to(self, agent_id: str, content: str) -> None:
        """Add a prompt to one agent's thread only.

        Some patterns ask one agent something its peers are not asked -- a judge
        shown the candidate answers, say. Keeping it out of the other threads is
        what stops a later fan-out from being contaminated by it.
        """
        self._threads[agent_id].append(user(content))
        self._trim()

    def add_assistant(self, agent_id: str, content: str) -> None:
        self._threads[agent_id].append(assistant(content, name=agent_id))
        self._trim()

    def record_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    def _trim(self) -> None:
        """Drop the oldest exchanges, but never the system message."""
        for agent_id, messages in self._threads.items():
            if len(messages) <= self.max_messages:
                continue
            head = [m for m in messages[:1] if m.role is Role.SYSTEM]
            keep = self.max_messages - len(head)
            self._threads[agent_id] = head + messages[-keep:]

    def __iter__(self) -> Iterator[tuple[str, list[Message]]]:
        return iter((a, list(m)) for a, m in self._threads.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cost_usd": self.usage.cost_usd,
            },
            "threads": {
                agent_id: [m.to_wire() for m in messages]
                for agent_id, messages in self._threads.items()
            },
        }

    def to_jsonl(self, agent_ids: Sequence[str] | None = None) -> str:
        """One line per agent's own thread, in the shape fine-tuning JSONL uses.

        Every agent by default: comparing several transcripts side by side is
        the whole point of a multi-agent deck, and silently dropping to one
        would defeat it. ``agent_ids`` narrows to a subset when that is
        genuinely what's wanted.
        """
        ids = list(agent_ids) if agent_ids is not None else self._agent_ids
        lines = [
            json.dumps(
                {
                    "agent_id": agent_id,
                    "messages": [m.to_wire() for m in self._threads[agent_id]],
                }
            )
            for agent_id in ids
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    def to_markdown(self, agent_ids: Sequence[str] | None = None) -> str:
        """Every agent's own thread as one human-readable document.

        Where ``to_jsonl`` is for a program to read, this is for a person --
        a tool call renders as one line (``call → result``) rather than the
        two separate wire messages (an assistant's ``tool_calls`` and a
        later ``tool``-role reply) it actually took to represent it.
        """
        ids = list(agent_ids) if agent_ids is not None else self._agent_ids
        stamp = self.created_at.strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# quorumdeck session",
            "",
            f"_{stamp} · {self.usage.total_tokens:,} tok · {format_usd(self.usage.cost_usd)}_",
        ]
        for agent_id in ids:
            lines += ["", f"## {agent_id}"]
            pending: dict[str, str] = {}  # tool_call_id -> "name(args)"
            for message in self._threads[agent_id]:
                if message.role is Role.USER:
                    lines += ["", f"**You:** {message.content}"]
                elif message.role is Role.ASSISTANT:
                    if message.content:
                        lines += ["", message.content]
                    for call in message.tool_calls or ():
                        function = call.get("function", {})
                        pending[call["id"]] = (
                            f"{function.get('name')}({function.get('arguments')})"
                        )
                elif message.role is Role.TOOL:
                    call = pending.pop(message.tool_call_id, "tool call")
                    lines += ["", f"> ⚙ {summarize(call)} → {summarize(message.content)}"]
                else:  # pragma: no cover - defensive: sessions never store system messages
                    lines += ["", f"**{message.role}:** {message.content}"]
        return "\n".join(lines).strip() + "\n"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Session:
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"session schema v{version} is not readable by this build "
                f"(expected v{SCHEMA_VERSION})"
            )
        threads = raw.get("threads", {})
        sess = cls(list(threads))
        for agent_id, messages in threads.items():
            sess._threads[agent_id] = [
                Message(
                    Role(m["role"]),
                    m.get("content", ""),
                    m.get("name"),
                    tuple(m["tool_calls"]) if m.get("tool_calls") else None,
                    m.get("tool_call_id"),
                )
                for m in messages
            ]
        usage = raw.get("usage", {})
        sess.usage = Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost_usd=usage.get("cost_usd", 0.0),
        )
        return sess

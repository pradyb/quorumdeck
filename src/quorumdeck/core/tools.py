"""What a tool looks like to an agent -- provider- and source-agnostic.

MCP is the only source today (``quorumdeck/mcp_pool.py`` is the one adapter),
but nothing here says so, the same reasoning as ``providers/base.py``: if a
second tool source ever shows up, it targets this vocabulary, not the MCP SDK.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One callable an agent may be offered.

    ``name`` is globally unique within a deck -- the tool pool prefixes it
    with the server it came from (``filesystem.read_file``), so two servers
    can each expose a ``search`` tool without colliding, and a tool call
    rendered in a panel always says which server it went to.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]  # JSON Schema for the arguments object


class ToolError(RuntimeError):
    """A tool call failed -- rejected, timed out, or the server is unreachable.

    Distinct from :class:`~quorumdeck.providers.base.ProviderError`: this is
    the tool misbehaving, not the model.
    """


def summarize(text: str, limit: int = 200) -> str:
    """A tool's arguments or result as shown to a person, in a panel or on
    stdout -- the model gets the text in full either way; this is only ever
    what gets displayed alongside it."""
    text = " ".join(text.split())  # tool JSON/output is rarely worth its own newlines here
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text)} chars)"

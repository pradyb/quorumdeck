"""The one adapter for MCP tool servers -- the only module that imports ``mcp``.

Opens every server referenced by at least one agent's allowlist, once, and
keeps the connections open for as long as the deck runs; several agents that
share a server share one connection to it. A deck that configures no tools at
all never imports ``mcp`` in the first place -- the same lazy-import
discipline as ``litellm`` and ``textual_serve``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Literal

from quorumdeck.config.loader import data_home
from quorumdeck.config.schema import DeckFile
from quorumdeck.core.tools import ToolError, ToolSpec


class ToolPool:
    """A deck's whole view of its tools: every server's tools, merged, and
    the narrower per-agent view each ``Agent`` actually gets offered.

    Tool names are prefixed with the *config key* the server was declared
    under (``filesystem.read_file``), not whatever the server calls itself in
    its own MCP handshake -- the prefix has to match what a person wrote in
    their own ``tools:`` allowlist, and a third-party server's self-reported
    name is not that.
    """

    def __init__(self, group: Any, tools: Mapping[str, ToolSpec]) -> None:
        self._group = group
        self._tools = dict(tools)

    def specs_for(
        self, allowlist: Mapping[str, list[str] | Literal["*"]] | None
    ) -> list[ToolSpec]:
        if not allowlist:
            return []
        offered: list[ToolSpec] = []
        for name, spec in self._tools.items():
            server_name, _, bare_name = name.partition(".")
            allowed = allowlist.get(server_name)
            if allowed == "*" or (allowed and bare_name in allowed):
                offered.append(spec)
        return offered

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise ToolError(f"'{name}' is not a tool this deck's mcp_servers exposes")
        try:
            result = await self._group.call_tool(name, arguments)
        except Exception as exc:  # a misbehaving server must not crash the deck
            raise ToolError(str(exc)) from exc
        text = _content_to_text(getattr(result, "content", None))
        if getattr(result, "isError", False):
            raise ToolError(text)
        return text


def _content_to_text(content: Any) -> str:
    """MCP tool results are a list of content blocks; text is all a model
    gets fed, so anything else is named rather than silently dropped."""
    parts = []
    for block in content or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else f"[{getattr(block, 'type', '?')} content]")
    return "\n".join(parts) if parts else "(empty result)"


@asynccontextmanager
async def open_tool_pool(config: DeckFile) -> AsyncIterator[ToolPool]:
    """Connect every server some agent's allowlist references.

    Yields immediately, with an empty pool, for a deck that configures no
    tools -- not even the ``mcp`` import happens in that case.
    """
    used = sorted({name for agent in config.agents for name in (agent.tools or {})})
    if not used:
        yield ToolPool(group=None, tools={})
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.session_group import ClientSessionGroup
    from mcp.client.stdio import stdio_client
    from mcp.types import Implementation

    # A stdio server's stderr goes straight to ours by default -- fine for a
    # one-off script, not for a full-screen TUI (litellm's own debug printing
    # is kept out of the UI for exactly this reason) or for `deck run`'s
    # otherwise-clean output. Logged to one file instead of discarded, so a
    # misbehaving server is still diagnosable.
    log_path = data_home() / "mcp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncExitStack() as stack:
        errlog = stack.enter_context(log_path.open("a", encoding="utf-8"))
        group = await stack.enter_async_context(
            ClientSessionGroup(
                # Every name gets prefixed, not only on collision -- picking
                # one scheme and applying it everywhere (rather than only
                # when two servers happen to clash) is what the SDK's own
                # docs recommend, and it is also what keeps a tool's name
                # stable across a config gaining or losing a second server.
                component_name_hook=lambda name, server_info: f"{server_info.name}.{name}"
            )
        )

        for server_name in used:
            server = config.mcp_servers[server_name]
            params = StdioServerParameters(
                command=server.command, args=server.args, env=server.env or None
            )
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            # connect_to_server would be one line instead of this, but it
            # names the connection from the server's own MCP handshake --
            # connect_with_session is the one path that lets *us* choose the
            # name, matching the config key the allowlist is keyed on.
            await group.connect_with_session(
                Implementation(name=server_name, version="0"), session
            )

        tools = {
            name: ToolSpec(
                name=name,
                description=tool.description or "",
                parameters=tool.input_schema or {},
            )
            for name, tool in group.tools.items()
        }
        yield ToolPool(group=group, tools=tools)

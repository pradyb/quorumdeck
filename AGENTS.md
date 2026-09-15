# Working on quorumdeck

Guidance for AI coding assistants and new contributors alike.

## What this project is

A terminal console for running several AI agents, on different models and
providers, side by side in one session. It is **not** an agent framework and not
a coding agent. See `ROADMAP.md` for what is explicitly out of scope.

## Layout

```
src/quorumdeck/
  core/        agent · orchestrator · session · events · messages · costs · tools
  providers/   base (the port) · litellm_provider (the only adapter)
  config/      schema (pydantic) · loader · secrets (OS keychain)
  tui/         app · widgets · screens
  mcp_pool.py  the only module that imports `mcp` (the tools port's one adapter)
  cli.py       argparse entry point
tests/         pytest, driven by a scripted FakeProvider -- never a network
```

## Rules that must not be broken

1. `core/` imports no vendor SDK and no UI framework. Provider concepts enter
   through the `Provider` protocol in `providers/base.py`; tool concepts
   through `ToolSpec`/`ToolError` in `core/tools.py`. Neither knows `mcp`
   exists -- `Agent` is handed a plain `call_tool` callable, the same
   reasoning as being handed a `Provider`.
2. `tui/` imports no provider. It consumes `core` events only.
3. `litellm`, `textual_serve`, and `mcp` are all imported lazily, inside
   `providers/litellm_provider.py`, `cli.py`'s `_cmd_serve`, and
   `mcp_pool.py`'s `open_tool_pool` respectively. Importing any of them at
   module scope adds real time to every `deck --help`; there are CLI/subprocess
   tests that notice if one leaks in for a deck that never uses it.
4. Credentials never go in `agents.yaml`, in session files, or in logs.
5. An MCP connection uses anyio task groups internally, which must be entered
   and exited from the *same* asyncio task. `open_tool_pool`'s
   `__aenter__`/`__aexit__` must not be split across two different Textual
   lifecycle hooks (`on_mount` vs `on_unmount`) -- Textual does not guarantee
   they share one, and splitting them crashed on exit the one time this was
   tried. Wrap the pool and `App.run_async()` in one `async with` in a single
   coroutine instead (see `tui/app.py`'s `run_async`).

## Commands

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
uv run deck --config examples/fanout.yaml agents   # no network needed
uv run textual run --dev quorumdeck.tui.app:QuorumDeckApp
```

## Conventions

- Comments say *why*. If a comment explains *what*, rename something instead.
- Tests are named after the behaviour they protect.
- User-facing errors say what to do next (`deck keys set anthropic`), not just
  what went wrong.
- New orchestration patterns must be rejected clearly while unimplemented,
  never silently downgraded to another pattern.

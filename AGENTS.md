# Working on agentdeck

Guidance for AI coding assistants and new contributors alike.

## What this project is

A terminal console for running several AI agents, on different models and
providers, side by side in one session. It is **not** an agent framework and not
a coding agent. See `ROADMAP.md` for what is explicitly out of scope.

## Layout

```
src/agentdeck/
  core/        agent · orchestrator · session · events · messages · costs
  providers/   base (the port) · litellm_provider (the only adapter)
  config/      schema (pydantic) · loader · secrets (OS keychain)
  tui/         app · widgets · screens
  cli.py       argparse entry point
tests/         pytest, driven by a scripted FakeProvider -- never a network
```

## Rules that must not be broken

1. `core/` imports no vendor SDK and no UI framework. Provider concepts enter
   through the `Provider` protocol in `providers/base.py`.
2. `tui/` imports no provider. It consumes `core` events only.
3. `litellm` is imported lazily inside `providers/litellm_provider.py`. Importing
   it at module scope adds seconds to `deck --help`; there is a CLI smoke test
   that will notice.
4. Credentials never go in `agents.yaml`, in session files, or in logs.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
uv run deck --config examples/fanout.yaml agents   # no network needed
uv run textual run --dev agentdeck.tui.app:AgentDeckApp
```

## Conventions

- Comments say *why*. If a comment explains *what*, rename something instead.
- Tests are named after the behaviour they protect.
- User-facing errors say what to do next (`deck keys set anthropic`), not just
  what went wrong.
- New orchestration patterns must be rejected clearly while unimplemented,
  never silently downgraded to another pattern.

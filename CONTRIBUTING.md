# Contributing

Thanks for taking a look.

## Getting set up

```bash
git clone https://github.com/pradyb/quorumdeck
cd quorumdeck
uv sync
uv run pytest
```

No API key is needed to develop or to run the tests: the suite drives a scripted
fake provider and never touches a network.

Working on `mcp_pool.py` or `cli.py`'s `serve` command needs their optional
extras, which plain `uv sync` does not install:

```bash
uv sync --extra mcp --extra web
```

## The two rules

Both are load-bearing, and a PR that breaks either will be asked to change:

1. **`core/` may not import a vendor SDK or a UI framework.** If you need a
   provider feature in `core`, add it to the `Provider` protocol in
   `providers/base.py` as a neutral concept instead; a tool feature goes into
   `ToolSpec`/`ToolError` in `core/tools.py` the same way. `core/` does not
   know `mcp` exists.
2. **`tui/` may not import a provider.** The UI consumes `core` events only.

## Adding a provider

Implement the `Provider` protocol in a new `providers/<name>_provider.py`. It must
emit exactly one `Completed` event last, and raise `ProviderError` rather than
letting an SDK exception escape. Add it to `providers/__init__.py`.

## Adding a tool source

MCP is the only one today, in `mcp_pool.py` -- the one module allowed to
import `mcp`. A second source would expose the same `ToolPool` interface
(`specs_for(allowlist)`, `async call(name, arguments)`) so `Agent` and the UI
need no changes; they already only know `core.tools.ToolSpec` and a plain
`call_tool` callable, not where either one comes from.

## Adding an orchestration pattern

Add the variant to `Pattern`, teach `config/schema.py` what a valid config for it
looks like (including what it should *reject*), and implement a `_run_<name>`
method that returns an `AsyncIterator[Event]`. Test it against `FakeProvider`
with at least: the happy path, one agent failing, and a config the schema should
refuse.

## Style

- `uv run ruff check .` and `uv run ruff format .` before opening a PR.
- Comments explain *why*, not *what*. If a line needs a comment to say what it
  does, rename something instead.
- Tests are named after the behaviour they protect, not the function they call.

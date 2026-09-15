## What this changes

<!-- One or two sentences. -->

## Why

<!-- The problem, not the patch. -->

## Checks

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` and `uv run ruff format .` are clean
- [ ] `core/` still imports no vendor SDK and no UI framework
- [ ] `tui/` still imports no provider
- [ ] New behaviour has a test named after what it protects

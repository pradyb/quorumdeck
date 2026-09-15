# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/pradyb/quorumdeck/security/advisories/new)
rather than opening a public issue. You can expect an acknowledgement within a
few days.

## What this tool does with your credentials

- API keys are read from the OS keychain (via `keyring`) or from environment
  variables. They are never written to `agents.yaml`, to session files, or to logs.
- `deck keys set` reads the key from a terminal prompt without echoing it.
- Keys are exported into the process environment only for the lifetime of the
  process, so the engine can use them.
- Saved sessions under `~/.local/share/quorumdeck/sessions/` contain your prompts
  and the model's replies in plain text. Treat them like any other chat log.

## What it sends where

Each configured agent's prompts go to that agent's provider, and nowhere else.
There is no telemetry, no analytics, and no phone-home.

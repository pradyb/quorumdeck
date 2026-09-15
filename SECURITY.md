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

## Tools (MCP)

An agent's `tools:` allowlist grants real capability, not just data access --
a filesystem server's `write_file` tool can overwrite a file the same as
running the command yourself would.

- **A tool call runs without asking for confirmation first.** Putting a tool
  on an agent's allowlist in `agents.yaml` is the trust decision; there is no
  second gate at call time. Do not put a tool there you would not want an
  unattended model to use.
- **The model chooses which of its allowed tools to call and with what
  arguments.** quorumdeck does not inspect or restrict arguments beyond what
  the allowlist itself limits -- a server that lets you read any path you ask
  for will let the model do the same.
- **A stdio MCP server is a subprocess you configured.** It runs with your
  own user's permissions, same as any other command in `agents.yaml`'s
  `command`/`args`. Scope what it can reach yourself -- the reference
  filesystem server does this by taking an explicit allowed directory as an
  argument; not every server does.
- A tool's result is fed back to the model as plain text, the same as a
  reply from any other agent -- if a tool's output could contain adversarial
  instructions (fetching an untrusted web page's contents, say), the model
  reads them as data, with whatever susceptibility to prompt injection the
  model itself has. quorumdeck adds no defense against this beyond what the
  allowlist already limits the tool to doing.

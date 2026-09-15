# quorumdeck

**A terminal console for running several AI agents — on different models, from different providers — side by side in one session.**

Most terminal LLM clients give you one model at a time and a dropdown to switch.
`quorumdeck` starts from the opposite premise: the interesting thing is what happens
when a Claude agent, a GPT agent and a local Llama agent all answer the same
question at once, and you can see the answers, the latency and the cost next to
each other.

Bring your own model. Bring several.

```
┌─ Claude · anthropic/claude-opus-5 ─┬─ GPT · openai/gpt-5 ───────────────┐
│ › Design a token bucket limiter    │ › Design a token bucket limiter    │
│                                    │                                    │
│ Use a monotonic clock and lazy     │ A token bucket needs four pieces   │
│ refill rather than a background…▌  │ of state: capacity, tokens,…▌      │
│                                    │                                    │
│      1204→318 tok · $0.0089 · 4.1s │      1204→402 tok · $0.0051 · 2.8s │
└────────────────────────────────────┴────────────────────────────────────┘
 ready  ·  fanout × 2  ·  1922 tok  ·  $0.01
```

> **Status: alpha.** All five orchestration patterns — `single`, `fanout`,
> `judge`, `debate`, `pipeline` — work end to end. See [ROADMAP.md](ROADMAP.md)
> for what's next (tool use, session resume).

---

## Why this exists

| | |
| --- | --- |
| **Compare honestly** | Same prompt, same moment, same screen. Per-agent tokens, cost and wall-clock, so "which model should we use for this" stops being a vibe. |
| **One bill, many providers** | 100+ backends through [LiteLLM](https://github.com/BerriAI/litellm) — Anthropic, OpenAI, Gemini, Bedrock, Groq, Mistral, OpenRouter, Ollama, anything OpenAI-shaped. |
| **Agents as config** | A deck is a YAML file you commit next to the code it is about. Share a deck, not a screenshot. |
| **Keys stay out of the repo** | Credentials live in the OS keychain (or the environment). `agents.yaml` is safe to check in. |

---

## Install

Requires Python 3.11+.

```bash
# Run without installing
uvx quorumdeck

# Or install
uv tool install quorumdeck   # or: pipx install quorumdeck
```

`0.1.0` is a normal release, not a pre-release, so no `--prerelease`/`--pre`
flag is needed -- but it's still pre-1.0: expect breaking changes to
`agents.yaml` between minor versions until 1.0.

## Quick start

```bash
deck config init                 # writes ~/.config/quorumdeck/agents.yaml
deck keys set anthropic          # prompts; stored in the OS keychain
deck                             # launch the TUI
```

One-shot, no UI — pipes and scripts welcome:

```bash
deck run "Explain this stack trace" < trace.txt
deck run --pattern fanout "Which index would you add here?"
```

---

## Configuring a deck

`agents.yaml` is the whole interface. Precedence: `--config` → `./quorumdeck.yaml`
→ `~/.config/quorumdeck/agents.yaml`.

```yaml
version: 1

deck:
  pattern: fanout        # single | fanout | debate | pipeline | judge
  title: Compare
  budget_usd: 2.00        # optional; a turn is refused once the deck has spent this

defaults:
  temperature: 0.7
  timeout_s: 120

agents:
  - id: claude
    name: Claude
    model: anthropic/claude-opus-5
    system_prompt: Answer concisely. Show the tradeoff, not just the answer.

  - id: gpt
    name: GPT
    model: openai/gpt-5

  - id: local
    name: Local
    model: ollama_chat/llama3.3
    api_base: http://localhost:11434
```

Model ids are LiteLLM ids: `provider/model`. More in [`examples/`](examples/).

`deck.budget_usd` is a hard stop, checked before a turn does anything: once
the deck's running total (shown in the status bar as it goes) reaches it, the
next turn is refused with a clear reason instead of a surprise bill. There is
no partial-turn cutoff mid-stream -- a turn already running is not
interrupted, only the *next* one is refused.

### Fully local

[`examples/ollama.yaml`](examples/ollama.yaml) compares three small models on a
running Ollama — no API key, no account, and no network:

```bash
ollama serve
deck --config examples/ollama.yaml
```

Use the `ollama_chat/` prefix rather than `ollama/`: it routes to Ollama's
`/api/chat` endpoint, which keeps a multi-turn conversation intact. A deck whose
agents are all local skips LiteLLM's remote price list too, so it stays offline
end to end.

### Free, but not local

[`examples/judge.yaml`](examples/judge.yaml),
[`examples/debate.yaml`](examples/debate.yaml) and
[`examples/pipeline.yaml`](examples/pipeline.yaml) run entirely on
OpenRouter's free tier — several models from different labs, one key, no
card:

```bash
deck keys set openrouter        # from https://openrouter.ai/keys
deck --config examples/judge.yaml
deck --config examples/debate.yaml
deck --config examples/pipeline.yaml
```

Or set `OPENROUTER_API_KEY` in the environment instead; it takes precedence over
the keychain. See [Keys](#keys) for the full prefix-to-variable table.

Free model ids rotate, so if one 404s pick a live replacement from
[openrouter.ai/models?q=free](https://openrouter.ai/models?q=free); no pattern
depends on a particular model.

### Orchestration patterns

| Pattern | What one prompt means | Status |
| --- | --- | --- |
| `single` | One agent answers. Ordinary chat. | ✅ shipped |
| `fanout` | Every agent answers the same prompt in parallel, side by side. | ✅ shipped |
| `debate` | One agent answers, another critiques, the first revises, for `rounds`. | ✅ shipped |
| `pipeline` | A `planner` decomposes the task; `worker` agents execute the steps. | ✅ shipped |
| `judge` | Agents answer in parallel, then a designated `judge` merges or scores. | ✅ shipped |

The schema was written to validate all five before any of them had a runtime,
on purpose — a pattern's config shape doesn't change the day its `_run_*`
method lands, and a request for one not yet built fails clearly rather than
silently falling back to another.

---

## Tools (MCP)

An agent with tools can do things, not just talk about them. quorumdeck talks
to [MCP](https://modelcontextprotocol.io) servers over stdio -- the same
protocol Claude Desktop and most other MCP hosts use, so the same servers
work here:

```bash
pip install "quorumdeck[mcp]"    # opt-in: pulls in the mcp package
```

```yaml
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]

agents:
  - id: assistant
    model: anthropic/claude-opus-5
    tools:
      filesystem: ["read_text_file", "list_directory"]   # or "*" for every tool it exposes
```

A tool's name in a panel or in `deck run`'s output is always `server.tool`
(`filesystem.read_text_file`) -- prefixed with the config key you gave the
server, not whatever the server calls itself internally, so it always matches
what you wrote under `tools:`.

**There is no confirmation prompt before a tool call runs.** A configured
tool is trusted the same way a configured model is: by choosing to put it in
`agents.yaml`. If a server exposes something you do not want an agent doing
unattended, leave it off that agent's allowlist -- `examples/tools.yaml`
deliberately admits `read_text_file` and not `write_file` from the same
server, for exactly this reason. A tool call that fails is reported to the
model as an error and fed back to it, not treated as fatal -- the agent gets
a chance to recover, the same way a person would if a command they ran
failed.

A tool-using turn is a bounded loop: ask, call whatever tools the model asks
for, feed the results back, ask again -- up to 10 rounds, so a model stuck
calling tools instead of ever answering cannot run forever or unboundedly
spend your `budget_usd`.

A stdio server's own diagnostic output goes to
`~/.local/share/quorumdeck/mcp.log`, not your terminal -- raw subprocess
output mixed into a running TUI corrupts it, the same reason LiteLLM's own
debug printing is suppressed too.

---

## Keys

```bash
deck keys list                   # where each provider's key comes from
deck keys set openai             # store in the OS keychain (never echoed)
deck keys rm openai
```

An environment variable always wins over the keychain, so CI and containers work
without a keyring backend. The variable is chosen by the model id's prefix:

| Model prefix | Environment variable |
| --- | --- |
| `anthropic/…` | `ANTHROPIC_API_KEY` |
| `azure/…` | `AZURE_API_KEY` |
| `cerebras/…` | `CEREBRAS_API_KEY` |
| `cohere/…` | `COHERE_API_KEY` |
| `deepseek/…` | `DEEPSEEK_API_KEY` |
| `fireworks_ai/…` | `FIREWORKS_API_KEY` |
| `gemini/…` | `GEMINI_API_KEY` |
| `groq/…` | `GROQ_API_KEY` |
| `mistral/…` | `MISTRAL_API_KEY` |
| `openai/…` | `OPENAI_API_KEY` |
| `openrouter/…` | `OPENROUTER_API_KEY` |
| `perplexity/…` | `PERPLEXITYAI_API_KEY` |
| `together_ai/…` | `TOGETHERAI_API_KEY` |
| `vertex_ai/…` | `VERTEXAI_PROJECT` |
| `xai/…` | `XAI_API_KEY` |

So `examples/judge.yaml`, whose models all start `openrouter/`, needs one
variable:

```bash
export OPENROUTER_API_KEY=sk-or-...     # or: deck keys set openrouter
deck --config examples/judge.yaml
```

A model id with no prefix at all (`gpt-5` rather than `openai/gpt-5`) is treated
as OpenAI, matching LiteLLM's own default. For a prefix outside this table, set
whatever variable LiteLLM expects for that backend yourself. `deck keys set`
rejects a provider outside this table rather than storing a key it could never
export, and it does so before prompting, so a typo never costs you a pasted
secret. `deck keys rm` stays permissive, so an entry stored under an old name
can still be cleaned up.

`ollama`, `ollama_chat`, `vllm`, `lm_studio`, `bedrock` and `sagemaker` need no
key at all: they authenticate over a local socket or through a credential chain.

`deck keys list` shows, per provider, whether the key is coming from the
environment, the keychain, or nowhere.

---

## Keyboard

| Key | Action |
| --- | --- |
| `enter` | Send |
| `ctrl+l` | Clear panels |
| `ctrl+s` | Save session to `~/.local/share/quorumdeck/sessions/` |
| `f1` / `?` | Help |
| `ctrl+q` | Quit |

A deck opens in Textual's `tokyo-night` theme -- several panels are read at
once, and it holds contrast between them better than Textual's default. The
command palette (`ctrl+p`) switches to any of Textual's other built-in themes
for the session; there is no config option for it yet.

---

## Sessions

`ctrl+s` in the TUI saves the session -- every agent's own thread, plus totals
-- as JSON to `~/.local/share/quorumdeck/sessions/`. `deck run` doesn't save
one unless asked:

```bash
deck run "..." --save          # write a session too, so --resume has something
deck sessions list             # everything saved, newest first
```

A bare filename in any of the commands below resolves under that directory,
so you don't need the full path for something you just saved.

**Resume** picks a saved session back up instead of starting fresh, in either
the TUI or a headless run:

```bash
deck --resume 20260915-153422.json               # reopens the TUI with history refilled
deck --resume 20260915-153422.json run "..." --save   # one more headless turn, then re-save
```

It refuses outright if the saved session's agents don't exactly match the
active config's -- resuming a mismatched deck would silently lose or
misassign history rather than continue it. The TUI replays the conversation
into every panel, but not the historical per-turn cost or latency figures:
`Session` only ever kept a running grand total across the whole session, not
a per-turn breakdown, so there's nothing to replay there -- only the running
total itself comes back, in the status bar.

**Export** turns a saved session into something outside quorumdeck can use,
in either of two shapes:

```bash
deck export 20260915-153422.json                             # JSONL, the default
deck export 20260915-153422.json --format markdown            # one readable document
deck export 20260915-153422.json --agent claude                # just one agent's thread
deck export 20260915-153422.json --format markdown -o notes.md # to a file instead
```

JSONL is `{"agent_id": ..., "messages": [...]}` per line, the same shape used
for fine-tuning data, so a multi-agent session becomes one file with one line
per model's own conversation -- ready for an eval harness or bulk analysis
without writing a parser first. Markdown is for a person instead of a
program: one `##` section per agent, a tool call rendered as a single
`call → result` line rather than the two separate messages it actually took
on the wire to represent it.

---

## Running in a browser

```bash
pip install "quorumdeck[web]"
deck serve                # or: deck --config X --resume Y serve --port 9000
```

An opt-in extra: it needs [`textual-serve`](https://github.com/Textualize/textual-serve),
which pulls in `aiohttp`, so it stays out of the default install. Every
browser tab that connects launches its own fresh `quorumdeck` subprocess with
whatever `--config`/`--resume` `deck serve` itself was given -- several tabs
pointed at the same `--resume` file each get their own independent copy of
that starting history, not a shared live session.

---

## Architecture

The layering is the load-bearing decision, not an aesthetic one:

```
src/quorumdeck/
  core/        agent · orchestrator · session · events · messages · costs
  providers/   base (the port) · litellm_provider (the only adapter today)
  config/      schema (pydantic) · loader · secrets (keychain)
  tui/         app · widgets · screens
  cli.py
```

Two rules hold it together, and both are enforced by the test suite:

1. **`core/` imports no vendor SDK and no UI framework.** It is pure async Python
   over its own event vocabulary, which is why every orchestration pattern is
   testable against a scripted fake provider with no network.
2. **`tui/` imports no provider.** It consumes `core` events. Adding a backend
   never touches the UI; adding a UI never touches a backend.

If LiteLLM turns out to be the wrong engine, replacing it means writing one new
module in `providers/` that satisfies the `Provider` protocol — roughly 80 lines —
and changing nothing else.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run textual run --dev quorumdeck.tui.app:QuorumDeckApp   # with devtools
```

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The most useful
places to start are listed in [ROADMAP.md](ROADMAP.md).

## License

MIT. See [LICENSE](LICENSE).

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

> **Status: alpha (0.1).** The `single` and `fanout` patterns work end to end.
> `debate`, `pipeline` and `judge` validate in config but are not implemented yet —
> see [ROADMAP.md](ROADMAP.md).

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
uvx quorumdeck@0.1.0a1

# Or install
uv tool install quorumdeck --prerelease=allow    # or: pipx install --pip-args=--pre quorumdeck
```

Releases are pre-1.0 alphas for now, so installers need to be told to accept
them. Once the orchestration patterns below are all implemented, `0.1.0` proper
drops the extra flag.

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

[`examples/judge.yaml`](examples/judge.yaml) runs the judge pattern entirely on
OpenRouter's free tier — three models from three different labs, one key, no
card:

```bash
deck keys set openrouter        # from https://openrouter.ai/keys
deck --config examples/judge.yaml
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
| `debate` | One agent answers, another critiques, the first revises, for `rounds`. | 🚧 0.2 |
| `pipeline` | A `planner` decomposes the task; `worker` agents execute the steps. | 🚧 0.2 |
| `judge` | Agents answer in parallel, then a designated `judge` merges or scores. | ✅ shipped |

Unimplemented patterns are rejected at runtime with a clear message rather than
silently falling back — the config schema is stable ahead of the runtime on purpose.

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
whatever variable LiteLLM expects for that backend yourself — `deck keys` only
manages the providers listed above.

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

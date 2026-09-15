# Roadmap

Ordered by what unblocks the most people, not by what is most fun.

## 0.2 — the remaining patterns (done)

The config schema validated all five patterns before any of them had a
runtime. All five now do.

- [x] **`debate`** — `author` answers, `critic` critiques, `author` revises, for
      `deck.rounds`. Shipped: the TUI panel architecture already generalized to
      several replies per agent per turn with no changes needed (`end_assistant`
      already resets the panel's stream, so a second `RunFinished` for the same
      agent just mounts a second bubble). The one real gap was that
      `Session.add_user_to` injects text invisibly, so a new `PromptInjected`
      event renders the injected critique/revision prompt the same way the
      human's own prompt renders -- added retroactively to `judge` too, which
      had the identical gap.
- [x] **`judge`** — fan out, then feed the candidate answers to `deck.judge` as a
      single synthesized prompt. Effectively `fanout` followed by `single`.
      Shipped: candidates are numbered rather than named, so the judge ranks
      answers instead of brands, and `Session.add_user_to` keeps the synthesized
      prompt out of every other agent's thread.
- [x] **`pipeline`** — the `planner` agent emits steps; `worker` agents execute
      them (one more LLM call each, not a tool call -- that's 0.3). Shipped: the
      planner is asked for a JSON step list; the parser is deliberately lenient
      -- fenced block, or the first balanced array or object anywhere in the
      text -- because response_format is only ever a hint some backends drop.
      Steps route to a named worker; one worker's own steps run in order (a
      later step can build on an earlier one), but different workers run
      concurrently through the same `merge()` fanout uses.

      Two real bugs found by running it against a live local model, not in
      review:
      - A 7B planner asked for "a JSON array" that decomposed the task into one
        step handed back that step bare, not wrapped. The parser now accepts a
        lone JSON object as a one-step plan.
      - A worker with more than one step raced its own session bookkeeping: it
        is one of several streams `merge()` drives concurrently, so nothing
        guarantees `Session.add_assistant` for step *N* has landed by the time
        step *N+1* needs it as context. Fixed by having every pattern's
        `_run_*` method stop mutating the session directly and only yield
        `PromptInjected`/`RunFinished`; `run_turn`'s single fold loop is now
        the only place session state changes, since it's the one consumer
        `merge()` guarantees sees every stream's own events in order.

## 0.3 — tools

- [ ] MCP client support, so agents can actually do things rather than only talk.
- [ ] Per-agent tool allowlists in `agents.yaml`.
- [ ] Tool-call events in the core vocabulary (`ToolCallStarted` / `ToolResult`)
      and a panel affordance for showing them.

## 0.4 — the session layer

- [x] Resume a saved session (`deck --resume`), into both `run` and the TUI.
      Refuses outright if the saved agent ids don't exactly match the active
      deck's, rather than silently losing or misassigning history. The TUI
      replays the conversation into every panel on mount, but not historical
      per-turn cost/latency -- Session only ever kept a running grand total,
      never a per-turn figure, so there is nothing to replay there.
      `run --save` (writes the session afterward, same as ctrl+s) shipped
      alongside it -- without it, `run --resume` could only ever continue a
      session the TUI had started, never chain two headless runs together.
- [x] `deck export` and `deck sessions list` -- a saved session to JSONL, one
      fine-tuning-shaped line per agent, and a listing of what is available to
      resume or export. `Session.save()`/`.load()` stay JSON on purpose;
      that's the round-trippable shape resume needs, so export is a one-way
      read of it, not a second save format.
- [ ] Export a session to Markdown too, alongside JSONL.
- [x] Cost budget per deck (`deck.budget_usd`), with a hard stop. Checked
      before a turn does anything, in both `run` and the TUI -- the TUI
      checks explicitly before mounting a prompt bubble into any panel,
      rather than letting `Orchestrator.run_turn` refuse only once the first
      event is pulled, which would have left an unanswered prompt sitting in
      every panel with no explanation in it (a bug the first version of this
      had, caught by its own test). The status bar shows progress toward the
      cap once one is set, not just the refusal after the fact.

## Unscheduled, wanted

- [ ] A second `Provider` implementation, mostly to prove the port is real.
- [ ] Per-agent temperature sweeps (same model, different settings, side by side).
- [ ] Web output via `textual serve`.
- [ ] `pipeline` steps that cross workers: right now a worker only ever sees
      its own steps, so a plan where step 2 (on worker B) genuinely needs step
      1's output (from worker A) has no way to get it. Fine for
      independently-parallelizable work; a real gap for anything else.

## Explicitly not planned

- **A general agent framework.** This is a console for watching agents work, not
  a library for building them.
- **Its own provider abstraction.** LiteLLM is one dependency doing one job well.
- **Autonomous file editing.** There are excellent coding agents already; this is
  the room where you compare them.

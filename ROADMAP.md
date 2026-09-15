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

- [ ] Resume a saved session (`deck --resume`), including into the TUI.
- [ ] Export a session to Markdown, not only JSON.
- [ ] Cost budget per deck, with a hard stop.

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

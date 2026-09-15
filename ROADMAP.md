# Roadmap

Ordered by what unblocks the most people, not by what is most fun.

## 0.2 — the remaining patterns

The config schema already validates all five patterns; two still need runtimes.
Each is a composition of what `0.1` proved, and each belongs in
`core/orchestrator.py` behind the same event stream.

- [ ] **`debate`** — `author` answers, `critic` critiques, `author` revises, for
      `deck.rounds`. `Session.add_user_to`, added for `judge`, is the mechanism
      for showing one agent a peer's text; what is still open is how a multi-round
      exchange reads in a panel built for one reply per turn.
- [x] **`judge`** — fan out, then feed the candidate answers to `deck.judge` as a
      single synthesized prompt. Effectively `fanout` followed by `single`.
      Shipped: candidates are numbered rather than named, so the judge ranks
      answers instead of brands, and `Session.add_user_to` keeps the synthesized
      prompt out of every other agent's thread.
- [ ] **`pipeline`** — the `planner` agent emits steps; `worker` agents execute
      them. The only pattern that needs structured output between agents, so it
      should land last.

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

## Explicitly not planned

- **A general agent framework.** This is a console for watching agents work, not
  a library for building them.
- **Its own provider abstraction.** LiteLLM is one dependency doing one job well.
- **Autonomous file editing.** There are excellent coding agents already; this is
  the room where you compare them.

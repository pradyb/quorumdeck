"""Command line entry point.

``argparse`` rather than a CLI framework: the surface is small, and every
dependency here is paid for on a tool people install with ``uvx``.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from quorumdeck import __version__
from quorumdeck.config import loader, secrets
from quorumdeck.config.loader import ConfigError
from quorumdeck.config.schema import DeckFile
from quorumdeck.core.agent import Agent
from quorumdeck.core.costs import format_usd
from quorumdeck.core.events import RunFailed, RunFinished, RunStarted, TextDelta
from quorumdeck.core.orchestrator import Orchestrator, Pattern
from quorumdeck.core.session import Session
from quorumdeck.providers import default_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deck",
        description="A terminal console for running several AI agents side by side.",
    )
    parser.add_argument("--version", action="version", version=f"quorumdeck {__version__}")
    parser.add_argument("-c", "--config", type=Path, help="path to agents.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--resume",
        type=Path,
        help="continue a saved session (see `deck sessions list`); works with the TUI or `run`",
    )

    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="run one prompt headlessly and print the result")
    run_cmd.add_argument("prompt", nargs="+", help="the prompt to send")
    run_cmd.add_argument(
        "--pattern",
        choices=[str(p) for p in Pattern],
        help="override the deck pattern for this run",
    )
    run_cmd.add_argument(
        "--save",
        action="store_true",
        help="save the session afterward (see `deck sessions list`), so a later "
        "`--resume` has something to continue",
    )

    sub.add_parser("agents", help="list the agents in the active config")

    config_cmd = sub.add_parser("config", help="inspect or create configuration")
    config_cmd.add_argument("action", choices=["init", "path", "show"])
    config_cmd.add_argument("--force", action="store_true", help="overwrite on init")

    keys_cmd = sub.add_parser("keys", help="manage API keys in the OS keychain")
    keys_cmd.add_argument("action", choices=["set", "list", "rm"])
    keys_cmd.add_argument("provider", nargs="?", help="e.g. anthropic, openai")

    export_cmd = sub.add_parser(
        "export", help="export a saved session to JSONL, one line per agent"
    )
    export_cmd.add_argument(
        "session", type=Path, help="a saved session file, or just its name under sessions_dir()"
    )
    export_cmd.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    export_cmd.add_argument(
        "--agent", action="append", dest="agents", help="only this agent (repeatable)"
    )

    sessions_cmd = sub.add_parser("sessions", help="inspect saved sessions")
    sessions_cmd.add_argument("action", choices=["list"])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        match args.command:
            case None:
                return _cmd_tui(args)
            case "run":
                return _cmd_run(args)
            case "agents":
                return _cmd_agents(args)
            case "config":
                return _cmd_config(args)
            case "keys":
                return _cmd_keys(args)
            case "export":
                return _cmd_export(args)
            case "sessions":
                return _cmd_sessions(args)
            case _:  # pragma: no cover - argparse rejects anything else
                parser.print_help()
                return 2
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _load(args: argparse.Namespace) -> DeckFile:
    return loader.load(args.config)


def _cmd_tui(args: argparse.Namespace) -> int:
    config = _load(args)
    _prepare_environment(config)
    from quorumdeck.tui.app import run  # imported late: Textual is not needed for `run`

    run(config, resume=args.resume)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.pattern:
        try:
            config = config.with_pattern(Pattern(args.pattern))
        except ValidationError as exc:
            raise ConfigError(f"--pattern {args.pattern}: {loader.render_errors(exc)}") from exc
    _prepare_environment(config)
    return asyncio.run(
        _stream_to_stdout(config, " ".join(args.prompt), resume=args.resume, save=args.save)
    )


async def _stream_to_stdout(
    config: DeckFile, prompt: str, *, resume: Path | None = None, save: bool = False
) -> int:
    provider = default_provider()
    agents = [
        Agent(spec, provider, timeout_s=config.defaults.timeout_s) for spec in config.specs()
    ]
    orchestrator = Orchestrator(
        agents,
        pattern=config.deck.pattern,
        rounds=config.deck.rounds,
        judge_id=config.deck.judge,
    )
    if resume is not None:
        session = loader.resume_session(
            resume, agent_ids=orchestrator.agent_ids, max_messages=config.defaults.max_messages
        )
        prior = sum(len(t) for _, t in session)
        print(f"resumed {resume} ({prior} prior messages)", file=sys.stderr)
    else:
        session = orchestrator.new_session(max_messages=config.defaults.max_messages)
    labels = {a.id: a.spec.label for a in agents}

    # Live streaming only makes sense when one agent is speaking. Interleaving
    # several concurrent streams into one file handle shreds every answer into
    # a few words per header, so a deck prints each reply whole instead.
    live = len(agents) == 1

    # Patterns that start agents at once produce replies in latency order, which
    # varies run to run. Hold those and print in config order, matching the panel
    # order in the TUI and making the same deck reproducible. Purely sequential
    # patterns finish in the order the agents spoke, which is already the order
    # worth reading, so they print as they go.
    ordered = config.deck.pattern in {Pattern.FANOUT, Pattern.JUDGE}
    position = {agent.id: index for index, agent in enumerate(agents)}
    # A judge rules on the candidates, so it reads last however it is declared.
    rank = {a.id: (a.id == config.deck.judge, position[a.id]) for a in agents}
    held: list[tuple[tuple[bool, int], str]] = []
    failures = 0

    try:
        async for event in orchestrator.run_turn(session, prompt):
            match event:
                case RunStarted(agent_id=agent_id, model=model):
                    if live:
                        print(f"\n── {labels[agent_id]} · {model} ──", flush=True)
                case TextDelta(text=text):
                    if live:
                        print(text, end="", flush=True)
                case RunFinished(
                    agent_id=agent_id, model=model, text=text, usage=usage, elapsed_s=elapsed
                ):
                    footer = (
                        f"\n   [{usage.input_tokens}→{usage.output_tokens} tok · "
                        f"{format_usd(usage.cost_usd)} · {elapsed:.1f}s]"
                    )
                    if live:
                        print(footer, flush=True)
                    else:
                        # RunFinished carries the whole reply, so grouping needs
                        # no buffer of deltas of its own.
                        block = f"\n── {labels[agent_id]} · {model} ──\n{text}{footer}"
                        if ordered:
                            held.append((rank[agent_id], block))
                        else:
                            print(block, flush=True)
                case RunFailed(agent_id=agent_id, error=error):
                    failures += 1
                    print(f"\n   error ({labels[agent_id]}): {error}", file=sys.stderr)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for _, block in sorted(held, key=lambda item: item[0]):
        print(block, flush=True)

    total = session.usage
    print(
        f"\ntotal: {total.total_tokens} tok · {format_usd(total.cost_usd)}",
        file=sys.stderr,
    )

    if save:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        saved_path = loader.sessions_dir() / f"{stamp}.json"
        session.save(saved_path)
        print(f"saved {saved_path}", file=sys.stderr)

    return 1 if failures else 0


def _cmd_agents(args: argparse.Namespace) -> int:
    config = _load(args)
    specs = config.specs()
    print(f"pattern: {config.deck.pattern}")

    # Sized to the deck: an OpenRouter id is roughly twice as long as an
    # Anthropic one, and a fixed width turns the table into a staircase.
    id_width = max((len(s.id) for s in specs), default=0)
    model_width = max((len(s.model) for s in specs), default=0)

    for spec in specs:
        provider = secrets.provider_of(spec.model)
        key = secrets.source(provider)
        mark = "ok" if key != "missing" or provider in secrets.NO_KEY_NEEDED else "no key"
        print(f"  {spec.id:<{id_width}}  {spec.model:<{model_width}}  {spec.role:<7} {mark}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    match args.action:
        case "init":
            path = loader.init(args.config, force=args.force)
            print(f"wrote {path}")
        case "path":
            found = loader.discover() if args.config is None else args.config
            if found is None:
                print(f"no config found; `deck config init` writes {loader.user_config_path()}")
                return 1
            print(found)
        case "show":
            config = _load(args)
            print(config.model_dump_json(indent=2))
    return 0


def _cmd_keys(args: argparse.Namespace) -> int:
    match args.action:
        case "list":
            for provider in secrets.known_providers():
                print(f"  {provider:<16} {secrets.source(provider)}")
        case "set":
            if not args.provider:
                print("error: `deck keys set` needs a provider name", file=sys.stderr)
                return 2
            try:
                # Before the prompt: a rejected name after you have typed a
                # secret means you have typed it into the void.
                secrets.ensure_storable(args.provider)
                value = getpass.getpass(f"API key for {args.provider}: ")
                secrets.set_key(args.provider, value)
            except (ValueError, secrets.KeyringUnavailable) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"stored key for {args.provider}")
        case "rm":
            if not args.provider:
                print("error: `deck keys rm` needs a provider name", file=sys.stderr)
                return 2
            print("removed" if secrets.delete(args.provider) else "no key stored")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    path = args.session
    if not path.is_file():
        # ctrl+s in the TUI always writes under sessions_dir(); this lets
        # `deck export 20260915-153422.json` work without the full path.
        candidate = loader.sessions_dir() / path.name
        if not candidate.is_file():
            print(f"error: no session file at {path}", file=sys.stderr)
            return 1
        path = candidate

    try:
        session = Session.load(path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    agent_ids = args.agents or session.agent_ids
    unknown = [a for a in agent_ids if a not in session.agent_ids]
    if unknown:
        print(
            f"error: unknown agent(s) {', '.join(unknown)}; this session has: "
            f"{', '.join(session.agent_ids)}",
            file=sys.stderr,
        )
        return 1

    jsonl = session.to_jsonl(agent_ids)
    if args.output:
        args.output.write_text(jsonl, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(jsonl, end="")
    return 0


def _cmd_sessions(args: argparse.Namespace) -> int:
    match args.action:
        case "list":
            directory = loader.sessions_dir()
            paths = sorted(directory.glob("*.json"), reverse=True) if directory.is_dir() else []
            if not paths:
                print("no saved sessions yet -- ctrl+s in the TUI writes one")
                return 0
            for path in paths:
                try:
                    session = Session.load(path)
                except (OSError, ValueError) as exc:
                    print(f"  {path.name:<28} unreadable: {exc}", file=sys.stderr)
                    continue
                stamp = session.created_at.strftime("%Y-%m-%d %H:%M")
                print(
                    f"  {path.name:<28} {stamp}  {len(session.agent_ids)} agent(s)  "
                    f"{session.usage.total_tokens} tok  {format_usd(session.usage.cost_usd)}"
                )
    return 0


def _prepare_environment(config: DeckFile) -> None:
    """Export the keys this deck needs, and keep a local deck genuinely offline."""
    models = [spec.model for spec in config.specs()]
    if secrets.all_local(models):
        # LiteLLM downloads a model price list at import. A deck of local models
        # has nothing to price, and behind an intercepting proxy the fetch spends
        # ~9s failing loudly before the first token arrives.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    missing = secrets.apply_to_env(models)
    if missing:
        names = ", ".join(missing)
        print(
            f"warning: no API key for {names} (set one with `deck keys set {missing[0]}`)",
            file=sys.stderr,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

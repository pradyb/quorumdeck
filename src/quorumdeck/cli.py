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
from quorumdeck.providers import default_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deck",
        description="A terminal console for running several AI agents side by side.",
    )
    parser.add_argument("--version", action="version", version=f"quorumdeck {__version__}")
    parser.add_argument("-c", "--config", type=Path, help="path to agents.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="run one prompt headlessly and print the result")
    run_cmd.add_argument("prompt", nargs="+", help="the prompt to send")
    run_cmd.add_argument(
        "--pattern",
        choices=[str(p) for p in Pattern],
        help="override the deck pattern for this run",
    )

    sub.add_parser("agents", help="list the agents in the active config")

    config_cmd = sub.add_parser("config", help="inspect or create configuration")
    config_cmd.add_argument("action", choices=["init", "path", "show"])
    config_cmd.add_argument("--force", action="store_true", help="overwrite on init")

    keys_cmd = sub.add_parser("keys", help="manage API keys in the OS keychain")
    keys_cmd.add_argument("action", choices=["set", "list", "rm"])
    keys_cmd.add_argument("provider", nargs="?", help="e.g. anthropic, openai")

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

    run(config)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.pattern:
        try:
            config = config.with_pattern(Pattern(args.pattern))
        except ValidationError as exc:
            raise ConfigError(f"--pattern {args.pattern}: {loader.render_errors(exc)}") from exc
    _prepare_environment(config)
    return asyncio.run(_stream_to_stdout(config, " ".join(args.prompt)))


async def _stream_to_stdout(config: DeckFile, prompt: str) -> int:
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
    session = orchestrator.new_session(max_messages=config.defaults.max_messages)
    labels = {a.id: a.spec.label for a in agents}

    # Live streaming only makes sense when one agent is speaking. Interleaving
    # several concurrent streams into one file handle shreds every answer into
    # a few words per header, so a deck prints each reply whole instead.
    live = len(agents) == 1

    # Fanout starts every agent at once, so replies arrive in latency order --
    # which varies run to run. Hold them and print in config order, matching the
    # panel order in the TUI and making the same deck reproducible. Sequential
    # patterns finish in the order the agents spoke, which is already the order
    # worth reading, so they print as they go.
    ordered = config.deck.pattern is Pattern.FANOUT
    position = {agent.id: index for index, agent in enumerate(agents)}
    held: list[tuple[int, str]] = []
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
                            held.append((position[agent_id], block))
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
    return 1 if failures else 0


def _cmd_agents(args: argparse.Namespace) -> int:
    config = _load(args)
    print(f"pattern: {config.deck.pattern}")
    for spec in config.specs():
        provider = secrets.provider_of(spec.model)
        key = secrets.source(provider)
        mark = "ok" if key != "missing" or provider in secrets.NO_KEY_NEEDED else "no key"
        print(f"  {spec.id:<12} {spec.model:<40} {spec.role:<8} {mark}")
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
            value = getpass.getpass(f"API key for {args.provider}: ")
            try:
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

"""Finding and reading ``agents.yaml``.

Precedence, highest first: an explicit ``--config`` path, a project-local file
in the working directory, then the per-user file under ``$XDG_CONFIG_HOME``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from agentdeck.config.schema import DeckFile

APP_NAME = "agentdeck"
PROJECT_FILENAMES = ("agentdeck.yaml", "agentdeck.yml", ".agentdeck/agents.yaml")


class ConfigError(Exception):
    """Raised with a message intended to be shown verbatim to the user."""


def config_home() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / APP_NAME


def user_config_path() -> Path:
    return config_home() / "agents.yaml"


def data_home() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / APP_NAME


def sessions_dir() -> Path:
    return data_home() / "sessions"


def discover(start: Path | None = None) -> Path | None:
    """Return the config that would be used, or ``None`` if there is none."""
    cwd = start or Path.cwd()
    for name in PROJECT_FILENAMES:
        candidate = cwd / name
        if candidate.is_file():
            return candidate
    user = user_config_path()
    return user if user.is_file() else None


def load(path: Path | None = None, *, start: Path | None = None) -> DeckFile:
    """Load and validate a deck, raising :class:`ConfigError` with usable text."""
    resolved = path or discover(start)
    if resolved is None:
        raise ConfigError(
            "no config found. Run `deck config init` to create "
            f"{user_config_path()}, or add agentdeck.yaml to this directory."
        )
    if not resolved.is_file():
        raise ConfigError(f"config not found: {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML\n{exc}") from exc

    if raw is None:
        raise ConfigError(f"{resolved}: file is empty")

    try:
        return DeckFile.from_mapping(raw)
    except ValidationError as exc:
        raise ConfigError(f"{resolved}: {_render(exc)}") from exc
    except ValueError as exc:
        raise ConfigError(f"{resolved}: {exc}") from exc


def _render(exc: ValidationError) -> str:
    """Pydantic's default repr is dense; flatten it to one line per problem."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "invalid config\n" + "\n".join(lines)


TEMPLATE = """\
# agentdeck -- https://github.com/pradyb/agentdeck
version: 1

deck:
  # single | fanout | debate | pipeline | judge
  pattern: single

defaults:
  temperature: 0.7
  timeout_s: 120

agents:
  - id: main
    name: Main
    model: anthropic/claude-opus-5
    system_prompt: You are a concise, technically precise assistant.

# Swap to a side-by-side comparison by setting pattern to `fanout`
# and uncommenting these:
#
#  - id: gpt
#    name: GPT
#    model: openai/gpt-5
#
#  - id: local
#    name: Local
#    model: ollama/llama3.3
#    api_base: http://localhost:11434
"""


def init(path: Path | None = None, *, force: bool = False) -> Path:
    """Write a starter config, refusing to clobber an existing one."""
    target = path or user_config_path()
    if target.exists() and not force:
        raise ConfigError(f"{target} already exists (use --force to overwrite)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(TEMPLATE, encoding="utf-8")
    return target

from __future__ import annotations

import pytest

from quorumdeck.cli import main

CONFIG = """
version: 1
deck: {pattern: fanout}
agents:
  - {id: a, model: openai/gpt-5}
  - {id: b, model: anthropic/claude-opus-5}
"""


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "quorumdeck.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "quorumdeck" in capsys.readouterr().out


def test_agents_lists_the_deck(config_file, capsys):
    assert main(["--config", str(config_file), "agents"]) == 0

    out = capsys.readouterr().out
    assert "pattern: fanout" in out
    assert "openai/gpt-5" in out
    assert "anthropic/claude-opus-5" in out


def test_missing_config_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    monkeypatch.chdir(tmp_path)

    assert main(["agents"]) == 1
    assert "error:" in capsys.readouterr().err


def test_config_show_emits_json(config_file, capsys):
    assert main(["--config", str(config_file), "config", "show"]) == 0

    import json

    parsed = json.loads(capsys.readouterr().out)
    assert [a["id"] for a in parsed["agents"]] == ["a", "b"]


def test_keys_set_without_a_provider_is_a_usage_error(capsys):
    assert main(["keys", "set"]) == 2
    assert "needs a provider" in capsys.readouterr().err


def test_litellm_is_not_imported_just_to_show_help():
    """Importing the engine at module scope adds seconds to every `deck --help`."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import quorumdeck.cli, sys; "
            "print('litellm' in sys.modules or 'textual' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_pattern_override_cannot_silently_downgrade_a_deck(config_file, capsys):
    """`--pattern single` on a two-agent deck used to run one agent and say nothing."""
    assert main(["--config", str(config_file), "run", "hi", "--pattern", "single"]) == 1

    err = capsys.readouterr().err
    assert "pattern 'single' takes exactly one agent" in err
    assert "fanout" in err  # the message points at the pattern that would work


def test_a_deck_prints_each_reply_whole(config_file, provider_factory, monkeypatch, capsys):
    """Interleaving concurrent streams into one file handle shreds every answer."""
    monkeypatch.setattr(
        "quorumdeck.cli.default_provider",
        lambda: provider_factory(chunks=["one ", "two ", "three"]),
    )

    assert main(["--config", str(config_file), "run", "hi"]) == 0

    out = capsys.readouterr().out
    assert out.count("one two three") == 2  # once per agent, unbroken


def test_a_single_agent_deck_prints_its_reply_once(
    tmp_path, provider_factory, monkeypatch, capsys
):
    """A solo deck streams live; printing the finished text too would double it."""
    path = tmp_path / "quorumdeck.yaml"
    path.write_text(
        "version: 1\nagents:\n  - {id: solo, model: openai/gpt-5}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "quorumdeck.cli.default_provider", lambda: provider_factory(chunks=["alpha", "beta"])
    )

    assert main(["--config", str(path), "run", "hi"]) == 0

    out = capsys.readouterr().out
    assert out.count("alphabeta") == 1


def test_fanout_prints_replies_in_config_order(config_file, monkeypatch, capsys):
    """Latency order varies between runs; the transcript of a deck should not."""
    import asyncio

    from quorumdeck.core.events import Usage
    from quorumdeck.providers.base import Chunk, Completed

    class Staggered:
        """Makes the first agent in the config the last one to finish."""

        name = "staggered"

        async def stream(self, request):
            slow = "gpt-5" in request.model  # agent "a", declared first
            await asyncio.sleep(0.05 if slow else 0.0)
            yield Chunk("declared-first" if slow else "declared-second")
            yield Completed(usage=Usage(), finish_reason="stop")

    monkeypatch.setattr("quorumdeck.cli.default_provider", Staggered)

    assert main(["--config", str(config_file), "run", "hi"]) == 0

    out = capsys.readouterr().out
    assert out.index("declared-first") < out.index("declared-second")

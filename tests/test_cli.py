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

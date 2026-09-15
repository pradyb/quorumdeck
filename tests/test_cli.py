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


def test_keys_set_rejects_an_unknown_provider_before_asking_for_the_key(monkeypatch, capsys):
    """Learning the name was wrong after typing a secret is learning it too late."""
    import getpass

    def refuse(_prompt):
        raise AssertionError("the user was prompted for a key that could never be used")

    monkeypatch.setattr(getpass, "getpass", refuse)

    assert main(["keys", "set", "openrouterr"]) == 1

    err = capsys.readouterr().err
    assert "unknown provider 'openrouterr'" in err
    assert "did you mean 'openrouter'" in err


def test_deck_run_prints_every_round_of_a_debate(tmp_path, monkeypatch, capsys):
    from quorumdeck.core.events import Usage
    from quorumdeck.providers.base import Chunk, Completed

    path = tmp_path / "quorumdeck.yaml"
    path.write_text(
        "version: 1\n"
        "deck: {pattern: debate, rounds: 1}\n"
        "agents:\n"
        "  - {id: author, name: Author, model: openai/gpt-5}\n"
        "  - {id: critic, name: Critic, model: anthropic/claude-opus-5, role: critic}\n",
        encoding="utf-8",
    )

    class Sequenced:
        name = "sequenced"

        def __init__(self):
            self._replies = iter(["draft", "critique", "revision"])

        async def stream(self, request):
            yield Chunk(next(self._replies))
            yield Completed(usage=Usage(), finish_reason="stop")

    monkeypatch.setattr("quorumdeck.cli.default_provider", Sequenced)

    assert main(["--config", str(path), "run", "hi"]) == 0

    out = capsys.readouterr().out
    # Sequential pattern: rounds print in the order they were spoken, not
    # held and re-sorted the way fanout and judge are.
    assert out.index("draft") < out.index("critique") < out.index("revision")
    assert out.count("── Author") == 2  # the draft, and the revision
    assert out.count("── Critic") == 1


def test_export_writes_one_jsonl_line_per_agent(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from quorumdeck.core.session import Session

    session = Session(["a", "b"])
    session.add_user("hi")
    session.add_assistant("a", "from a")
    session.add_assistant("b", "from b")
    saved = session.save(tmp_path / "quorumdeck" / "sessions" / "s.json")

    assert main(["export", str(saved)]) == 0

    import json

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["agent_id"] for line in lines} == {"a", "b"}


def test_export_resolves_a_bare_filename_under_sessions_dir(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from quorumdeck.core.session import Session

    session = Session(["a"])
    session.save(tmp_path / "quorumdeck" / "sessions" / "s.json")

    assert main(["export", "s.json"]) == 0
    assert capsys.readouterr().out.strip() != ""


def test_export_can_be_narrowed_to_one_agent(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from quorumdeck.core.session import Session

    session = Session(["a", "b"])
    session.add_user("hi")
    saved = session.save(tmp_path / "s.json")

    assert main(["export", str(saved), "--agent", "a"]) == 0

    import json

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["agent_id"] == "a"


def test_export_rejects_an_agent_not_in_the_session(tmp_path, capsys):
    from quorumdeck.core.session import Session

    session = Session(["a"])
    saved = session.save(tmp_path / "s.json")

    assert main(["export", str(saved), "--agent", "nobody"]) == 1
    assert "unknown agent" in capsys.readouterr().err


def test_export_writes_to_a_file_when_asked(tmp_path):
    from quorumdeck.core.session import Session

    session = Session(["a"])
    session.add_user("hi")
    saved = session.save(tmp_path / "s.json")
    out = tmp_path / "out.jsonl"

    assert main(["export", str(saved), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").strip()


def test_export_reports_a_missing_session_file(tmp_path, capsys):
    assert main(["export", str(tmp_path / "nope.json")]) == 1
    assert "no session file" in capsys.readouterr().err

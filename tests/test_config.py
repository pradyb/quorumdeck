from __future__ import annotations

import pytest
from pydantic import ValidationError

from quorumdeck.config import loader
from quorumdeck.config.loader import ConfigError
from quorumdeck.config.schema import DeckFile
from quorumdeck.core.orchestrator import Pattern

MINIMAL = """
version: 1
agents:
  - id: main
    model: anthropic/claude-opus-5
"""


def write(tmp_path, text, name="quorumdeck.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_minimal_config_loads_with_defaults(tmp_path):
    config = loader.load(write(tmp_path, MINIMAL))

    assert config.deck.pattern is Pattern.SINGLE
    assert config.defaults.timeout_s == 120.0
    spec = config.specs()[0]
    assert spec.id == "main"
    assert spec.label == "main"


def test_defaults_flow_into_specs_but_agent_wins(tmp_path):
    text = """
version: 1
deck: {pattern: fanout}
defaults: {temperature: 0.2, max_tokens: 100}
agents:
  - {id: a, model: openai/gpt-5}
  - {id: b, model: openai/gpt-5, temperature: 0.9}
"""
    specs = loader.load(write(tmp_path, text)).specs()

    assert specs[0].temperature == 0.2
    assert specs[1].temperature == 0.9
    assert specs[0].max_tokens == specs[1].max_tokens == 100


def test_project_config_beats_user_config(tmp_path, monkeypatch):
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    user_file = loader.user_config_path()
    user_file.parent.mkdir(parents=True)
    user_file.write_text(MINIMAL.replace("main", "from-user"), encoding="utf-8")

    project = tmp_path / "proj"
    project.mkdir()
    (project / "quorumdeck.yaml").write_text(
        MINIMAL.replace("main", "from-project"), encoding="utf-8"
    )

    assert loader.discover(project) == project / "quorumdeck.yaml"
    assert loader.load(start=project).specs()[0].id == "from-project"


def test_missing_config_explains_the_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    with pytest.raises(ConfigError, match="deck config init"):
        loader.load(start=tmp_path)


def test_invalid_yaml_names_the_file(tmp_path):
    path = write(tmp_path, "version: 1\nagents: [oops\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        loader.load(path)


def test_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    text = MINIMAL + "  \nnotakey: true\n"
    with pytest.raises(ConfigError, match="notakey"):
        loader.load(write(tmp_path, text))


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate agent id"):
        DeckFile.from_mapping(
            {
                "version": 1,
                "deck": {"pattern": "fanout"},
                "agents": [
                    {"id": "a", "model": "m"},
                    {"id": "a", "model": "m"},
                ],
            }
        )


def test_single_pattern_refuses_multiple_agents():
    with pytest.raises(ValueError, match="use pattern 'fanout'"):
        DeckFile.from_mapping(
            {
                "version": 1,
                "agents": [{"id": "a", "model": "m"}, {"id": "b", "model": "m"}],
            }
        )


def test_fanout_needs_two_agents():
    with pytest.raises(ValueError, match="at least 2 agents"):
        DeckFile.from_mapping(
            {"version": 1, "deck": {"pattern": "fanout"}, "agents": [{"id": "a", "model": "m"}]}
        )


def test_judge_must_name_a_real_agent():
    with pytest.raises(ValueError, match="is not one of"):
        DeckFile.from_mapping(
            {
                "version": 1,
                "deck": {"pattern": "judge", "judge": "nobody"},
                "agents": [{"id": "a", "model": "m"}, {"id": "b", "model": "m"}],
            }
        )


def test_debate_needs_exactly_one_critic():
    with pytest.raises(ValueError, match="exactly one author and one agent with role 'critic'"):
        DeckFile.from_mapping(
            {
                "version": 1,
                "deck": {"pattern": "debate"},
                "agents": [{"id": "a", "model": "m"}, {"id": "b", "model": "m"}],
            }
        )


def test_debate_rejects_a_third_wheel():
    with pytest.raises(ValueError, match="exactly one author and one agent with role 'critic'"):
        DeckFile.from_mapping(
            {
                "version": 1,
                "deck": {"pattern": "debate"},
                "agents": [
                    {"id": "a", "model": "m"},
                    {"id": "b", "model": "m", "role": "critic"},
                    {"id": "c", "model": "m"},
                ],
            }
        )


def test_debate_with_one_author_and_one_critic_is_valid():
    deck = DeckFile.from_mapping(
        {
            "version": 1,
            "deck": {"pattern": "debate"},
            "agents": [
                {"id": "author", "model": "m"},
                {"id": "critic", "model": "m", "role": "critic"},
            ],
        }
    )
    assert deck.deck.pattern.value == "debate"


def test_future_schema_version_is_refused():
    with pytest.raises(ValueError, match="not supported"):
        DeckFile.from_mapping({"version": 99, "agents": [{"id": "a", "model": "m"}]})


def test_init_writes_a_loadable_template(tmp_path):
    path = loader.init(tmp_path / "agents.yaml")
    assert loader.load(path).specs()

    with pytest.raises(ConfigError, match="already exists"):
        loader.init(path)
    assert loader.init(path, force=True) == path


def test_shipped_examples_are_valid():
    from pathlib import Path

    examples = Path(__file__).resolve().parent.parent / "examples"
    found = sorted(examples.glob("*.yaml"))
    assert found, "examples/ should not be empty"
    for path in found:
        loader.load(path)


def test_with_pattern_keeps_the_agents_and_switches_the_pattern():
    deck = DeckFile.from_mapping(
        {
            "version": 1,
            "deck": {"pattern": "fanout"},
            "agents": [
                {"id": "a", "model": "x/y"},
                {"id": "b", "model": "x/z", "role": "critic"},
            ],
        }
    )

    switched = deck.with_pattern(Pattern.DEBATE)

    assert switched.deck.pattern is Pattern.DEBATE
    assert [a.id for a in switched.agents] == ["a", "b"]
    assert deck.deck.pattern is Pattern.FANOUT  # the original is untouched


def test_with_pattern_rejects_a_pattern_the_deck_cannot_run():
    """model_copy would skip this check and quietly run one of the two agents."""
    deck = DeckFile.from_mapping(
        {
            "version": 1,
            "deck": {"pattern": "fanout"},
            "agents": [{"id": "a", "model": "x/y"}, {"id": "b", "model": "x/z"}],
        }
    )

    with pytest.raises(ValidationError, match="takes exactly one agent"):
        deck.with_pattern(Pattern.SINGLE)


def test_resume_session_loads_matching_history(tmp_path):
    from quorumdeck.core.session import Session

    session = Session(["a", "b"])
    session.add_user("hi")
    saved = session.save(tmp_path / "s.json")

    resumed = loader.resume_session(saved, agent_ids=["a", "b"], max_messages=50)

    assert [m.content for m in resumed.thread("a")] == ["hi"]
    assert resumed.max_messages == 50  # not Session's own default of 200


def test_resume_session_resolves_a_bare_filename_under_sessions_dir(tmp_path, monkeypatch):
    from pathlib import Path

    from quorumdeck.core.session import Session

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    Session(["a"]).save(tmp_path / "quorumdeck" / "sessions" / "s.json")

    resumed = loader.resume_session(Path("s.json"), agent_ids=["a"], max_messages=200)

    assert resumed.agent_ids == ["a"]


def test_resume_session_rejects_a_mismatched_deck(tmp_path):
    from quorumdeck.core.session import Session

    saved = Session(["a", "b"]).save(tmp_path / "s.json")

    with pytest.raises(ConfigError, match="was saved with agents"):
        loader.resume_session(saved, agent_ids=["a", "c"], max_messages=200)


def test_resume_session_reports_a_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="no session file"):
        loader.resume_session(tmp_path / "nope.json", agent_ids=["a"], max_messages=200)


def test_budget_usd_must_be_positive():
    with pytest.raises(ValidationError):
        DeckFile.from_mapping(
            {
                "version": 1,
                "deck": {"budget_usd": 0},
                "agents": [{"id": "a", "model": "m"}],
            }
        )


def test_budget_usd_is_optional_and_off_by_default():
    deck = DeckFile.from_mapping({"version": 1, "agents": [{"id": "a", "model": "m"}]})
    assert deck.deck.budget_usd is None

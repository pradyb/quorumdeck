from __future__ import annotations

import pytest

from agentdeck.core.events import Usage
from agentdeck.core.messages import Role, system
from agentdeck.core.session import Session


def test_user_prompt_reaches_every_thread():
    session = Session(["a", "b"])
    session.add_user("hi")
    session.add_assistant("a", "from a")

    assert [m.content for m in session.thread("a")] == ["hi", "from a"]
    assert [m.content for m in session.thread("b")] == ["hi"]


def test_usage_accumulates():
    session = Session(["a"])
    session.record_usage(Usage(1, 2, 0.5))
    session.record_usage(Usage(3, 4, 0.25))

    assert session.usage == Usage(4, 6, 0.75)
    assert session.usage.total_tokens == 10


def test_trimming_keeps_the_system_message():
    session = Session(["a"], max_messages=4)
    session._threads["a"].append(system("stay"))
    for i in range(20):
        session.add_user(f"m{i}")

    thread = session.thread("a")
    assert len(thread) == 4
    assert thread[0].role is Role.SYSTEM
    assert thread[-1].content == "m19"


def test_round_trips_through_disk(tmp_path):
    session = Session(["a", "b"])
    session.add_user("hi")
    session.add_assistant("a", "yes")
    session.record_usage(Usage(5, 6, 0.01))

    path = session.save(tmp_path / "nested" / "s.json")
    restored = Session.load(path)

    assert restored.agent_ids == ["a", "b"]
    assert [m.content for m in restored.thread("a")] == ["hi", "yes"]
    assert restored.usage == Usage(5, 6, 0.01)


def test_unknown_schema_version_is_refused(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"schema_version": 99, "threads": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="not readable"):
        Session.load(path)

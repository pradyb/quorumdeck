from __future__ import annotations

import json

import pytest

from quorumdeck.core.events import Usage
from quorumdeck.core.messages import Role, system
from quorumdeck.core.session import Session


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


def test_tool_calls_survive_a_round_trip_through_disk(tmp_path):
    """Found while building markdown export: to_wire() already wrote these,
    but load() silently dropped them -- a resumed session with tool calls
    would have sent a tool-role message with no tool_call_id back to the
    provider, which most reject outright."""
    from quorumdeck.core.messages import assistant, tool_result

    session = Session(["a"])
    session.add_user("hi")
    session._threads["a"].append(
        assistant(
            "",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        )
    )
    session._threads["a"].append(tool_result("42", tool_call_id="c1"))

    restored = Session.load(session.save(tmp_path / "s.json"))

    assistant_message, tool_message = restored.thread("a")[1:]
    assert assistant_message.tool_calls[0]["id"] == "c1"
    assert tool_message.tool_call_id == "c1"


def test_unknown_schema_version_is_refused(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"schema_version": 99, "threads": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="not readable"):
        Session.load(path)


def test_to_jsonl_is_one_line_per_agent():
    session = Session(["a", "b"])
    session.add_user("hi")
    session.add_assistant("a", "from a")
    session.add_assistant("b", "from b")

    lines = session.to_jsonl().splitlines()
    assert len(lines) == 2

    a_line = json.loads(lines[0])
    assert a_line["agent_id"] == "a"
    assert [m["content"] for m in a_line["messages"]] == ["hi", "from a"]
    # An agent's own line never contains a peer's reply.
    assert "from b" not in lines[0]


def test_to_jsonl_can_be_narrowed_to_one_agent():
    session = Session(["a", "b"])
    session.add_user("hi")

    lines = session.to_jsonl(["a"]).splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["agent_id"] == "a"


def test_to_jsonl_of_an_empty_deck_is_an_empty_string():
    session = Session([])
    assert session.to_jsonl() == ""


def test_to_markdown_renders_a_readable_transcript():
    session = Session(["a", "b"])
    session.add_user("hi")
    session.add_assistant("a", "from a")
    session.add_assistant("b", "from b")
    session.record_usage(Usage(10, 5, 0.01))

    text = session.to_markdown()

    assert text.startswith("# quorumdeck session")
    assert "## a" in text and "## b" in text
    assert "**You:** hi" in text
    assert "from a" in text
    assert "from b" in text
    # An agent's own section never contains a peer's reply.
    a_section = text.split("## a", 1)[1].split("## b", 1)[0]
    assert "from b" not in a_section


def test_to_markdown_can_be_narrowed_to_one_agent():
    session = Session(["a", "b"])
    session.add_user("hi")

    text = session.to_markdown(["a"])

    assert "## a" in text
    assert "## b" not in text


def test_to_markdown_renders_a_tool_call_as_one_line():
    from quorumdeck.core.messages import assistant, tool_result

    session = Session(["a"])
    session.add_user("hi")
    session._threads["a"].append(
        assistant(
            "",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"x": 1}'},
                }
            ],
        )
    )
    session._threads["a"].append(tool_result("42", tool_call_id="c1"))
    session.add_assistant("a", "the answer is 42")

    text = session.to_markdown()

    assert '⚙ echo({"x": 1}) → 42' in text
    assert "the answer is 42" in text


def test_to_markdown_of_an_empty_deck_still_has_a_header():
    session = Session([])
    text = session.to_markdown()
    assert text.startswith("# quorumdeck session")

from __future__ import annotations

from quorumdeck.config.schema import DeckFile
from quorumdeck.core.tools import ToolSpec
from quorumdeck.mcp_pool import ToolPool, _content_to_text, open_tool_pool


def make_pool(*names: str) -> ToolPool:
    tools = {name: ToolSpec(name=name, description="", parameters={}) for name in names}
    return ToolPool(group=None, tools=tools)


def test_specs_for_filters_to_one_agents_allowlist():
    pool = make_pool("filesystem.read_file", "filesystem.write_file", "git.log")

    offered = pool.specs_for({"filesystem": ["read_file"]})

    assert [s.name for s in offered] == ["filesystem.read_file"]


def test_specs_for_star_means_every_tool_on_that_server():
    pool = make_pool("filesystem.read_file", "filesystem.write_file", "git.log")

    offered = pool.specs_for({"filesystem": "*"})

    assert {s.name for s in offered} == {"filesystem.read_file", "filesystem.write_file"}


def test_specs_for_never_offers_a_tool_from_an_unlisted_server():
    pool = make_pool("filesystem.read_file", "git.log")

    offered = pool.specs_for({"filesystem": "*"})

    assert "git.log" not in {s.name for s in offered}


def test_specs_for_with_no_allowlist_offers_nothing():
    pool = make_pool("filesystem.read_file")

    assert pool.specs_for(None) == []
    assert pool.specs_for({}) == []


def test_content_to_text_joins_text_blocks():
    class Block:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    assert _content_to_text([Block("a"), Block("b")]) == "a\nb"


def test_content_to_text_names_a_non_text_block_rather_than_dropping_it():
    class ImageBlock:
        text = None
        type = "image"

    assert "[image content]" in _content_to_text([ImageBlock()])


def test_content_to_text_of_nothing_is_not_an_empty_string():
    """An empty string fed back to a model reads as 'it worked and said
    nothing', which is a different claim than 'there was no content'."""
    assert _content_to_text(None) == "(empty result)"
    assert _content_to_text([]) == "(empty result)"


async def test_open_tool_pool_does_nothing_when_no_agent_uses_tools():
    """Not even the mcp import happens -- this must work with mcp uninstalled,
    so it must not be exercised by any test that assumes mcp is available."""
    config = DeckFile.from_mapping({"version": 1, "agents": [{"id": "a", "model": "m"}]})

    async with open_tool_pool(config) as pool:
        assert pool.specs_for({"anything": "*"}) == []


def test_a_deck_with_no_tools_never_imports_mcp():
    """mcp is a real dependency (aiohttp-adjacent transports, pydantic
    models); a deck that never configures a tool should not pay for it."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, sys\n"
            "from quorumdeck.config.schema import DeckFile\n"
            "from quorumdeck.mcp_pool import open_tool_pool\n"
            "config = DeckFile.from_mapping("
            '{"version": 1, "agents": [{"id": "a", "model": "m"}]})\n'
            "async def main():\n"
            "    async with open_tool_pool(config):\n"
            "        pass\n"
            "asyncio.run(main())\n"
            "print('mcp' in sys.modules)\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"

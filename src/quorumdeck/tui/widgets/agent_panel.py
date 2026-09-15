"""One agent's column: its transcript, its state, its running cost."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Markdown, Static

if TYPE_CHECKING:
    # Textual does not re-export MarkdownStream from textual.widgets, so this
    # is a type-only import to avoid depending on a private module at runtime.
    from textual.widgets._markdown import MarkdownStream

from quorumdeck.core.agent import AgentSpec
from quorumdeck.core.costs import format_usd
from quorumdeck.core.events import Usage


class AgentPanel(Widget):
    """Renders a single agent's side of the conversation.

    Assistant replies stream straight into a ``Markdown`` widget via
    ``Markdown.get_stream``, so code fences and tables render as they arrive
    rather than only once the turn completes.
    """

    def __init__(self, spec: AgentSpec, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.spec = spec
        self.usage = Usage()
        self._stream: MarkdownStream | None = None

    def compose(self):
        yield VerticalScroll(id=f"transcript-{self.spec.id}")

    def on_mount(self) -> None:
        self.border_title = f" {self.spec.label} "
        self.border_subtitle = f" {self.spec.model} "
        self.transcript.anchor()

    @property
    def transcript(self) -> VerticalScroll:
        return self.query_one(VerticalScroll)

    async def add_user(self, text: str) -> None:
        await self.transcript.mount(Static(f"› {text}", classes="user-turn"))

    def begin_assistant(self) -> None:
        """Mark the panel busy. The reply widget is created on first output."""
        self.add_class("-active")
        self.remove_class("-failed")

    async def append(self, text: str) -> None:
        # Created lazily: a turn that fails before emitting anything should not
        # leave an empty bubble behind, and MarkdownStream.stop() raises
        # CancelledError if it is stopped without ever having been written to.
        if self._stream is None:
            widget = Markdown()
            await self.transcript.mount(widget)
            self._stream = Markdown.get_stream(widget)
        await self._stream.write(text)

    async def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        self.remove_class("-active")
        if stream is None:
            return
        try:
            await stream.stop()
        except asyncio.CancelledError:
            # Textual cancels the stream's pump task to stop it; that is not a
            # cancellation of *our* turn, so only re-raise if we are the target.
            if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                raise

    async def end_assistant(self, usage: Usage, elapsed_s: float) -> None:
        await self._close_stream()
        self.usage = self.usage + usage
        meta = (
            f"{usage.input_tokens}→{usage.output_tokens} tok · "
            f"{format_usd(usage.cost_usd)} · {elapsed_s:.1f}s"
        )
        await self.transcript.mount(Static(meta, classes="run-meta"))

    async def fail(self, error: str) -> None:
        await self._close_stream()
        self.add_class("-failed")
        await self.transcript.mount(Static(f"✗ {error}", classes="run-error"))

    async def clear(self) -> None:
        await self._close_stream()
        await self.transcript.remove_children()
        self.usage = Usage()
        self.remove_class("-failed")

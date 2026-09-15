"""LiteLLM adapter -- the one engine shipped today.

``litellm`` is imported lazily: it pulls in a large dependency tree and costs
seconds at import time, which would be paid on every ``deck --help``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from quorumdeck.core.costs import cost_usd
from quorumdeck.core.events import Usage
from quorumdeck.providers.base import (
    Chunk,
    Completed,
    CompletionRequest,
    ProviderError,
    ProviderEvent,
    Reasoning,
    ToolCallRequested,
)

log = logging.getLogger(__name__)


class LiteLLMProvider:
    """Talks to 100+ backends through a single OpenAI-shaped call."""

    name = "litellm"

    def __init__(self, *, drop_params: bool = True) -> None:
        # Not every backend accepts every knob (temperature on reasoning
        # models, say). Dropping unsupported params beats a hard 400.
        self._drop_params = drop_params
        self._configured = False

    def _ensure_configured(self) -> Any:
        import litellm

        if not self._configured:
            litellm.drop_params = self._drop_params
            # We surface errors in the UI ourselves; LiteLLM's own banner
            # printing corrupts a full-screen TUI.
            litellm.suppress_debug_info = True
            self._configured = True
        return litellm

    def _build_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [m.to_wire() for m in request.messages],
            "stream": True,
            # Without this, streamed responses carry no token counts at all.
            "stream_options": {"include_usage": True},
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.timeout_s is not None:
            kwargs["timeout"] = request.timeout_s
        if request.api_base is not None:
            kwargs["api_base"] = request.api_base
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in request.tools
            ]
        kwargs.update(request.extra)
        return kwargs

    async def stream(self, request: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        litellm = self._ensure_configured()
        kwargs = self._build_kwargs(request)

        input_tokens = 0
        output_tokens = 0
        finish_reason: str | None = None
        # Streamed tool calls arrive as fragments keyed by position, id/name on
        # the first fragment only, arguments accumulating as raw JSON text
        # across however many chunks it takes -- there is no "this one is
        # done" marker, so nothing is finalized until the stream itself ends.
        tool_calls: dict[int, dict[str, str]] = {}

        try:
            response = await litellm.acompletion(**kwargs)
            async for chunk in response:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    # Arrives on the final chunk when include_usage is set.
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                thinking = getattr(delta, "reasoning_content", None)
                if thinking:
                    yield Reasoning(thinking)

                text = getattr(delta, "content", None)
                if text:
                    yield Chunk(text)

                for fragment in getattr(delta, "tool_calls", None) or []:
                    index = getattr(fragment, "index", 0)
                    call = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if getattr(fragment, "id", None):
                        call["id"] = fragment.id
                    function = getattr(fragment, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            call["name"] += function.name
                        if getattr(function, "arguments", None):
                            call["arguments"] += function.arguments
        except Exception as exc:  # translated into our own type below
            raise ProviderError(_explain(exc)) from exc

        for index in sorted(tool_calls):
            call = tool_calls[index]
            # A local model with weak tool support can emit a call with no id;
            # position in the response is still a stable enough handle for
            # matching the eventual tool result back to it.
            yield ToolCallRequested(
                id=call["id"] or f"call_{index}",
                name=call["name"],
                arguments=call["arguments"],
            )

        yield Completed(
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd(request.model, input_tokens, output_tokens),
            ),
            finish_reason=finish_reason,
        )


def _explain(exc: Exception) -> str:
    """Turn an SDK exception into something worth showing a user."""
    name = type(exc).__name__
    detail = str(exc).strip() or name
    hints = {
        "AuthenticationError": "check your API key with `deck keys set <provider>`",
        "RateLimitError": "rate limited or out of quota",
        "NotFoundError": "model not available on this account",
        "BadRequestError": "the model rejected these parameters",
        "Timeout": "request timed out",
        "APIConnectionError": "could not reach the provider",
    }
    hint = hints.get(name)
    return f"{detail} ({hint})" if hint else detail

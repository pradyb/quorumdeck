"""Cost lookup, isolated so it can be stubbed in tests.

LiteLLM ships a price table for every model it knows; unknown models degrade to
zero cost rather than raising -- a wrong price is worse than a blank one.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Best-effort price for a completed run, in USD.

    Returns ``0.0`` when the model is not in LiteLLM's price table (common for
    local Ollama models and freshly released hosted ones).
    """
    if not input_tokens and not output_tokens:
        return 0.0
    try:
        from litellm import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception as exc:  # noqa: BLE001 - pricing must never break a run
        log.debug("no price entry for %s: %s", model, exc)
        return 0.0


def format_usd(amount: float) -> str:
    """Render a cost without pretending to precision we do not have."""
    if amount <= 0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.2f}"

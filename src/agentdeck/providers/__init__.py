"""Engine adapters. Import from here, never from a vendor SDK directly."""

from agentdeck.providers.base import (
    Chunk,
    Completed,
    CompletionRequest,
    Provider,
    ProviderError,
    ProviderEvent,
    Reasoning,
)
from agentdeck.providers.litellm_provider import LiteLLMProvider

__all__ = [
    "Chunk",
    "Completed",
    "CompletionRequest",
    "LiteLLMProvider",
    "Provider",
    "ProviderError",
    "ProviderEvent",
    "Reasoning",
]


def default_provider() -> Provider:
    """The engine used when a deck does not ask for anything specific."""
    return LiteLLMProvider()

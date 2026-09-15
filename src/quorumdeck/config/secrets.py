"""API keys: OS keychain first-class, environment variables still honoured.

Keys never touch the config file. ``agents.yaml`` is meant to be committed and
shared; credentials live in the system keychain or the environment.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from difflib import get_close_matches

log = logging.getLogger(__name__)

SERVICE = "quorumdeck"

# Provider prefix (the part before "/" in a model id) -> env var LiteLLM reads.
ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "vertex_ai": "VERTEXAI_PROJECT",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "fireworks_ai": "FIREWORKS_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}

# Backends that authenticate some other way (local socket, AWS profile, ...).
NO_KEY_NEEDED = frozenset(
    {"ollama", "ollama_chat", "bedrock", "sagemaker", "vllm", "lm_studio"}
)

# The subset that runs on this machine. A deck made only of these should not
# reach the network at all -- not even for a price list.
LOCAL_PROVIDERS = frozenset({"ollama", "ollama_chat", "vllm", "lm_studio"})


class KeyringUnavailable(RuntimeError):
    """No usable keychain backend on this machine."""


def provider_of(model: str) -> str:
    """``anthropic/claude-opus-5`` -> ``anthropic``.

    A bare model id with no prefix is assumed to be OpenAI, matching LiteLLM.
    """
    prefix, _, rest = model.partition("/")
    return prefix if rest else "openai"


def known_providers() -> list[str]:
    return sorted(ENV_VARS)


def _keyring():
    try:
        import keyring
        from keyring.errors import NoKeyringError
    except ImportError as exc:  # pragma: no cover - keyring is a hard dep
        raise KeyringUnavailable("keyring is not installed") from exc
    try:
        backend = keyring.get_keyring()
        if backend.__class__.__name__ == "FailKeyring":
            raise KeyringUnavailable("no keychain backend available")
    except NoKeyringError as exc:
        raise KeyringUnavailable("no keychain backend available") from exc
    return keyring


def get(provider: str) -> str | None:
    """Look up a key: environment wins, then the keychain."""
    env_var = ENV_VARS.get(provider)
    if env_var and (value := os.environ.get(env_var)):
        return value
    try:
        return _keyring().get_password(SERVICE, provider)
    except KeyringUnavailable as exc:
        log.debug("keychain unavailable: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - a locked keychain must not crash us
        log.warning("could not read key for %s: %s", provider, exc)
        return None


def ensure_storable(provider: str) -> None:
    """Raise unless a key stored for ``provider`` would actually be used.

    Checked before the key is asked for, not after: being told the provider name
    was wrong is only useful before you have typed a secret.
    """
    if provider in NO_KEY_NEEDED:
        raise ValueError(
            f"'{provider}' authenticates without an API key, so there is nothing to store"
        )
    if provider not in ENV_VARS:
        close = get_close_matches(provider, known_providers(), n=1)
        hint = f" -- did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"unknown provider '{provider}'{hint}\n"
            "  `deck keys list` shows every provider a key can be stored for.\n"
            "  For any other backend set its environment variable yourself: "
            "there is no variable name to export this one to."
        )


def set_key(provider: str, value: str) -> None:
    ensure_storable(provider)
    if not value.strip():
        raise ValueError("refusing to store an empty key")
    _keyring().set_password(SERVICE, provider, value)


def delete(provider: str) -> bool:
    try:
        _keyring().delete_password(SERVICE, provider)
        return True
    except Exception:  # noqa: BLE001 - "not there" and "cannot read" are the same to us
        return False


def source(provider: str) -> str:
    """Where a key would come from: ``env``, ``keychain``, or ``missing``."""
    env_var = ENV_VARS.get(provider)
    if env_var and os.environ.get(env_var):
        return "env"
    try:
        if _keyring().get_password(SERVICE, provider):
            return "keychain"
    except Exception:  # noqa: BLE001
        return "missing"
    return "missing"


def apply_to_env(models: Iterable[str]) -> list[str]:
    """Export keychain-held keys so the engine can see them.

    Returns providers that still have no key, so a caller can warn before
    burning a round-trip on a request that is going to 401.
    """
    missing: list[str] = []
    for provider in {provider_of(m) for m in models}:
        if provider in NO_KEY_NEEDED:
            continue
        env_var = ENV_VARS.get(provider)
        if env_var and os.environ.get(env_var):
            continue
        key = get(provider)
        if key is None:
            missing.append(provider)
        elif env_var:
            os.environ[env_var] = key
    return sorted(missing)


def all_local(models: Iterable[str]) -> bool:
    """True when every model in the deck is served from this machine."""
    listed = list(models)
    return bool(listed) and all(provider_of(m) in LOCAL_PROVIDERS for m in listed)

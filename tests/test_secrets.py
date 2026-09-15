from __future__ import annotations

import pytest

from quorumdeck.config import secrets


@pytest.fixture(autouse=True)
def stub_keyring(monkeypatch):
    """Never touch the real keychain from a test run."""
    store: dict[tuple[str, str], str] = {}

    class StubKeyring:
        @staticmethod
        def get_password(service, name):
            return store.get((service, name))

        @staticmethod
        def set_password(service, name, value):
            store[(service, name)] = value

        @staticmethod
        def delete_password(service, name):
            del store[(service, name)]

    monkeypatch.setattr(secrets, "_keyring", lambda: StubKeyring)
    return store


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-opus-5", "anthropic"),
        ("openai/gpt-5", "openai"),
        ("ollama/llama3.3", "ollama"),
        ("gpt-4o", "openai"),
    ],
)
def test_provider_is_read_from_the_model_id(model, expected):
    assert secrets.provider_of(model) == expected


def test_environment_beats_the_keychain(monkeypatch, stub_keyring):
    stub_keyring[(secrets.SERVICE, "openai")] = "from-keychain"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    assert secrets.get("openai") == "from-env"
    assert secrets.source("openai") == "env"


def test_keychain_is_used_when_the_environment_is_empty(monkeypatch, stub_keyring):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    secrets.set_key("anthropic", "sk-test")

    assert secrets.get("anthropic") == "sk-test"
    assert secrets.source("anthropic") == "keychain"


def test_empty_keys_are_refused():
    with pytest.raises(ValueError, match="empty key"):
        secrets.set_key("openai", "   ")


def test_apply_to_env_exports_and_reports_gaps(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secrets.set_key("anthropic", "sk-anthropic")

    missing = secrets.apply_to_env(
        ["anthropic/claude-opus-5", "openai/gpt-5", "ollama/llama3.3"]
    )

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-anthropic"
    # Local backends need no key, so they must not be reported as missing.
    assert missing == ["openai"]


def test_a_broken_keychain_does_not_raise(monkeypatch):
    def explode():
        raise secrets.KeyringUnavailable("no backend")

    monkeypatch.setattr(secrets, "_keyring", explode)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert secrets.get("openai") is None
    assert secrets.source("openai") == "missing"


def test_all_local_spots_a_deck_that_needs_no_network():
    assert secrets.all_local(["ollama_chat/qwen2.5:3b", "lm_studio/phi-4"])
    assert not secrets.all_local(["ollama_chat/qwen2.5:3b", "openai/gpt-5"])
    assert not secrets.all_local(["bedrock/claude"])  # no key, but not on this machine
    assert not secrets.all_local([])


def test_the_readme_key_table_matches_the_code():
    """A hand-written table of env vars drifts the moment a provider is added."""
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    for provider, env_var in secrets.ENV_VARS.items():
        assert f"| `{provider}/…` | `{env_var}` |" in text, f"README is missing {provider}"

    for provider in secrets.NO_KEY_NEEDED:
        assert f"`{provider}`" in text, f"README does not mention {provider}"

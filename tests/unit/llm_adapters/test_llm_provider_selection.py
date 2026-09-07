# © Artur Czarnecki. All rights reserved.

"""Explicit LLM provider selection contract tests (NPSC-3B-R3V-R5)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.llm_resolver import (
    resolve_environment_llm_adapter,
    resolve_llm_adapter,
    resolve_optional_llm_adapter,
    resolve_optional_llm_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterDependencyError,
    LLMProviderNotConfiguredError,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.catalog_capabilities import unwrap_catalog_capability_adapter
from intergrax.llm_adapters.providers.openai_compat_providers import GroqChatAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _StubCustomEnvAdapter(LLMAdapter):
    provider = "custom_gateway_env"
    model = "custom-model"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.model = str(kwargs.get("model", self.model))

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response

        return build_adapter_response(content="ok")


@pytest.fixture()
def _restore_registry_state():
    snapshot = dict(LLMAdapterRegistry._factories)
    try:
        yield snapshot
    finally:
        LLMAdapterRegistry._factories = snapshot


def test_llm_profile_from_env_absent_provider_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_MODEL", raising=False)
    assert llm_profile_from_env() is None


def test_llm_profile_from_env_explicit_ollama() -> None:
    with patch.dict("os.environ", {"INTERGRAX_LLM_PROVIDER": "ollama"}, clear=False):
        profile = llm_profile_from_env()
    assert profile is not None
    assert profile.provider == LLMProvider.OLLAMA


def test_llm_profile_from_env_explicit_non_ollama_does_not_touch_ollama_factory(
    _restore_registry_state,
) -> None:
    LLMAdapterRegistry.register(
        "custom_gateway_env",
        lambda **kwargs: _StubCustomEnvAdapter(**kwargs),
        override=True,
    )
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_LLM_PROVIDER": "custom_gateway_env",
            "INTERGRAX_LLM_MODEL": "custom-model",
        },
        clear=False,
    ):
        profile = llm_profile_from_env()
    assert profile is not None
    adapter = profile.create_adapter()
    assert isinstance(adapter, LLMAdapter)
    assert adapter.provider == "custom_gateway_env"


def test_mandatory_resolver_without_provider_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="test.no-llm")
    with pytest.raises(LLMProviderNotConfiguredError, match="not explicitly configured"):
        resolve_llm_adapter(env)


def test_optional_resolver_without_provider_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="test.no-llm")
    assert resolve_optional_llm_profile(env) is None
    assert resolve_optional_llm_adapter(env) is None


def test_llm_profile_lab_remains_explicit_ollama() -> None:
    profile = LLMProfile.lab()
    assert profile.provider == LLMProvider.OLLAMA
    assert profile.model == "llama3.1:latest"


def test_explicit_ollama_missing_sdk_raises_dependency_error() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="test.ollama")
    env = env.model_copy(update={"llm_profile": LLMProfile.lab()})
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMAdapterDependencyError, match="provider 'ollama'"):
            resolve_environment_llm_adapter(env)


def test_explicit_groq_does_not_import_ollama() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
    with patch.dict("os.environ", {"GROQ_API_KEY": "k"}, clear=False):
        adapter = profile.create_adapter(client=MagicMock())
    inner = unwrap_catalog_capability_adapter(adapter)
    assert isinstance(inner, GroqChatAdapter)

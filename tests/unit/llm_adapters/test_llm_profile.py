# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.registry.catalog_capabilities import unwrap_catalog_capability_adapter
from intergrax.llm_adapters.providers.openai_compat_providers import GroqChatAdapter


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


def test_llm_profile_create_adapter() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile", options={"max_retries": 1})
    with patch.dict("os.environ", {"GROQ_API_KEY": "k"}, clear=False):
        adapter = profile.create_adapter(client=MagicMock())
    inner = unwrap_catalog_capability_adapter(adapter)
    assert isinstance(inner, GroqChatAdapter)
    assert adapter.model == "llama-3.3-70b-versatile"


def test_llm_profile_from_env_builtin_provider() -> None:
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_LLM_PROVIDER": "openrouter",
            "INTERGRAX_LLM_MODEL": "openai/gpt-4o-mini",
        },
        clear=False,
    ):
        profile = llm_profile_from_env()
    assert profile.provider == LLMProvider.OPENROUTER
    assert profile.model == "openai/gpt-4o-mini"


def test_llm_profile_from_env_registered_custom_provider(_restore_registry_state) -> None:
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
    assert profile.provider == "custom_gateway_env"
    assert profile.model == "custom-model"
    adapter = profile.create_adapter()
    assert isinstance(adapter, LLMAdapter)
    assert adapter.provider == "custom_gateway_env"


def test_llm_profile_from_env_unknown_provider(_restore_registry_state) -> None:
    with patch.dict(
        "os.environ",
        {"INTERGRAX_LLM_PROVIDER": "not_registered_slug_env"},
        clear=False,
    ):
        with pytest.raises(ValueError, match="unknown LLM provider slug"):
            llm_profile_from_env()


def test_llm_profile_lab_default() -> None:
    profile = LLMProfile.lab()
    assert profile.provider == LLMProvider.OLLAMA

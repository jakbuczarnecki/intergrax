# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.providers.openai_compat_providers import GroqChatAdapter


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
    assert isinstance(adapter, GroqChatAdapter)
    assert adapter.model == "llama-3.3-70b-versatile"


def test_llm_profile_from_env() -> None:
    with patch.dict(
        "os.environ",
        {"INTERGRAX_LLM_PROVIDER": "vllm", "INTERGRAX_LLM_MODEL": "local-model"},
        clear=False,
    ):
        profile = llm_profile_from_env()
    assert profile.provider == LLMProvider.VLLM
    assert profile.model == "local-model"


def test_llm_profile_lab_default() -> None:
    profile = LLMProfile.lab()
    assert profile.provider == LLMProvider.OLLAMA

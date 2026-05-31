# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry, _BUILTIN_ADAPTERS


def test_all_llm_provider_enum_values_are_registered() -> None:
    registered = set(LLMAdapterRegistry.registered_providers())
    for provider in LLMProvider:
        assert provider.value in registered, f"missing builtin adapter for {provider.value}"
        assert provider.value in _BUILTIN_ADAPTERS


def test_registered_provider_count() -> None:
    assert len(LLMAdapterRegistry.registered_providers()) == 19
    assert len(LLMProvider) == 19

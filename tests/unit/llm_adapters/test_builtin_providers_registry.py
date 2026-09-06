# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry


def test_all_llm_provider_enum_values_are_registered() -> None:
    registered = set(LLMAdapterRegistry.registered_providers())
    enum_values = {provider.value for provider in LLMProvider}
    assert enum_values == registered


def test_registered_provider_count() -> None:
    assert len(LLMAdapterRegistry.registered_providers()) == 19
    assert len(LLMProvider) == 19

# © Artur Czarnecki. All rights reserved.

"""Autouse env for agent distribution tests that compose Research hosts."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from testing_support.builder import FakeLLMAdapter


@pytest.fixture(autouse=True)
def _research_explicit_llm_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = dict(LLMAdapterRegistry._factories)
    LLMAdapterRegistry.register(
        LLMProvider.GROQ,
        lambda **_kwargs: FakeLLMAdapter(),
        override=True,
    )
    monkeypatch.setenv("RESEARCH_LLM_PROVIDER", "groq")
    monkeypatch.setenv("RESEARCH_LLM_MODEL", "llama-3.3-70b-versatile")
    yield
    LLMAdapterRegistry._factories = snapshot

# © Artur Czarnecki. All rights reserved.

"""Explicit Research LLM test support (NPSC-3B-R3V-R7C)."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from testing_support.builder import FakeLLMAdapter

_RESEARCH_TEST_LLM_PROVIDER = "groq"
_RESEARCH_TEST_LLM_MODEL = "llama-3.3-70b-versatile"


@pytest.fixture
def configured_research_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure explicit Research LLM provider/model with a deterministic fake adapter."""
    snapshot = dict(LLMAdapterRegistry._factories)
    LLMAdapterRegistry.register(
        LLMProvider.GROQ,
        lambda **_kwargs: FakeLLMAdapter(),
        override=True,
    )
    monkeypatch.setenv("RESEARCH_LLM_PROVIDER", _RESEARCH_TEST_LLM_PROVIDER)
    monkeypatch.setenv("RESEARCH_LLM_MODEL", _RESEARCH_TEST_LLM_MODEL)
    yield
    LLMAdapterRegistry._factories = snapshot

# © Artur Czarnecki. All rights reserved.

"""Autouse env for diagnostic Problem list cursor secret in application unit tests."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from testing_support.builder import FakeLLMAdapter

_CURSOR_SECRET_ENV = "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"
_CURSOR_SECRET_VALUE = "unit-test-diagnostic-problem-list-cursor-secret"


@pytest.fixture(autouse=True)
def _diagnostic_problem_list_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = dict(LLMAdapterRegistry._factories)
    LLMAdapterRegistry.register(
        LLMProvider.GROQ,
        lambda **_kwargs: FakeLLMAdapter(),
        override=True,
    )
    monkeypatch.setenv(_CURSOR_SECRET_ENV, _CURSOR_SECRET_VALUE)
    monkeypatch.setenv("RESEARCH_LLM_PROVIDER", "groq")
    monkeypatch.setenv("RESEARCH_LLM_MODEL", "llama-3.3-70b-versatile")
    yield
    LLMAdapterRegistry._factories = snapshot

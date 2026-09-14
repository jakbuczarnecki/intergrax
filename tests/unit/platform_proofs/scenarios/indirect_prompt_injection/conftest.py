"""Pytest fixtures for IPI E2E qualification runs."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


@pytest.fixture
def patch_scenario_llm(monkeypatch: pytest.MonkeyPatch):
    def _apply(adapter: LLMAdapter) -> LLMAdapter:
        monkeypatch.setattr(
            "intergrax.applications._shared.environment_wiring.resolve_optional_environment_llm_adapter",
            lambda _env, **_: adapter,
        )
        monkeypatch.setattr(
            "intergrax.applications._shared.scenario_runtime_baseline.resolve_environment_llm_adapter",
            lambda _env, **_: adapter,
        )
        return adapter

    return _apply

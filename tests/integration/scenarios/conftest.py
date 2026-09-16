# © Artur Czarnecki. All rights reserved.

"""Shared offline doubles for scenario integration qualification."""

from __future__ import annotations

import pytest

from testing_support.builder import FakeLLMAdapter


@pytest.fixture(autouse=True)
def _patch_scenario_llm_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeLLMAdapter(fixed_text="ok")

    def _fake_resolve(*_args, agent_override=None, **_kwargs):
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_optional_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_optional_environment_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.runtime_config_bridge.resolve_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition.resolve_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition.resolve_llm_adapter",
        _fake_resolve,
    )

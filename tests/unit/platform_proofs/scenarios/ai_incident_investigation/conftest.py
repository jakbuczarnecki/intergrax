# © Artur Czarnecki. All rights reserved.

"""Test boundary doubles for scenario runtime composition (APP-1)."""

from __future__ import annotations

import pytest

from testing_support.builder import FakeLLMAdapter


@pytest.fixture(autouse=True)
def _patch_scenario_llm_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scenario tests offline while canonical code uses production resolver paths."""

    def _fake_resolve(*_args, agent_override=None, **_kwargs):
        if agent_override is not None:
            return agent_override
        return FakeLLMAdapter(fixed_text="investigate")

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.runtime_composition.resolve_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.runtime_config_bridge.resolve_llm_adapter",
        _fake_resolve,
    )

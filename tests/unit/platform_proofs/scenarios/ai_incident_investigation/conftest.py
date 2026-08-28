# © Artur Czarnecki. All rights reserved.

"""Test boundary doubles for scenario runtime composition (APP-1 / APP-2A)."""

from __future__ import annotations

import pytest

from tests.unit.platform_proofs.scenarios.ai_incident_investigation.planner_doubles import (
    ScriptedIncidentInvestigationLLM,
)


@pytest.fixture(autouse=True)
def _cleanup_bound_execution_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset identity bindings left open by direct run_step / gather_incident_evidence tests."""
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity as _bind_active_execution_identity,
        reset_active_execution_identity,
    )

    bound_tokens: list[object] = []

    def _tracking_bind(**kwargs: object) -> object:
        token = _bind_active_execution_identity(**kwargs)
        bound_tokens.append(token)
        return token

    monkeypatch.setattr(
        "intergrax.contracts.execution_identity.bind_active_execution_identity",
        _tracking_bind,
    )
    yield
    from intergrax.contracts.execution_identity import peek_active_execution_identity

    for token in reversed(bound_tokens):
        if peek_active_execution_identity() is None:
            continue
        reset_active_execution_identity(token)


@pytest.fixture(autouse=True)
def _patch_scenario_llm_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scenario tests offline while canonical code uses production resolver paths."""

    monkeypatch.delenv("SCENARIO_AI_INCIDENT_LAB_PLANNER", raising=False)

    def _fake_resolve(*_args, agent_override=None, **_kwargs):
        if agent_override is not None:
            return agent_override
        return ScriptedIncidentInvestigationLLM()

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition.resolve_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.runtime_config_bridge.resolve_llm_adapter",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _fake_resolve,
    )

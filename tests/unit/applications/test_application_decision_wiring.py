# © Artur Czarnecki. All rights reserved.

"""Application Decision wiring unit tests (DS-MIG-02 / DS-MIG-05)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.decision_wiring import (
    DEFAULT_APPLICATION_DECISION_WIRING_SPEC,
    ApplicationDecisionWiringSpec,
    application_decision_wiring_spec,
    application_decision_wiring_spec_from_environment,
    resolve_application_decision_agent_id,
    wire_application_decision,
    wire_application_decision_flow,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DecisionFlowProfile,
    DecisionProfile,
)
from intergrax.runtime.decision_flow import CanonicalDecisionFlowGate, DecisionFlowScope
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DECISION_WIRING_PATH = _REPO_ROOT / "intergrax/applications/_shared/decision_wiring.py"
_FORBIDDEN_DECISION_WIRING_TOKENS = frozenset(
    {
        "CriticProfile",
        "critic_profile",
        "resolve_critic_wiring_options",
        "evaluator_loop_max_iterations",
        "critic_runtime_bridge",
        "critic_wiring",
    },
)


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def test_default_spec_posture() -> None:
    spec = ApplicationDecisionWiringSpec()
    assert spec.verify_graph_final is True
    assert spec.verify_uaep_step is False
    assert spec.max_revisions == 0
    assert DEFAULT_APPLICATION_DECISION_WIRING_SPEC == spec


def test_explicit_uaep_spec_enables_uaep_scope() -> None:
    registry = _echo_registry()
    spec = application_decision_wiring_spec(verify_graph_final=False, verify_uaep_step=True)
    wiring = wire_application_decision(
        registry=registry,
        agent_id="echo",
        spec=spec,
    )
    gate = wiring.gate
    assert isinstance(gate, CanonicalDecisionFlowGate)
    assert gate.supports_scope(DecisionFlowScope.UAEP_STEP)
    assert not gate.supports_scope(DecisionFlowScope.GRAPH_FINAL)


def test_explicit_revision_budget_maps_to_revision_policy() -> None:
    registry = _echo_registry()
    wiring = wire_application_decision_flow(
        registry=registry,
        agent_id="echo",
        max_revisions=2,
    )
    gate = wiring.gate
    assert isinstance(gate, CanonicalDecisionFlowGate)
    assert gate.capabilities.revision_policy.max_revisions == 2


def test_application_decision_wiring_spec_rejects_negative_revisions() -> None:
    with pytest.raises(ValueError, match="max_revisions"):
        application_decision_wiring_spec(max_revisions=-1)


def test_application_decision_wiring_spec_requires_supported_scope() -> None:
    with pytest.raises(ValueError, match="at least one supported scope"):
        application_decision_wiring_spec(
            verify_graph_final=False,
            verify_uaep_step=False,
        )


def test_environment_decision_profile_drives_wiring_spec() -> None:
    registry = _echo_registry()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="decision.profile").model_copy(
        update={
            "decision_profile": DecisionProfile(
                flow=DecisionFlowProfile(
                    verify_graph_final=False,
                    verify_uaep_step=True,
                    max_revisions=3,
                ),
            ),
        },
    )
    spec = application_decision_wiring_spec_from_environment(env)
    wiring = wire_application_decision(
        registry=registry,
        agent_id=resolve_application_decision_agent_id(registry, env),
        spec=spec,
    )
    gate = wiring.gate
    assert isinstance(gate, CanonicalDecisionFlowGate)
    assert gate.supports_scope(DecisionFlowScope.UAEP_STEP)
    assert not gate.supports_scope(DecisionFlowScope.GRAPH_FINAL)
    assert gate.capabilities.revision_policy.max_revisions == 3


def test_decision_wiring_source_has_no_critic_authority_tokens() -> None:
    source = _DECISION_WIRING_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_DECISION_WIRING_PATH))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    for token in _FORBIDDEN_DECISION_WIRING_TOKENS:
        assert token not in source

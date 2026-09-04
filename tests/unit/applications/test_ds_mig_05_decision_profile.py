# © Artur Czarnecki. All rights reserved.

"""DS-MIG-05 — canonical Decision application profile migration gates."""

from __future__ import annotations

import json

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.decision_verifier_llm_resolver import (
    resolve_decision_verifier_llm_adapter,
)
from intergrax.applications._shared.decision_wiring import (
    application_decision_wiring_spec_from_profile,
    wire_application_decision_flow,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DecisionFlowProfile,
    DecisionProfile,
    DecisionVerificationProfile,
)
from intergrax.applications.contracts.environment_profile.bundles import CognitionBundle
from intergrax.applications.contracts.environment_profile.decision_profile_legacy import (
    migrate_legacy_critic_payload_to_decision,
)
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.decision_flow import CanonicalDecisionFlowGate
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


def test_default_decision_profile_validates() -> None:
    profile = DecisionProfile()
    assert profile.verification.semantic_enabled is False
    assert profile.flow.verify_graph_final is True
    assert profile.flow.max_revisions == 0


def test_regulated_preset_validates() -> None:
    cognition = CognitionBundle.regulated()
    assert cognition.decision.verification.semantic_enabled is True
    assert cognition.decision.flow.verify_graph_final is True
    assert "critic" not in cognition.model_dump(mode="json")


def test_canonical_v2_serialization_contains_no_critic() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ds-mig-05.v2").with_spec_v2_wire()
    wire = env.model_dump(mode="json")
    cognition = wire["cognition"]
    assert "critic" not in cognition
    assert "decision" in cognition
    flat = json.dumps(wire)
    for token in ("critic_profile", "critic_llm", "l2_human_required", '"critic"'):
        assert token not in flat


def test_safe_legacy_input_migrates_to_decision_profile() -> None:
    env = ApplicationEnvironmentProfile.model_validate(
        {
            "profile_id": "legacy.safe",
            "critic_profile": {
                "semantic_judge_enabled": True,
                "trajectory_eval_enabled": True,
                "scopes": {"graph_final": True, "uaep_step": True},
            },
        },
    )
    assert env.decision_profile.verification.semantic_enabled is True
    assert env.decision_profile.verification.trajectory_enabled is True
    assert env.decision_profile.flow.verify_graph_final is True
    assert env.decision_profile.flow.verify_uaep_step is True


def test_unsafe_legacy_l2_human_required_fails_migration() -> None:
    with pytest.raises(ValueError, match="l2_human_required"):
        ApplicationEnvironmentProfile.model_validate(
            {
                "profile_id": "legacy.unsafe.l2",
                "critic_profile": {"l2_human_required": True},
            },
        )


def test_unsafe_legacy_evaluator_loop_fails_migration() -> None:
    with pytest.raises(ValueError, match="evaluator_loop_max_iterations"):
        migrate_legacy_critic_payload_to_decision({"evaluator_loop_max_iterations": 10})
    with pytest.raises(ValueError, match="evaluator_loop_max_iterations"):
        ApplicationEnvironmentProfile.model_validate(
            {
                "profile_id": "legacy.unsafe.loop",
                "critic_profile": {"evaluator_loop_max_iterations": 10},
            },
        )


def test_decision_profile_flow_maps_to_application_decision_wiring_spec() -> None:
    profile = DecisionProfile(
        flow=DecisionFlowProfile(
            verify_graph_final=False,
            verify_uaep_step=True,
            max_revisions=2,
        ),
    )
    spec = application_decision_wiring_spec_from_profile(profile)
    assert spec.verify_graph_final is False
    assert spec.verify_uaep_step is True
    assert spec.max_revisions == 2


def test_independent_verifier_llm_resolves_separate_adapter() -> None:
    producer = FakeLLMAdapter()
    verifier_profile = LLMProfile.lab()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="verifier.split")
    env.decision_profile = DecisionProfile(
        verification=DecisionVerificationProfile(
            semantic_enabled=True,
            verifier_llm_profile=verifier_profile,
        ),
    )
    resolved = resolve_decision_verifier_llm_adapter(
        env,
        producer_adapter=producer,
    )
    assert resolved is not producer


def test_wire_application_decision_flow_uses_profile_revision_budget() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    wiring = wire_application_decision_flow(
        registry=registry,
        agent_id="echo",
        max_revisions=2,
    )
    gate = wiring.gate
    assert isinstance(gate, CanonicalDecisionFlowGate)
    assert gate.capabilities.revision_policy.max_revisions == 2


def test_regulated_environment_has_semantic_decision_verification_enabled() -> None:
    env = ApplicationEnvironmentProfile.strict_multi_agent_defaults()
    assert env.decision_profile.verification.semantic_enabled is True
    assert env.decision_profile.flow.verify_graph_final is True
    assert env.spec_version.startswith("1.")

# © Artur Czarnecki. All rights reserved.

"""DS-MIG-02 production retirement boundary tests."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.critic_wiring import (
    apply_application_critic_wiring,
    wire_application_critic,
)
from intergrax.applications._shared.decision_wiring import (
    apply_application_decision_wiring,
    wire_application_decision_from_environment,
)
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.builder import FakeLLMAdapter
from echo.echo_agent import EchoAgent

pytestmark = pytest.mark.unit


def test_application_critic_wiring_is_inert() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="retire.critic")
    wiring = wire_application_critic(env)
    nexus = NexusLoop(AgentRegistry())
    apply_application_critic_wiring(nexus, wiring)
    assert nexus.peek_decision_flow_gate() is None


def test_harness_nexus_composes_with_decision_not_critic_authority() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="retire.decision")
    decision_wiring = wire_application_decision_from_environment(registry, env)
    assert decision_wiring is not None
    nexus = build_nexus_loop_from_environment(
        registry,
        env=env,
        decision_wiring=decision_wiring,
        llm_adapter=FakeLLMAdapter(),
    )
    apply_application_decision_wiring(nexus, decision_wiring)
    assert nexus.peek_decision_flow_gate() is decision_wiring.gate

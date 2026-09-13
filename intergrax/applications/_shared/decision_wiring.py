# © Artur Czarnecki. All rights reserved.

"""Tier-3 Decision flow wiring (DS-MIG-01 / DS-MIG-02 / P0-A)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.application_decision_composition import (
    ApplicationDecisionComposition,
    ApplicationDecisionWiringSpec,
    compose_application_decision,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import DecisionProfile
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.decision_flow import DecisionFlowGate
from intergrax.runtime.decision_verification_composition import ToolWiringEvalVerificationBridge
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


def application_decision_wiring_spec(
    *,
    verify_graph_final: bool = True,
    verify_uaep_step: bool = False,
    max_revisions: int = 0,
) -> ApplicationDecisionWiringSpec:
    """Build and validate one immutable Decision wiring spec."""
    if max_revisions < 0:
        raise ValueError("ApplicationDecisionWiringSpec.max_revisions must be >= 0")
    if not verify_graph_final and not verify_uaep_step:
        raise ValueError(
            "ApplicationDecisionWiringSpec requires at least one supported scope",
        )
    return ApplicationDecisionWiringSpec(
        verify_graph_final=verify_graph_final,
        verify_uaep_step=verify_uaep_step,
        max_revisions=max_revisions,
    )


DEFAULT_APPLICATION_DECISION_WIRING_SPEC = application_decision_wiring_spec()


def application_decision_wiring_spec_from_profile(
    profile: DecisionProfile,
) -> ApplicationDecisionWiringSpec:
    """Translate host ``DecisionProfile`` into immutable Decision wiring spec."""
    return application_decision_wiring_spec(
        verify_graph_final=profile.flow.verify_graph_final,
        verify_uaep_step=profile.flow.verify_uaep_step,
        max_revisions=profile.flow.max_revisions,
    )


def application_decision_wiring_spec_from_environment(
    env: ApplicationEnvironmentProfile,
) -> ApplicationDecisionWiringSpec:
    """Resolve Decision wiring spec from the host environment profile."""
    return application_decision_wiring_spec_from_profile(env.decision_profile)


@dataclass(frozen=True, slots=True)
class ApplicationDecisionWiring:
    """Resolved Decision flow artifacts for a Tier-3 host."""

    gate: DecisionFlowGate[AgentExecutionResult]
    verify_graph_final: bool
    verify_uaep_step: bool
    composition: ApplicationDecisionComposition | None = None


def resolve_application_decision_agent_id(
    registry: AgentRegistryRead,
    env: ApplicationEnvironmentProfile,
) -> str:
    """Resolve the primary agent id used for Decision verification pipeline wiring."""
    graph_spec = env.graph_spec
    if graph_spec is not None and graph_spec.nodes:
        return graph_spec.nodes[0].agent_id
    agent_ids = registry.list_agent_ids()
    if not agent_ids:
        raise ValueError("registry must contain at least one agent for decision wiring")
    return agent_ids[0]


def wire_application_decision(
    *,
    registry: AgentRegistryRead,
    agent_id: str,
    spec: ApplicationDecisionWiringSpec,
    environment: ApplicationEnvironmentProfile,
    capability: str | None = None,
    eval_bridge: ToolWiringEvalVerificationBridge | None = None,
) -> ApplicationDecisionWiring:
    """Materialize one reusable Decision flow gate from explicit composition spec."""
    contract = registry.get_contract(agent_id)
    composed = compose_application_decision(
        environment=environment,
        contract=contract,
        spec=spec,
        capability=capability,
        eval_bridge=eval_bridge,
    )
    return ApplicationDecisionWiring(
        gate=composed.gate,
        verify_graph_final=composed.verify_graph_final,
        verify_uaep_step=composed.verify_uaep_step,
        composition=composed,
    )


def wire_application_decision_flow(
    *,
    registry: AgentRegistryRead,
    agent_id: str,
    environment: ApplicationEnvironmentProfile,
    capability: str | None = None,
    verify_graph_final: bool = True,
    verify_uaep_step: bool = False,
    max_revisions: int = 0,
    eval_bridge: ToolWiringEvalVerificationBridge | None = None,
) -> ApplicationDecisionWiring:
    """Materialize Decision flow wiring from explicit scope and revision flags."""
    spec = application_decision_wiring_spec(
        verify_graph_final=verify_graph_final,
        verify_uaep_step=verify_uaep_step,
        max_revisions=max_revisions,
    )
    return wire_application_decision(
        registry=registry,
        agent_id=agent_id,
        spec=spec,
        environment=environment,
        capability=capability,
        eval_bridge=eval_bridge,
    )


def apply_application_decision_wiring(
    nexus: NexusLoop,
    wiring: ApplicationDecisionWiring,
) -> None:
    """Attach resolved Decision flow gate to an existing ``NexusLoop`` instance."""
    nexus.apply_decision_flow_gate(
        wiring.gate,
        verify_uaep_step=wiring.verify_uaep_step,
    )

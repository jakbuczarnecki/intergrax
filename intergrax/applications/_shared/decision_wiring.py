# © Artur Czarnecki. All rights reserved.

"""Tier-3 Decision flow wiring (DS-MIG-01 / DS-MIG-02)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import build_agent_execution_verification_pipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry


@dataclass(frozen=True, slots=True)
class ApplicationDecisionWiringSpec:
    """Explicit application-composition contract for canonical Decision wiring."""

    verify_graph_final: bool = True
    verify_uaep_step: bool = False
    max_revisions: int = 0


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


@dataclass(frozen=True, slots=True)
class ApplicationDecisionWiring:
    """Resolved Decision flow artifacts for a Tier-3 host."""

    gate: DecisionFlowGate[AgentExecutionResult]
    verify_graph_final: bool
    verify_uaep_step: bool


def resolve_application_decision_agent_id(
    registry: AgentRegistry,
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
    registry: AgentRegistry,
    agent_id: str,
    spec: ApplicationDecisionWiringSpec,
    capability: str | None = None,
) -> ApplicationDecisionWiring:
    """Materialize one reusable Decision flow gate from explicit composition spec."""
    contract = registry.get_contract(agent_id)
    scopes: set[DecisionFlowScope] = set()
    if spec.verify_graph_final:
        scopes.add(DecisionFlowScope.GRAPH_FINAL)
    if spec.verify_uaep_step:
        scopes.add(DecisionFlowScope.UAEP_STEP)
    pipeline = build_agent_execution_verification_pipeline(
        contract=contract,
        capability=capability,
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=pipeline,
            revision_policy=decision_revision_policy(max_revisions=spec.max_revisions),
            scopes=frozenset(scopes),
        ),
    )
    return ApplicationDecisionWiring(
        gate=gate,
        verify_graph_final=spec.verify_graph_final,
        verify_uaep_step=spec.verify_uaep_step,
    )


def wire_application_decision_flow(
    *,
    registry: AgentRegistry,
    agent_id: str,
    capability: str | None = None,
    verify_graph_final: bool = True,
    verify_uaep_step: bool = False,
    max_revisions: int = 0,
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
        capability=capability,
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

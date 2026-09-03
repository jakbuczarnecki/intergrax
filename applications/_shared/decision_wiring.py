# © Artur Czarnecki. All rights reserved.

"""Tier-3 Decision flow wiring (DS-MIG-01)."""

from __future__ import annotations

from dataclasses import dataclass

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
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry


@dataclass(frozen=True, slots=True)
class ApplicationDecisionWiring:
    """Resolved Decision flow artifacts for a Tier-3 host."""

    gate: DecisionFlowGate[AgentExecutionResult]
    verify_graph_final: bool
    verify_uaep_step: bool


def wire_application_decision_flow(
    *,
    registry: AgentRegistry,
    agent_id: str,
    capability: str | None = None,
    verify_graph_final: bool = True,
    verify_uaep_step: bool = False,
    max_revisions: int = 0,
    validation_engine: NexusValidationEngine | None = None,
) -> ApplicationDecisionWiring:
    """Materialize one reusable Decision flow gate for Graph and UAEP hosts."""
    contract = registry.get_contract(agent_id)
    scopes: set[DecisionFlowScope] = set()
    if verify_graph_final:
        scopes.add(DecisionFlowScope.GRAPH_FINAL)
    if verify_uaep_step:
        scopes.add(DecisionFlowScope.UAEP_STEP)
    pipeline = build_agent_execution_verification_pipeline(
        contract=contract,
        capability=capability,
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=pipeline,
            revision_policy=decision_revision_policy(max_revisions=max_revisions),
            scopes=frozenset(scopes),
        ),
    )
    del validation_engine
    return ApplicationDecisionWiring(
        gate=gate,
        verify_graph_final=verify_graph_final,
        verify_uaep_step=verify_uaep_step,
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

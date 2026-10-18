# © Artur Czarnecki. All rights reserved.

"""Typed host dependency injection after authoritative AC3 factory materialization (R3-F01)."""

from __future__ import annotations

from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkOperationPort,
)
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)


class StrictExternalWorkHostAgentInjectionError(TypeError):
    """Authoritative materialization cannot receive strict host injection."""


def materialize_with_strict_external_work_host_dependencies(
    agent: Tier2Agent,
    *,
    external_work: ExternalWorkIntegration,
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary | None,
    sanctioned_operation_port: SanctionedExternalWorkOperationPort,
    require_sanctioned_external_work_mutations: bool,
) -> ExternalContractorAdapterAgent:
    """Apply frozen host ports to an artifact-authorized external contractor agent."""
    if not isinstance(agent, ExternalContractorAdapterAgent):
        raise StrictExternalWorkHostAgentInjectionError(
            "strict external work host injection requires ExternalContractorAdapterAgent "
            f"from authoritative factory; got {type(agent).__name__}",
        )
    if not require_sanctioned_external_work_mutations:
        raise StrictExternalWorkHostAgentInjectionError(
            "strict external work host injection requires "
            "require_sanctioned_external_work_mutations=True",
        )
    return agent.inject_sanctioned_external_work_host_dependencies(
        external_work=external_work,
        authorization_boundary=authorization_boundary,
        sanctioned_operation_port=sanctioned_operation_port,
        require_sanctioned_external_work_mutations=True,
    )


__all__ = [
    "StrictExternalWorkHostAgentInjectionError",
    "materialize_with_strict_external_work_host_dependencies",
]

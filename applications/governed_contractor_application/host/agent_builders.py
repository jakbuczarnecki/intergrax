# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings


def build_external_contractor_adapter_factory(
    *,
    external_work: ExternalWorkIntegration | None,
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary | None,
) -> AgentFactory[GovernedContractorBackendSettings]:
    """Configured factory — captures host runtime ports (not settings)."""

    def _factory(
        ctx: ApplicationBuildContext[GovernedContractorBackendSettings],
        _binding: AgentBinding,
    ) -> Agent:
        _ = ctx
        return ExternalContractorAdapterAgent(
            external_work=external_work,
            authorization_boundary=authorization_boundary,
        )

    return _factory


def build_governed_contractor_agent_builders(
    runtime: GovernedContractorHostRuntimeComposition,
) -> dict[type[Agent], AgentFactory[GovernedContractorBackendSettings]]:
    external_factory = build_external_contractor_adapter_factory(
        external_work=runtime.external_work_integration,
        authorization_boundary=runtime.meaningful_side_effect_authorization_boundary,
    )
    return {
        ExternalContractorAdapterAgent: external_factory,
    }


GOVERNED_CONTRACTOR_AGENT_BUILDERS: dict[
    type[Agent],
    AgentFactory[GovernedContractorBackendSettings],
] = build_governed_contractor_agent_builders(GovernedContractorHostRuntimeComposition())

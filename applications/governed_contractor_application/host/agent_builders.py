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
from governed_contractor_application.host.settings import GovernedContractorBackendSettings


def _backend_settings(
    ctx: ApplicationBuildContext[GovernedContractorBackendSettings],
) -> GovernedContractorBackendSettings | None:
    return ctx.settings


def _external_work_from_context(
    ctx: ApplicationBuildContext[GovernedContractorBackendSettings],
) -> ExternalWorkIntegration | None:
    """Optional host injection via settings — Tier-2 never constructs providers."""
    settings = _backend_settings(ctx)
    if settings is None:
        return None
    return settings.external_work_integration


def _authorization_boundary_from_context(
    ctx: ApplicationBuildContext[GovernedContractorBackendSettings],
) -> MeaningfulSideEffectAuthorizationBoundary | None:
    """Optional host injection of canonical meaningful side-effect authorization."""
    settings = _backend_settings(ctx)
    if settings is None:
        return None
    return settings.meaningful_side_effect_authorization_boundary


def _build_external_contractor_adapter(
    ctx: ApplicationBuildContext[GovernedContractorBackendSettings],
    _binding: AgentBinding,
) -> Agent:
    return ExternalContractorAdapterAgent(
        external_work=_external_work_from_context(ctx),
        authorization_boundary=_authorization_boundary_from_context(ctx),
    )


GOVERNED_CONTRACTOR_AGENT_BUILDERS: dict[
    type[Agent],
    AgentFactory[GovernedContractorBackendSettings],
] = {
    ExternalContractorAdapterAgent: _build_external_contractor_adapter,
}

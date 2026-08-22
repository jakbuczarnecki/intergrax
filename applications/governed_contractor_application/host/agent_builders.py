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


def _external_work_from_context(ctx: ApplicationBuildContext) -> ExternalWorkIntegration | None:
    """Optional host injection via settings — Tier-2 never constructs providers."""
    settings = ctx.settings
    if settings is None:
        return None
    raw = getattr(settings, "external_work_integration", None)
    if raw is None:
        return None
    if not isinstance(raw, ExternalWorkIntegration):
        raise TypeError(
            "settings.external_work_integration must implement ExternalWorkIntegration"
        )
    return raw


def _authorization_boundary_from_context(
    ctx: ApplicationBuildContext,
) -> MeaningfulSideEffectAuthorizationBoundary | None:
    """Optional host injection of canonical meaningful side-effect authorization."""
    settings = ctx.settings
    if settings is None:
        return None
    raw = getattr(settings, "meaningful_side_effect_authorization_boundary", None)
    if raw is None:
        return None
    if not isinstance(raw, MeaningfulSideEffectAuthorizationBoundary):
        raise TypeError(
            "settings.meaningful_side_effect_authorization_boundary must be a "
            "MeaningfulSideEffectAuthorizationBoundary"
        )
    return raw


def _build_external_contractor_adapter(
    ctx: ApplicationBuildContext,
    _binding: AgentBinding,
) -> Agent:
    return ExternalContractorAdapterAgent(
        external_work=_external_work_from_context(ctx),
        authorization_boundary=_authorization_boundary_from_context(ctx),
    )


GOVERNED_CONTRACTOR_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    ExternalContractorAdapterAgent: _build_external_contractor_adapter,
}

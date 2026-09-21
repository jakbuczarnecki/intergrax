# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent
from governed_contractor_application.host.settings import GovernedContractorBackendSettings


def build_governed_contractor_external_contractor_adapter_from_context(
    ctx: ApplicationBuildContext[GovernedContractorBackendSettings],
    binding: AgentBinding,
) -> ExternalContractorAdapterAgent:
    """Manifest default factory — development scaffold without host runtime wiring."""
    _ = ctx, binding
    return ExternalContractorAdapterAgent(
        external_work=None,
        authorization_boundary=None,
    )

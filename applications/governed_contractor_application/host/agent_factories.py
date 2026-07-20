# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent
from governed_contractor_application.host.agent_builders import GOVERNED_CONTRACTOR_AGENT_BUILDERS


def build_governed_contractor_external_contractor_adapter_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> ExternalContractorAdapterAgent:
    _ = ctx, binding
    factory = GOVERNED_CONTRACTOR_AGENT_BUILDERS.get(ExternalContractorAdapterAgent)
    if factory is None:
        raise ValueError(f"No builder registered for {binding.import_path!r}")
    return factory(ctx, binding)

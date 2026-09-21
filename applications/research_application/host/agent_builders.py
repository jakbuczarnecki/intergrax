# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for the research application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.agents.reference_harness import LabHarnessContext
from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.tool_enablement_binding import resolve_tool_enablement
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from research_application.host.settings import ResearchBackendSettings


def build_research_agent_builders(
    *,
    tool_profile: ToolEnablementProfile | None = None,
    lab_harness: LabHarnessContext | None = None,
) -> dict[type[Agent], AgentFactory[ResearchBackendSettings]]:
    """Compose research builder map with host-bound harness / tool-profile deps."""

    def build_research_agent_from_context(
        ctx: ApplicationBuildContext[ResearchBackendSettings],
        binding: AgentBinding,
    ) -> ResearchAgent:
        _ = binding
        settings = ctx.settings
        enable_websearch = (
            settings.enable_websearch
            if settings is not None
            else ResearchBackendSettings().enable_websearch
        )
        environment = ctx.environment
        env_profile = environment.tool_profile if environment is not None else None
        resolved_profile = resolve_tool_enablement(
            tool_profile,
            environment_tool_profile=env_profile,
        )
        return ResearchAgent(
            lab_harness,
            tool_profile=resolved_profile,
            enable_websearch=enable_websearch,
        )

    def build_summary_agent_from_context(
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> SummaryAgent:
        _ = ctx, binding
        return SummaryAgent()

    return {
        ResearchAgent: build_research_agent_from_context,
        SummaryAgent: build_summary_agent_from_context,
    }


# Unbound defaults for tests / callers that do not yet wire composition deps.
RESEARCH_AGENT_BUILDERS: dict[type[Agent], AgentFactory[ResearchBackendSettings]] = (
    build_research_agent_builders()
)

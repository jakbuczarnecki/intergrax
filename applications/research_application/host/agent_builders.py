# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for the research application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.application_composition_context import (
    optional_factory_composition,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from research_application.host.settings import ResearchBackendSettings


def build_research_agent_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> ResearchAgent:
    _ = binding
    settings = ctx.settings
    enable_websearch = (
        settings.enable_websearch
        if isinstance(settings, ResearchBackendSettings)
        else ResearchBackendSettings().enable_websearch
    )
    composition = optional_factory_composition()
    environment = ctx.environment
    tool_profile = (
        composition.tool_profile
        if composition is not None and composition.tool_profile is not None
        else (environment.tool_profile if environment is not None else None)
    )
    tool_wiring_context = (
        composition.tool_wiring_context if composition is not None else None
    )
    return ResearchAgent(
        tool_profile=tool_profile,
        tool_wiring_context=tool_wiring_context,
        enable_websearch=enable_websearch,
    )


def build_summary_agent_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> SummaryAgent:
    _ = ctx, binding
    return SummaryAgent()


RESEARCH_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    ResearchAgent: build_research_agent_from_context,
    SummaryAgent: build_summary_agent_from_context,
}

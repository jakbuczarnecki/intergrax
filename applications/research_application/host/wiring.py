# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.registry.agent_registry import AgentRegistry
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.settings import ResearchBackendSettings
from research_application.host.tool_wiring import wire_research_tools
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST


def build_research_registry(
    *,
    settings: ResearchBackendSettings | None = None,
) -> AgentRegistry:
    """Compose research + summary agents via unified Tier-3 wiring."""
    settings = settings or ResearchBackendSettings.from_env()
    tool_wiring = wire_research_tools(settings=settings)
    ctx = ApplicationBuildContext.for_manifest(
        RESEARCH_APPLICATION_MANIFEST,
        settings=settings,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
    )
    return build_application_registry(
        RESEARCH_APPLICATION_MANIFEST,
        ctx,
        builders=RESEARCH_AGENT_BUILDERS,
    )


def build_research_registry_for_host() -> AgentRegistry:
    """Backward-compatible alias for existing imports."""
    return build_research_registry()

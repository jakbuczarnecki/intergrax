# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for research_application."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile
from research_application.host.settings import ResearchBackendSettings


def wire_research_tools(
    *,
    settings: ResearchBackendSettings,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    """Research prototype — web search catalog tool when enabled in settings."""
    enabled = list(settings.enabled_tool_ids)
    profile = ToolProfile(enabled=enabled) if enabled else ToolProfile()
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
        websearch_executor=settings.websearch_executor,
    )
# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for research_application."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from research_application.host.settings import ResearchBackendSettings

_RESEARCH_SKILL_TOOL_IDS: tuple[str, ...] = RESEARCH_LITERATURE_SCAN.tool_ids


def wire_research_tools(
    *,
    settings: ResearchBackendSettings,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    """Research host — settings-driven tools plus ids required by ``research.literature_scan``."""
    enabled = list(settings.enabled_tool_ids)
    for tool_id in _RESEARCH_SKILL_TOOL_IDS:
        if tool_id not in enabled:
            enabled.append(tool_id)
    profile = ToolProfile(enabled=enabled)
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
        websearch_executor=settings.websearch_executor,
    )
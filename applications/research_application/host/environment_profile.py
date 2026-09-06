# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment profile for research_application."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from research_application.host.settings import ResearchBackendSettings
from research_application.host.skill_wiring import (
    RESEARCH_BUNDLE_ID,
    build_research_skill_profile,
)


def build_research_environment_profile(
    settings: ResearchBackendSettings | None = None,
) -> ApplicationEnvironmentProfile:
    """Product environment for research host (DX-1.4)."""
    settings = settings or ResearchBackendSettings.from_env()
    enabled_tools = list(settings.enabled_tool_ids)
    for tool_id in RESEARCH_LITERATURE_SCAN.tool_ids:
        if tool_id not in enabled_tools:
            enabled_tools.append(tool_id)
    return (
        ApplicationEnvironmentProfile.product_defaults(
            profile_id="research.product",
            skill_bundles=[RESEARCH_BUNDLE_ID],
            tool_ids=enabled_tools,
        )
        .model_copy(
            update={
                "integration_profile": IntegrationProfile.research_product(),
                "skill_profile": build_research_skill_profile(),
            },
        )
        .with_harness_memory()
        .with_reference_host_platform_defaults()
    )

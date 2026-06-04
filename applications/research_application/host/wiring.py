# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from research_application.host.settings import ResearchBackendSettings
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST


def build_research_environment_profile(
    settings: ResearchBackendSettings | None = None,
) -> ApplicationEnvironmentProfile:
    """Product environment for research host (DX-1.4)."""
    settings = settings or ResearchBackendSettings.from_env()
    manifest = RESEARCH_APPLICATION_MANIFEST
    enabled_tools = list(settings.enabled_tool_ids)
    for tool_id in RESEARCH_LITERATURE_SCAN.tool_ids:
        if tool_id not in enabled_tools:
            enabled_tools.append(tool_id)
    return ApplicationEnvironmentProfile.product_defaults(
        profile_id="research.product",
        skill_bundles=["research"],
        tool_ids=enabled_tools,
    ).model_copy(update={"integration_profile": manifest.integration_profile})


def build_research_registry(
    *,
    settings: ResearchBackendSettings | None = None,
) -> AgentRegistry:
    """Compose research + summary agents via unified Tier-3 wiring."""
    settings = settings or ResearchBackendSettings.from_env()
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(
        manifest,
        env,
        settings=settings,
        websearch_executor=settings.websearch_executor,
    )
    return build_application_registry(
        manifest,
        env_wiring.build_context,
        builders=RESEARCH_AGENT_BUILDERS,
    )


def build_research_registry_for_host() -> AgentRegistry:
    """Backward-compatible alias for existing imports."""
    return build_research_registry()

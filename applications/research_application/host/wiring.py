# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.runtime.registry.agent_registry import AgentRegistry
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.environment_profile import build_research_environment_profile
from research_application.host.settings import ResearchBackendSettings
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

__all__ = [
    "build_research_environment_profile",
    "build_research_registry",
    "build_research_registry_for_host",
]


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
    return build_manifest_development_registry(
        manifest,
        env_wiring.build_context,
        builders=RESEARCH_AGENT_BUILDERS,
    )


def build_research_registry_for_host() -> AgentRegistry:
    """Backward-compatible alias for existing imports."""
    return build_research_registry()

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax_assistant_application.host.agent_builders import INTERGRAX_ASSISTANT_AGENT_BUILDERS
from intergrax_assistant_application.host.environment_profile import build_intergrax_assistant_environment_profile
from intergrax_assistant_application.host.settings import IntergraxAssistantApplicationSettings
from intergrax_assistant_application.manifest import build_intergrax_assistant_manifest


def build_intergrax_assistant_registry(
    *,
    settings: IntergraxAssistantApplicationSettings | None = None,
) -> AgentRegistry:
    settings = settings or IntergraxAssistantApplicationSettings.from_env()
    manifest = build_intergrax_assistant_manifest(settings)
    env = manifest.environment or build_intergrax_assistant_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env, settings=settings)
    return build_manifest_development_registry(
        manifest,
        env_wiring.build_context,
        builders=INTERGRAX_ASSISTANT_AGENT_BUILDERS,
    )

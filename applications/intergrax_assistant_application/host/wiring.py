# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.lab_harness_context import lab_harness_context_from_build_context
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax_assistant_application.host.agent_builders import (
    build_intergrax_assistant_agent_builders,
)
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
    composition = env_wiring.composition
    builders = build_intergrax_assistant_agent_builders(
        tool_profile=composition.tool_profile,
        lab_harness=lab_harness_context_from_build_context(
            env_wiring.build_context,
            policy_bundle=composition.policy_bundle,
            tool_wiring_context=composition.tool_wiring_context,
            tool_registry=composition.tool_registry,
        ),
    )
    return build_manifest_development_registry(
        manifest,
        env_wiring.build_context,
        builders=builders,
        composition=composition,
    )

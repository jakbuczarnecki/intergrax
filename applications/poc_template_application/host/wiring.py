# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry
from poc_template_application.host.agent_builders import POC_TEMPLATE_AGENT_BUILDERS
from poc_template_application.host.settings import PocTemplateApplicationSettings
from poc_template_application.manifest import build_poc_template_manifest


def build_poc_template_registry(
    *,
    settings: PocTemplateApplicationSettings | None = None,
) -> AgentRegistry:
    settings = settings or PocTemplateApplicationSettings.from_env()
    manifest = build_poc_template_manifest()
    env = manifest.environment or ApplicationEnvironmentProfile.lab_defaults(
        profile_id="poc_template",
        harness_tools=False,
    ).with_reference_host_platform_defaults()
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env, settings=settings)
    return build_manifest_development_registry(
        manifest,
        env_wiring.build_context,
        builders=POC_TEMPLATE_AGENT_BUILDERS,
    )

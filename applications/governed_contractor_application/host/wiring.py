# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from governed_contractor_application.host.agent_builders import build_governed_contractor_agent_builders
from governed_contractor_application.host.environment_profile import build_governed_contractor_environment_profile
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import APPLICATION_MANIFEST


def build_governed_contractor_manifest(settings: GovernedContractorBackendSettings) -> ApplicationManifest:
    binding = APPLICATION_MANIFEST.agents[0].model_copy(
        update={"contract_id": settings.default_agent_id}
    )
    agents = [binding, *APPLICATION_MANIFEST.agents[1:]]
    return APPLICATION_MANIFEST.model_copy(update={"agents": agents})


def build_governed_contractor_registry(
    settings: GovernedContractorBackendSettings | None = None,
    *,
    host_runtime: GovernedContractorHostRuntimeComposition | None = None,
) -> AgentRegistry:
    settings = settings or GovernedContractorBackendSettings.from_env()
    resolved_runtime = host_runtime or GovernedContractorHostRuntimeComposition()
    manifest = build_governed_contractor_manifest(settings)
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env, settings=settings)
    return build_manifest_development_registry(
        manifest,
        env_wiring.build_context,
        builders=build_governed_contractor_agent_builders(resolved_runtime),
    )

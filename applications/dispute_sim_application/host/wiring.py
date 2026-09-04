# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from dispute_sim_application.host.agent_builders import DISPUTE_SIM_AGENT_BUILDERS
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from dispute_sim_application.manifest import DISPUTE_SIM_APPLICATION_MANIFEST


def build_dispute_sim_manifest(settings: DisputeSimBackendSettings) -> ApplicationManifest:
    default_idx = next(
        (i for i, b in enumerate(DISPUTE_SIM_APPLICATION_MANIFEST.agents) if b.default),
        0,
    )
    agents = list(DISPUTE_SIM_APPLICATION_MANIFEST.agents)
    agents[default_idx] = agents[default_idx].model_copy(
        update={"contract_id": settings.default_agent_id}
    )
    return DISPUTE_SIM_APPLICATION_MANIFEST.model_copy(update={"agents": agents})


def build_dispute_sim_registry(
    settings: DisputeSimBackendSettings | None = None,
) -> AgentRegistry:
    settings = settings or DisputeSimBackendSettings.from_env()
    manifest = build_dispute_sim_manifest(settings)
    env = manifest.environment or build_dispute_sim_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env, settings=settings)
    return build_manifest_development_registry(
        manifest,
        env_wiring.build_context,
        builders=DISPUTE_SIM_AGENT_BUILDERS,
    )

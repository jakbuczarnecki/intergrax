# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications.contracts.platform_plugin_evidence import (
    ApplicationPlatformPluginEvidence,
)
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS
from local_workspace_application.host.environment_profile import build_local_workspace_environment_profile
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST


def build_local_workspace_manifest(settings: LocalWorkspaceBackendSettings) -> ApplicationManifest:
    default_idx = next(
        (i for i, b in enumerate(LOCAL_WORKSPACE_APPLICATION_MANIFEST.agents) if b.default),
        1,
    )
    agents = list(LOCAL_WORKSPACE_APPLICATION_MANIFEST.agents)
    agents[default_idx] = agents[default_idx].model_copy(
        update={"contract_id": settings.default_agent_id}
    )
    return LOCAL_WORKSPACE_APPLICATION_MANIFEST.model_copy(update={"agents": agents})


@dataclass(frozen=True, slots=True)
class LocalWorkspaceHostComposition:
    """LKW host registry build with canonical Tier-3 platform plugin evidence."""

    registry: AgentRegistryRead
    platform_plugin_evidence: ApplicationPlatformPluginEvidence


def build_local_workspace_host_composition(
    settings: LocalWorkspaceBackendSettings | None = None,
) -> LocalWorkspaceHostComposition:
    settings = settings or LocalWorkspaceBackendSettings.from_env()
    manifest = build_local_workspace_manifest(settings)
    env = manifest.environment or build_local_workspace_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env, settings=settings)
    return LocalWorkspaceHostComposition(
        registry=build_manifest_development_registry(
            manifest,
            env_wiring.build_context,
            builders=LOCAL_WORKSPACE_AGENT_BUILDERS,
        ),
        platform_plugin_evidence=env_wiring.platform_plugin_evidence,
    )


def build_local_workspace_registry(
    settings: LocalWorkspaceBackendSettings | None = None,
) -> AgentRegistryRead:
    return build_local_workspace_host_composition(settings).registry

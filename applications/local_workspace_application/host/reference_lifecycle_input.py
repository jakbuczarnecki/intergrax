# © Artur Czarnecki. All rights reserved.

"""Reference production lifecycle input for the Local Workspace STRICT host."""

from __future__ import annotations

from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.registry_projection_input_bundle import (
    build_reference_activation_request,
    build_reference_registry_projection_input_bundle,
)
from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST


def build_local_workspace_reference_lifecycle_input(
    settings: LocalWorkspaceBackendSettings | None = None,
    *,
    runtime_revision_id: str = "local-workspace-reference-runtime-revision",
) -> tuple[RegistryProjectionInputBundle, ActivateRuntimeRevisionRequest]:
    """Explicit deploy input for reference Local Workspace production (not host startup)."""
    resolved_settings = settings or LocalWorkspaceBackendSettings.from_env()
    manifest = LOCAL_WORKSPACE_APPLICATION_MANIFEST
    env = build_local_workspace_environment_profile(resolved_settings)
    projection_input = build_reference_registry_projection_input_bundle(
        manifest,
        env,
        builders=LOCAL_WORKSPACE_AGENT_BUILDERS,
        runtime_revision_id=runtime_revision_id,
        settings=resolved_settings,
    )
    activation_request = build_reference_activation_request(projection_input)
    return projection_input, activation_request


__all__ = ["build_local_workspace_reference_lifecycle_input"]

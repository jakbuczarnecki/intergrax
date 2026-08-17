# © Artur Czarnecki. All rights reserved.

"""LKW test registry projection helper (AC-3)."""

from __future__ import annotations

from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection


def build_lkw_test_registry_projection(
    settings: LocalWorkspaceBackendSettings | None = None,
    *,
    revision_id: str = "lkw-test-runtime-revision",
) -> MaterializedRegistryProjection:
    """Revision-bound projection covering the full LKW manifest roster."""
    resolved_settings = settings or LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(resolved_settings)
    return build_test_registry_projection(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        builders=LOCAL_WORKSPACE_AGENT_BUILDERS,
        revision_id=revision_id,
    )


def lkw_test_environment(
    settings: LocalWorkspaceBackendSettings | None = None,
) -> ApplicationEnvironmentProfile:
    resolved_settings = settings or LocalWorkspaceBackendSettings.from_env()
    return build_local_workspace_environment_profile(resolved_settings)

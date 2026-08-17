# © Artur Czarnecki. All rights reserved.

"""LKW test registry projection helper (AC-3)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
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


def create_lkw_hosted_test_process_composition(
    *,
    seed_active_projection: bool = False,
    settings: LocalWorkspaceBackendSettings | None = None,
    revision_id: str = "lkw-hosted-test-runtime-revision",
) -> ProductionProcessComposition:
    """Process composition for hosted-runtime tests; optionally seed one active projection."""
    composition = create_reference_production_process_composition()
    if not seed_active_projection:
        return composition
    env = lkw_test_environment(settings)
    projection = build_lkw_test_registry_projection(settings, revision_id=revision_id)
    _seed_active_projection(
        composition=composition,
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    return composition


def _seed_active_projection(
    *,
    composition: ProductionProcessComposition,
    application_id: str,
    application_environment_id: str,
    projection: MaterializedRegistryProjection,
) -> str:
    stores = composition.agent_platform_runtime.stores
    revision_id = projection.evidence.runtime_revision_id
    stores.registry_projection_store.put(projection)
    stores.serving_store.atomic_swap_serving_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )
    return revision_id

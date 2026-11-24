# © Artur Czarnecki. All rights reserved.

"""LKW test registry projection helper (AC-3)."""

from __future__ import annotations

import os

from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_runtime_materialization import (
    prepare_reference_runtime_materialization,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.registry_projection_input_bundle import (
    build_reference_activation_request,
    build_reference_registry_projection_input_bundle,
    reference_admission_mutation_id,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection

from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.host_runtime_composition import (
    resolve_local_workspace_host_tenant_binding,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST


def build_lkw_test_registry_projection(
    settings: LocalWorkspaceBackendSettings | None = None,
    *,
    revision_id: str = "lkw-test-runtime-revision",
) -> MaterializedRegistryProjection:
    """Revision-bound projection covering the full LKW manifest roster."""
    os.environ.setdefault("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    resolved_settings = settings or LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(resolved_settings)
    return build_test_registry_projection(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        builders=LOCAL_WORKSPACE_AGENT_BUILDERS,
        revision_id=revision_id,
        settings=resolved_settings,
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
    """Process composition for hosted-runtime tests; optionally activate via AP lifecycle."""
    composition = create_reference_production_process_composition()
    if not seed_active_projection:
        return composition
    resolved_settings = settings or LocalWorkspaceBackendSettings.from_env()
    env = lkw_test_environment(resolved_settings)
    host_tenant = resolve_local_workspace_host_tenant_binding(resolved_settings)
    projection_input = build_reference_registry_projection_input_bundle(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        builders=LOCAL_WORKSPACE_AGENT_BUILDERS,
        runtime_revision_id=revision_id,
        settings=resolved_settings,
        tenant_id=host_tenant.tenant_id,
    )
    activation_request = build_reference_activation_request(projection_input)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    prepare_reference_runtime_materialization(
        composition.agent_platform_runtime.stores,
        projection_input,
        artifact_locator=activation_request.artifact_locator,
    )
    launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
        admission_mutation_id=reference_admission_mutation_id(projection_input.runtime_revision.runtime_revision_id),
    )
    return composition

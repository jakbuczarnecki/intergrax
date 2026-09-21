# © Artur Czarnecki. All rights reserved.

"""Legal PRODUCT harness host builders for task-control CLA-04 composition proofs."""

from __future__ import annotations

from pathlib import Path

from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.orchestration_decision_requirement_policy import (
    default_governed_contractor_harness_orchestration_decision_requirement_policy,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
    resolve_reference_production_strict_host_environment,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


class _InMemoryCollaborativeWorkStoreOwner:
    def close(self) -> None:
        return None


def in_memory_collaborative_work_repositories_for_tests() -> CollaborativeWorkRepositories:
    return CollaborativeWorkRepositories(
        membership=InMemoryWorkspaceMembershipRepository(),
        delegation=InMemoryAuthorityDelegationRepository(),
        principal_authority=InMemoryPrincipalAuthorityRepository(),
        policy=InMemoryCollaborativePolicyRepository(),
        operation_profile=InMemoryCollaborativeOperationPolicyProfileRepository(),
        store=_InMemoryCollaborativeWorkStoreOwner(),
    )


def durable_execution_continuation_state_store_for_tests() -> object:
    backing = ExecutionContinuationDurableBacking()
    backing_execution_continuation_state_store(backing)
    export = export_durable_continuation_state(backing)
    return execution_continuation_state_store_from_durable_export(export)


def build_task_control_product_harness_host_runtime(
    tmp_path: Path,
    *,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
) -> HarnessHostRuntime:
    """Real ``build_harness_host_runtime`` path with legal strict PRODUCT prerequisites."""
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    base_env = manifest.environment or build_governed_contractor_environment_profile(settings)
    env = resolve_reference_production_strict_host_environment(base_env)
    manifest_for_runtime = manifest.model_copy(update={"environment": env})
    platform = build_reference_production_platform_persistence(
        db_path=tmp_path / "task-control-product-kv.db",
    )
    return build_harness_host_runtime(
        manifest_for_runtime,
        env,
        settings=settings,
        tenant_id=manifest.app_id,
        registry_projection=build_governed_contractor_test_registry_projection(),
        key_value_cache=platform.kv_store,
        document_store=platform.document_store,
        execution_continuation_state_store=durable_execution_continuation_state_store_for_tests(),
        collaborative_work_repositories=in_memory_collaborative_work_repositories_for_tests(),
        orchestration_decision_requirement_policy=(
            default_governed_contractor_harness_orchestration_decision_requirement_policy()
        ),
        mutation_authorization_boundary=mutation_authorization_boundary,
    )

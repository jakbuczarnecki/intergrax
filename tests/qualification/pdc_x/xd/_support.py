# © Artur Czarnecki. All rights reserved.

"""Shared helpers for the PDC-X B-P6 cross-domain integration scenarios (real stores, no mocks)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.admission_wiring import (
    validate_background_execution_admission_durability,
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from tests.unit.runtime.background_execution.reentry_admission_doubles import admission_kwargs

REPO_ROOT = Path(__file__).resolve().parents[4]


def build_sqlite_worker(
    data_dir: Path,
    definitions: Iterable[TaskDefinition],
    *,
    idempotency_store: object | None = None,
    event_emitter: object | None = None,
) -> WorkerRuntime:
    """Real ``WorkerRuntime`` over the durable SQLite KV admission wiring (fail-closed durability gate)."""
    kv = SqliteDistributedKVStore(data_dir / "worker_kv.sqlite3")
    registry = TaskRegistry()
    for definition in definitions:
        registry.register(definition)
    execution_registry = TaskExecutionRegistry()
    registry.bind_execution_registry(execution_registry)
    deps = wire_background_execution_admission_dependencies(kv_store=kv)
    validate_background_execution_admission_durability(deps)
    return WorkerRuntime(
        registry=registry,
        state_store=TaskStateStore(kv_store=kv, provider="sqlite"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="sqlite",
        idempotency_store=idempotency_store,  # type: ignore[arg-type]
        event_emitter=event_emitter,  # type: ignore[arg-type]
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        **admission_kwargs(deps),  # pyright: ignore[reportArgumentType]
    )


def build_sqlite_agent_stack(db_path: Path, *, tenant: str = "tenant-test"):  # type: ignore[no-untyped-def]
    """The B-D1 admin stack composed over the durable SQLite agent-distribution store bundle.

    Same composition as ``d1._d1_harness.build_stack`` with the in-memory stores replaced by
    ``build_sqlite_agent_distribution_store_bundle`` so state survives a close and reopen.
    """
    from intergrax.agent_distribution.activation import (
        ActivationService,
        FakeRuntimeServingProjectionCoordinator,
    )
    from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
    from intergrax.agent_distribution.binding_service import BindingService
    from intergrax.agent_distribution.control_plane_governance import (
        StaticApplicationEnvironmentTenantResolver,
    )
    from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
    from intergrax.agent_distribution.effective_roster import (
        EffectiveRosterBuilder,
        InstalledAgentRequirementSetBuilder,
    )
    from intergrax.agent_distribution.effective_roster_authority import EffectiveRosterAuthorityService
    from intergrax.agent_distribution.in_memory_stores import AgentDistributionStoreState
    from intergrax.agent_distribution.installation_service import InstallationService
    from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
    from intergrax.agent_distribution.runtime_graph_service import CandidateRuntimeGraphBuilder
    from intergrax.agent_distribution.runtime_revision import MaterializationTopology
    from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
    from intergrax.agent_distribution.sqlite_stores import build_sqlite_agent_distribution_store_bundle
    from testing_support.agent_platform_admin_harness import (
        AgentProjectMetadataTestProvider,
        DeterministicAgentDistributionAdapter,
        allow_mutation_boundary,
    )
    from testing_support.agent_platform_dependency_resolver import make_identity_dependency_resolver
    from tests.qualification.pdc_x.d1 import _d1_harness as h

    bundle = build_sqlite_agent_distribution_store_bundle(db_path)
    installation_service = InstallationService(bundle.installation_store)
    binding_service = BindingService(bundle.binding_store, installation_service)
    revision_service = RuntimeRevisionService(bundle.revision_store)
    roster_authority = EffectiveRosterAuthorityService(snapshot_store=bundle.effective_roster_snapshot_store)
    metadata_provider = AgentProjectMetadataTestProvider({})
    adapter = FakeInMemoryRuntimeDeploymentAdapter()
    activation = ActivationService(
        revision_store=bundle.revision_store,
        deployment_instance_store=bundle.deployment_instance_store,
        serving_store=bundle.serving_store,
        activation_store=bundle.activation_store,
        deployment_adapter=adapter,
        projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        materialization_store=bundle.materialization_store,
    )
    revocation = h.MutableRevocation()
    service = AgentPlatformAdminService(
        installation_store=bundle.installation_store,
        binding_store=bundle.binding_store,
        revision_store=bundle.revision_store,
        serving_store=bundle.serving_store,
        deployment_instance_store=bundle.deployment_instance_store,
        lock_store=bundle.lock_store,
        materialization_store=bundle.materialization_store,
        effective_roster_snapshot_store=bundle.effective_roster_snapshot_store,
        effective_roster_authority=roster_authority,
        artifact_metadata_store=bundle.artifact_metadata_store,
        installation_service=installation_service,
        binding_service=binding_service,
        revision_service=revision_service,
        roster_builder=EffectiveRosterBuilder(bundle.installation_store),
        requirement_set_builder=InstalledAgentRequirementSetBuilder(bundle.artifact_metadata_store),
        activation_service=activation,
        graph_builder=CandidateRuntimeGraphBuilder(metadata_provider),
        materialization_service=RuntimeMaterializationService(
            {MaterializationTopology.OCI_IMAGE: DeterministicAgentDistributionAdapter()}
        ),
        metadata_provider=metadata_provider,
        dependency_resolver=make_identity_dependency_resolver(),
        mutation_authorization_boundary=allow_mutation_boundary(),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(tenant),
        package_trust_revocation_state_source=revocation,
        package_trust_policy_source=None,
        package_trust_evaluation_time_source=lambda: h.EVALUATED_AT,
    )
    return h.D1Stack(
        service=service,
        state=AgentDistributionStoreState(),
        adapter=adapter,
        revocation=revocation,
        installation_service=installation_service,
        binding_service=binding_service,
        installation_store=bundle.installation_store,
        serving_store=bundle.serving_store,
        revision_store=bundle.revision_store,
        deployment_store=bundle.deployment_instance_store,
        roster_authority=roster_authority,
        activation=activation,
        revision_service=revision_service,
        metadata_provider=metadata_provider,
    )

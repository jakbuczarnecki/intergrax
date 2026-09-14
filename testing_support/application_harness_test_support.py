# © Artur Czarnecki. All rights reserved.

"""Canonical harness host builders for application-layer unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications.contracts.profile_resolution import ProfileLayerInput
from intergrax.applications.contracts.profile_resolution.activation import (
    ActiveEffectiveProfileRevisionStore,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionPinningStore,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.registry.agent_registry import AgentRegistry

if TYPE_CHECKING:
    from intergrax.harness.application_host import ApplicationHost

CANONICAL_HARNESS_TEST_TENANT = "test-tenant"


def build_harness_host_runtime_for_tests(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    tenant_id: str = CANONICAL_HARNESS_TEST_TENANT,
    settings: object | None = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    idempotency_db_path: Path | None = None,
    use_in_memory_trace: bool = False,
    builders: dict[type, object] | None = None,
    registry: AgentRegistry | None = None,
    registry_projection: MaterializedRegistryProjection | None = None,
    registry_assembly_mode: RegistryAssemblyMode | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    agent_checkpoint_store: AgentCheckpointStore | None = None,
    notification_adapter: NotificationAdapter | None = None,
    application_host: ApplicationHost | None = None,
    document_store: DocumentStore | None = None,
    key_value_cache: DistributedKVStore | None = None,
    boundary_event_buffer: BoundaryEventBuffer | None = None,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    profile_layers: tuple[ProfileLayerInput, ...] = (),
    revision_store: EffectiveProfileRevisionStore | None = None,
    pinning_store: EffectiveProfileExecutionPinningStore | None = None,
    active_store: ActiveEffectiveProfileRevisionStore | None = None,
) -> HarnessHostRuntime:
    """``build_harness_host_runtime`` with explicit governance tenant identity."""
    return build_harness_host_runtime(
        manifest,
        environment,
        settings=settings,
        tenant_id=tenant_id,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        idempotency_db_path=idempotency_db_path,
        use_in_memory_trace=use_in_memory_trace,
        builders=builders,
        registry=registry,
        registry_projection=registry_projection,
        registry_assembly_mode=registry_assembly_mode,
        checkpoint_store=checkpoint_store,
        agent_checkpoint_store=agent_checkpoint_store,
        notification_adapter=notification_adapter,
        application_host=application_host,
        document_store=document_store,
        key_value_cache=key_value_cache,
        boundary_event_buffer=boundary_event_buffer,
        mutation_authorization_boundary=mutation_authorization_boundary,
        profile_layers=profile_layers,
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )

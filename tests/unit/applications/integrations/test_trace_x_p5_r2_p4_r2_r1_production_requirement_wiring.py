# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4-R2-R1 production requirement wiring, Case C/D, integrity matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import pytest

from intergrax.applications._shared.integrations.active_execution_requirement_recovery_staging import (
    ActiveExecutionIdentityRequirementRecoveryStagingSource,
)
from intergrax.applications._shared.integrations.integration_configuration_provenance_requirement_commit import (
    RuntimeEventBusIntegrationConfigurationProvenanceRequirementCommitPort,
)
from intergrax.applications._shared.integrations.persistence import (
    KvExecutionIntegrationConfigurationPinningStore,
)
from intergrax.applications._shared.uca6c_marketplace_qualified_execution_composition import (
    Uca6cMarketplaceQualifiedExecutionCompositionError,
    build_production_marketplace_configured_execution_composition,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    derive_integration_configuration_provenance_requirement_event_id,
)
from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
    InMemoryKVStore,
)
from intergrax.integrations.configured_relational_store_execution_binding import (
    ConfiguredRelationalStoreExecutionBindingError,
    build_default_configured_relational_store_execution_binding,
)
from intergrax.integrations.contracts.configured_relational_store_execution import (
    RelationalQueryRequest,
)
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
    ExecutionIntegrationConfigurationAdoptionError,
)
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationMaterializationPort,
    ExecutionBoundIntegrationResolution,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import MandatoryEvidencePersistenceError
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
    _EXEC,
    _provenance_configured_adopted,
    _recovery_staging,
    _subject,
)
from tests.unit.applications.integrations.test_trace_x_p5_r2_p4_r1_r1_r1_r1_r1_r1_pin_ambiguous_outcome_contract import (
    _staging,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TASK = TaskId("task_00000000000000000000000000000001")
_RUN = RunId("run_" + "a" * 32)
_ATTEMPT = AttemptId("attempt_" + "b" * 32)


@dataclass
class _IoCounter:
    calls: int = 0


class _FakeRelational(RelationalStoreIntegrationContract):
    def __init__(self, counter: _IoCounter) -> None:
        super().__init__(
            **RelationalStoreIntegrationContract.for_provider(
                provider_id="sqlite",
                display_name="Fake Relational",
            ).model_dump(),
        )
        self._counter = counter

    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._counter.calls += 1

    def fetch_all(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> Sequence[Mapping[str, Any]]:
        self._counter.calls += 1
        return ({"n": 1},)

    def close(self) -> None:
        return None


@dataclass
class _Materialization(ExecutionBoundIntegrationMaterializationPort):
    integration: _FakeRelational

    def resolve_catalog(self, category, *, slug: str, profile=None):
        return self.integration

    def resolve_from_profile(self, profile, category):
        return self.integration


def _adoption(tenant: str = _TENANT) -> ExecutionIntegrationConfigurationAdoption:
    binding = ConfiguredCapabilityBinding(
        tenant_id=tenant,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        configuration_type="test",
        configuration_version="v1",
        configuration_fingerprint="fp-a",
        realization_evidence_refs=(),
    )
    return ExecutionIntegrationConfigurationAdoption(
        configured_binding=binding,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        resource_scope="default",
    )


def _identity_scope(execution_id: ExecutionId, *, tenant_id: str = _TENANT):
    identity_token = bind_active_execution_identity(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=execution_id,
        task_id=_TASK,
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=tenant_id,
            workspace_id="workspace-1",
            principal_id="principal-1",
        ),
    )
    return identity_token, governance_token


def _reset_identity_scope(identity_token, governance_token) -> None:
    reset_active_execution_governance_identity(governance_token)
    reset_active_execution_identity(identity_token)


def _production_binding(
    *,
    pinning_store,
    event_bus: RuntimeEventBus,
    counter: _IoCounter,
):
    resolution = ExecutionBoundIntegrationResolution(
        pinning_store=pinning_store,
        materialization=_Materialization(_FakeRelational(counter)),
    )
    commit = RuntimeEventBusIntegrationConfigurationProvenanceRequirementCommitPort(event_bus)
    staging_source = ActiveExecutionIdentityRequirementRecoveryStagingSource()
    return build_default_configured_relational_store_execution_binding(
        resolution=resolution,
        requirement_commit_port=commit,
        requirement_recovery_staging_source=staging_source,
        require_configured_adopted_requirement_evidence=True,
    )


def test_production_binding_requires_staging_source_and_commit_port() -> None:
    resolution = ExecutionBoundIntegrationResolution(
        pinning_store=KvExecutionIntegrationConfigurationPinningStore(
            InMemoryKVStore(),
        ),
    )
    with pytest.raises(ConfiguredRelationalStoreExecutionBindingError):
        build_default_configured_relational_store_execution_binding(
            resolution=resolution,
            require_configured_adopted_requirement_evidence=True,
        )


def test_marketplace_production_composition_requires_runtime_event_bus() -> None:
    from unittest.mock import MagicMock

    with pytest.raises(Uca6cMarketplaceQualifiedExecutionCompositionError):
        build_production_marketplace_configured_execution_composition(
            intent_repository=MagicMock(),
            stage_repository=MagicMock(),
            activation_read=MagicMock(),
            acquisition=MagicMock(),
            host_profile_id="host",
            material_provider=MagicMock(),
            catalog_tool_invoker=MagicMock(),
            configuration_pinning_kv_store=InMemoryKVStore(),
        )


class _FailingOncePersistence:
    def __init__(self) -> None:
        self._inner = InMemoryRuntimeEventStore()
        self._failed = False

    def append(self, event, *, tenant_id: str):
        if not self._failed:
            self._failed = True
            raise MandatoryEvidencePersistenceError("forced spine failure")
        return self._inner.append(event, tenant_id=tenant_id)

    def query_run(self, tenant_id, run_id, *, limit=1000):
        return self._inner.query_run(tenant_id, run_id, limit=limit)


def test_case_c_spine_failure_blocks_io_then_recovery_allows_io() -> None:
    execution_id = mint_execution_id()
    counter = _IoCounter()
    kv = InMemoryKVStore()
    pinning = KvExecutionIntegrationConfigurationPinningStore(kv)
    failing_bus = RuntimeEventBus(persistence=_FailingOncePersistence())
    binding = _production_binding(
        pinning_store=pinning,
        event_bus=failing_bus,
        counter=counter,
    )
    identity_token, governance_token = _identity_scope(execution_id)
    try:
        port = binding.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        with pytest.raises(ExecutionIntegrationConfigurationAdoptionError):
            port.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter.calls == 0
        pins = pinning.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        assert len(pins) == 1
        prepared = pins[0].requirement_recovery_staging
        assert prepared is not None
    finally:
        _reset_identity_scope(identity_token, governance_token)

    counter2 = _IoCounter()
    binding2 = _production_binding(
        pinning_store=pinning,
        event_bus=RuntimeEventBus(persistence=InMemoryRuntimeEventStore()),
        counter=counter2,
    )
    identity_token2, governance_token2 = _identity_scope(execution_id)
    try:
        port2 = binding2.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port2.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter2.calls == 1
        pins2 = pinning.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        assert len(pins2) == 1
        assert pins2[0].requirement_recovery_staging == prepared
    finally:
        _reset_identity_scope(identity_token2, governance_token2)


def test_case_d_idempotent_spine_then_single_io() -> None:
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    counter = _IoCounter()
    pinning = KvExecutionIntegrationConfigurationPinningStore(InMemoryKVStore())
    binding = _production_binding(pinning_store=pinning, event_bus=bus, counter=counter)
    identity_token, governance_token = _identity_scope(execution_id)
    try:
        port = binding.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port._initialize_adapter()  # noqa: SLF001 — crash-before-I/O recovery proof
        assert counter.calls == 0
        pins = pinning.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        prepared = pins[0].requirement_recovery_staging
        event_id = derive_integration_configuration_provenance_requirement_event_id(
            tenant_id=_TENANT,
            execution_id=execution_id,
            subject=pins[0].subject,
        )
        events_after_crash = store.list_positioned_for_run(_RUN, tenant_id=_TENANT)
        assert len(events_after_crash) == 1
        assert events_after_crash[0].event.event_id == event_id

        port2 = binding.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port2.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter.calls == 1
        events = store.list_positioned_for_run(_RUN, tenant_id=_TENANT)
        assert len(events) == 1
        assert events[0].position.value == 1
        assert pins[0].requirement_recovery_staging == prepared
    finally:
        _reset_identity_scope(identity_token, governance_token)


def _fresh_sqlite_event_store(db_path: Path) -> SQLiteRuntimeEventStore:
    return SQLiteRuntimeEventStore(db_path=db_path)


def test_case_c_durable_sqlite_pin_restart_spine_recovery(tmp_path: Path) -> None:
    """Unit durable restart: fresh SQLite spine adapter after pin-only phase (blocker 33)."""
    db_path = tmp_path / "runtime_events.sqlite"
    execution_id = mint_execution_id()
    counter = _IoCounter()
    kv = InMemoryKVStore()
    pinning = KvExecutionIntegrationConfigurationPinningStore(kv)
    failing_bus = RuntimeEventBus(persistence=_FailingOncePersistence())
    binding = _production_binding(
        pinning_store=pinning,
        event_bus=failing_bus,
        counter=counter,
    )
    identity_token, governance_token = _identity_scope(execution_id)
    try:
        port = binding.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        with pytest.raises(ExecutionIntegrationConfigurationAdoptionError):
            port.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter.calls == 0
        pins = pinning.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        assert len(pins) == 1
        prepared = pins[0].requirement_recovery_staging
        assert prepared is not None
        assert pins[0].provenance.execution_id == execution_id
    finally:
        _reset_identity_scope(identity_token, governance_token)

    counter2 = _IoCounter()
    pinning2 = KvExecutionIntegrationConfigurationPinningStore(kv)
    durable_store_b = _fresh_sqlite_event_store(db_path)
    binding2 = _production_binding(
        pinning_store=pinning2,
        event_bus=RuntimeEventBus(persistence=durable_store_b),
        counter=counter2,
    )
    identity_token2, governance_token2 = _identity_scope(execution_id)
    try:
        port2 = binding2.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port2.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter2.calls == 1
        pins2 = pinning2.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        assert len(pins2) == 1
        assert pins2[0].requirement_recovery_staging == prepared
        assert len(durable_store_b.list_positioned_for_run(_RUN, tenant_id=_TENANT)) == 1
    finally:
        _reset_identity_scope(identity_token2, governance_token2)


def test_case_d_durable_sqlite_restart_idempotent_spine_single_io(tmp_path: Path) -> None:
    """Unit durable restart: requirement event survives fresh bus + store adapter (blocker 33)."""
    db_path = tmp_path / "runtime_events.sqlite"
    execution_id = mint_execution_id()
    pin_kv = InMemoryKVStore()
    store_a = _fresh_sqlite_event_store(db_path)
    bus_a = RuntimeEventBus(persistence=store_a)
    counter = _IoCounter()
    pinning_a = KvExecutionIntegrationConfigurationPinningStore(pin_kv)
    binding_a = _production_binding(pinning_store=pinning_a, event_bus=bus_a, counter=counter)
    identity_token, governance_token = _identity_scope(execution_id)
    try:
        port_a = binding_a.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port_a._initialize_adapter()  # noqa: SLF001
        assert counter.calls == 0
        pins = pinning_a.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        event_id = derive_integration_configuration_provenance_requirement_event_id(
            tenant_id=_TENANT,
            execution_id=execution_id,
            subject=pins[0].subject,
        )
        assert len(store_a.list_positioned_for_run(_RUN, tenant_id=_TENANT)) == 1
    finally:
        _reset_identity_scope(identity_token, governance_token)

    store_b = _fresh_sqlite_event_store(db_path)
    bus_b = RuntimeEventBus(persistence=store_b)
    counter_b = _IoCounter()
    pinning_b = KvExecutionIntegrationConfigurationPinningStore(pin_kv)
    binding_b = _production_binding(pinning_store=pinning_b, event_bus=bus_b, counter=counter_b)
    identity_token2, governance_token2 = _identity_scope(execution_id)
    try:
        port_b = binding_b.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port_b.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter_b.calls == 1
        events = store_b.list_positioned_for_run(_RUN, tenant_id=_TENANT)
        assert len(events) == 1
        assert events[0].event.event_id == event_id
        assert events[0].position.value == 1
    finally:
        _reset_identity_scope(identity_token2, governance_token2)


@pytest.mark.docker
def test_case_d_docker_redis_idempotent_spine_single_io(tmp_path: Path) -> None:
    try:
        import redis
    except ModuleNotFoundError:
        pytest.skip("redis package not installed")
    from intergrax.distributed.providers.redis_kv_store import RedisKVStore

    try:
        client = redis.Redis(host="localhost", port=6379, db=15)
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis unavailable: {exc}")
    client.flushdb()
    db_path = tmp_path / "docker_runtime_events.sqlite"
    execution_id = mint_execution_id()
    store_a = _fresh_sqlite_event_store(db_path)
    bus_a = RuntimeEventBus(persistence=store_a)
    counter = _IoCounter()
    pinning_a = KvExecutionIntegrationConfigurationPinningStore(
        RedisKVStore(client=client, key_prefix="p4r2r1d"),
    )
    binding_a = _production_binding(pinning_store=pinning_a, event_bus=bus_a, counter=counter)
    identity_token, governance_token = _identity_scope(execution_id)
    try:
        port_a = binding_a.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port_a._initialize_adapter()  # noqa: SLF001
        pins = pinning_a.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        event_id = derive_integration_configuration_provenance_requirement_event_id(
            tenant_id=_TENANT,
            execution_id=execution_id,
            subject=pins[0].subject,
        )
        assert counter.calls == 0
        assert len(store_a.list_positioned_for_run(_RUN, tenant_id=_TENANT)) == 1
    finally:
        _reset_identity_scope(identity_token, governance_token)

    store_b = _fresh_sqlite_event_store(db_path)
    bus_b = RuntimeEventBus(persistence=store_b)
    counter_b = _IoCounter()
    pinning_b = KvExecutionIntegrationConfigurationPinningStore(
        RedisKVStore(client=client, key_prefix="p4r2r1d"),
    )
    binding_b = _production_binding(pinning_store=pinning_b, event_bus=bus_b, counter=counter_b)
    identity_token2, governance_token2 = _identity_scope(execution_id)
    try:
        port_b = binding_b.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port_b.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter_b.calls == 1
        events = store_b.list_positioned_for_run(_RUN, tenant_id=_TENANT)
        assert len(events) == 1
        assert events[0].event.event_id == event_id
        assert events[0].position.value == 1
    finally:
        _reset_identity_scope(identity_token2, governance_token2)


@pytest.mark.docker
def test_case_c_docker_redis_pin_and_spine_recovery(tmp_path: Path) -> None:
    try:
        import redis
    except ModuleNotFoundError:
        pytest.skip("redis package not installed")
    from intergrax.distributed.providers.redis_kv_store import RedisKVStore

    try:
        client = redis.Redis(host="localhost", port=6379, db=15)
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis unavailable: {exc}")
    client.flushdb()
    db_path = tmp_path / "docker_runtime_events_case_c.sqlite"
    execution_id = mint_execution_id()
    counter = _IoCounter()
    pinning = KvExecutionIntegrationConfigurationPinningStore(
        RedisKVStore(client=client, key_prefix="p4r2r1"),
    )
    failing_bus = RuntimeEventBus(persistence=_FailingOncePersistence())
    binding = _production_binding(pinning_store=pinning, event_bus=failing_bus, counter=counter)
    identity_token, governance_token = _identity_scope(execution_id)
    prepared = None
    try:
        port = binding.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        with pytest.raises(ExecutionIntegrationConfigurationAdoptionError):
            port.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter.calls == 0
        pins = pinning.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        assert len(pins) == 1
        prepared = pins[0].requirement_recovery_staging
    finally:
        _reset_identity_scope(identity_token, governance_token)

    counter2 = _IoCounter()
    pinning2 = KvExecutionIntegrationConfigurationPinningStore(
        RedisKVStore(client=client, key_prefix="p4r2r1"),
    )
    durable_store_b = _fresh_sqlite_event_store(db_path)
    binding2 = _production_binding(
        pinning_store=pinning2,
        event_bus=RuntimeEventBus(persistence=durable_store_b),
        counter=counter2,
    )
    identity_token2, governance_token2 = _identity_scope(execution_id)
    try:
        port2 = binding2.create_bound_port(
            tenant_id=_TENANT,
            execution_id=execution_id,
            adoption=_adoption(),
        )
        port2.query(RelationalQueryRequest(sql="SELECT 1"))
        assert counter2.calls == 1
        pins2 = pinning2.read_pin_records(tenant_id=_TENANT, execution_id=execution_id)
        assert len(pins2) == 1
        assert pins2[0].requirement_recovery_staging == prepared
        assert len(durable_store_b.list_positioned_for_run(_RUN, tenant_id=_TENANT)) == 1
    finally:
        _reset_identity_scope(identity_token2, governance_token2)


def test_cross_tenant_pin_scope_isolation() -> None:
    pinning = KvExecutionIntegrationConfigurationPinningStore(InMemoryKVStore())
    subject = _subject()
    provenance = _provenance_configured_adopted()
    pinning.pin(
        subject=subject,
        provenance=provenance,
        requirement_recovery_staging=_recovery_staging(),
    )
    assert pinning.read_pin_records(tenant_id="tenant-b", execution_id=_EXEC) == ()


def test_integrity_matrix_timezone_invalid_staging_rejected() -> None:
    from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
        ExecutionIntegrationConfigurationRequirementRecoveryStaging,
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        ExecutionIntegrationConfigurationRequirementRecoveryStaging(
            requirement_boundary_prepared_at=datetime(2026, 1, 1, 12, 0, 0),
            task_id=_TASK,
            run_id=_RUN,
            attempt_id=_ATTEMPT,
        )


def test_active_execution_tenant_mismatch_rejects_pin_spine_and_io() -> None:
    execution_id = mint_execution_id()
    counter = _IoCounter()
    store = InMemoryRuntimeEventStore()
    pinning = KvExecutionIntegrationConfigurationPinningStore(InMemoryKVStore())
    binding = _production_binding(
        pinning_store=pinning,
        event_bus=RuntimeEventBus(persistence=store),
        counter=counter,
    )
    identity_token, governance_token = _identity_scope(execution_id, tenant_id=_TENANT)
    try:
        with pytest.raises(ValueError, match="active execution tenant"):
            binding.create_bound_port(
                tenant_id="tenant-b",
                execution_id=execution_id,
                adoption=_adoption(tenant="tenant-b"),
            )
        assert counter.calls == 0
        assert len(store.list_positioned_for_run(_RUN, tenant_id="tenant-b")) == 0
        assert len(pinning.read_pin_records(tenant_id="tenant-b", execution_id=execution_id)) == 0
    finally:
        _reset_identity_scope(identity_token, governance_token)


def test_integrity_matrix_first_pin_requires_candidate_staging() -> None:
    from intergrax.applications._shared.integrations.persistence import (
        InMemoryExecutionIntegrationConfigurationPinningStore,
    )
    from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
        ExecutionIntegrationConfigurationPinningError,
    )
    from intergrax.integrations.execution_integration_configuration_pin_reconciliation import (
        pin_with_reconcile,
    )

    store = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    provenance = _provenance_configured_adopted()
    with pytest.raises(ExecutionIntegrationConfigurationPinningError, match="staging required"):
        pin_with_reconcile(
            pinning_store=store,
            subject=subject,
            provenance=provenance,
            candidate_staging=None,
        )


def test_integrity_matrix_legacy_pin_without_staging_fails_reconcile() -> None:
    from intergrax.applications._shared.integrations.persistence import (
        InMemoryExecutionIntegrationConfigurationPinningStore,
    )
    from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
        ExecutionIntegrationConfigurationPinningError,
    )
    from intergrax.integrations.execution_integration_configuration_pin_reconciliation import (
        pin_with_reconcile,
    )

    store = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    provenance = _provenance_configured_adopted()
    store.pin(subject=subject, provenance=provenance, requirement_recovery_staging=None)
    with pytest.raises(ExecutionIntegrationConfigurationPinningError):
        pin_with_reconcile(
            pinning_store=store,
            subject=subject,
            provenance=provenance,
            candidate_staging=_staging("retry"),
        )

# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1 pin ambiguous-outcome contract tests (P4 API)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.applications._shared.integrations.persistence import (
    DocumentStoreExecutionIntegrationConfigurationPinningStore,
    InMemoryExecutionIntegrationConfigurationPinningStore,
    KvExecutionIntegrationConfigurationPinningStore,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationRequirementRecoveryStaging,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
    ExecutionIntegrationConfigurationPinningFailureReason,
)
from intergrax.integrations.execution_integration_configuration_pin_reconciliation import (
    pin_with_reconcile,
)
from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
    InMemoryConditionalDocumentStore,
    InMemoryKVStore,
    _EXEC,
    _provenance_configured_adopted,
    _subject,
)

pytestmark = pytest.mark.unit


def _staging(label: str) -> ExecutionIntegrationConfigurationRequirementRecoveryStaging:
    base = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return ExecutionIntegrationConfigurationRequirementRecoveryStaging(
        requirement_boundary_prepared_at=base,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        correlation_id=label,
    )


@pytest.fixture
def pin_stores() -> list[Any]:
    return [
        InMemoryExecutionIntegrationConfigurationPinningStore(),
        KvExecutionIntegrationConfigurationPinningStore(InMemoryKVStore()),
        DocumentStoreExecutionIntegrationConfigurationPinningStore(InMemoryConditionalDocumentStore()),
    ]


def test_ambiguous_pin_lost_acknowledgement_retry_reuses_stored_staging(
    pin_stores: list[Any],
) -> None:
    subject = _subject()
    provenance = _provenance_configured_adopted()
    s1 = _staging("s1")
    for store in pin_stores:
        store.pin(subject=subject, provenance=provenance, requirement_recovery_staging=s1)
        s2 = _staging("s2")
        pin_with_reconcile(
            pinning_store=store,
            subject=subject,
            provenance=provenance,
            candidate_staging=s2,
        )
        records = store.read_pin_records(tenant_id="tenant-a", execution_id=_EXEC)
        assert len(records) == 1
        assert records[0].requirement_recovery_staging == s1


def test_ambiguous_pin_no_row_retry_may_create_new_staging(pin_stores: list[Any]) -> None:
    subject = _subject()
    provenance = _provenance_configured_adopted()
    s1 = _staging("first")
    for store in pin_stores:
        pin_with_reconcile(
            pinning_store=store,
            subject=subject,
            provenance=provenance,
            candidate_staging=s1,
        )
        stored = store.read_pin_records(tenant_id="tenant-a", execution_id=_EXEC)[0]
        assert stored.requirement_recovery_staging == s1


def test_ambiguous_pin_concurrent_same_obligation_loser_reconciles_winner_staging(
    pin_stores: list[Any],
) -> None:
    subject = _subject()
    provenance = _provenance_configured_adopted()
    s1 = _staging("winner")
    s2 = _staging("loser")
    for store in pin_stores:
        store.pin(subject=subject, provenance=provenance, requirement_recovery_staging=s1)
        pin_with_reconcile(
            pinning_store=store,
            subject=subject,
            provenance=provenance,
            candidate_staging=s2,
        )
        assert store.read_pin_records(tenant_id="tenant-a", execution_id=_EXEC)[0].requirement_recovery_staging == s1


def test_ambiguous_pin_concurrent_different_provenance_conflict() -> None:
    from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
        _configured_slice,
    )
    from intergrax.contracts.execution_integration_configuration_provenance import (
        ExecutionIntegrationConfigurationProvenance,
    )

    store = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    first = _provenance_configured_adopted()
    second = ExecutionIntegrationConfigurationProvenance(
        tenant_id=first.tenant_id,
        execution_id=first.execution_id,
        mode=first.mode,
        effective=first.effective,
        configured=_configured_slice(configuration_fingerprint="fp-other"),
    )
    store.pin(subject=subject, provenance=first, requirement_recovery_staging=_staging("a"))
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        pin_with_reconcile(
            pinning_store=store,
            subject=subject,
            provenance=second,
            candidate_staging=_staging("b"),
        )
    assert exc.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CONFLICT


def test_ambiguous_pin_stored_staging_never_overwritten() -> None:
    store = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    provenance = _provenance_configured_adopted()
    s1 = _staging("canonical")
    store.pin(subject=subject, provenance=provenance, requirement_recovery_staging=s1)
    divergent = _staging("divergent")
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        store.pin(subject=subject, provenance=provenance, requirement_recovery_staging=divergent)
    assert exc.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CONFLICT


def test_ambiguous_pin_restart_recovers_staging_before_spine() -> None:
    store_a = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    provenance = _provenance_configured_adopted()
    s1 = _staging("restart")
    store_a.pin(subject=subject, provenance=provenance, requirement_recovery_staging=s1)
    store_b = InMemoryExecutionIntegrationConfigurationPinningStore()
    for record in store_a.read_pin_records(tenant_id="tenant-a", execution_id=_EXEC):
        store_b.pin(
            subject=record.subject,
            provenance=record.provenance,
            requirement_recovery_staging=record.requirement_recovery_staging,
        )
    recovered = store_b.read_pin_records(tenant_id="tenant-a", execution_id=_EXEC)[0]
    assert recovered.requirement_recovery_staging == s1


def test_ambiguous_pin_no_second_durable_staging_store_module() -> None:
    from pathlib import Path

    persistence = (
        Path(__file__).resolve().parents[4]
        / "intergrax/applications/_shared/integrations/persistence.py"
    )
    source = persistence.read_text(encoding="utf-8")
    assert "RequirementRecoveryStagingStore" not in source
    assert "recovery_staging_store" not in source.lower()


@pytest.mark.docker
def test_ambiguous_pin_docker_backed_write_recovery() -> None:
    try:
        import redis
    except ModuleNotFoundError:
        pytest.skip("redis package not installed")
    from intergrax.distributed.providers.redis_kv_store import RedisKVStore

    try:
        client = redis.Redis(host="localhost", port=6379, db=15)
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis unavailable for docker durability proof: {exc}")
    client.flushdb()
    store = KvExecutionIntegrationConfigurationPinningStore(RedisKVStore(client=client, key_prefix="p4r2"))
    subject = _subject()
    provenance = _provenance_configured_adopted()
    s1 = _staging("docker")
    store.pin(subject=subject, provenance=provenance, requirement_recovery_staging=s1)
    store2 = KvExecutionIntegrationConfigurationPinningStore(
        RedisKVStore(client=client, key_prefix="p4r2"),
    )
    pin_with_reconcile(
        pinning_store=store2,
        subject=subject,
        provenance=provenance,
        candidate_staging=_staging("fresh"),
    )
    assert store2.read_pin_records(tenant_id="tenant-a", execution_id=_EXEC)[0].requirement_recovery_staging == s1

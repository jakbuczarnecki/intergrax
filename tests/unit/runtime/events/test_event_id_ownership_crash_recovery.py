# © Artur Czarnecki. All rights reserved.

"""EventId ownership fingerprint and crash-recovery semantics (DG-002 R2)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.events.persistence_contract import (
    EVENT_ID_OWNERSHIP_SCHEMA_V1,
    RuntimeEventPersistenceIntegrityError,
    build_runtime_event_identity_claim,
    encode_event_identity_claim,
    runtime_event_identity_fingerprint,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
    _event_identity_claim_document,
    _global_event_id_partition,
    _run_partition,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _put_ownership_only(
    backend: InMemoryDocumentStore,
    *,
    tenant_id: str,
    event,
) -> None:
    claim = build_runtime_event_identity_claim(
        persistence_tenant_id=tenant_id,
        event=event,
    )
    backend.put(_event_identity_claim_document(event.event_id, claim))


def _put_legacy_ownership(
    backend: InMemoryDocumentStore,
    *,
    tenant_id: str,
    event_id: str,
    run_id: str,
) -> None:
    backend.put(
        DocumentRecord(
            partition_key=_global_event_id_partition(),
            row_key=event_id,
            data={"tenant_id": tenant_id, "run_id": run_id},
        )
    )


def test_fingerprint_same_object_twice() -> None:
    event = sample_runtime_event(tenant_id="tenant-fp-a")
    assert runtime_event_identity_fingerprint(event) == runtime_event_identity_fingerprint(event)


def test_fingerprint_model_copy_same_data() -> None:
    event = sample_runtime_event(tenant_id="tenant-fp-b")
    copy = event.model_copy(deep=True)
    assert runtime_event_identity_fingerprint(event) == runtime_event_identity_fingerprint(copy)


def test_fingerprint_payload_dict_insertion_order() -> None:
    event = sample_runtime_event(tenant_id="tenant-fp-c")
    first = event.model_copy(update={"payload": {"z": 1, "a": 2}})
    second = event.model_copy(update={"payload": {"a": 2, "z": 1}})
    assert runtime_event_identity_fingerprint(first) == runtime_event_identity_fingerprint(second)


def test_fingerprint_changes_on_task_id() -> None:
    event = sample_runtime_event(tenant_id="tenant-fp-d")
    changed = event.model_copy(update={"task_id": mint_task_id()})
    assert runtime_event_identity_fingerprint(event) != runtime_event_identity_fingerprint(changed)


def test_fingerprint_changes_on_payload() -> None:
    event = sample_runtime_event(tenant_id="tenant-fp-e")
    changed = event.model_copy(update={"payload": {"marker": "changed"}})
    assert runtime_event_identity_fingerprint(event) != runtime_event_identity_fingerprint(changed)


def test_fingerprint_changes_on_event_type() -> None:
    event = sample_runtime_event(tenant_id="tenant-fp-f")
    changed = event.model_copy(update={"event_type": RuntimeEventType.STEP_COMPLETED})
    assert runtime_event_identity_fingerprint(event) != runtime_event_identity_fingerprint(changed)


def test_fingerprint_timezone_aware_timestamp_deterministic() -> None:
    event = sample_runtime_event(tenant_id="tenant-fp-g").model_copy(
        update={"timestamp": datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)},
    )
    assert runtime_event_identity_fingerprint(event) == runtime_event_identity_fingerprint(event)


def test_crash_recovery_exact_retry_writes_canonical_and_repairs() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-crash-exact"
    event = sample_runtime_event(tenant_id=tenant_id)
    _put_ownership_only(backend, tenant_id=tenant_id, event=event)
    positioned = store.append(event, tenant_id=tenant_id)
    lookup = store.get_by_event_id(tenant_id=tenant_id, event_id=event.event_id)
    assert lookup is not None
    assert lookup.position == positioned.position
    assert backend.get(_run_partition(tenant_id, event.run_id), event.event_id) is not None
    task_partition = f"{tenant_id}|task|{event.task_id}"
    assert backend.get(task_partition, event.event_id) is not None
    store.close()


def test_crash_recovery_task_mismatch_hard_conflict() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-crash-task"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_t1 = mint_task_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_t1,
    )
    _put_ownership_only(backend, tenant_id=tenant_id, event=original)
    conflicting = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=mint_task_id(),
    )
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="task_id"):
        store.append(conflicting, tenant_id=tenant_id)
    assert backend.get(_run_partition(tenant_id, run_id), event_id) is None
    store.close()


def test_crash_recovery_payload_mismatch_hard_conflict() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-crash-payload"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_id = mint_task_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
    ).model_copy(update={"payload": {"value": "A"}})
    _put_ownership_only(backend, tenant_id=tenant_id, event=original)
    conflicting = original.model_copy(update={"payload": {"value": "B"}})
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="fingerprint"):
        store.append(conflicting, tenant_id=tenant_id)
    assert backend.get(_run_partition(tenant_id, run_id), event_id) is None
    store.close()


def test_crash_recovery_event_type_mismatch_hard_conflict() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-crash-type"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_id = mint_task_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
    )
    _put_ownership_only(backend, tenant_id=tenant_id, event=original)
    conflicting = original.model_copy(update={"event_type": RuntimeEventType.STEP_COMPLETED})
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="fingerprint"):
        store.append(conflicting, tenant_id=tenant_id)
    assert backend.get(_run_partition(tenant_id, run_id), event_id) is None
    store.close()


def test_crash_recovery_execution_identity_mismatch_hard_conflict() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-crash-exec"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_id = mint_task_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    _put_ownership_only(backend, tenant_id=tenant_id, event=original)
    conflicting = original.model_copy(update={"execution_id": mint_execution_id()})
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="fingerprint"):
        store.append(conflicting, tenant_id=tenant_id)
    assert backend.get(_run_partition(tenant_id, run_id), event_id) is None
    store.close()


def test_crash_recovery_cross_tenant_hard_conflict_no_leakage() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_a = "tenant-crash-cross-a"
    tenant_b = "tenant-crash-cross-b"
    event_id = mint_event_id()
    original = sample_runtime_event(tenant_id=tenant_a, event_id=event_id)
    _put_ownership_only(backend, tenant_id=tenant_a, event=original)
    foreign = sample_runtime_event(tenant_id=tenant_b, event_id=event_id)
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="tenant"):
        store.append(foreign, tenant_id=tenant_b)
    assert backend.get(_run_partition(tenant_b, foreign.run_id), event_id) is None
    assert store.get_by_event_id(tenant_id=tenant_b, event_id=event_id) is None
    store.close()


def test_legacy_ownership_with_canonical_upgrades_and_accepts_exact_retry() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-legacy-upgrade"
    event = sample_runtime_event(tenant_id=tenant_id)
    positioned = store.append(event, tenant_id=tenant_id)
    claim = build_runtime_event_identity_claim(
        persistence_tenant_id=tenant_id,
        event=event,
    )
    backend.put(
        DocumentRecord(
            partition_key=_global_event_id_partition(),
            row_key=event.event_id,
            data={"tenant_id": tenant_id, "run_id": event.run_id},
        )
    )
    repaired = store.append(event, tenant_id=tenant_id)
    assert repaired.position == positioned.position
    ownership = backend.get(_global_event_id_partition(), event.event_id)
    assert ownership is not None
    assert ownership.data == encode_event_identity_claim(claim)
    store.close()


def test_legacy_ownership_without_canonical_fails_closed() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-legacy-orphan"
    event = sample_runtime_event(tenant_id=tenant_id)
    _put_legacy_ownership(
        backend,
        tenant_id=tenant_id,
        event_id=event.event_id,
        run_id=event.run_id,
    )
    with pytest.raises(
        RuntimeEventPersistenceIntegrityError,
        match="legacy runtime event ownership lacks canonical identity fingerprint",
    ):
        store.append(event, tenant_id=tenant_id)
    assert backend.get(_run_partition(tenant_id, event.run_id), event.event_id) is None
    store.close()


def test_new_ownership_record_uses_v1_schema() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-v1-schema"
    event = sample_runtime_event(tenant_id=tenant_id)
    store.append(event, tenant_id=tenant_id)
    ownership = backend.get(_global_event_id_partition(), event.event_id)
    assert ownership is not None
    assert ownership.data["schema_version"] == EVENT_ID_OWNERSHIP_SCHEMA_V1
    assert "fingerprint" in ownership.data
    assert "task_id" in ownership.data
    store.close()


def test_concurrent_exact_duplicate_single_canonical_record() -> None:
    backend = InMemoryDocumentStore()
    store_a = DocumentBackedRuntimeEventStore(backend)
    store_b = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-concurrent-exact"
    event = sample_runtime_event(tenant_id=tenant_id)
    barrier = threading.Barrier(2)
    results: list[int] = []
    lock = threading.Lock()

    def _append(store: DocumentBackedRuntimeEventStore) -> None:
        barrier.wait()
        position = store.append(event, tenant_id=tenant_id).position.value
        with lock:
            results.append(position)

    threads = [
        threading.Thread(target=_append, args=(store_a,)),
        threading.Thread(target=_append, args=(store_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert results[0] == results[1]
    assert len(store_a.list_positioned_for_run(event.run_id, tenant_id=tenant_id)) == 1
    store_a.close()


def test_concurrent_conflicting_duplicate_second_fails() -> None:
    backend = InMemoryDocumentStore()
    store_a = DocumentBackedRuntimeEventStore(backend)
    store_b = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-concurrent-conflict"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_a = mint_task_id()
    task_b = mint_task_id()
    event_a = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_a,
    )
    event_b = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_b,
    )
    first_claim = threading.Event()
    release_second = threading.Event()
    errors: list[BaseException] = []
    accepted_positions: list[int] = []
    lock = threading.Lock()

    def _append_first() -> None:
        position = store_a.append(event_a, tenant_id=tenant_id).position.value
        with lock:
            accepted_positions.append(position)
        first_claim.set()
        release_second.wait(timeout=5)

    def _append_second() -> None:
        first_claim.wait(timeout=5)
        try:
            store_b.append(event_b, tenant_id=tenant_id)
        except BaseException as exc:
            errors.append(exc)
        finally:
            release_second.set()

    thread_a = threading.Thread(target=_append_first)
    thread_b = threading.Thread(target=_append_second)
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeEventPersistenceIntegrityError)
    assert len(accepted_positions) == 1
    assert backend.get(_run_partition(tenant_id, run_id), event_id) is not None
    store_a.close()

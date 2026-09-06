# © Artur Czarnecki. All rights reserved.

"""EventId identity and accepted persistence tenant semantics (DG-002 R1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.persistence_contract import (
    EVENT_ID_OWNERSHIP_SCHEMA_V1,
    NullRuntimeEventPersistence,
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
    build_runtime_event_identity_claim,
    encode_event_identity_claim,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
    _global_event_id_partition,
    _run_partition,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_BACKEND_LABELS = ("null", "memory", "sqlite", "document", "validating")


@pytest.fixture(params=_BACKEND_LABELS)
def persistence_backend(request: pytest.FixtureRequest, tmp_path: Path):
    label = request.param
    if label == "sqlite":
        store: RuntimeEventPersistence = SQLiteRuntimeEventStore(
            db_path=tmp_path / f"{label}_event_id.db",
        )
    elif label == "null":
        store = NullRuntimeEventPersistence()
    elif label == "memory":
        store = InMemoryRuntimeEventStore()
    elif label == "document":
        store = DocumentBackedRuntimeEventStore(InMemoryDocumentStore())
    elif label == "validating":
        store = ValidatingRuntimeEventPersistence(InMemoryRuntimeEventStore())
    else:
        raise AssertionError(f"unknown backend label: {label}")
    yield label, store
    store.close()


def test_explicit_tenant_lookup_when_event_tenant_none(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_id = f"{label}-explicit-tenant"
    event = sample_runtime_event(tenant_id=None)
    positioned = store.append(event, tenant_id=tenant_id)
    lookup = store.get_by_event_id(tenant_id=tenant_id, event_id=event.event_id)
    assert lookup is not None, label
    assert lookup.position == positioned.position, label


def test_wrong_tenant_lookup_returns_none(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_a = f"{label}-tenant-a"
    tenant_b = f"{label}-tenant-b"
    event = sample_runtime_event(tenant_id=tenant_a)
    store.append(event, tenant_id=tenant_a)
    assert store.get_by_event_id(tenant_id=tenant_b, event_id=event.event_id) is None, label


def test_exact_duplicate_returns_same_positioned_event(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_id = f"{label}-duplicate"
    event = sample_runtime_event(tenant_id=tenant_id)
    first = store.append(event, tenant_id=tenant_id)
    second = store.append(event, tenant_id=tenant_id)
    assert second.position == first.position, label
    assert store.get_by_event_id(tenant_id=tenant_id, event_id=event.event_id) == first, label


def test_conflicting_run_rejected(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_id = f"{label}-run-conflict"
    event_id = mint_event_id()
    task_id = mint_task_id()
    original = sample_runtime_event(tenant_id=tenant_id, event_id=event_id, task_id=task_id)
    store.append(original, tenant_id=tenant_id)
    conflicting = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        task_id=task_id,
        run_id=mint_run_id(),
    )
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="conflicts"):
        store.append(conflicting, tenant_id=tenant_id)


def test_conflicting_task_rejected(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_id = f"{label}-task-conflict"
    event_id = mint_event_id()
    run_id = mint_run_id()
    original = sample_runtime_event(tenant_id=tenant_id, event_id=event_id, run_id=run_id)
    store.append(original, tenant_id=tenant_id)
    conflicting = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=mint_task_id(),
    )
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="conflicts"):
        store.append(conflicting, tenant_id=tenant_id)


def test_conflicting_payload_rejected(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_id = f"{label}-payload-conflict"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_id = mint_task_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
    )
    store.append(original, tenant_id=tenant_id)
    conflicting = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
    )
    conflicting = conflicting.model_copy(
        update={"event_type": RuntimeEventType.STEP_COMPLETED},
    )
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="conflicts"):
        store.append(conflicting, tenant_id=tenant_id)


def test_cross_tenant_same_event_id_rejected(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_a = f"{label}-cross-a"
    tenant_b = f"{label}-cross-b"
    event_id = mint_event_id()
    original = sample_runtime_event(tenant_id=tenant_a, event_id=event_id)
    store.append(original, tenant_id=tenant_a)
    foreign = sample_runtime_event(tenant_id=tenant_b, event_id=event_id)
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="tenant"):
        store.append(foreign, tenant_id=tenant_b)


def test_same_event_different_explicit_tenant_rejected(
    persistence_backend: tuple[str, RuntimeEventPersistence],
) -> None:
    label, store = persistence_backend
    tenant_a = f"{label}-scope-a"
    tenant_b = f"{label}-scope-b"
    event = sample_runtime_event(tenant_id=None)
    store.append(event, tenant_id=tenant_a)
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="tenant"):
        store.append(event, tenant_id=tenant_b)


def test_document_backed_conflict_leaves_no_second_canonical_record(tmp_path: Path) -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-structural"
    event_id = mint_event_id()
    task_id = mint_task_id()
    run_a = mint_run_id()
    run_b = mint_run_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        task_id=task_id,
        run_id=run_a,
    )
    store.append(original, tenant_id=tenant_id)
    conflicting = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        task_id=task_id,
        run_id=run_b,
    )
    with pytest.raises(RuntimeEventPersistenceIntegrityError):
        store.append(conflicting, tenant_id=tenant_id)

    assert backend.get(_run_partition(tenant_id, run_a), event_id) is not None
    assert backend.get(_run_partition(tenant_id, run_b), event_id) is None
    ownership = backend.get(_global_event_id_partition(), event_id)
    assert ownership is not None
    expected_claim = build_runtime_event_identity_claim(
        persistence_tenant_id=tenant_id,
        event=original,
    )
    assert ownership.data == encode_event_identity_claim(expected_claim)
    assert ownership.data["schema_version"] == EVENT_ID_OWNERSHIP_SCHEMA_V1
    store.close()


def test_document_backed_exact_duplicate_repairs_task_projection() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-task-repair"
    event = sample_runtime_event(tenant_id=tenant_id)
    positioned = store.append(event, tenant_id=tenant_id)
    task_partition = f"{tenant_id}|task|{event.task_id}"
    backend.delete(task_partition, event.event_id)
    repaired = store.append(event, tenant_id=tenant_id)
    assert repaired.position == positioned.position
    assert backend.get(task_partition, event.event_id) is not None
    store.close()

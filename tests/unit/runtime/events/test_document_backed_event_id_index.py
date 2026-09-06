# © Artur Czarnecki. All rights reserved.

"""DocumentBackedRuntimeEventStore EventId index behavior (DG-002 persistence slice)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentQueryPageV1, DocumentRecord
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistenceIntegrityError
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
    _event_index_partition,
    _run_partition,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _QueryTrackingDocumentStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__()
        self.query_calls = 0

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        self.query_calls += 1
        return super().query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )


def test_event_index_exists_after_append() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-event-index"
    event = sample_runtime_event(tenant_id=tenant_id)
    store.append(event, tenant_id=tenant_id)
    index_record = backend.get(_event_index_partition(tenant_id), event.event_id)
    assert index_record is not None
    assert index_record.data == {"run_id": event.run_id}
    store.close()


def test_missing_event_index_repaired_by_idempotent_append() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-index-repair"
    event = sample_runtime_event(tenant_id=tenant_id)
    positioned = store.append(event, tenant_id=tenant_id)
    backend.delete(_event_index_partition(tenant_id), event.event_id)
    assert store.get_by_event_id(tenant_id=tenant_id, event_id=event.event_id) is None
    repaired = store.append(event, tenant_id=tenant_id)
    assert repaired.position == positioned.position
    lookup = store.get_by_event_id(tenant_id=tenant_id, event_id=event.event_id)
    assert lookup is not None
    assert lookup.position == positioned.position
    store.close()


def test_conflicting_event_index_fails_closed() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-index-conflict"
    event = sample_runtime_event(tenant_id=tenant_id)
    store.append(event, tenant_id=tenant_id)
    backend.put(
        DocumentRecord(
            partition_key=_event_index_partition(tenant_id),
            row_key=event.event_id,
            data={"run_id": mint_run_id()},
        )
    )
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="conflicts with expected run_id"):
        store.append(event, tenant_id=tenant_id)
    store.close()


def test_index_with_missing_canonical_record_fails_closed() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-missing-canonical"
    event = sample_runtime_event(tenant_id=tenant_id)
    store.append(event, tenant_id=tenant_id)
    backend.delete(_run_partition(tenant_id, event.run_id), event.event_id)
    with pytest.raises(
        RuntimeEventPersistenceIntegrityError,
        match="missing canonical run record",
    ):
        store.get_by_event_id(tenant_id=tenant_id, event_id=event.event_id)
    store.close()


def test_get_by_event_id_does_not_scan_partitions() -> None:
    backend = _QueryTrackingDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_id = "tenant-no-scan"
    event = sample_runtime_event(tenant_id=tenant_id)
    store.append(event, tenant_id=tenant_id)
    backend.query_calls = 0
    lookup = store.get_by_event_id(tenant_id=tenant_id, event_id=event.event_id)
    assert lookup is not None
    assert lookup.event.event_id == event.event_id
    assert backend.query_calls == 0
    store.close()


def test_tenant_isolation_for_event_id_lookup() -> None:
    backend = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(backend)
    tenant_a = "tenant-a-isolation"
    tenant_b = "tenant-b-isolation"
    event = sample_runtime_event(tenant_id=tenant_a)
    store.append(event, tenant_id=tenant_a)
    assert store.get_by_event_id(tenant_id=tenant_b, event_id=event.event_id) is None
    store.close()

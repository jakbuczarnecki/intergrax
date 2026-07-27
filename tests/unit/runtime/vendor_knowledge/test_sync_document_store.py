# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for DocumentStore Vendor Knowledge sync repositories."""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor, KnowledgeItemRevision
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCorruptState,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStatus,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncCheckpoint,
)


class _PlainDocumentStore:
    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        return None

    def put(self, document: DocumentRecord) -> None:
        return None

    def delete(self, partition_key: str, row_key: str) -> None:
        return None

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        return DocumentQueryResult(documents=[], total=0)

    def close(self) -> None:
        return None


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint(
    *,
    tenant_id: str = "tenant-1",
    binding_id: str = "binding-1",
    cursor_value: str = "cursor-1",
    version: int = 1,
) -> KnowledgeSyncCheckpoint:
    return KnowledgeSyncCheckpoint(
        tenant_id=tenant_id,
        binding_id=binding_id,
        binding_configuration_version=version,
        cursor=KnowledgeCursor(value=cursor_value, version="v1"),
    )


def _state(
    *,
    remote_id: str,
    delivery_id: str,
    status: KnowledgeRemoteItemStatus = KnowledgeRemoteItemStatus.ACTIVE,
    tenant_id: str = "tenant-1",
    binding_id: str = "binding-1",
    version: str = "1",
) -> KnowledgeRemoteItemState:
    return KnowledgeRemoteItemState(
        tenant_id=tenant_id,
        binding_id=binding_id,
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        remote_id=remote_id,
        status=status,
        revision=(
            KnowledgeItemRevision(version=version)
            if status is KnowledgeRemoteItemStatus.ACTIVE
            else None
        ),
        last_delivery_id=delivery_id,
    )


@pytest.mark.unit
def test_repositories_require_conditional_document_store() -> None:
    plain = _PlainDocumentStore()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentStoreKnowledgeSourceLeaseRepository(plain)
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentStoreKnowledgeSyncCheckpointRepository(plain)
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentStoreKnowledgeRemoteItemStateRepository(plain)


@pytest.mark.unit
def test_lease_acquire_missing_and_active_busy() -> None:
    store = InMemoryDocumentStore()
    clock = {"now": 100.0}
    repo = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: "token-a",
        record_version_factory=lambda: "rv-1",
    )
    lease = repo.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-1",
        ttl_seconds=30,
    )
    assert lease is not None
    assert lease.token == "token-a"
    busy = repo.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-2",
        ttl_seconds=30,
    )
    assert busy is None


@pytest.mark.unit
def test_lease_expired_takeover_and_single_winner() -> None:
    store = InMemoryDocumentStore()
    clock = {"now": 10.0}
    versions = iter(["rv-1", "rv-2", "rv-3"])
    tokens = iter(["token-old", "token-new", "token-race"])
    repo_a = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: next(tokens),
        record_version_factory=lambda: next(versions),
    )
    first = repo_a.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-1",
        ttl_seconds=5,
    )
    assert first is not None
    clock["now"] = 20.0
    repo_b = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: "token-winner",
        record_version_factory=lambda: "rv-winner",
    )
    winner = repo_b.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-2",
        ttl_seconds=5,
    )
    assert winner is not None
    assert winner.owner_id == "owner-2"
    loser = repo_a.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-3",
        ttl_seconds=5,
    )
    assert loser is None


@pytest.mark.unit
def test_lease_concurrent_acquire_single_winner() -> None:
    store = InMemoryDocumentStore()
    results: list[KnowledgeSourceLeaseToken | None] = []
    lock = threading.Lock()

    def _worker(owner: str) -> None:
        repo = DocumentStoreKnowledgeSourceLeaseRepository(store)
        lease = repo.acquire(
            tenant_id="tenant-1",
            binding_id="binding-1",
            owner_id=owner,
            ttl_seconds=30,
        )
        with lock:
            results.append(lease)

    threads = [threading.Thread(target=_worker, args=(f"owner-{idx}",)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    winners = [item for item in results if item is not None]
    assert len(winners) == 1


@pytest.mark.unit
def test_lease_release_idempotent_and_stale_token_safe() -> None:
    store = InMemoryDocumentStore()
    clock = {"now": 1.0}
    tokens = iter(["token-1", "token-candidate", "token-2"])
    versions = iter(["rv-1", "rv-candidate", "rv-2"])
    repo = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: next(tokens),
        record_version_factory=lambda: next(versions),
    )
    first = repo.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-1",
        ttl_seconds=5,
    )
    assert first is not None
    clock["now"] = 10.0
    second = repo.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-2",
        ttl_seconds=5,
    )
    assert second is not None
    repo.release(lease=first)
    still = store.get(
        "vendor_knowledge.source_lease.v1:tenant-1",
        "binding:binding-1",
    )
    assert still is not None
    assert still.data["token"] == "token-2"
    repo.release(lease=second)
    assert (
        store.get("vendor_knowledge.source_lease.v1:tenant-1", "binding:binding-1")
        is None
    )
    repo.release(lease=second)


@pytest.mark.unit
def test_lease_tenant_isolation_and_corrupt_hides_token() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: 1.0,
        token_factory=lambda: "secret-token-value",
        record_version_factory=lambda: "rv-1",
    )
    assert (
        repo.acquire(
            tenant_id="tenant-a",
            binding_id="binding-1",
            owner_id="owner-1",
            ttl_seconds=10,
        )
        is not None
    )
    assert (
        repo.acquire(
            tenant_id="tenant-b",
            binding_id="binding-1",
            owner_id="owner-1",
            ttl_seconds=10,
        )
        is not None
    )
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.source_lease.v1:tenant-1",
            row_key="binding:binding-1",
            data={"schema_version": "bad", "token": "secret-token-value"},
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState) as exc_info:
        repo.acquire(
            tenant_id="tenant-1",
            binding_id="binding-1",
            owner_id="owner-x",
            ttl_seconds=10,
        )
    assert "secret-token-value" not in str(exc_info.value)


@pytest.mark.unit
def test_checkpoint_create_get_and_conflicts() -> None:
    store = InMemoryDocumentStore()
    versions = iter(["rv-1", "rv-2", "rv-3"])
    repo = DocumentStoreKnowledgeSyncCheckpointRepository(
        store,
        record_version_factory=lambda: next(versions),
    )
    first = _checkpoint(cursor_value="cp-1")
    repo.commit(first, expected_previous=None)
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1") == first
    with pytest.raises(KnowledgeSyncCheckpointConflict):
        repo.commit(first, expected_previous=None)
    second = _checkpoint(cursor_value="cp-2")
    repo.commit(second, expected_previous=first)
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1") == second
    with pytest.raises(KnowledgeSyncCheckpointConflict):
        repo.commit(_checkpoint(cursor_value="cp-3"), expected_previous=first)
    with pytest.raises(KnowledgeSyncCheckpointConflict):
        DocumentStoreKnowledgeSyncCheckpointRepository(store).commit(
            _checkpoint(tenant_id="tenant-missing", binding_id="binding-1"),
            expected_previous=_checkpoint(tenant_id="tenant-missing"),
        )


@pytest.mark.unit
def test_checkpoint_aba_and_competing_repos() -> None:
    store = InMemoryDocumentStore()
    versions = iter([f"rv-{idx}" for idx in range(1, 20)])
    repo = DocumentStoreKnowledgeSyncCheckpointRepository(
        store,
        record_version_factory=lambda: next(versions),
    )
    cp1 = _checkpoint(cursor_value="cp-1")
    repo.commit(cp1, expected_previous=None)
    raw_before = store.get(
        "vendor_knowledge.sync_checkpoint.v1:tenant-1",
        "binding:binding-1",
    )
    assert raw_before is not None
    cp2 = _checkpoint(cursor_value="cp-2")
    repo.commit(cp2, expected_previous=cp1)
    repo.commit(cp1, expected_previous=cp2)
    # Public checkpoint restored to cp1, but stale raw document (old record_version) must not CAS.
    assert (
        store.replace_if_match(
            expected=raw_before,
            replacement=DocumentRecord(
                partition_key=raw_before.partition_key,
                row_key=raw_before.row_key,
                data={**dict(raw_before.data), "record_version": "stale"},
            ),
        )
        is False
    )

    store2 = InMemoryDocumentStore()
    repo_a = DocumentStoreKnowledgeSyncCheckpointRepository(
        store2,
        record_version_factory=lambda: "a",
    )
    repo_b = DocumentStoreKnowledgeSyncCheckpointRepository(
        store2,
        record_version_factory=lambda: "b",
    )
    repo_a.commit(_checkpoint(cursor_value="x"), expected_previous=None)
    with pytest.raises(KnowledgeSyncCheckpointConflict):
        repo_b.commit(_checkpoint(cursor_value="y"), expected_previous=None)


@pytest.mark.unit
def test_checkpoint_corrupt_hides_cursor() -> None:
    store = InMemoryDocumentStore()
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.sync_checkpoint.v1:tenant-1",
            row_key="binding:binding-1",
            data={
                "schema_version": "vendor_knowledge.sync_checkpoint.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": "rv",
                "checkpoint": {"cursor": {"value": "SECRET-CURSOR"}},
            },
        )
    )
    repo = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState) as exc_info:
        repo.get(tenant_id="tenant-1", binding_id="binding-1")
    assert "SECRET-CURSOR" not in str(exc_info.value)


@pytest.mark.unit
def test_remote_item_round_trip_and_statuses() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = _sha("d1")
    active = _state(remote_id="item-1", delivery_id=delivery)
    deleted = _state(
        remote_id="item-2",
        delivery_id=delivery,
        status=KnowledgeRemoteItemStatus.DELETED,
    )
    revoked = _state(
        remote_id="item-3",
        delivery_id=delivery,
        status=KnowledgeRemoteItemStatus.REVOKED,
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=(active, deleted, revoked),
    )
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="item-1") == active
    assert (
        repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="item-2") == deleted
    )
    assert (
        repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="item-3") == revoked
    )


@pytest.mark.unit
def test_remote_item_idempotent_and_fingerprint_fail_closed() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = _sha("same-delivery")
    states = (_state(remote_id="item-1", delivery_id=delivery),)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=(_state(remote_id="item-2", delivery_id=delivery),),
        )


@pytest.mark.unit
def test_remote_item_partial_resume_and_completed_noop() -> None:
    store = InMemoryDocumentStore()
    versions = iter([f"rv-{idx}" for idx in range(1, 50)])
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(
        store,
        record_version_factory=lambda: next(versions),
    )
    delivery = _sha("partial")
    first = _state(remote_id="a", delivery_id=delivery, version="1")
    second = _state(remote_id="b", delivery_id=delivery, version="1")
    # Seed applying marker and only first item.
    partition = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
    fingerprint = hashlib.sha256(
        __import__("json")
        .dumps(
            [first.model_dump(mode="json"), second.model_dump(mode="json")],
            sort_keys=True,
            separators=(",", ":"),
        )
        .encode("utf-8")
    ).hexdigest()
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=f"delivery:{delivery}",
            data={
                "schema_version": "vendor_knowledge.delivery_marker.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "delivery_id": delivery,
                "batch_fingerprint": fingerprint,
                "status": "applying",
                "record_version": "marker-1",
            },
        )
    )
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=f"item:{_sha('a')}",
            data={
                "schema_version": "vendor_knowledge.remote_item_state.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": "item-1",
                "state": first.model_dump(mode="json"),
            },
        )
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=(first, second),
    )
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="b") == second
    marker = store.get(partition, f"delivery:{delivery}")
    assert marker is not None
    assert marker.data["status"] == "completed"
    # completed marker: no further writes required
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=(first, second),
    )


@pytest.mark.unit
def test_remote_item_newer_delivery_cas_and_row_key_hash() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    d1 = _sha("delivery-1")
    d2 = _sha("delivery-2")
    remote_id = "ISSUE-42"
    first = _state(remote_id=remote_id, delivery_id=d1, version="1")
    second = _state(remote_id=remote_id, delivery_id=d2, version="2")
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=d1,
        states=(first,),
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=d2,
        states=(second,),
    )
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id=remote_id) == second
    row_key = f"item:{_sha(remote_id)}"
    assert remote_id not in row_key
    assert store.get(
        "vendor_knowledge.remote_item.v1:tenant-1:binding-1",
        row_key,
    ) is not None


@pytest.mark.unit
def test_remote_item_cas_conflict_fail_closed() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    d1 = _sha("d-cas-1")
    d2 = _sha("d-cas-2")
    state = _state(remote_id="item-1", delivery_id=d1)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=d1,
        states=(state,),
    )
    partition = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
    row_key = f"item:{_sha('item-1')}"
    current = store.get(partition, row_key)
    assert current is not None

    class _ConflictStore(InMemoryDocumentStore):
        def replace_if_match(
            self,
            *,
            expected: DocumentRecord,
            replacement: DocumentRecord,
        ) -> bool:
            if expected.row_key.startswith("item:"):
                return False
            return super().replace_if_match(expected=expected, replacement=replacement)

    conflict_store = _ConflictStore()
    # copy existing docs
    for doc in store.query(partition, limit=100).documents:
        conflict_store.put(doc)
    conflict_repo = DocumentStoreKnowledgeRemoteItemStateRepository(conflict_store)
    with pytest.raises(KnowledgeSyncCorruptState):
        conflict_repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=d2,
            states=(_state(remote_id="item-1", delivery_id=d2, version="9"),),
        )


@pytest.mark.unit
def test_remote_item_isolation_and_corrupt_records() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = _sha("iso")
    repo.apply_batch(
        tenant_id="tenant-a",
        binding_id="binding-1",
        delivery_id=delivery,
        states=(_state(remote_id="item-1", delivery_id=delivery, tenant_id="tenant-a"),),
    )
    assert (
        repo.get(tenant_id="tenant-b", binding_id="binding-1", remote_id="item-1") is None
    )
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.remote_item.v1:tenant-1:binding-1",
            row_key=f"item:{_sha('x')}",
            data={"schema_version": "bad", "state": {"remote_id": "x"}},
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="x")
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.remote_item.v1:tenant-1:binding-1",
            row_key=f"delivery:{delivery}",
            data={"schema_version": "bad", "batch_fingerprint": "fp", "status": "applying"},
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=(_state(remote_id="z", delivery_id=delivery),),
        )


@pytest.mark.unit
def test_batch_fingerprint_uses_sha256_not_python_hash() -> None:
    import json

    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = _sha("fp-check")
    states = (
        _state(remote_id="b", delivery_id=delivery),
        _state(remote_id="a", delivery_id=delivery),
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    marker = store.get(
        "vendor_knowledge.remote_item.v1:tenant-1:binding-1",
        f"delivery:{delivery}",
    )
    assert marker is not None
    fingerprint = str(marker.data["batch_fingerprint"])
    assert len(fingerprint) == 64
    ordered = [
        _state(remote_id="a", delivery_id=delivery).model_dump(mode="json"),
        _state(remote_id="b", delivery_id=delivery).model_dump(mode="json"),
    ]
    expected = hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert fingerprint == expected
    assert fingerprint != format(hash(json.dumps(ordered)), "x")

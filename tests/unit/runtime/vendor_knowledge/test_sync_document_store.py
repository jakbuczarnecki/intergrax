# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for DocumentStore Vendor Knowledge sync repositories."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from typing import Optional

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import (
    DocumentQueryResult,
    DocumentRecord,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeCursor,
    KnowledgeItemRevision,
)
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
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncPublicationFenceConflict,
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationInProgress,
    KnowledgeSyncPublicationPermitV1,
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
    store = InMemoryDocumentStore()
    DocumentStoreKnowledgeSourceLeaseRepository(store)
    DocumentStoreKnowledgeSyncCheckpointRepository(store)
    DocumentStoreKnowledgeRemoteItemStateRepository(store)


@pytest.mark.unit
def test_publication_fence_revision_cas_rejects_stale_writer() -> None:
    store = InMemoryDocumentStore()
    repository = DocumentStoreKnowledgeSyncPublicationFenceRepository(store)
    first = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        enabled=True,
        detached=False,
    )
    second = first.model_copy(
        update={"lifecycle_revision": 2, "lifecycle_token": "token-b"}
    )
    repository.write_fence(first, expected_revision=None)
    repository.write_fence(second, expected_revision=1)
    assert repository.read_fence(
        tenant_id="tenant-1", binding_id="binding-1"
    ) == second
    with pytest.raises(KnowledgeSyncPublicationFenceConflict):
        repository.write_fence(first, expected_revision=1)


@pytest.mark.unit
def test_publication_permit_linearizes_lifecycle_and_recovers_after_expiry() -> None:
    store = InMemoryDocumentStore()
    now = {"value": datetime(2026, 1, 1, 12, 0, tzinfo=UTC)}
    repository_a = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: now["value"],
        permit_id_factory=lambda: "permit-a",
    )
    repository_b = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: now["value"],
        permit_id_factory=lambda: "permit-b",
    )
    fence = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        enabled=True,
        detached=False,
    )
    repository_a.write_fence(fence, expected_revision=None)

    permit_a = repository_a.acquire_publication_permit(
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_revision=1,
        expected_token="token-a",
        owner_id="worker-a",
        ttl_seconds=5,
    )
    assert permit_a is not None
    assert isinstance(permit_a, KnowledgeSyncPublicationPermitV1)
    assert repository_b.acquire_publication_permit(
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_revision=1,
        expected_token="token-a",
        owner_id="worker-b",
        ttl_seconds=5,
    ) is None

    with pytest.raises(KnowledgeSyncPublicationInProgress, match="publication_in_progress"):
        repository_b.disable(
            tenant_id="tenant-1",
            binding_id="binding-1",
            lifecycle_revision=2,
            lifecycle_token="token-b",
            expected_revision=1,
        )

    assert repository_a.release_publication_permit(permit=permit_a) is True
    assert repository_a.release_publication_permit(permit=permit_a) is True
    disabled = repository_b.disable(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=2,
        lifecycle_token="token-b",
        expected_revision=1,
    )
    assert disabled.detached is False
    assert disabled.enabled is False
    enabled = repository_b.enable(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=3,
        lifecycle_token="token-c",
        expected_revision=2,
    )
    assert enabled.enabled is True
    assert repository_a.acquire_publication_permit(
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_revision=1,
        expected_token="token-a",
        owner_id="stale-worker",
        ttl_seconds=5,
    ) is None
    permit_c = repository_a.acquire_publication_permit(
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_revision=3,
        expected_token="token-c",
        owner_id="worker-c",
        ttl_seconds=5,
    )
    assert permit_c is not None
    with pytest.raises(KnowledgeSyncPublicationInProgress):
        repository_b.detach(
            tenant_id="tenant-1",
            binding_id="binding-1",
            lifecycle_revision=4,
            lifecycle_token="token-d",
            expected_revision=3,
        )
    assert repository_a.release_publication_permit(permit=permit_c) is True
    detached = repository_b.detach(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=4,
        lifecycle_token="token-d",
        expected_revision=3,
    )
    assert detached.detached is True


@pytest.mark.unit
def test_publication_permit_stale_owner_cannot_release_newer_permit() -> None:
    store = InMemoryDocumentStore()
    now = {"value": datetime(2026, 1, 1, 12, 0, tzinfo=UTC)}
    permits = iter(("permit-a", "permit-b"))
    repository_a = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: now["value"],
        permit_id_factory=lambda: next(permits),
    )
    repository_b = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: now["value"],
        permit_id_factory=lambda: "unused",
    )
    repository_a.write_fence(
        KnowledgeSyncPublicationFenceV1(
            tenant_id="tenant-1",
            binding_id="binding-1",
            lifecycle_revision=1,
            lifecycle_token="token-a",
            enabled=True,
            detached=False,
        ),
        expected_revision=None,
    )
    first = repository_a.acquire_publication_permit(
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_revision=1,
        expected_token="token-a",
        owner_id="worker-a",
        ttl_seconds=5,
    )
    assert first is not None
    now["value"] = datetime(2026, 1, 1, 12, 0, 6, tzinfo=UTC)
    second = repository_a.acquire_publication_permit(
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_revision=1,
        expected_token="token-a",
        owner_id="worker-b",
        ttl_seconds=5,
    )
    assert second is not None
    assert repository_b.release_publication_permit(permit=first) is False
    assert repository_a.is_current_publication_permit(permit=second) is True
    now["value"] = datetime(2026, 1, 1, 12, 0, 12, tzinfo=UTC)
    assert repository_a.is_current_publication_permit(permit=second) is False
    assert repository_a.release_publication_permit(permit=second) is True


@pytest.mark.unit
def test_two_independent_repositories_have_one_permit_winner() -> None:
    store = InMemoryDocumentStore()
    repository = DocumentStoreKnowledgeSyncPublicationFenceRepository(store)
    repository.write_fence(
        KnowledgeSyncPublicationFenceV1(
            tenant_id="tenant-1",
            binding_id="binding-1",
            lifecycle_revision=1,
            lifecycle_token="token-a",
            enabled=True,
            detached=False,
        ),
        expected_revision=None,
    )
    repositories = [
        DocumentStoreKnowledgeSyncPublicationFenceRepository(store),
        DocumentStoreKnowledgeSyncPublicationFenceRepository(store),
    ]
    results: list[KnowledgeSyncPublicationPermitV1 | None] = []
    lock = threading.Lock()

    def _acquire(repo: DocumentStoreKnowledgeSyncPublicationFenceRepository) -> None:
        permit = repo.acquire_publication_permit(
            tenant_id="tenant-1",
            binding_id="binding-1",
            expected_revision=1,
            expected_token="token-a",
            owner_id="independent-worker",
            ttl_seconds=30,
        )
        with lock:
            results.append(permit)

    threads = [threading.Thread(target=_acquire, args=(repo,)) for repo in repositories]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len([permit for permit in results if permit is not None]) == 1


@pytest.mark.unit
def test_lease_acquire_missing_and_active_busy() -> None:
    store = InMemoryDocumentStore()
    clock = {"now": 100.0}
    repo = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: "token-a",
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
def test_lease_expired_takeover_via_replace_if_match() -> None:
    store = InMemoryDocumentStore()
    clock = {"now": 10.0}
    tokens = iter(["token-old", "token-new"])
    versions = iter(["rv-old", "rv-new"])
    repo = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: next(tokens),
        version_factory=lambda: next(versions),
    )
    first = repo.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-1",
        ttl_seconds=5,
    )
    assert first is not None
    raw = store.get("vendor_knowledge.source_lease.v1:tenant-1", "binding:binding-1")
    assert raw is not None
    assert set(raw.data) == {
        "schema_version",
        "tenant_id",
        "binding_id",
        "owner_id",
        "token",
        "acquired_at_epoch",
        "expires_at_epoch",
        "record_version",
    }
    clock["now"] = 20.0
    winner = repo.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-2",
        ttl_seconds=5,
    )
    assert winner is not None
    assert winner.owner_id == "owner-2"
    assert winner.token == "token-new"
    taken = store.get("vendor_knowledge.source_lease.v1:tenant-1", "binding:binding-1")
    assert taken is not None
    assert taken.data["record_version"] == "rv-new"


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
def test_lease_stale_release_cannot_delete_new_lease() -> None:
    store = InMemoryDocumentStore()
    clock = {"now": 1.0}
    tokens = iter(["token-1", "token-2"])
    repo = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: next(tokens),
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


@pytest.mark.unit
def test_lease_tenant_isolation_and_corrupt_hides_token() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: 1.0,
        token_factory=lambda: "secret-token-value",
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
def test_checkpoint_create_get_cas_and_conflicts() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1") is None
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
def test_checkpoint_stale_writer_and_tenant_isolation() -> None:
    store = InMemoryDocumentStore()
    repo_a = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    repo_b = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    cp_a = _checkpoint(cursor_value="a")
    repo_a.commit(cp_a, expected_previous=None)
    cp_b = _checkpoint(cursor_value="b")
    repo_b.commit(cp_b, expected_previous=cp_a)
    with pytest.raises(KnowledgeSyncCheckpointConflict):
        repo_a.commit(_checkpoint(cursor_value="stale"), expected_previous=cp_a)

    repo_a.commit(
        _checkpoint(tenant_id="tenant-a", cursor_value="t-a"),
        expected_previous=None,
    )
    repo_b.commit(
        _checkpoint(tenant_id="tenant-b", cursor_value="t-b"),
        expected_previous=None,
    )
    assert (
        repo_a.get(tenant_id="tenant-a", binding_id="binding-1")
        == _checkpoint(tenant_id="tenant-a", cursor_value="t-a")
    )
    assert repo_a.get(tenant_id="tenant-b", binding_id="binding-1") == _checkpoint(
        tenant_id="tenant-b",
        cursor_value="t-b",
    )


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
                "record_version": "rv-1",
                "checkpoint": {
                    "tenant_id": "tenant-1",
                    "binding_id": "binding-1",
                    "binding_configuration_version": "not-an-int",
                    "cursor": {"value": "SECRET-CURSOR", "version": "v1"},
                },
            },
        )
    )
    repo = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState) as exc_info:
        repo.get(tenant_id="tenant-1", binding_id="binding-1")
    assert "SECRET-CURSOR" not in str(exc_info.value)


@pytest.mark.unit
def test_remote_item_round_trip_statuses_and_row_key_hash() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = _sha("d1")
    remote_id = "https://vendor.example/issues/42"
    active = _state(remote_id=remote_id, delivery_id=delivery)
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
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id=remote_id) == active
    assert (
        repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="item-2") == deleted
    )
    row_key = f"item:{_sha(remote_id)}"
    assert remote_id not in row_key
    assert store.get(
        "vendor_knowledge.remote_item.v1:tenant-1:binding-1",
        row_key,
    ) is not None


@pytest.mark.unit
def test_remote_item_replay_partial_and_marker_after_states() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = _sha("partial")
    first = _state(remote_id="a", delivery_id=delivery, version="1")
    second = _state(remote_id="b", delivery_id=delivery, version="1")
    partition = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
    # Crash after first state write — no delivery marker yet.
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=f"item:{_sha('a')}",
            data={
                "schema_version": "vendor_knowledge.remote_item_state.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": "rv-partial",
                "state": first.model_dump(mode="json"),
            },
        )
    )
    assert store.get(partition, f"delivery:{delivery}") is None
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=(first, second),
    )
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="b") == second
    marker = store.get(partition, f"delivery:{delivery}")
    assert marker is not None
    assert marker.data["batch_fingerprint"]
    # identical delivery replay is no-op
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=(first, second),
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=(_state(remote_id="only-other", delivery_id=delivery),),
        )


@pytest.mark.unit
def test_remote_item_cas_conflict_does_not_overwrite() -> None:
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
    kept = conflict_repo.get(
        tenant_id="tenant-1",
        binding_id="binding-1",
        remote_id="item-1",
    )
    assert kept == state


@pytest.mark.unit
def test_remote_item_tenant_isolation_and_active_requires_revision() -> None:
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
    with pytest.raises(Exception):
        KnowledgeRemoteItemState(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            remote_id="item-x",
            status=KnowledgeRemoteItemStatus.ACTIVE,
            revision=None,
            last_delivery_id=delivery,
        )


@pytest.mark.unit
def test_delivery_marker_rejects_foreign_partition() -> None:
    delivery = _sha("foreign-partition")
    states = (_state(remote_id="item-1", delivery_id=delivery),)
    fingerprint = hashlib.sha256(
        json.dumps(
            [states[0].model_dump(mode="json")],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    marker_row = f"delivery:{delivery}"
    marker_data = {
        "schema_version": "vendor_knowledge.delivery_marker.v1",
        "tenant_id": "tenant-1",
        "binding_id": "binding-1",
        "delivery_id": delivery,
        "batch_fingerprint": fingerprint,
        "status": "completed",
        "record_version": "rv-marker",
    }
    foreign_partition = "vendor_knowledge.remote_item.v1:other-tenant:other-binding"
    expected_partition = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
    writes: list[DocumentRecord] = []

    class _ForeignPartitionMarkerStore:
        def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
            if row_key == marker_row:
                return DocumentRecord(
                    partition_key=foreign_partition,
                    row_key=marker_row,
                    data=dict(marker_data),
                )
            return None

        def put(self, document: DocumentRecord) -> None:
            writes.append(document)

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

        def put_if_absent(self, document: DocumentRecord) -> bool:
            if document.row_key == marker_row:
                return False
            writes.append(document)
            return True

        def replace_if_match(
            self,
            *,
            expected: DocumentRecord,
            replacement: DocumentRecord,
        ) -> bool:
            writes.append(replacement)
            return True

        def delete_if_match(self, *, expected: DocumentRecord) -> bool:
            return False

    repo = DocumentStoreKnowledgeRemoteItemStateRepository(_ForeignPartitionMarkerStore())
    with pytest.raises(KnowledgeSyncCorruptState) as exc_info:
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=states,
        )
    message = str(exc_info.value)
    assert message == "delivery marker partition is invalid"
    assert foreign_partition not in message
    assert delivery not in message
    assert expected_partition not in message
    assert writes == []


@pytest.mark.unit
def test_batch_fingerprint_uses_sha256_not_python_hash() -> None:
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
    ordered = [
        _state(remote_id="a", delivery_id=delivery).model_dump(mode="json"),
        _state(remote_id="b", delivery_id=delivery).model_dump(mode="json"),
    ]
    expected = hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert fingerprint == expected
    assert fingerprint != format(hash(json.dumps(ordered)), "x")


@pytest.mark.unit
def test_delivery_marker_applying_resume_then_completed_noop() -> None:
    store = InMemoryDocumentStore()
    versions = iter([f"rv-{idx}" for idx in range(1, 20)])
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(
        store,
        version_factory=lambda: next(versions),
    )
    delivery = _sha("resume-delivery")
    states = (_state(remote_id="item-1", delivery_id=delivery),)
    partition = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
    fingerprint = hashlib.sha256(
        json.dumps(
            [states[0].model_dump(mode="json")],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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
                "record_version": "rv-applying",
            },
        )
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    assert (
        repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="item-1")
        == states[0]
    )
    marker = store.get(partition, f"delivery:{delivery}")
    assert marker is not None
    assert marker.data["status"] == "completed"
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    marker_again = store.get(partition, f"delivery:{delivery}")
    assert marker_again is not None
    assert marker_again.data["status"] == "completed"
    assert marker_again.data["record_version"] == marker.data["record_version"]

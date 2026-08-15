# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Active-index completeness lifecycle tests for durable reconciliation."""

from __future__ import annotations

import hashlib

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.vendor_knowledge.models import KnowledgeItemRevision
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeCandidateInventoryIncomplete,
    KnowledgeSyncCorruptState,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeReconciliationCandidateInventoryRepository,
    DocumentStoreKnowledgeRemoteItemStateRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStatus,
)

_ITEM_PARTITION = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
_INDEX_PARTITION = "vendor_knowledge.active_item_index.v1:tenant-1:binding-1:v1"


def _state(
    *,
    remote_id: str = "item-1",
    status: KnowledgeRemoteItemStatus = KnowledgeRemoteItemStatus.ACTIVE,
    delivery_id: str = "a" * 64,
    version: str = "1",
) -> KnowledgeRemoteItemState:
    return KnowledgeRemoteItemState(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        remote_id=remote_id,
        status=status,
        revision=KnowledgeItemRevision(version=version),
        last_delivery_id=delivery_id,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _legacy_state_row(
    remote_id: str,
    *,
    state: KnowledgeRemoteItemState | None = None,
) -> DocumentRecord:
    stored_state = state or _state(remote_id=remote_id)
    return DocumentRecord(
        partition_key=_ITEM_PARTITION,
        row_key=f"item:{_sha(remote_id)}",
        data={
            "schema_version": "vendor_knowledge.remote_item_state.v1",
            "tenant_id": "tenant-1",
            "binding_id": "binding-1",
            "record_version": "1",
            "state": stored_state.model_dump(mode="json"),
        },
    )


@pytest.mark.unit
def test_legacy_states_plus_new_active_write_remain_incomplete() -> None:
    store = InMemoryDocumentStore()
    store.put(_legacy_state_row("legacy-item"))
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    with pytest.raises(KnowledgeCandidateInventoryIncomplete):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id="b" * 64,
            states=(_state(remote_id="new-item", delivery_id="b" * 64),),
        )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeCandidateInventoryIncomplete):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_partial_legacy_replay_is_incomplete() -> None:
    store = InMemoryDocumentStore()
    delivery = "b" * 64
    expected = tuple(
        _state(remote_id=remote_id, delivery_id=delivery)
        for remote_id in ("item-a", "item-b", "item-c")
    )
    for state in expected[:2]:
        store.put(_legacy_state_row(state.remote_id, state=state))

    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    with pytest.raises(KnowledgeCandidateInventoryIncomplete):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=expected,
        )


@pytest.mark.unit
def test_extra_legacy_replay_row_is_incomplete() -> None:
    store = InMemoryDocumentStore()
    delivery = "c" * 64
    durable = tuple(
        _state(remote_id=remote_id, delivery_id=delivery)
        for remote_id in ("item-a", "item-b", "item-c")
    )
    for state in durable:
        store.put(_legacy_state_row(state.remote_id, state=state))

    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    with pytest.raises(KnowledgeCandidateInventoryIncomplete):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=durable[:2],
        )


@pytest.mark.unit
def test_wrong_content_legacy_replay_is_incomplete() -> None:
    store = InMemoryDocumentStore()
    delivery = "d" * 64
    expected = _state(remote_id="item-a", delivery_id=delivery, version="expected")
    durable = _state(remote_id="item-a", delivery_id=delivery, version="durable")
    store.put(_legacy_state_row(durable.remote_id, state=durable))

    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    with pytest.raises(KnowledgeCandidateInventoryIncomplete):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=(expected,),
        )


@pytest.mark.unit
def test_wrong_row_identity_legacy_replay_is_incomplete() -> None:
    store = InMemoryDocumentStore()
    delivery = "e" * 64
    expected = tuple(
        _state(remote_id=remote_id, delivery_id=delivery)
        for remote_id in ("item-a", "item-b")
    )
    durable = tuple(
        _state(remote_id=remote_id, delivery_id=delivery)
        for remote_id in ("item-a", "item-c")
    )
    for state in durable:
        store.put(_legacy_state_row(state.remote_id, state=state))

    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    with pytest.raises(KnowledgeCandidateInventoryIncomplete):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=expected,
        )


@pytest.mark.unit
def test_exact_legacy_replay_recovers_clean_inventory_without_manifest() -> None:
    store = InMemoryDocumentStore()
    delivery = "f" * 64
    states = tuple(
        _state(remote_id=remote_id, delivery_id=delivery)
        for remote_id in ("item-a", "item-b", "item-c")
    )
    for state in states:
        store.put(_legacy_state_row(state.remote_id, state=state))
    assert store.get(_INDEX_PARTITION, "manifest") is None

    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )

    manifest = store.get(_INDEX_PARTITION, "manifest")
    assert manifest is not None
    assert manifest.data["completeness_state"] == "CLEAN"
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    assert inventory.list_active_remote_ids(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        limit=10,
    ) == ("item-a", "item-b", "item-c")


@pytest.mark.unit
def test_fresh_empty_repository_supports_clean_empty_inventory() -> None:
    store = InMemoryDocumentStore()
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    assert (
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )
        == ()
    )


@pytest.mark.unit
def test_first_tombstone_only_batch_remains_complete() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id="c" * 64,
        states=(
            _state(
                remote_id="gone",
                status=KnowledgeRemoteItemStatus.DELETED,
                delivery_id="c" * 64,
            ),
        ),
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    assert (
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )
        == ()
    )


@pytest.mark.unit
def test_dirty_manifest_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id="d" * 64,
        states=(_state(delivery_id="d" * 64),),
    )
    manifest = store.get(_INDEX_PARTITION, "manifest")
    assert manifest is not None
    dirty = DocumentRecord(
        partition_key=_INDEX_PARTITION,
        row_key="manifest",
        data={
            **manifest.data,
            "completeness_state": "DIRTY",
            "inflight_delivery_id": "d" * 64,
            "inflight_batch_fingerprint": "e" * 64,
        },
    )
    store.put(dirty)
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeCandidateInventoryIncomplete):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_index_points_to_missing_state_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id="e" * 64,
        states=(_state(remote_id="orphan", delivery_id="e" * 64),),
    )
    item_rows = store.query(_ITEM_PARTITION, limit=10, row_key_prefix="item:")
    assert item_rows.documents
    store.delete(_ITEM_PARTITION, item_rows.documents[0].row_key)
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_completed_replay_repairs_missing_index_row() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = "f" * 64
    states = (_state(delivery_id=delivery),)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    index_rows = store.query(_INDEX_PARTITION, limit=10, row_key_prefix="active:")
    assert index_rows.documents
    store.delete(_INDEX_PARTITION, index_rows.documents[0].row_key)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    assert inventory.list_active_remote_ids(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        limit=10,
    ) == ("item-1",)


@pytest.mark.unit
def test_partial_index_mutation_retains_dirty_until_exact_retry() -> None:
    class _FailSecondIndexPut(InMemoryDocumentStore):
        def __init__(self) -> None:
            super().__init__()
            self._index_puts = 0

        def put(self, document: DocumentRecord) -> None:
            if (
                document.partition_key == _INDEX_PARTITION
                and document.row_key.startswith("active:")
            ):
                self._index_puts += 1
                if self._index_puts == 2:
                    raise RuntimeError("simulated index write failure")
            super().put(document)

    store = _FailSecondIndexPut()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = "a1" * 32
    states = (
        _state(remote_id="item-1", delivery_id=delivery),
        _state(remote_id="item-2", delivery_id=delivery),
    )
    with pytest.raises(RuntimeError, match="simulated index write failure"):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=states,
        )
    manifest = store.get(_INDEX_PARTITION, "manifest")
    assert manifest is not None
    assert manifest.data["completeness_state"] == "DIRTY"
    assert manifest.data["inflight_delivery_id"] == delivery
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    manifest = store.get(_INDEX_PARTITION, "manifest")
    assert manifest is not None
    assert manifest.data["completeness_state"] == "CLEAN"
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    assert inventory.list_active_remote_ids(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        limit=10,
    ) == ("item-1", "item-2")


@pytest.mark.unit
def test_foreign_delivery_cannot_take_over_dirty_manifest() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = "b1" * 32
    foreign = "c1" * 32
    states = (_state(remote_id="item-1", delivery_id=delivery),)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=delivery,
        states=states,
    )
    manifest = store.get(_INDEX_PARTITION, "manifest")
    assert manifest is not None
    dirty = DocumentRecord(
        partition_key=_INDEX_PARTITION,
        row_key="manifest",
        data={
            **manifest.data,
            "completeness_state": "DIRTY",
            "inflight_delivery_id": delivery,
            "inflight_batch_fingerprint": "d1" * 32,
        },
    )
    store.put(dirty)
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=foreign,
            states=(_state(remote_id="item-2", delivery_id=foreign),),
        )


@pytest.mark.unit
def test_mixed_configuration_batch_rejected_before_io() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    delivery = "e1" * 32
    mixed = (
        _state(remote_id="item-1", delivery_id=delivery),
        _state(
            remote_id="item-2",
            delivery_id=delivery,
        ).model_copy(update={"binding_configuration_version": 2}),
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=delivery,
            states=mixed,
        )


def _seed_clean_index(
    store: InMemoryDocumentStore,
    *,
    remote_id: str = "item-1",
) -> None:
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id="f1" * 32,
        states=(_state(remote_id=remote_id, delivery_id="f1" * 32),),
    )


@pytest.mark.unit
def test_invalid_dirty_manifest_delivery_digest_raises_corrupt_state() -> None:
    store = InMemoryDocumentStore()
    _seed_clean_index(store)
    manifest = store.get(_INDEX_PARTITION, "manifest")
    assert manifest is not None
    store.put(
        DocumentRecord(
            partition_key=_INDEX_PARTITION,
            row_key="manifest",
            data={
                **manifest.data,
                "completeness_state": "DIRTY",
                "inflight_delivery_id": "not-a-digest",
                "inflight_batch_fingerprint": "e" * 64,
            },
        )
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_invalid_dirty_manifest_batch_digest_raises_corrupt_state() -> None:
    store = InMemoryDocumentStore()
    _seed_clean_index(store)
    manifest = store.get(_INDEX_PARTITION, "manifest")
    assert manifest is not None
    store.put(
        DocumentRecord(
            partition_key=_INDEX_PARTITION,
            row_key="manifest",
            data={
                **manifest.data,
                "completeness_state": "DIRTY",
                "inflight_delivery_id": "d" * 64,
                "inflight_batch_fingerprint": "bad",
            },
        )
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_wrong_index_row_partition_raises_corrupt_state() -> None:
    store = InMemoryDocumentStore()
    _seed_clean_index(store)
    row = store.query(_INDEX_PARTITION, limit=1, row_key_prefix="active:").documents[0]
    store._rows[(_INDEX_PARTITION, row.row_key)] = DocumentRecord(  # type: ignore[attr-defined]
        partition_key="vendor_knowledge.active_item_index.v1:wrong",
        row_key=row.row_key,
        data=row.data,
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_wrong_index_payload_tenant_raises_corrupt_state() -> None:
    store = InMemoryDocumentStore()
    _seed_clean_index(store)
    row = store.query(_INDEX_PARTITION, limit=1, row_key_prefix="active:").documents[0]
    store.put(
        DocumentRecord(
            partition_key=_INDEX_PARTITION,
            row_key=row.row_key,
            data={**row.data, "tenant_id": "other-tenant"},
        )
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_wrong_index_payload_binding_raises_corrupt_state() -> None:
    store = InMemoryDocumentStore()
    _seed_clean_index(store)
    row = store.query(_INDEX_PARTITION, limit=1, row_key_prefix="active:").documents[0]
    store.put(
        DocumentRecord(
            partition_key=_INDEX_PARTITION,
            row_key=row.row_key,
            data={**row.data, "binding_id": "other-binding"},
        )
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )


@pytest.mark.unit
def test_wrong_index_payload_configuration_version_raises_corrupt_state() -> None:
    store = InMemoryDocumentStore()
    _seed_clean_index(store)
    row = store.query(_INDEX_PARTITION, limit=1, row_key_prefix="active:").documents[0]
    store.put(
        DocumentRecord(
            partition_key=_INDEX_PARTITION,
            row_key=row.row_key,
            data={**row.data, "binding_configuration_version": 2},
        )
    )
    inventory = DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(store)
    with pytest.raises(KnowledgeSyncCorruptState):
        inventory.list_active_remote_ids(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            limit=10,
        )

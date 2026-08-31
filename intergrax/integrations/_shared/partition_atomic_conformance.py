# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Conformance helpers for PartitionAtomicDocumentStore."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicBatch,
    PartitionAtomicBatchResult,
    PartitionAtomicDocumentStore,
    PartitionPutIfAbsentOnCreated,
    PartitionReplaceIfMatchOnCreated,
)
from intergrax.integrations._shared.conformance import assert_implements


def assert_partition_atomic_document_store(
    instance: object,
) -> PartitionAtomicDocumentStore:
    return assert_implements(instance, PartitionAtomicDocumentStore)


def assert_partition_atomic_batch_commit_both_or_neither(
    store: PartitionAtomicDocumentStore,
) -> None:
    primary = DocumentRecord(partition_key="p1", row_key="occ:1", data={"v": 1})
    metadata = DocumentRecord(partition_key="p1", row_key="meta:1", data={"generation": 1})
    result = store.execute_partition_atomic_batch(
        PartitionAtomicBatch(
            partition_key="p1",
            primary_put_if_absent=primary,
            on_created_ops=(PartitionPutIfAbsentOnCreated(document=metadata),),
        ),
    )
    assert result == PartitionAtomicBatchResult(primary_created=True)
    assert store.get("p1", "occ:1") == primary
    assert store.get("p1", "meta:1") == metadata


def assert_partition_atomic_batch_duplicate_skips_metadata(
    store: PartitionAtomicDocumentStore,
) -> None:
    primary = DocumentRecord(partition_key="p2", row_key="occ:1", data={"v": 1})
    metadata = DocumentRecord(partition_key="p2", row_key="meta:1", data={"generation": 1})
    first = store.execute_partition_atomic_batch(
        PartitionAtomicBatch(
            partition_key="p2",
            primary_put_if_absent=primary,
            on_created_ops=(PartitionPutIfAbsentOnCreated(document=metadata),),
        ),
    )
    assert first.primary_created is True
    second = store.execute_partition_atomic_batch(
        PartitionAtomicBatch(
            partition_key="p2",
            primary_put_if_absent=DocumentRecord(
                partition_key="p2",
                row_key="occ:1",
                data={"v": 2},
            ),
            on_created_ops=(
                PartitionPutIfAbsentOnCreated(
                    document=DocumentRecord(
                        partition_key="p2",
                        row_key="meta:1",
                        data={"generation": 99},
                    ),
                ),
            ),
        ),
    )
    assert second.primary_created is False
    assert store.get("p2", "meta:1") == metadata


def assert_partition_atomic_batch_replace_on_created(
    store: PartitionAtomicDocumentStore,
) -> None:
    initial_metadata = DocumentRecord(partition_key="p3", row_key="meta:1", data={"generation": 1})
    store.put(initial_metadata)
    primary = DocumentRecord(partition_key="p3", row_key="occ:2", data={"v": 1})
    replacement = DocumentRecord(partition_key="p3", row_key="meta:1", data={"generation": 2})
    result = store.execute_partition_atomic_batch(
        PartitionAtomicBatch(
            partition_key="p3",
            primary_put_if_absent=primary,
            on_created_ops=(
                PartitionReplaceIfMatchOnCreated(
                    expected=initial_metadata,
                    replacement=replacement,
                ),
            ),
        ),
    )
    assert result.primary_created is True
    assert store.get("p3", "occ:2") == primary
    assert store.get("p3", "meta:1") == replacement


def assert_partition_atomic_batch_sequential_metadata_bumps(
    store: PartitionAtomicDocumentStore,
) -> None:
    metadata = DocumentRecord(partition_key="p4", row_key="meta:1", data={"generation": 0})
    store.put(metadata)
    for index in range(1, 4):
        result = store.execute_partition_atomic_batch(
            PartitionAtomicBatch(
                partition_key="p4",
                primary_put_if_absent=DocumentRecord(
                    partition_key="p4",
                    row_key=f"occ:{index}",
                    data={"v": index},
                ),
                on_created_ops=(
                    PartitionReplaceIfMatchOnCreated(
                        expected=metadata,
                        replacement=DocumentRecord(
                            partition_key="p4",
                            row_key="meta:1",
                            data={"generation": index},
                        ),
                    ),
                ),
            ),
        )
        assert result.primary_created is True
        metadata = store.get("p4", "meta:1")
        assert metadata is not None
        assert metadata.data["generation"] == index


def assert_partition_atomic_document_store_semantics(
    store: PartitionAtomicDocumentStore,
) -> None:
    assert_partition_atomic_batch_commit_both_or_neither(store)
    assert_partition_atomic_batch_duplicate_skips_metadata(store)
    assert_partition_atomic_batch_replace_on_created(store)
    assert_partition_atomic_batch_sequential_metadata_bumps(store)

# © Artur Czarnecki. All rights reserved.

"""PostgreSQL partition atomic batch — cross-client concurrency qualification."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from intergrax.integrations._shared.partition_atomic_conformance import (
    assert_partition_atomic_document_store_semantics,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicBatch,
    PartitionAtomicDocumentStore,
    PartitionPutIfAbsentOnCreated,
    PartitionReplaceIfMatchOnCreated,
)
from intergrax.integrations.providers.document_store.postgresql.bundle import (
    create_postgresql_document_store,
)
from testing_support.uca6c_r6_r5_9_r4_postgresql_durable_document_store import (
    drop_postgresql_document_schema,
    fresh_postgresql_document_store_client,
    resolve_uca6c_postgresql_document_dsn,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]

_SCHEMA_PREFIX = "uca6c_r59_r4_atomic_"


@pytest.fixture
def isolated_postgresql_partition_atomic_store() -> Iterator[
    tuple[str, str, PartitionAtomicDocumentStore]
]:
    dsn = resolve_uca6c_postgresql_document_dsn()
    schema_name = f"{_SCHEMA_PREFIX}{uuid.uuid4().hex}"
    store = create_postgresql_document_store(tenant_schema=schema_name, dsn=dsn)
    assert isinstance(store, PartitionAtomicDocumentStore)
    yield schema_name, dsn, store
    drop_postgresql_document_schema(schema_name, dsn=dsn)


def test_postgresql_partition_atomic_conformance_semantics(
    isolated_postgresql_partition_atomic_store: tuple[
        str, str, PartitionAtomicDocumentStore
    ],
) -> None:
    _, _, store = isolated_postgresql_partition_atomic_store
    assert_partition_atomic_document_store_semantics(store)


def test_concurrent_same_primary_put_if_absent(
    isolated_postgresql_partition_atomic_store: tuple[
        str, str, PartitionAtomicDocumentStore
    ],
) -> None:
    schema_name, dsn, _ = isolated_postgresql_partition_atomic_store
    store_a = fresh_postgresql_document_store_client(tenant_schema=schema_name, dsn=dsn)
    store_b = fresh_postgresql_document_store_client(tenant_schema=schema_name, dsn=dsn)
    assert isinstance(store_a, PartitionAtomicDocumentStore)
    assert isinstance(store_b, PartitionAtomicDocumentStore)
    batch = PartitionAtomicBatch(
        partition_key="race-primary",
        primary_put_if_absent=DocumentRecord(
            partition_key="race-primary",
            row_key="occ:1",
            data={"host": "contender"},
        ),
    )

    def _run(store: PartitionAtomicDocumentStore) -> bool:
        return store.execute_partition_atomic_batch(batch).primary_created

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(_run, (store_a, store_b)))

    assert sorted(outcomes) == [False, True]
    winner_count = sum(1 for created in outcomes if created)
    assert winner_count == 1


def test_concurrent_claim_like_metadata_bump(
    isolated_postgresql_partition_atomic_store: tuple[
        str, str, PartitionAtomicDocumentStore
    ],
) -> None:
    schema_name, dsn, seed_store = isolated_postgresql_partition_atomic_store
    metadata = DocumentRecord(
        partition_key="claim-partition",
        row_key="meta:1",
        data={"generation": 0},
    )
    seed_store.put(metadata)

    store_a = fresh_postgresql_document_store_client(tenant_schema=schema_name, dsn=dsn)
    store_b = fresh_postgresql_document_store_client(tenant_schema=schema_name, dsn=dsn)
    assert isinstance(store_a, PartitionAtomicDocumentStore)
    assert isinstance(store_b, PartitionAtomicDocumentStore)

    def _claim(store: PartitionAtomicDocumentStore, host: str) -> bool:
        expected = seed_store.get("claim-partition", "meta:1")
        assert expected is not None
        return store.execute_partition_atomic_batch(
            PartitionAtomicBatch(
                partition_key="claim-partition",
                primary_put_if_absent=DocumentRecord(
                    partition_key="claim-partition",
                    row_key="occ:claim",
                    data={"owner": host},
                ),
                on_created_ops=(
                    PartitionReplaceIfMatchOnCreated(
                        expected=expected,
                        replacement=DocumentRecord(
                            partition_key="claim-partition",
                            row_key="meta:1",
                            data={"generation": 1, "owner": host},
                        ),
                    ),
                ),
            ),
        ).primary_created

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_claim, store_a, "A")
        future_b = pool.submit(_claim, store_b, "B")
        created_a = future_a.result()
        created_b = future_b.result()

    assert sorted([created_a, created_b]) == [False, True]
    final_meta = seed_store.get("claim-partition", "meta:1")
    assert final_meta is not None
    assert final_meta.data.get("generation") == 1
    owner = final_meta.data.get("owner")
    assert owner in {"A", "B"}
    claim_row = seed_store.get("claim-partition", "occ:claim")
    assert claim_row is not None
    assert claim_row.data.get("owner") == owner


def test_on_created_cas_failure_rolls_back_entire_batch(
    isolated_postgresql_partition_atomic_store: tuple[
        str, str, PartitionAtomicDocumentStore
    ],
) -> None:
    _, _, store = isolated_postgresql_partition_atomic_store
    stale_expected = DocumentRecord(
        partition_key="rollback-partition",
        row_key="meta:1",
        data={"generation": 99},
    )
    batch = PartitionAtomicBatch(
        partition_key="rollback-partition",
        primary_put_if_absent=DocumentRecord(
            partition_key="rollback-partition",
            row_key="occ:1",
            data={"v": 1},
        ),
        on_created_ops=(
            PartitionReplaceIfMatchOnCreated(
                expected=stale_expected,
                replacement=DocumentRecord(
                    partition_key="rollback-partition",
                    row_key="meta:1",
                    data={"generation": 100},
                ),
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="partition_atomic_batch_on_created_stale"):
        store.execute_partition_atomic_batch(batch)

    assert store.get("rollback-partition", "occ:1") is None
    assert store.get("rollback-partition", "meta:1") is None


def test_concurrent_different_partitions_both_succeed(
    isolated_postgresql_partition_atomic_store: tuple[
        str, str, PartitionAtomicDocumentStore
    ],
) -> None:
    schema_name, dsn, _ = isolated_postgresql_partition_atomic_store
    store_a = fresh_postgresql_document_store_client(tenant_schema=schema_name, dsn=dsn)
    store_b = fresh_postgresql_document_store_client(tenant_schema=schema_name, dsn=dsn)
    assert isinstance(store_a, PartitionAtomicDocumentStore)
    assert isinstance(store_b, PartitionAtomicDocumentStore)

    def _run_partition(
        store: PartitionAtomicDocumentStore,
        partition_key: str,
    ) -> bool:
        return store.execute_partition_atomic_batch(
            PartitionAtomicBatch(
                partition_key=partition_key,
                primary_put_if_absent=DocumentRecord(
                    partition_key=partition_key,
                    row_key="occ:1",
                    data={"partition": partition_key},
                ),
                on_created_ops=(
                    PartitionPutIfAbsentOnCreated(
                        document=DocumentRecord(
                            partition_key=partition_key,
                            row_key="meta:1",
                            data={"ok": True},
                        ),
                    ),
                ),
            ),
        ).primary_created

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_run_partition, store_a, "partition-a"),
            pool.submit(_run_partition, store_b, "partition-b"),
        ]
        outcomes = [future.result() for future in as_completed(futures)]

    assert outcomes == [True, True]
    assert store_a.get("partition-a", "occ:1") is not None
    assert store_a.get("partition-b", "occ:1") is not None
    assert store_b.get("partition-a", "meta:1") is not None
    assert store_b.get("partition-b", "meta:1") is not None

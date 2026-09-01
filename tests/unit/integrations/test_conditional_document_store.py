# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for optional ConditionalDocumentStore capability."""

from __future__ import annotations

import threading

import pytest

from intergrax.integrations._shared.conformance import (
    assert_conditional_document_store,
    assert_document_store,
)
from intergrax.integrations._shared.partition_atomic_conformance import (
    assert_partition_atomic_document_store,
    assert_partition_atomic_document_store_semantics,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicDocumentStore,
)

pytestmark = pytest.mark.unit


def test_in_memory_implements_document_store() -> None:
    store = InMemoryDocumentStore()
    assert isinstance(store, DocumentStore)
    assert_document_store(store)


def test_in_memory_implements_conditional_document_store() -> None:
    store = InMemoryDocumentStore()
    assert isinstance(store, ConditionalDocumentStore)
    assert_conditional_document_store(store)


def test_in_memory_implements_partition_atomic_document_store() -> None:
    store = InMemoryDocumentStore()
    assert isinstance(store, PartitionAtomicDocumentStore)
    assert_partition_atomic_document_store(store)
    assert_partition_atomic_document_store_semantics(store)


def test_put_if_absent_writes_missing_record() -> None:
    store = InMemoryDocumentStore()
    document = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})

    assert store.put_if_absent(document) is True
    assert store.get("p1", "r1") == document


def test_second_put_if_absent_returns_false() -> None:
    store = InMemoryDocumentStore()
    first = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    second = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 2})

    assert store.put_if_absent(first) is True
    assert store.put_if_absent(second) is False


def test_second_put_if_absent_does_not_overwrite() -> None:
    store = InMemoryDocumentStore()
    first = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    second = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 2})

    store.put_if_absent(first)
    store.put_if_absent(second)

    assert store.get("p1", "r1") == first


def test_replace_if_match_updates_current_expected() -> None:
    store = InMemoryDocumentStore()
    expected = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    replacement = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 2})
    store.put(expected)

    assert store.replace_if_match(expected=expected, replacement=replacement) is True
    assert store.get("p1", "r1") == replacement


def test_stale_replace_if_match_returns_false() -> None:
    store = InMemoryDocumentStore()
    current = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    stale = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 0})
    replacement = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 2})
    store.put(current)

    assert store.replace_if_match(expected=stale, replacement=replacement) is False


def test_stale_replace_if_match_does_not_modify() -> None:
    store = InMemoryDocumentStore()
    current = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    stale = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 0})
    replacement = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 2})
    store.put(current)

    store.replace_if_match(expected=stale, replacement=replacement)
    assert store.get("p1", "r1") == current


def test_replace_rejects_different_partition_key() -> None:
    store = InMemoryDocumentStore()
    expected = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    replacement = DocumentRecord(partition_key="p2", row_key="r1", data={"v": 2})

    with pytest.raises(ValueError, match="partition_key"):
        store.replace_if_match(expected=expected, replacement=replacement)


def test_replace_rejects_different_row_key() -> None:
    store = InMemoryDocumentStore()
    expected = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    replacement = DocumentRecord(partition_key="p1", row_key="r2", data={"v": 2})

    with pytest.raises(ValueError, match="row_key"):
        store.replace_if_match(expected=expected, replacement=replacement)


def test_delete_if_match_removes_matching_record() -> None:
    store = InMemoryDocumentStore()
    expected = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    store.put(expected)

    assert store.delete_if_match(expected=expected) is True
    assert store.get("p1", "r1") is None


def test_stale_delete_if_match_returns_false() -> None:
    store = InMemoryDocumentStore()
    current = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    stale = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 0})
    store.put(current)

    assert store.delete_if_match(expected=stale) is False
    assert store.get("p1", "r1") == current


def test_ttl_seconds_is_not_part_of_conditional_match() -> None:
    store = InMemoryDocumentStore()
    stored = DocumentRecord(
        partition_key="p1",
        row_key="r1",
        data={"v": 1},
        ttl_seconds=10,
    )
    expected = DocumentRecord(
        partition_key="p1",
        row_key="r1",
        data={"v": 1},
        ttl_seconds=999,
    )
    replacement = DocumentRecord(
        partition_key="p1",
        row_key="r1",
        data={"v": 2},
        ttl_seconds=None,
    )
    store.put(stored)

    assert store.replace_if_match(expected=expected, replacement=replacement) is True
    assert store.get("p1", "r1") == replacement


def test_concurrent_put_if_absent_has_single_winner() -> None:
    store = InMemoryDocumentStore()
    document = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    thread_count = 16
    barrier = threading.Barrier(thread_count)
    results: list[bool] = []
    lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()
        won = store.put_if_absent(document)
        with lock:
            results.append(won)

    threads = [threading.Thread(target=_worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == thread_count - 1
    assert store.get("p1", "r1") == document
    assert store.query("p1").total == 1


def test_base_crud_still_works() -> None:
    store = InMemoryDocumentStore()
    document = DocumentRecord(partition_key="p1", row_key="r1", data={"v": 1})
    store.put(document)
    assert store.get("p1", "r1") == document
    assert store.query("p1").total == 1
    store.delete("p1", "r1")
    assert store.get("p1", "r1") is None
    store.put(document)
    store.close()
    assert store.get("p1", "r1") is None


def test_cursor_query_returns_1200_rows_without_duplicates() -> None:
    store = InMemoryDocumentStore()
    for index in range(1200):
        store.put(
            DocumentRecord(
                partition_key="p1",
                row_key=f"row-{index:04d}",
                data={"index": index},
            )
        )

    cursor = None
    rows: list[str] = []
    while True:
        page = store.query("p1", limit=137, cursor=cursor)
        rows.extend(document.row_key for document in page.documents)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert rows == [f"row-{index:04d}" for index in range(1200)]
    assert len(set(rows)) == 1200


def test_cursor_is_bound_to_partition_and_prefix() -> None:
    store = InMemoryDocumentStore()
    store.put(DocumentRecord(partition_key="p1", row_key="a:1"))
    store.put(DocumentRecord(partition_key="p1", row_key="b:1"))
    cursor = store.query("p1", row_key_prefix="a:", limit=1).next_cursor
    assert cursor is None

    store.put(DocumentRecord(partition_key="p1", row_key="a:2"))
    cursor = store.query("p1", row_key_prefix="a:", limit=1).next_cursor
    assert cursor is not None
    with pytest.raises(ValueError, match="query_mismatch"):
        store.query("p2", row_key_prefix="a:", limit=1, cursor=cursor)
    with pytest.raises(ValueError, match="query_mismatch"):
        store.query("p1", row_key_prefix="b:", limit=1, cursor=cursor)


def test_cursor_rejects_malformed_tokens_and_invalid_limits() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(ValueError, match="cursor_invalid"):
        store.query("p1", limit=1, cursor="tampered")
    for limit in (0, -1, 5001):
        with pytest.raises(ValueError, match="limit_invalid"):
            store.query("p1", limit=limit)


def test_cursor_continues_after_returned_rows_are_deleted() -> None:
    store = InMemoryDocumentStore()
    for index in range(5):
        store.put(DocumentRecord(partition_key="p1", row_key=f"row-{index}"))

    first = store.query("p1", limit=2)
    assert first.next_cursor is not None
    for document in first.documents:
        store.delete(document.partition_key, document.row_key)

    second = store.query("p1", limit=10, cursor=first.next_cursor)
    assert [document.row_key for document in second.documents] == [
        "row-2",
        "row-3",
        "row-4",
    ]
    assert second.next_cursor is None

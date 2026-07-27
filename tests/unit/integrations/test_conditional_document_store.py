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
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
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

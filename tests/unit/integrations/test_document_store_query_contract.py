# © Artur Czarnecki. All rights reserved.

"""Generic DocumentStore query contract tests (PROVIDER-QUAL-4-R2)."""

from __future__ import annotations

import base64
import json

import pytest

from intergrax.integrations._shared.document_store_query_support import (
    build_mongo_keyset_filter,
    compare_document_sort_keys,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentRecord,
)
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from tests.unit.integrations.providers.document_store.test_mongodb import (
    _collection_factory,
    _mongodb_config,
)

pytestmark = pytest.mark.unit


def _codec() -> DocumentQueryCursorCodec:
    return DocumentQueryCursorCodec(secret=b"query-contract-test-secret")


def _store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore(cursor_codec=_codec())


def _record(
    row_key: str,
    *,
    partition_key: str = "partition-a",
    data: dict[str, object] | None = None,
) -> DocumentRecord:
    return DocumentRecord(partition_key=partition_key, row_key=row_key, data=data or {})


def _seed_numeric_sort(
    store: InMemoryDocumentStore,
    *,
    field: str,
    values: tuple[int, ...],
    prefix: str = "row-",
) -> list[str]:
    row_keys: list[str] = []
    for index, value in enumerate(values):
        row_key = f"{prefix}{index:03d}"
        row_keys.append(row_key)
        store.put(_record(row_key, data={field: value}))
    return row_keys


def _paginate_all(
    store: InMemoryDocumentStore,
    *,
    partition_key: str = "partition-a",
    sort: tuple[DocumentDataSort, ...] = (),
    equalities: tuple[DocumentDataEquality, ...] = (),
    row_key_upper_bound: str | None = None,
    limit: int = 2,
) -> list[str]:
    collected: list[str] = []
    cursor: str | None = None
    while True:
        page = store.query(
            partition_key,
            limit=limit,
            sort=sort,
            data_equalities=equalities,
            row_key_upper_bound=row_key_upper_bound,
            cursor=cursor,
        )
        collected.extend(doc.row_key for doc in page.documents)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return collected


def test_ascending_pagination_returns_all_rows_in_order() -> None:
    store = _store()
    row_keys = _seed_numeric_sort(store, field="score", values=(10, 20, 30, 40, 50))
    sort = (DocumentDataSort(path="score", direction="asc"), DocumentDataSort(path="$row_key", direction="asc"))

    actual = _paginate_all(store, sort=sort, limit=2)

    assert actual == row_keys
    assert len(actual) == len(set(actual)) == 5


def test_descending_pagination_returns_all_rows_in_order() -> None:
    store = _store()
    row_keys = _seed_numeric_sort(store, field="score", values=(100, 90, 80, 70, 60))
    sort = (DocumentDataSort(path="score", direction="desc"), DocumentDataSort(path="$row_key", direction="desc"))

    actual = _paginate_all(store, sort=sort, limit=2)

    assert actual == row_keys
    assert len(actual) == len(set(actual)) == 5


def test_ties_resolved_by_secondary_sort_and_row_key() -> None:
    store = _store()
    store.put(_record("row-a", data={"group": 1, "seq": 2}))
    store.put(_record("row-b", data={"group": 1, "seq": 1}))
    store.put(_record("row-c", data={"group": 2, "seq": 9}))
    sort = (
        DocumentDataSort(path="group", direction="desc"),
        DocumentDataSort(path="seq", direction="asc"),
        DocumentDataSort(path="$row_key", direction="asc"),
    )

    actual = _paginate_all(store, sort=sort, limit=1)

    assert actual == ["row-c", "row-b", "row-a"]


def test_mixed_direction_sort_paginates_lexicographically() -> None:
    store = _store()
    store.put(_record("row-1", data={"primary": 10, "secondary": 5}))
    store.put(_record("row-2", data={"primary": 10, "secondary": 1}))
    store.put(_record("row-3", data={"primary": 20, "secondary": 9}))
    sort = (
        DocumentDataSort(path="primary", direction="desc"),
        DocumentDataSort(path="secondary", direction="asc"),
        DocumentDataSort(path="$row_key", direction="asc"),
    )

    actual = _paginate_all(store, sort=sort, limit=1)

    assert actual == ["row-3", "row-2", "row-1"]


def test_cursor_from_equality_query_rejected_for_different_equality() -> None:
    store = _store()
    store.put(_record("pg-a", data={"provider_id": "postgresql"}))
    store.put(_record("pg-b", data={"provider_id": "postgresql"}))
    store.put(_record("ora-a", data={"provider_id": "oracle"}))
    equalities_a = (DocumentDataEquality(path="provider_id", value="postgresql"),)
    first = store.query("partition-a", limit=1, data_equalities=equalities_a)
    assert first.next_cursor is not None

    equalities_b = (DocumentDataEquality(path="provider_id", value="oracle"),)
    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        store.query("partition-a", limit=1, data_equalities=equalities_b, cursor=first.next_cursor)


def test_cursor_from_sorted_query_rejected_with_changed_sort() -> None:
    store = _store()
    _seed_numeric_sort(store, field="score", values=(1, 2, 3))
    sort_desc = (DocumentDataSort(path="score", direction="desc"),)
    sort_asc = (DocumentDataSort(path="score", direction="asc"),)
    first = store.query("partition-a", limit=1, sort=sort_desc)
    assert first.next_cursor is not None

    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        store.query("partition-a", limit=1, sort=sort_asc, cursor=first.next_cursor)


def test_cursor_rejected_with_changed_row_key_upper_bound() -> None:
    store = _store()
    for row_key in ("row-010", "row-020", "row-030"):
        store.put(_record(row_key))
    first = store.query("partition-a", limit=1, row_key_upper_bound="row-025")
    assert first.next_cursor is not None

    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        store.query(
            "partition-a",
            limit=1,
            row_key_upper_bound="row-030",
            cursor=first.next_cursor,
        )


def test_legacy_simple_v1_cursor_still_works() -> None:
    store = _store()
    for row_key in ("row-1", "row-2", "row-3"):
        store.put(_record(row_key))

    first = store.query("partition-a", limit=1)
    assert first.next_cursor is not None
    envelope = json.loads(
        base64.urlsafe_b64decode(first.next_cursor + "==").decode("utf-8"),
    )
    assert envelope["payload"]["schema_version"] == "document_store.cursor.v1"

    second = store.query("partition-a", limit=1, cursor=first.next_cursor)
    assert [doc.row_key for doc in second.documents] == ["row-2"]


def test_extended_equality_query_uses_v2_cursor() -> None:
    store = _store()
    store.put(_record("row-1", data={"provider_id": "postgresql"}))
    store.put(_record("row-2", data={"provider_id": "postgresql"}))
    equalities = (DocumentDataEquality(path="provider_id", value="postgresql"),)
    first = store.query("partition-a", limit=1, data_equalities=equalities)
    assert first.next_cursor is not None
    envelope = json.loads(
        base64.urlsafe_b64decode(first.next_cursor + "==").decode("utf-8"),
    )
    assert envelope["payload"]["schema_version"] == "document_store.cursor.v2"

    second = store.query(
        "partition-a",
        limit=1,
        data_equalities=equalities,
        cursor=first.next_cursor,
    )
    assert [doc.row_key for doc in second.documents] == ["row-2"]


def test_equality_order_is_canonical_for_cursor_binding() -> None:
    store = _store()
    store.put(_record("row-1", data={"alpha": 1, "beta": 2}))
    store.put(_record("row-2", data={"alpha": 1, "beta": 2}))
    equalities_ab = (
        DocumentDataEquality(path="alpha", value=1),
        DocumentDataEquality(path="beta", value=2),
    )
    equalities_ba = (
        DocumentDataEquality(path="beta", value=2),
        DocumentDataEquality(path="alpha", value=1),
    )
    first = store.query("partition-a", limit=1, data_equalities=equalities_ab)
    assert first.next_cursor is not None

    second = store.query(
        "partition-a",
        limit=1,
        data_equalities=equalities_ba,
        cursor=first.next_cursor,
    )
    assert [doc.row_key for doc in second.documents] == ["row-2"]


def test_mongo_keyset_filter_uses_correct_operators() -> None:
    sort = (
        DocumentDataSort(path="a", direction="desc"),
        DocumentDataSort(path="b", direction="asc"),
    )
    keyset = build_mongo_keyset_filter(sort, ("a-value", "b-value"))

    assert keyset == {
        "$or": [
            {"data.a": {"$lt": "a-value"}},
            {"data.a": "a-value", "data.b": {"$gt": "b-value"}},
        ],
    }


def test_compare_document_sort_keys_supports_mixed_directions() -> None:
    sort = (
        DocumentDataSort(path="a", direction="desc"),
        DocumentDataSort(path="b", direction="asc"),
    )
    assert compare_document_sort_keys((10, 1), (10, 2), sort) < 0
    assert compare_document_sort_keys((10, 3), (10, 2), sort) > 0
    assert compare_document_sort_keys((11, 0), (10, 9), sort) < 0


def test_mongodb_adapter_ascending_pagination_matches_in_memory_semantics() -> None:
    factory, collection = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    for index, score in enumerate((10, 20, 30, 40)):
        collection.storage[("partition-a", f"row-{index}")] = {
            "partition_key": "partition-a",
            "row_key": f"row-{index}",
            "data": {"score": score},
            "expires_at": None,
        }
    sort = (DocumentDataSort(path="score", direction="asc"), DocumentDataSort(path="$row_key", direction="asc"))

    first = store.query("partition-a", limit=2, sort=sort)
    second = store.query("partition-a", limit=2, sort=sort, cursor=first.next_cursor)

    assert [doc.row_key for doc in first.documents + second.documents] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]

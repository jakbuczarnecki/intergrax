# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import base64
import json

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import (
    DocumentQueryCursorCodec,
    DocumentRecord,
)

pytestmark = pytest.mark.unit


def _record(partition_key: str, row_key: str) -> DocumentRecord:
    return DocumentRecord(partition_key=partition_key, row_key=row_key)


def _decode_envelope(cursor: str) -> dict[str, object]:
    padding = "=" * (-len(cursor) % 4)
    return json.loads(
        base64.urlsafe_b64decode((cursor + padding).encode("ascii")).decode("utf-8")
    )


def _encode_envelope(envelope: dict[str, object]) -> str:
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _store_with_rows(codec: DocumentQueryCursorCodec) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore(cursor_codec=codec)
    for row_key in ("row-1", "row-2", "row-3"):
        store.put(_record("partition-a", row_key))
    return store


def test_modified_cursor_payload_is_rejected_without_a_forgeable_mac() -> None:
    codec = DocumentQueryCursorCodec(secret=b"deterministic-test-secret")
    store = _store_with_rows(codec)
    cursor = store.query("partition-a", limit=1).next_cursor
    assert cursor is not None

    envelope = _decode_envelope(cursor)
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    payload["last_row_key"] = "row-3"
    forged_cursor = _encode_envelope(envelope)

    with pytest.raises(
        ValueError,
        match="document_store_cursor_authentication_failed",
    ):
        store.query("partition-a", limit=1, cursor=forged_cursor)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("partition_key", "partition-b"),
        ("row_key_prefix", "other-"),
        ("schema_version", "document_store.cursor.v2"),
    ),
)
def test_cursor_payload_field_tampering_is_rejected(
    field: str,
    value: str,
) -> None:
    codec = DocumentQueryCursorCodec(secret=b"deterministic-test-secret")
    store = _store_with_rows(codec)
    cursor = store.query("partition-a", limit=1, row_key_prefix=None).next_cursor
    assert cursor is not None

    envelope = _decode_envelope(cursor)
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    payload[field] = value

    expected_error = (
        "document_store_cursor_invalid"
        if field == "schema_version"
        else "document_store_cursor_authentication_failed"
    )
    with pytest.raises(ValueError, match=expected_error):
        store.query("partition-a", limit=1, cursor=_encode_envelope(envelope))


def test_truncated_and_oversized_cursors_are_rejected() -> None:
    codec = DocumentQueryCursorCodec(secret=b"deterministic-test-secret")
    store = _store_with_rows(codec)
    cursor = store.query("partition-a", limit=1).next_cursor
    assert cursor is not None

    with pytest.raises(ValueError, match="document_store_cursor_invalid"):
        store.query("partition-a", limit=1, cursor=cursor[:-8])
    with pytest.raises(ValueError, match="document_store_cursor_invalid"):
        store.query("partition-a", limit=1, cursor="x" * 4097)


def test_wrong_signature_and_wrong_codec_key_are_rejected() -> None:
    codec = DocumentQueryCursorCodec(secret=b"deterministic-test-secret")
    store = _store_with_rows(codec)
    cursor = store.query("partition-a", limit=1).next_cursor
    assert cursor is not None

    envelope = _decode_envelope(cursor)
    mac = envelope["mac"]
    assert isinstance(mac, str)
    envelope["mac"] = ("A" if mac[0] != "A" else "B") + mac[1:]
    with pytest.raises(
        ValueError,
        match="document_store_cursor_authentication_failed",
    ):
        store.query("partition-a", limit=1, cursor=_encode_envelope(envelope))

    other_store = _store_with_rows(
        DocumentQueryCursorCodec(secret=b"different-test-secret")
    )
    with pytest.raises(
        ValueError,
        match="document_store_cursor_authentication_failed",
    ):
        other_store.query("partition-a", limit=1, cursor=cursor)


def test_shared_codec_allows_cross_instance_continuation() -> None:
    codec = DocumentQueryCursorCodec(secret=b"restart-test-secret")
    first_store = _store_with_rows(codec)
    second_store = _store_with_rows(codec)

    first_page = first_store.query("partition-a", limit=1)
    assert [doc.row_key for doc in first_page.documents] == ["row-1"]
    assert first_page.next_cursor is not None

    second_page = second_store.query(
        "partition-a",
        limit=1,
        cursor=first_page.next_cursor,
    )
    assert [doc.row_key for doc in second_page.documents] == ["row-2"]


def test_cursor_is_bound_to_partition_and_prefix_query_scope() -> None:
    codec = DocumentQueryCursorCodec(secret=b"scope-test-secret")
    store = _store_with_rows(codec)
    store.put(_record("partition-a", "scope-row-1"))
    cursor = store.query(
        "partition-a",
        limit=1,
        row_key_prefix="scope-",
    ).next_cursor
    assert cursor is not None

    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        store.query("partition-b", limit=1, cursor=cursor)
    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        store.query(
            "partition-a",
            limit=1,
            row_key_prefix="other-",
            cursor=cursor,
        )
    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        store.query("partition-a", limit=1, cursor=cursor)


def test_1200_records_paginate_once_with_page_size_137() -> None:
    codec = DocumentQueryCursorCodec(secret=b"pagination-test-secret")
    store = InMemoryDocumentStore(cursor_codec=codec)
    expected = [f"row-{index:04d}" for index in range(1200)]
    for row_key in expected:
        store.put(_record("partition-a", row_key))

    actual: list[str] = []
    cursor: str | None = None
    while True:
        page = store.query("partition-a", limit=137, cursor=cursor)
        actual.extend(doc.row_key for doc in page.documents)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert actual == expected
    assert len(actual) == len(set(actual)) == 1200

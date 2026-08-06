# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Document / wide-column store integration contract (§7.1.2, Phase M.6 P2)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class DocumentRecord(BaseModel):
    """Normalized document row scoped by partition key."""

    partition_key: str
    row_key: str
    data: Mapping[str, Any] = Field(default_factory=dict)
    ttl_seconds: int | None = None


class DocumentQueryResult(BaseModel):
    documents: Sequence[DocumentRecord] = Field(default_factory=list)
    total: int = 0


_DOCUMENT_QUERY_CURSOR_SCHEMA = "document_store.cursor.v1"
_DOCUMENT_QUERY_MAX_LIMIT = 5000


class DocumentQueryPageV1(BaseModel):
    """Bounded, immutable page returned by a cursor-aware document query."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    documents: tuple[DocumentRecord, ...] = ()
    next_cursor: str | None = None

    @property
    def total(self) -> int:
        """Compatibility view for callers of the pre-cursor result contract."""
        return len(self.documents)


class _DocumentQueryCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["document_store.cursor.v1"]
    partition_key: str
    row_key_prefix: str | None
    last_row_key: str


def validate_document_query_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("document_store_query_limit_invalid")
    if limit < 1 or limit > _DOCUMENT_QUERY_MAX_LIMIT:
        raise ValueError("document_store_query_limit_invalid")
    return limit


def encode_document_query_cursor(
    *,
    partition_key: str,
    row_key_prefix: str | None,
    last_row_key: str,
) -> str:
    payload = _DocumentQueryCursorPayloadV1(
        schema_version=_DOCUMENT_QUERY_CURSOR_SCHEMA,
        partition_key=partition_key,
        row_key_prefix=row_key_prefix,
        last_row_key=last_row_key,
    ).model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    envelope = {
        "payload": payload,
        "checksum": hashlib.sha256(canonical).hexdigest(),
    }
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def decode_document_query_cursor(
    cursor: str,
    *,
    partition_key: str,
    row_key_prefix: str | None,
) -> str:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 4096:
        raise ValueError("document_store_cursor_invalid")
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = json.loads(
            base64.urlsafe_b64decode((cursor + padding).encode("ascii")).decode("utf-8")
        )
        payload = _DocumentQueryCursorPayloadV1.model_validate(raw["payload"], strict=True)
        checksum = raw["checksum"]
        canonical = json.dumps(
            payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        binascii.Error,
        json.JSONDecodeError,
    ):
        raise ValueError("document_store_cursor_invalid") from None
    if (
        not isinstance(checksum, str)
        or checksum != hashlib.sha256(canonical).hexdigest()
        or payload.partition_key != partition_key
        or payload.row_key_prefix != row_key_prefix
    ):
        raise ValueError("document_store_cursor_query_mismatch")
    return payload.last_row_key


@runtime_checkable
class DocumentStore(Protocol):
    """
    Backend-agnostic document store facade.

    Implementations: cassandra, mongodb, dynamodb, …
    """

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        """Fetch a single document or return ``None`` when missing."""

    def put(self, document: DocumentRecord) -> None:
        """Insert or upsert a document."""

    def delete(self, partition_key: str, row_key: str) -> None:
        """Remove a document."""

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        """List documents in deterministic bounded pages."""
        ...

    def close(self) -> None:
        """Release resources."""


@runtime_checkable
class ConditionalDocumentStore(DocumentStore, Protocol):
    """
    Optional single-record conditional write capability.

    Compare partition_key, row_key and data only — not ttl_seconds.
    Normal conflicts return False; they are not errors.
    """

    def put_if_absent(
        self,
        document: DocumentRecord,
    ) -> bool:
        """Atomically insert when missing; return False if the key already exists."""
        ...

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        """Atomically replace when current record matches expected keys+data."""
        ...

    def delete_if_match(
        self,
        *,
        expected: DocumentRecord,
    ) -> bool:
        """Atomically delete when current record matches expected keys+data."""
        ...

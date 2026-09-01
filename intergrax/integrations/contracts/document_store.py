# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Document / wide-column store integration contract (§7.1.2, Phase M.6 P2)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError


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


class DocumentQueryCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["document_store.cursor.v1"]
    partition_key: str
    row_key_prefix: str | None
    last_row_key: str


class DocumentQueryCursorCodec:
    """Authenticated, query-bound codec for document-store continuation cursors."""

    _MAX_TOKEN_LENGTH = 4096

    def __init__(self, *, secret: bytes) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("document_store_cursor_secret_invalid")
        self._secret = secret

    @staticmethod
    def _canonical_payload(payload: DocumentQueryCursorPayloadV1) -> bytes:
        return json.dumps(
            payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _encode_base64url(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_base64url(value: object) -> bytes:
        if not isinstance(value, str) or not value or len(value) % 4 == 1:
            raise ValueError
        padding = "=" * (-len(value) % 4)
        return base64.b64decode(
            (value + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )

    def encode(
        self,
        *,
        partition_key: str,
        row_key_prefix: str | None,
        last_row_key: str,
    ) -> str:
        payload = DocumentQueryCursorPayloadV1(
            schema_version=_DOCUMENT_QUERY_CURSOR_SCHEMA,
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
            last_row_key=last_row_key,
        )
        canonical = self._canonical_payload(payload)
        mac = hmac.new(self._secret, canonical, digestmod="sha256").digest()
        envelope = {
            "mac": self._encode_base64url(mac),
            "payload": payload.model_dump(mode="json"),
        }
        encoded = self._encode_base64url(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if len(encoded) > self._MAX_TOKEN_LENGTH:
            raise ValueError("document_store_cursor_invalid")
        return encoded

    def decode(
        self,
        cursor: str,
        *,
        partition_key: str,
        row_key_prefix: str | None,
    ) -> DocumentQueryCursorPayloadV1:
        if not isinstance(cursor, str) or not cursor or len(cursor) > self._MAX_TOKEN_LENGTH:
            raise ValueError("document_store_cursor_invalid")
        try:
            raw = json.loads(self._decode_base64url(cursor).decode("utf-8"))
            if not isinstance(raw, dict) or set(raw) != {"mac", "payload"}:
                raise ValueError
            payload = DocumentQueryCursorPayloadV1.model_validate(
                raw["payload"],
                strict=True,
            )
            mac = self._decode_base64url(raw["mac"])
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeError,
            binascii.Error,
            json.JSONDecodeError,
            ValidationError,
        ):
            raise ValueError("document_store_cursor_invalid") from None

        expected_mac = hmac.new(
            self._secret,
            self._canonical_payload(payload),
            digestmod="sha256",
        ).digest()
        if len(mac) != len(expected_mac) or not hmac.compare_digest(mac, expected_mac):
            raise ValueError("document_store_cursor_authentication_failed")
        if (
            payload.partition_key != partition_key
            or payload.row_key_prefix != row_key_prefix
        ):
            raise ValueError("document_store_cursor_query_mismatch")
        return payload


def validate_document_query_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("document_store_query_limit_invalid")
    if limit < 1 or limit > _DOCUMENT_QUERY_MAX_LIMIT:
        raise ValueError("document_store_query_limit_invalid")
    return limit


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
        row_key_upper_bound: str | None = None,
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

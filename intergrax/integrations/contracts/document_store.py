# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Document / wide-column store integration contract (§7.1.2, Phase M.6 P2)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
_DOCUMENT_QUERY_CURSOR_SCHEMA_V2 = "document_store.cursor.v2"
_DOCUMENT_QUERY_MAX_LIMIT = 5000
_ROW_KEY_SORT_PATH = "$row_key"
_DATA_PATH_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


@dataclass(frozen=True, slots=True)
class DocumentDataEquality:
    """Exact-match filter on a dot path within ``DocumentRecord.data``."""

    path: str
    value: Any


@dataclass(frozen=True, slots=True)
class DocumentDataSort:
    """Sort key on ``DocumentRecord.data`` or the special ``$row_key`` path."""

    path: str
    direction: Literal["asc", "desc"]


def validate_document_data_path(path: str) -> str:
    if path == _ROW_KEY_SORT_PATH:
        return path
    if not isinstance(path, str) or not path or not _DATA_PATH_PATTERN.match(path):
        raise ValueError("document_store_data_path_invalid")
    return path


def normalize_document_data_equalities(
    equalities: Sequence[DocumentDataEquality] | None,
) -> tuple[DocumentDataEquality, ...]:
    if equalities is None:
        return ()
    normalized: list[DocumentDataEquality] = []
    for item in equalities:
        if not isinstance(item, DocumentDataEquality):
            raise TypeError("document_store_data_equality_invalid")
        normalized.append(
            DocumentDataEquality(
                path=validate_document_data_path(item.path),
                value=item.value,
            ),
        )
    normalized.sort(key=lambda item: item.path)
    return tuple(normalized)


def query_requires_v2_cursor(
    *,
    data_equalities: Sequence[DocumentDataEquality],
    sort: Sequence[DocumentDataSort],
    row_key_upper_bound: str | None,
) -> bool:
    """Return True when continuation cursors must bind extended query dimensions."""
    return bool(data_equalities or sort or row_key_upper_bound is not None)


def normalize_document_data_sort(
    sort: Sequence[DocumentDataSort] | None,
) -> tuple[DocumentDataSort, ...]:
    if sort is None:
        return ()
    normalized: list[DocumentDataSort] = []
    for item in sort:
        if not isinstance(item, DocumentDataSort):
            raise TypeError("document_store_data_sort_invalid")
        if item.direction not in ("asc", "desc"):
            raise ValueError("document_store_data_sort_invalid")
        normalized.append(
            DocumentDataSort(
                path=validate_document_data_path(item.path),
                direction=item.direction,
            ),
        )
    return tuple(normalized)


def document_data_path_value(data: Mapping[str, Any], path: str) -> Any:
    validate_document_data_path(path)
    if path == _ROW_KEY_SORT_PATH:
        raise ValueError("document_store_data_path_invalid")
    current: Any = data
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def document_sort_key_values(
    document: DocumentRecord,
    sort: Sequence[DocumentDataSort],
) -> tuple[Any, ...]:
    values: list[Any] = []
    for spec in sort:
        if spec.path == _ROW_KEY_SORT_PATH:
            values.append(document.row_key)
        else:
            values.append(document_data_path_value(document.data, spec.path))
    return tuple(values)


def _canonical_json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_canonical_json_value(value_json: str) -> Any:
    return json.loads(value_json)


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


class DocumentQueryCursorPayloadV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["document_store.cursor.v2"]
    partition_key: str
    row_key_prefix: str | None
    row_key_upper_bound: str | None
    data_equalities: list[list[str]]
    sort: list[list[str]]
    last_row_key: str
    last_sort_values: list[str]


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

    @staticmethod
    def _canonical_payload_v2(payload: DocumentQueryCursorPayloadV2) -> bytes:
        return json.dumps(
            payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def encode_v2(
        self,
        *,
        partition_key: str,
        row_key_prefix: str | None,
        row_key_upper_bound: str | None,
        data_equalities: tuple[DocumentDataEquality, ...],
        sort: tuple[DocumentDataSort, ...],
        last_row_key: str,
        last_sort_values: tuple[Any, ...],
    ) -> str:
        if len(last_sort_values) != len(sort):
            raise ValueError("document_store_cursor_invalid")
        payload = DocumentQueryCursorPayloadV2(
            schema_version=_DOCUMENT_QUERY_CURSOR_SCHEMA_V2,
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
            row_key_upper_bound=row_key_upper_bound,
            data_equalities=[
                [item.path, _canonical_json_value(item.value)] for item in data_equalities
            ],
            sort=[[item.path, item.direction] for item in sort],
            last_row_key=last_row_key,
            last_sort_values=[_canonical_json_value(value) for value in last_sort_values],
        )
        canonical = self._canonical_payload_v2(payload)
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

    def decode_v2(
        self,
        cursor: str,
        *,
        partition_key: str,
        row_key_prefix: str | None,
        row_key_upper_bound: str | None,
        data_equalities: tuple[DocumentDataEquality, ...],
        sort: tuple[DocumentDataSort, ...],
    ) -> DocumentQueryCursorPayloadV2:
        if not isinstance(cursor, str) or not cursor or len(cursor) > self._MAX_TOKEN_LENGTH:
            raise ValueError("document_store_cursor_invalid")
        try:
            raw = json.loads(self._decode_base64url(cursor).decode("utf-8"))
            if not isinstance(raw, dict) or set(raw) != {"mac", "payload"}:
                raise ValueError
            payload = DocumentQueryCursorPayloadV2.model_validate(
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
            self._canonical_payload_v2(payload),
            digestmod="sha256",
        ).digest()
        if len(mac) != len(expected_mac) or not hmac.compare_digest(mac, expected_mac):
            raise ValueError("document_store_cursor_authentication_failed")
        expected_equalities = [
            [item.path, _canonical_json_value(item.value)] for item in data_equalities
        ]
        expected_sort = [[item.path, item.direction] for item in sort]
        if (
            payload.partition_key != partition_key
            or payload.row_key_prefix != row_key_prefix
            or payload.row_key_upper_bound != row_key_upper_bound
            or payload.data_equalities != expected_equalities
            or payload.sort != expected_sort
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
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
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

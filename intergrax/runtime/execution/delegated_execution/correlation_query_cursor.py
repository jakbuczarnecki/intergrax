# © Artur Czarnecki. All rights reserved.

"""Authenticated opaque keyset cursor codec for delegated correlation queries (P2.1-S2C3)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.contracts.delegated_execution_query import (
    DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
    DelegatedExecutionQueryInvalidCursorError,
    DelegatedInvocationCorrelationQuery,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id

_CURSOR_SCHEMA = "intergrax.delegated_correlation_query_cursor.v1"
_MAX_TOKEN_LENGTH = 4096
_MIN_CURSOR_SECRET_BYTES = 32


class _DelegatedCorrelationQueryCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["intergrax.delegated_correlation_query_cursor.v1"]
    query_fingerprint: str
    last_persisted_at: datetime | None = None
    last_execution_id: str | None = None
    document_store_cursor: str | None = None


class DelegatedCorrelationQueryCursorCodec:
    """Authenticated, query-bound codec for delegated correlation pagination cursors."""

    def __init__(self, *, secret: bytes) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("delegated_correlation_query_cursor_secret_invalid")
        if len(secret) < _MIN_CURSOR_SECRET_BYTES:
            raise ValueError("delegated_correlation_query_cursor_secret_too_short")
        self._secret = secret

    @staticmethod
    def _canonical_payload(payload: _DelegatedCorrelationQueryCursorPayloadV1) -> bytes:
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
        query: DelegatedInvocationCorrelationQuery,
        last_persisted_at: datetime | None,
        last_execution_id: ExecutionId | None,
        document_store_cursor: str | None,
    ) -> str:
        payload = _DelegatedCorrelationQueryCursorPayloadV1(
            schema_version=_CURSOR_SCHEMA,
            query_fingerprint=delegated_correlation_query_fingerprint(query),
            last_persisted_at=last_persisted_at,
            last_execution_id=(
                str(last_execution_id) if last_execution_id is not None else None
            ),
            document_store_cursor=document_store_cursor,
        )
        canonical = self._canonical_payload(payload)
        signature = hmac.new(self._secret, canonical, digestmod="sha256").digest()
        envelope = {
            "payload": self._encode_base64url(canonical),
            "signature": self._encode_base64url(signature),
        }
        encoded = self._encode_base64url(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        if len(encoded) > _MAX_TOKEN_LENGTH:
            raise DelegatedExecutionQueryInvalidCursorError("cursor token too large")
        return encoded

    def decode(
        self,
        *,
        query: DelegatedInvocationCorrelationQuery,
        cursor: str,
    ) -> tuple[datetime | None, ExecutionId | None, str | None]:
        if not isinstance(cursor, str) or not cursor or len(cursor) > _MAX_TOKEN_LENGTH:
            raise DelegatedExecutionQueryInvalidCursorError(
                DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
            )
        try:
            envelope_bytes = self._decode_base64url(cursor)
            envelope = json.loads(envelope_bytes.decode("utf-8"))
            payload_bytes = self._decode_base64url(envelope["payload"])
            provided_signature = self._decode_base64url(envelope["signature"])
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            binascii.Error,
            UnicodeError,
        ) as exc:
            raise DelegatedExecutionQueryInvalidCursorError(
                DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
            ) from exc

        try:
            payload = _DelegatedCorrelationQueryCursorPayloadV1.model_validate_json(
                payload_bytes.decode("utf-8"),
                strict=True,
            )
        except ValidationError as exc:
            raise DelegatedExecutionQueryInvalidCursorError(
                DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
            ) from exc

        expected_signature = hmac.new(
            self._secret,
            payload_bytes,
            digestmod="sha256",
        ).digest()
        if not hmac.compare_digest(provided_signature, expected_signature):
            raise DelegatedExecutionQueryInvalidCursorError(
                DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
            )
        if payload.query_fingerprint != delegated_correlation_query_fingerprint(query):
            raise DelegatedExecutionQueryInvalidCursorError(
                DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
            )
        last_execution_id = (
            validate_execution_id(payload.last_execution_id)
            if payload.last_execution_id is not None
            else None
        )
        return (
            payload.last_persisted_at,
            last_execution_id,
            payload.document_store_cursor,
        )


def delegated_correlation_query_fingerprint(
    query: DelegatedInvocationCorrelationQuery,
) -> str:
    """Stable fingerprint for cursor binding (excludes page_size and cursor)."""
    payload = {
        "parent_execution_id": (
            str(query.parent_execution_id) if query.parent_execution_id is not None else None
        ),
        "provider_id": query.provider_id,
        "run_id": str(query.run_id) if query.run_id is not None else None,
        "persisted_from": (
            query.persisted_from.isoformat() if query.persisted_from is not None else None
        ),
        "persisted_to": (
            query.persisted_to.isoformat() if query.persisted_to is not None else None
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def encode_delegated_correlation_query_cursor(
    *,
    codec: DelegatedCorrelationQueryCursorCodec,
    query: DelegatedInvocationCorrelationQuery,
    last_persisted_at: datetime | None,
    last_execution_id: ExecutionId | None,
    document_store_cursor: str | None = None,
) -> str:
    return codec.encode(
        query=query,
        last_persisted_at=last_persisted_at,
        last_execution_id=last_execution_id,
        document_store_cursor=document_store_cursor,
    )


def decode_delegated_correlation_query_cursor(
    *,
    codec: DelegatedCorrelationQueryCursorCodec,
    query: DelegatedInvocationCorrelationQuery,
    cursor: str,
) -> tuple[datetime | None, ExecutionId | None, str | None]:
    return codec.decode(query=query, cursor=cursor)


__all__ = [
    "DelegatedCorrelationQueryCursorCodec",
    "decode_delegated_correlation_query_cursor",
    "delegated_correlation_query_fingerprint",
    "encode_delegated_correlation_query_cursor",
]

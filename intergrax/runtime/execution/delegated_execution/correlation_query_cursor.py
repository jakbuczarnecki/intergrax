# © Artur Czarnecki. All rights reserved.

"""Opaque keyset cursor codec for delegated correlation queries (P2.1-S2C3)."""

from __future__ import annotations

import base64
import binascii
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


class _DelegatedCorrelationQueryCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["intergrax.delegated_correlation_query_cursor.v1"]
    query_fingerprint: str
    last_persisted_at: datetime
    last_execution_id: str
    document_store_cursor: str | None = None


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
    query: DelegatedInvocationCorrelationQuery,
    last_persisted_at: datetime,
    last_execution_id: ExecutionId,
    document_store_cursor: str | None = None,
) -> str:
    payload = _DelegatedCorrelationQueryCursorPayloadV1(
        schema_version=_CURSOR_SCHEMA,
        query_fingerprint=delegated_correlation_query_fingerprint(query),
        last_persisted_at=last_persisted_at,
        last_execution_id=str(last_execution_id),
        document_store_cursor=document_store_cursor,
    )
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(canonical).decode("ascii").rstrip("=")
    if len(token) > _MAX_TOKEN_LENGTH:
        raise DelegatedExecutionQueryInvalidCursorError("cursor token too large")
    return token


def decode_delegated_correlation_query_cursor(
    *,
    query: DelegatedInvocationCorrelationQuery,
    cursor: str,
) -> tuple[datetime, ExecutionId, str | None]:
    if not isinstance(cursor, str) or not cursor or len(cursor) > _MAX_TOKEN_LENGTH:
        raise DelegatedExecutionQueryInvalidCursorError(
            DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
        )
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.b64decode(
            (cursor + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        payload = _DelegatedCorrelationQueryCursorPayloadV1.model_validate_json(raw)
    except (
        ValueError,
        ValidationError,
        UnicodeError,
        binascii.Error,
        json.JSONDecodeError,
    ) as exc:
        raise DelegatedExecutionQueryInvalidCursorError(
            DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
        ) from exc
    if payload.query_fingerprint != delegated_correlation_query_fingerprint(query):
        raise DelegatedExecutionQueryInvalidCursorError(
            DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
        )
    return (
        payload.last_persisted_at,
        validate_execution_id(payload.last_execution_id),
        payload.document_store_cursor,
    )


__all__ = [
    "decode_delegated_correlation_query_cursor",
    "delegated_correlation_query_fingerprint",
    "encode_delegated_correlation_query_cursor",
]

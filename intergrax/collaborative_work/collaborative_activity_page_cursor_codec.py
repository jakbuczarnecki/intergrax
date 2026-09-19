# © Artur Czarnecki. All rights reserved.

"""Opaque append-position cursor codec for MP-6E activity timeline pagination."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Final

from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityPageCursor,
    CollaborativeActivityQuery,
)
from intergrax.contracts.collaborative_activity_read import CollaborativeActivityCursorInvalid

_CURSOR_PAYLOAD_SCHEMA: Final = "collaborative_activity_page_cursor_payload.v1"


def collaborative_activity_query_filter_binding(query: CollaborativeActivityQuery) -> str:
    """Stable digest of filter dimensions bound into continuation cursors."""
    payload = {
        "activity_types": tuple(item.qualified_id for item in query.activity_types),
        "actor_principal_id": query.actor_principal_id,
        "work_item_id": query.work_item_id,
        "occurred_after": (
            query.occurred_after.isoformat() if query.occurred_after is not None else None
        ),
        "occurred_before": (
            query.occurred_before.isoformat() if query.occurred_before is not None else None
        ),
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:32]


def decode_collaborative_activity_page_cursor(
    cursor: CollaborativeActivityPageCursor,
    *,
    query: CollaborativeActivityQuery,
) -> int:
    """Return ``after_append_position`` when cursor matches query scope and filters."""
    try:
        padded = cursor.token + "=" * (-len(cursor.token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeError) as exc:
        raise CollaborativeActivityCursorInvalid("cursor token is not valid opaque data") from exc

    if payload.get("schema_version") != _CURSOR_PAYLOAD_SCHEMA:
        raise CollaborativeActivityCursorInvalid("unsupported cursor schema version")

    tenant_id = str(payload.get("tenant_id", "")).strip()
    workspace_id = str(payload.get("workspace_id", "")).strip()
    if tenant_id != query.tenant_id.strip() or workspace_id != query.workspace_id.strip():
        raise CollaborativeActivityCursorInvalid("cursor scope does not match query scope")

    filter_binding = str(payload.get("filter_binding", "")).strip()
    expected_binding = collaborative_activity_query_filter_binding(query)
    if filter_binding != expected_binding:
        raise CollaborativeActivityCursorInvalid("cursor filter binding does not match query")

    try:
        after_append_position = int(payload["after_append_position"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CollaborativeActivityCursorInvalid("cursor append position is invalid") from exc

    if after_append_position < 0:
        raise CollaborativeActivityCursorInvalid("cursor append position must be non-negative")

    return after_append_position


def encode_collaborative_activity_page_cursor(
    *,
    query: CollaborativeActivityQuery,
    after_append_position: int,
) -> CollaborativeActivityPageCursor:
    if after_append_position < 0:
        raise ValueError("after_append_position must be non-negative")
    payload = {
        "schema_version": _CURSOR_PAYLOAD_SCHEMA,
        "tenant_id": query.tenant_id.strip(),
        "workspace_id": query.workspace_id.strip(),
        "filter_binding": collaborative_activity_query_filter_binding(query),
        "after_append_position": after_append_position,
    }
    token = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).decode("ascii").rstrip("=")
    return CollaborativeActivityPageCursor(token=token)


def resolve_after_append_position(query: CollaborativeActivityQuery) -> int:
    """Continuation watermark for provider keyset pagination (0 = first page)."""
    if query.cursor is None:
        return 0
    return decode_collaborative_activity_page_cursor(query.cursor, query=query)

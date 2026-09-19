# © Artur Czarnecki. All rights reserved.

"""MP-6E provider-neutral Collaborative Activity read adapters (SQLite / PostgreSQL)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from intergrax.collaborative_work.collaborative_activity_page_cursor_codec import (
    encode_collaborative_activity_page_cursor,
    resolve_after_append_position,
)
from intergrax.collaborative_work.postgresql_repository import PostgreSQLCollaborativeWorkStore
from intergrax.collaborative_work.serialization import collaborative_activity_from_json
from intergrax.collaborative_work.sqlite_repository import SQLiteCollaborativeWorkStore
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivity,
    CollaborativeActivityPage,
    CollaborativeActivityQuery,
    CollaborativeActivityReadPort,
)
from intergrax.contracts.collaborative_activity_read import (
    CollaborativeActivityCursorInvalid,
    CollaborativeActivityReadPersistenceError,
)


class CollaborativeActivityReadIntegrityError(CollaborativeActivityReadPersistenceError):
    """Persisted activity row violates canonical contract."""


ParamStyle = Literal["qmark", "pyformat"]


@dataclass(frozen=True, slots=True)
class _CompiledActivityReadQuery:
    sql: str
    params: tuple[Any, ...]


def _placeholder(param_style: ParamStyle, index: int) -> str:
    if param_style == "qmark":
        return "?"
    return "%s"


def compile_collaborative_activity_read_query(
    query: CollaborativeActivityQuery,
    *,
    after_append_position: int,
    fetch_limit: int,
    param_style: ParamStyle,
) -> _CompiledActivityReadQuery:
    params: list[Any] = [
        query.tenant_id.strip(),
        query.workspace_id.strip(),
        after_append_position,
    ]
    clauses = [
        "tenant_id = "
        f"{_placeholder(param_style, 1)}",
        "workspace_id = "
        f"{_placeholder(param_style, 2)}",
        "append_position > "
        f"{_placeholder(param_style, 3)}",
    ]

    if query.activity_types:
        placeholders = ", ".join(
            _placeholder(param_style, len(params) + index + 1)
            for index in range(len(query.activity_types))
        )
        clauses.append(f"activity_type_qualified_id IN ({placeholders})")
        params.extend(item.qualified_id for item in query.activity_types)

    if query.work_item_id is not None:
        if param_style == "qmark":
            clauses.append(
                "json_extract(record_json, '$.scope.work_item_id') = "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        else:
            clauses.append(
                "record_json::jsonb -> 'scope' ->> 'work_item_id' = "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        params.append(query.work_item_id.strip())

    if query.actor_principal_id is not None:
        if param_style == "qmark":
            clauses.append(
                "json_extract(record_json, '$.actor.principal_id') = "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        else:
            clauses.append(
                "record_json::jsonb -> 'actor' ->> 'principal_id' = "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        params.append(query.actor_principal_id.strip())

    if query.occurred_after is not None:
        if param_style == "qmark":
            clauses.append(
                "json_extract(record_json, '$.occurred_at') >= "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        else:
            clauses.append(
                "record_json::jsonb ->> 'occurred_at' >= "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        params.append(query.occurred_after.isoformat())

    if query.occurred_before is not None:
        if param_style == "qmark":
            clauses.append(
                "json_extract(record_json, '$.occurred_at') <= "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        else:
            clauses.append(
                "record_json::jsonb ->> 'occurred_at' <= "
                f"{_placeholder(param_style, len(params) + 1)}",
            )
        params.append(query.occurred_before.isoformat())

    limit_ph = _placeholder(param_style, len(params) + 1)
    params.append(fetch_limit)
    sql = (
        "SELECT record_json, append_position FROM collaborative_activities WHERE "
        + " AND ".join(clauses)
        + f" ORDER BY append_position ASC LIMIT {limit_ph}"
    )
    return _CompiledActivityReadQuery(sql=sql, params=tuple(params))


def _decode_rows(rows: Sequence[Any]) -> tuple[CollaborativeActivity, ...]:
    activities: list[CollaborativeActivity] = []
    for row in rows:
        if isinstance(row, dict):
            record_json = row["record_json"]
        else:
            record_json = row["record_json"]
        try:
            activities.append(collaborative_activity_from_json(record_json))
        except Exception as exc:
            raise CollaborativeActivityReadIntegrityError(
                "collaborative activity record_json is corrupt or incompatible",
            ) from exc
    return tuple(activities)


def _materialize_page(
    *,
    query: CollaborativeActivityQuery,
    rows: Sequence[Any],
) -> CollaborativeActivityPage:
    activities = _decode_rows(rows)
    limit = query.limit
    if len(activities) <= limit:
        return CollaborativeActivityPage(activities=activities, next_cursor=None)

    page_items = activities[:limit]
    last_position = page_items[-1].append_position
    next_cursor = encode_collaborative_activity_page_cursor(
        query=query,
        after_append_position=last_position,
    )
    return CollaborativeActivityPage(activities=page_items, next_cursor=next_cursor)


def execute_collaborative_activity_read_query(
    query: CollaborativeActivityQuery,
    *,
    param_style: ParamStyle,
    fetch: Any,
) -> CollaborativeActivityPage:
    try:
        after_append_position = resolve_after_append_position(query)
    except CollaborativeActivityCursorInvalid:
        raise
    compiled = compile_collaborative_activity_read_query(
        query,
        after_append_position=after_append_position,
        fetch_limit=query.limit + 1,
        param_style=param_style,
    )
    try:
        rows = fetch(compiled)
    except CollaborativeActivityCursorInvalid:
        raise
    except Exception as exc:
        raise CollaborativeActivityReadPersistenceError(
            "collaborative activity read provider query failed",
        ) from exc
    return _materialize_page(query=query, rows=rows)


class SQLiteCollaborativeActivityReadStore:
    """SQLite read adapter — scoped keyset pagination by ``append_position``."""

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    def query(self, query: CollaborativeActivityQuery) -> CollaborativeActivityPage:
        def _fetch(compiled: _CompiledActivityReadQuery) -> list[Any]:
            with self._store._lock:
                self._store._ensure_open()
                cursor = self._store._connection.execute(compiled.sql, compiled.params)
                return cursor.fetchall()

        return execute_collaborative_activity_read_query(
            query,
            param_style="qmark",
            fetch=_fetch,
        )


class PostgreSQLCollaborativeActivityReadStore:
    """PostgreSQL read adapter — semantic parity with SQLite read port."""

    def __init__(self, store: PostgreSQLCollaborativeWorkStore) -> None:
        self._store = store
        self._lock = threading.RLock()

    def query(self, query: CollaborativeActivityQuery) -> CollaborativeActivityPage:
        def _fetch(compiled: _CompiledActivityReadQuery) -> list[Any]:
            with self._lock:
                self._store._ensure_open()
                with self._store.transaction() as session:
                    result = session.execute(compiled.sql, compiled.params)
                    return result.fetchall()

        return execute_collaborative_activity_read_query(
            query,
            param_style="pyformat",
            fetch=_fetch,
        )


def sqlite_collaborative_activity_read_store(db_path: str) -> CollaborativeActivityReadPort:
    from pathlib import Path

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteCollaborativeWorkStore(str(path))
    return SQLiteCollaborativeActivityReadStore(store)


__all__ = [
    "PostgreSQLCollaborativeActivityReadStore",
    "SQLiteCollaborativeActivityReadStore",
    "compile_collaborative_activity_read_query",
    "execute_collaborative_activity_read_query",
    "sqlite_collaborative_activity_read_store",
]

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

import sqlite3

from intergrax.tools.providers.database.contracts import DatabaseDescribeSchemaInput, DatabaseExecuteInput, DatabaseQueryInput
from intergrax.tools.providers.database.service import database_describe_schema, database_execute, database_query
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemoryRelationalStore:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")

    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()


def test_database_query_returns_rows() -> None:
    ctx = ToolWiringContext(relational_store=InMemoryRelationalStore())
    database_execute(ctx, DatabaseExecuteInput(sql="INSERT INTO items (id, name) VALUES (?, ?)", params=[1, "alpha"]))
    out = database_query(ctx, DatabaseQueryInput(sql="SELECT * FROM items"))
    assert out.row_count == 1
    assert out.rows[0]["name"] == "alpha"


def test_database_execute_inserts_row() -> None:
    store = InMemoryRelationalStore()
    ctx = ToolWiringContext(relational_store=store)
    database_execute(ctx, DatabaseExecuteInput(sql="INSERT INTO items (id, name) VALUES (?, ?)", params=[1, "alpha"]))
    database_execute(ctx, DatabaseExecuteInput(sql="INSERT INTO items (id, name) VALUES (?, ?)", params=[2, "beta"]))
    out = database_query(ctx, DatabaseQueryInput(sql="SELECT * FROM items"))
    assert out.row_count == 2


def test_database_not_configured() -> None:
    with pytest.raises(RuntimeError, match="relational_store_not_configured"):
        database_query(ToolWiringContext(), DatabaseQueryInput(sql="SELECT 1"))


def test_database_describe_schema() -> None:
    ctx = ToolWiringContext(relational_store=InMemoryRelationalStore())
    out = database_describe_schema(ctx, DatabaseDescribeSchemaInput())
    assert out.used is True
    assert any(table.name == "items" for table in out.tables)
    items = next(table for table in out.tables if table.name == "items")
    assert any(column.name == "name" for column in items.columns)

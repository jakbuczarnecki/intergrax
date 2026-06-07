# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from intergrax.tools.providers.database.contracts import DatabaseExecuteInput, DatabaseQueryInput
from intergrax.tools.providers.database.service import database_execute, database_query
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemoryRelationalStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = [{"id": 1, "name": "alpha"}]

    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        if sql.strip().upper().startswith("INSERT"):
            self.rows.append({"id": params[0], "name": params[1]})

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        return list(self.rows)

    def close(self) -> None:
        return None


def test_database_query_returns_rows() -> None:
    ctx = ToolWiringContext(relational_store=InMemoryRelationalStore())
    out = database_query(ctx, DatabaseQueryInput(sql="SELECT * FROM items"))
    assert out.row_count == 1
    assert out.rows[0]["name"] == "alpha"


def test_database_execute_inserts_row() -> None:
    store = InMemoryRelationalStore()
    ctx = ToolWiringContext(relational_store=store)
    database_execute(ctx, DatabaseExecuteInput(sql="INSERT INTO items VALUES (?, ?)", params=[2, "beta"]))
    out = database_query(ctx, DatabaseQueryInput(sql="SELECT * FROM items"))
    assert out.row_count == 2


def test_database_not_configured() -> None:
    with pytest.raises(RuntimeError, match="relational_store_not_configured"):
        database_query(ToolWiringContext(), DatabaseQueryInput(sql="SELECT 1"))

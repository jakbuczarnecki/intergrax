# © Artur Czarnecki. All rights reserved.

"""Unit tests for DB-level bounded SQL fetch (no post-fetch Python slicing)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

import pytest

from platform_proofs.tools.iterative_sql_investigation.contracts import MAX_VISIBLE_ROWS
from platform_proofs.tools.iterative_sql_investigation.sql_tool import execute_bounded_query, wrap_bounded_query

pytestmark = pytest.mark.unit


class _RecordingStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.executed_sql: list[str] = []
        self._rows = rows

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        self.executed_sql.append(sql)
        if "LIMIT 201" not in sql:
            return self._rows
        return self._rows[:201]

    def connect(self) -> None: ...

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None: ...

    def close(self) -> None: ...


def test_bounded_query_uses_database_limit_not_python_slice_of_unbounded_result() -> None:
    source_rows = [{"parcel_id": index} for index in range(250)]
    store = _RecordingStore(source_rows)
    output = execute_bounded_query(store, "SELECT parcel_id FROM proof.parcel_events")
    assert store.executed_sql == [wrap_bounded_query("SELECT parcel_id FROM proof.parcel_events")]
    assert output.row_count == MAX_VISIBLE_ROWS
    assert output.truncated is True
    assert len(output.rows) == MAX_VISIBLE_ROWS


def test_bounded_query_not_truncated_when_source_within_cap() -> None:
    store = _RecordingStore([{"parcel_id": 1}, {"parcel_id": 2}])
    output = execute_bounded_query(store, "SELECT parcel_id FROM proof.parcel_events")
    assert output.row_count == 2
    assert output.truncated is False


def test_cte_query_is_wrapped_safely() -> None:
    sql = "WITH ids AS (SELECT parcel_id FROM proof.parcel_events) SELECT parcel_id FROM ids"
    wrapped = wrap_bounded_query(sql)
    assert wrapped.count("LIMIT 201") == 1
    store = _RecordingStore([{"parcel_id": 1}])
    output = execute_bounded_query(store, sql)
    assert output.row_count == 1
    assert output.truncated is False

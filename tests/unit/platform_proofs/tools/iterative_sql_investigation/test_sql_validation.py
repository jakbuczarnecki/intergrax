# © Artur Czarnecki. All rights reserved.

"""Unit tests for proof-local read-only SQL validation and bounded wrapping."""

from __future__ import annotations

import pytest

from platform_proofs.tools.iterative_sql_investigation.sql_tool import (
    SqlValidationError,
    validate_read_only_sql,
    wrap_bounded_query,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select parcel_id from proof.parcel_events",
        "WITH recent AS (SELECT parcel_id FROM proof.parcel_events) SELECT * FROM recent",
        "SELECT parcel_id FROM proof.parcel_events ORDER BY parcel_id",
        "SELECT parcel_id FROM proof.parcel_events LIMIT 10",
        "SELECT parcel_id FROM proof.parcel_events;",
    ],
)
def test_validation_accepts_read_only_queries(sql: str) -> None:
    assert validate_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO proof.parcel_events(parcel_id) VALUES (1)",
        "UPDATE proof.parcel_events SET delayed = true",
        "DELETE FROM proof.parcel_events",
        "CREATE TABLE proof.evil(id int)",
        "COPY proof.parcel_events FROM STDIN",
        "CALL pg_sleep(1)",
        "BEGIN; SELECT 1",
        "SELECT 1; SELECT 2",
    ],
)
def test_validation_rejects_mutating_or_multi_statement_sql(sql: str) -> None:
    with pytest.raises(SqlValidationError):
        validate_read_only_sql(sql)


def test_wrap_bounded_query_preserves_inner_query() -> None:
    wrapped = wrap_bounded_query("SELECT parcel_id FROM proof.parcel_events ORDER BY parcel_id LIMIT 5")
    assert wrapped.startswith("SELECT * FROM (")
    assert "ORDER BY parcel_id LIMIT 5" in wrapped
    assert wrapped.endswith(") AS _proof_bounded LIMIT 201")

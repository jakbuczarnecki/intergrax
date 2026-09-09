"""Unit tests for structured CONTAINS capability contract in PostgreSQL schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from intergrax.integrations.providers.relational_store.postgresql.session import (
    import_psycopg,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapSchemaError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    StructuredAttributeTableSpec,
    _STRUCTURED_CANONICAL_EQUALS_INDEX_NAME,
    _STRUCTURED_CONTAINS_INDEX_NAME,
    _STRUCTURED_SOURCE_EQUALS_INDEX_NAME,
    pg_trgm_extension_available,
    structured_contains_capability_available,
    verify_structured_attribute_table_compatible,
)

pytestmark = pytest.mark.unit

_, _, _, _PSYCOPG_SQL = import_psycopg()

SqlParam = str | int | None
SqlParams = tuple[SqlParam, ...]
SqlStatement = str | _PSYCOPG_SQL.Composable

_TABLE_SPEC = StructuredAttributeTableSpec(
    schema_name="vpi_test_schema",
    table_name="vpi_structured_attribute",
)

_STRUCTURED_COLUMNS: list[Mapping[str, str]] = [
    {"column_name": "catalog_id", "data_type": "text", "is_nullable": "NO"},
    {"column_name": "offer_id", "data_type": "text", "is_nullable": "NO"},
    {"column_name": "source_revision_norm", "data_type": "text", "is_nullable": "NO"},
    {"column_name": "source_revision", "data_type": "text", "is_nullable": "YES"},
    {"column_name": "attr_identity", "data_type": "text", "is_nullable": "NO"},
    {"column_name": "canonical_key", "data_type": "text", "is_nullable": "YES"},
    {"column_name": "source_key", "data_type": "text", "is_nullable": "NO"},
    {"column_name": "source_value", "data_type": "text", "is_nullable": "NO"},
    {"column_name": "normalized_text_value", "data_type": "text", "is_nullable": "NO"},
    {"column_name": "typed_value_text", "data_type": "text", "is_nullable": "YES"},
    {"column_name": "source_field", "data_type": "text", "is_nullable": "NO"},
]


def _executed_sql_text(statement: SqlStatement) -> str:
    if isinstance(statement, str):
        return statement
    return statement.as_string(None)


@dataclass
class _FakeCursor:
    _rows: list[Mapping[str, str | int | None]] = field(default_factory=list)

    def fetchall(self) -> list[Mapping[str, str | int | None]]:
        return list(self._rows)

    def fetchone(self) -> Mapping[str, str | int | None] | None:
        if not self._rows:
            return None
        return self._rows[0]


@dataclass
class _FakeSession:
    pg_trgm_available: bool = False
    contains_index_present: bool = False
    contains_index_schema: str = "vpi_test_schema"
    contains_index_table: str = "vpi_structured_attribute"
    contains_index_valid_definition: bool = True
    executed: list[tuple[SqlStatement, SqlParams]] = field(default_factory=list)

    def execute(self, sql: SqlStatement, params: SqlParams = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        sql_text = _executed_sql_text(sql).lower()
        if "information_schema.columns" in sql_text:
            return _FakeCursor(_rows=_STRUCTURED_COLUMNS)
        if "information_schema.table_constraints" in sql_text:
            return _FakeCursor(_rows=[{"constraint_name": "vpi_structured_attribute_pk"}])
        if "pg_indexes" in sql_text and len(params) == 3:
            return _FakeCursor(_rows=[{"indexname": str(params[2])}])
        if "pg_extension" in sql_text and "extname" in sql_text:
            if self.pg_trgm_available and params == ("pg_trgm",):
                return _FakeCursor(_rows=[{"present": 1}])
            return _FakeCursor(_rows=[])
        if "pg_index" in sql_text and "gin_trgm_ops" in sql_text:
            if (
                self.contains_index_present
                and self.contains_index_valid_definition
                and len(params) == 3
                and str(params[0]) == self.contains_index_schema
                and str(params[1]) == self.contains_index_table
                and str(params[2]) == _STRUCTURED_CONTAINS_INDEX_NAME
            ):
                return _FakeCursor(_rows=[{"present": 1}])
            return _FakeCursor(_rows=[])
        return _FakeCursor(_rows=[])


def test_structured_contains_capability_requires_pg_trgm_and_index() -> None:
    session = _FakeSession(pg_trgm_available=True, contains_index_present=True)
    assert structured_contains_capability_available(session, _TABLE_SPEC) is True


def test_structured_contains_capability_unavailable_without_pg_trgm() -> None:
    session = _FakeSession(pg_trgm_available=False, contains_index_present=True)
    assert structured_contains_capability_available(session, _TABLE_SPEC) is False


def test_structured_contains_capability_unavailable_without_index() -> None:
    session = _FakeSession(pg_trgm_available=True, contains_index_present=False)
    assert structured_contains_capability_available(session, _TABLE_SPEC) is False


def test_structured_contains_capability_rejects_wrong_schema() -> None:
    session = _FakeSession(
        pg_trgm_available=True,
        contains_index_present=True,
        contains_index_schema="other_schema",
    )
    assert structured_contains_capability_available(session, _TABLE_SPEC) is False


def test_structured_contains_capability_rejects_wrong_table() -> None:
    session = _FakeSession(
        pg_trgm_available=True,
        contains_index_present=True,
        contains_index_table="other_table",
    )
    assert structured_contains_capability_available(session, _TABLE_SPEC) is False


def test_structured_contains_capability_rejects_incompatible_definition() -> None:
    session = _FakeSession(
        pg_trgm_available=True,
        contains_index_present=True,
        contains_index_valid_definition=False,
    )
    assert structured_contains_capability_available(session, _TABLE_SPEC) is False


def test_pg_trgm_extension_available_reports_presence() -> None:
    assert pg_trgm_extension_available(_FakeSession(pg_trgm_available=True)) is True
    assert pg_trgm_extension_available(_FakeSession(pg_trgm_available=False)) is False


def test_verify_structured_table_requires_compatible_contains_index() -> None:
    session = _FakeSession(pg_trgm_available=True, contains_index_present=True)
    verify_structured_attribute_table_compatible(
        session,
        _TABLE_SPEC,
        require_contains_index=True,
    )


def test_verify_structured_table_rejects_missing_contains_index() -> None:
    session = _FakeSession(pg_trgm_available=True, contains_index_present=False)
    with pytest.raises(PostgreSqlBootstrapSchemaError, match=_STRUCTURED_CONTAINS_INDEX_NAME):
        verify_structured_attribute_table_compatible(
            session,
            _TABLE_SPEC,
            require_contains_index=True,
        )


def test_verify_structured_table_does_not_require_contains_index_by_default() -> None:
    session = _FakeSession(pg_trgm_available=False, contains_index_present=False)
    verify_structured_attribute_table_compatible(session, _TABLE_SPEC)


def test_contains_index_name_is_canonical() -> None:
    assert _STRUCTURED_CONTAINS_INDEX_NAME == "vpi_structured_value_trgm_idx"
    assert _STRUCTURED_CANONICAL_EQUALS_INDEX_NAME == "vpi_structured_canonical_equals_idx"
    assert _STRUCTURED_SOURCE_EQUALS_INDEX_NAME == "vpi_structured_source_equals_idx"

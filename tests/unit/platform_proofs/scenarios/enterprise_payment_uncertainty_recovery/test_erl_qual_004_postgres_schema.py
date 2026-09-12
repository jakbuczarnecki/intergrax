"""Architecture tests for ERL-QUAL-004 PostgreSQL schema implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.database.contract import (
    CANONICAL_SCHEMA_SNAPSHOT,
    COMMERCE_SCHEMA,
    EXTERNAL_SOR_SCHEMA,
    INITIAL_MIGRATION,
    RECONCILIATION_SCHEMA,
    REQUIRED_TABLES,
    list_migration_files,
    read_sql,
    validate_migration_sql,
    validate_schema_artifacts,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_ROOT = (
    _REPO_ROOT / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
)
_IMPL_DOC = _SCENARIO_ROOT / "docs" / "ERL_QUAL_004_POSTGRESQL_SCHEMA_IMPLEMENTATION.md"


def test_schema_migration_and_snapshot_files_exist() -> None:
    assert INITIAL_MIGRATION.is_file()
    assert CANONICAL_SCHEMA_SNAPSHOT.is_file()
    assert list_migration_files() == [INITIAL_MIGRATION]


def test_schema_contract_has_no_violations() -> None:
    assert not validate_schema_artifacts()


def test_required_entities_and_constraints_present() -> None:
    sql = read_sql(INITIAL_MIGRATION)
    violations = validate_migration_sql(sql)
    assert not violations
    lowered = sql.lower()
    for table in REQUIRED_TABLES:
        assert f"create table {table}" in lowered
    assert "primary key" in lowered
    assert "references commerce.payment_intents" in lowered
    assert "references external_sor.external_payment_effects" in lowered


def test_external_reality_separate_from_application_knowledge() -> None:
    sql = read_sql(INITIAL_MIGRATION).lower()
    assert f"create table {COMMERCE_SCHEMA}.application_knowledge" in sql
    assert f"create table {EXTERNAL_SOR_SCHEMA}.external_reality" in sql
    assert sql.find(f"{COMMERCE_SCHEMA}.application_knowledge") < sql.find(
        f"{EXTERNAL_SOR_SCHEMA}.external_reality"
    )


def test_unknown_semantics_not_equated_to_failure() -> None:
    sql = read_sql(INITIAL_MIGRATION)
    assert "'UNKNOWN'" in sql
    assert "'FAILED'" in sql
    assert "application_knowledge_unknown_explicit_check" in sql.lower()
    assert "known_status in ('unknown', 'confirmed', 'failed')" in sql.lower()


def test_reconciliation_zone_tables_defined() -> None:
    sql = read_sql(INITIAL_MIGRATION).lower()
    assert f"create table {RECONCILIATION_SCHEMA}.reconciliation_cases" in sql
    assert f"create table {RECONCILIATION_SCHEMA}.investigation_attempts" in sql
    assert f"create table {RECONCILIATION_SCHEMA}.evidence_references" in sql
    assert f"create table {RECONCILIATION_SCHEMA}.resolution_records" in sql


def test_schema_implementation_documentation_present() -> None:
    assert _IMPL_DOC.is_file()
    body = _IMPL_DOC.read_text(encoding="utf-8")
    assert "External Reality" in body
    assert "Application Knowledge" in body
    assert "UNKNOWN" in body

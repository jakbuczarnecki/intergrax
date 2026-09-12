"""Static contract for ERL-QUAL-004 PostgreSQL schema artifacts."""

from __future__ import annotations

import re
from pathlib import Path

DATABASE_ROOT = Path(__file__).resolve().parent
MIGRATIONS_DIR = DATABASE_ROOT / "migrations"
SCHEMA_DIR = DATABASE_ROOT / "schema"
INITIAL_MIGRATION = MIGRATIONS_DIR / "001_erl_qual_004_core_schema.sql"
CANONICAL_SCHEMA_SNAPSHOT = SCHEMA_DIR / "erl_qual_004_core_schema.sql"

COMMERCE_SCHEMA = "commerce"
EXTERNAL_SOR_SCHEMA = "external_sor"
RECONCILIATION_SCHEMA = "reconciliation"

REQUIRED_TABLES: frozenset[str] = frozenset(
    {
        f"{COMMERCE_SCHEMA}.organizations",
        f"{COMMERCE_SCHEMA}.orders",
        f"{COMMERCE_SCHEMA}.payment_intents",
        f"{COMMERCE_SCHEMA}.application_knowledge",
        f"{EXTERNAL_SOR_SCHEMA}.external_payment_effects",
        f"{EXTERNAL_SOR_SCHEMA}.external_reality",
        f"{RECONCILIATION_SCHEMA}.reconciliation_cases",
        f"{RECONCILIATION_SCHEMA}.investigation_attempts",
        f"{RECONCILIATION_SCHEMA}.evidence_references",
        f"{RECONCILIATION_SCHEMA}.resolution_records",
    }
)

REQUIRED_FOREIGN_KEY_HINTS: frozenset[str] = frozenset(
    {
        "REFERENCES commerce.organizations",
        "REFERENCES commerce.orders",
        "REFERENCES commerce.payment_intents",
        "REFERENCES external_sor.external_payment_effects",
        "REFERENCES reconciliation.reconciliation_cases",
        "REFERENCES reconciliation.investigation_attempts",
    }
)

REQUIRED_SEMANTIC_MARKERS: frozenset[str] = frozenset(
    {
        "known_status IN ('UNKNOWN', 'CONFIRMED', 'FAILED')",
        "application_knowledge_unknown_explicit_check",
        "terminal_outcome IN",
        "TRUTH_INDETERMINATE",
        "truth_availability_state",
    }
)

_MIGRATION_FILE_PATTERN = re.compile(r"^\d{3}_[\w]+\.sql$", re.IGNORECASE)


def list_migration_files() -> list[Path]:
    """Return migration SQL files in lexical order."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    return sorted(
        path
        for path in MIGRATIONS_DIR.iterdir()
        if path.is_file() and _MIGRATION_FILE_PATTERN.match(path.name)
    )


def read_sql(path: Path) -> str:
    if not path.is_file():
        msg = f"missing SQL artifact: {path}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


def validate_migration_sql(sql: str) -> list[str]:
    """Return human-readable schema contract violations (empty when valid)."""
    violations: list[str] = []
    lowered = sql.lower()
    if "begin;" not in lowered or "commit;" not in lowered:
        violations.append("migration must be wrapped in BEGIN/COMMIT transaction")
    for table in sorted(REQUIRED_TABLES):
        if f"create table {table}" not in lowered:
            violations.append(f"missing CREATE TABLE for {table}")
    for fk in sorted(REQUIRED_FOREIGN_KEY_HINTS):
        if fk.lower() not in lowered:
            violations.append(f"missing foreign key reference: {fk}")
    for marker in sorted(REQUIRED_SEMANTIC_MARKERS):
        if marker.lower() not in lowered:
            violations.append(f"missing semantic marker: {marker}")
    if (
        "commerce.application_knowledge" in lowered
        and "external_sor.external_reality" in lowered
    ):
        if lowered.find("external_sor.external_reality") < lowered.find(
            "commerce.application_knowledge"
        ):
            violations.append(
                "application_knowledge must be declared before external_reality "
                "(commerce vs external_sor separation)"
            )
    return violations


def validate_schema_artifacts() -> list[str]:
    """Validate migration files and canonical schema snapshot."""
    violations: list[str] = []
    migrations = list_migration_files()
    if not migrations:
        violations.append("no numbered migration files under database/migrations/")
    if INITIAL_MIGRATION not in migrations:
        violations.append("missing initial migration 001_erl_qual_004_core_schema.sql")
    if not CANONICAL_SCHEMA_SNAPSHOT.is_file():
        violations.append("missing canonical schema snapshot under database/schema/")
    for migration in migrations:
        violations.extend(validate_migration_sql(read_sql(migration)))
    if CANONICAL_SCHEMA_SNAPSHOT.is_file():
        violations.extend(validate_migration_sql(read_sql(CANONICAL_SCHEMA_SNAPSHOT)))
    return violations

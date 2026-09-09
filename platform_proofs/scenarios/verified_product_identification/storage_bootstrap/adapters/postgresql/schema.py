"""PostgreSQL DDL and compatibility checks for relational Data Pack bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg.sql import Composable

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLSession,
    import_psycopg,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapSchemaError,
)

_REQUIRED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("catalog_id", "text", "NO"),
    ("offer_id", "text", "NO"),
    ("source_revision_norm", "text", "NO"),
    ("source_revision", "text", "YES"),
    ("global_row_index", "bigint", "NO"),
    ("record_json", "jsonb", "NO"),
    ("semantic_text", "text", "NO"),
    ("semantic_text_hash", "text", "NO"),
    ("derivation_version", "text", "NO"),
)

_REQUIRED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "vpi_dpr_source_identity_pk",
        "vpi_dpr_global_row_index_uq",
    }
)


@dataclass(frozen=True, slots=True)
class RelationalTableSpec:
    schema_name: str
    table_name: str


def create_table_ddl(spec: RelationalTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )
    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            catalog_id TEXT NOT NULL,
            offer_id TEXT NOT NULL,
            source_revision_norm TEXT NOT NULL DEFAULT '',
            source_revision TEXT,
            global_row_index BIGINT NOT NULL,
            record_json JSONB NOT NULL,
            semantic_text TEXT NOT NULL,
            semantic_text_hash TEXT NOT NULL,
            derivation_version TEXT NOT NULL,
            CONSTRAINT vpi_dpr_source_identity_pk
                PRIMARY KEY (catalog_id, offer_id, source_revision_norm),
            CONSTRAINT vpi_dpr_global_row_index_uq UNIQUE (global_row_index)
        )
        """
    ).format(table=qualified)


def verify_table_compatible(session: PostgreSQLSession, spec: RelationalTableSpec) -> None:
    columns = session.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (spec.schema_name, spec.table_name),
    ).fetchall()
    if not columns:
        raise PostgreSqlBootstrapSchemaError("POSTGRESQL_SCHEMA_INCOMPATIBLE: table missing")

    actual_columns = {
        (str(row["column_name"]), str(row["data_type"]), str(row["is_nullable"]))
        for row in columns
    }
    for required_name, required_type, required_nullable in _REQUIRED_COLUMNS:
        if (required_name, required_type, required_nullable) not in actual_columns:
            raise PostgreSqlBootstrapSchemaError(
                "POSTGRESQL_SCHEMA_INCOMPATIBLE: "
                f"missing or incompatible column {required_name}"
            )

    constraints = session.execute(
        """
        SELECT tc.constraint_name
        FROM information_schema.table_constraints tc
        WHERE tc.table_schema = %s
          AND tc.table_name = %s
          AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
        """,
        (spec.schema_name, spec.table_name),
    ).fetchall()
    present = {str(row["constraint_name"]) for row in constraints}
    missing = sorted(_REQUIRED_CONSTRAINTS - present)
    if missing:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing constraints "
            + ", ".join(missing)
        )

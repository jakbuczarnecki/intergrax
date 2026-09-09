"""PgVector DDL and compatibility checks for vector Data Pack bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg.sql import Composable

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLSession,
    import_psycopg,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.errors import (
    PgVectorBootstrapSchemaError,
)

_REQUIRED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("logical_point_id", "text", "NO"),
    ("catalog_id", "text", "NO"),
    ("offer_id", "text", "NO"),
    ("source_revision_norm", "text", "NO"),
    ("source_revision", "text", "YES"),
    ("semantic_text_hash", "text", "NO"),
    ("embedding_provider", "text", "NO"),
    ("embedding_model", "text", "NO"),
    ("embedding_revision", "text", "YES"),
    ("embedding_dimension", "integer", "NO"),
    ("derivation_version", "text", "NO"),
    ("dense_embedding", "USER-DEFINED", "NO"),
)

_REQUIRED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "vpi_dpv_logical_point_id_pk",
        "vpi_dpv_source_identity_uq",
    }
)


@dataclass(frozen=True, slots=True)
class PgVectorTableSpec:
    schema_name: str
    table_name: str
    dimension: int


def ensure_pgvector_extension(
    session: PostgreSQLSession,
    *,
    allow_create: bool,
) -> None:
    row = session.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS exists"
    ).fetchone()
    if row is not None and bool(row["exists"]):
        return
    if allow_create:
        _, _, _, sql = import_psycopg()
        session.execute_statement(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))
        return
    raise PgVectorBootstrapSchemaError("PGVECTOR_EXTENSION_UNAVAILABLE")


def create_table_ddl(spec: PgVectorTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )
    vector_type = sql.SQL("vector({})").format(sql.Literal(spec.dimension))
    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            logical_point_id TEXT NOT NULL,
            catalog_id TEXT NOT NULL,
            offer_id TEXT NOT NULL,
            source_revision_norm TEXT NOT NULL DEFAULT '',
            source_revision TEXT,
            semantic_text_hash TEXT NOT NULL,
            embedding_provider TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_revision TEXT,
            embedding_dimension INTEGER NOT NULL,
            derivation_version TEXT NOT NULL,
            dense_embedding {vector_type} NOT NULL,
            CONSTRAINT vpi_dpv_logical_point_id_pk PRIMARY KEY (logical_point_id),
            CONSTRAINT vpi_dpv_source_identity_uq
                UNIQUE (catalog_id, offer_id, source_revision_norm)
        )
        """
    ).format(table=qualified, vector_type=vector_type)


def verify_table_compatible(session: PostgreSQLSession, spec: PgVectorTableSpec) -> None:
    columns = session.execute(
        """
        SELECT column_name, data_type, is_nullable, udt_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (spec.schema_name, spec.table_name),
    ).fetchall()
    if not columns:
        raise PgVectorBootstrapSchemaError("PGVECTOR_SCHEMA_INCOMPATIBLE: table missing")

    actual_columns = {
        (
            str(row["column_name"]),
            str(row["data_type"]),
            str(row["is_nullable"]),
        )
        for row in columns
    }
    for required_name, required_type, required_nullable in _REQUIRED_COLUMNS:
        if (required_name, required_type, required_nullable) not in actual_columns:
            raise PgVectorBootstrapSchemaError(
                "PGVECTOR_SCHEMA_INCOMPATIBLE: "
                f"missing or incompatible column {required_name}"
            )

    embedding_udt = next(
        (
            str(row["udt_name"])
            for row in columns
            if str(row["column_name"]) == "dense_embedding"
        ),
        "",
    )
    if embedding_udt != "vector":
        raise PgVectorBootstrapSchemaError(
            "PGVECTOR_SCHEMA_INCOMPATIBLE: dense_embedding is not vector type"
        )

    type_row = session.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS vector_type
        FROM pg_attribute AS a
        JOIN pg_class AS c ON c.oid = a.attrelid
        JOIN pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = %s
          AND c.relname = %s
          AND a.attname = 'dense_embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        (spec.schema_name, spec.table_name),
    ).fetchone()
    expected_type = f"vector({spec.dimension})"
    actual_type = str(type_row["vector_type"]) if type_row is not None else ""
    if actual_type != expected_type:
        raise PgVectorBootstrapSchemaError(
            "PGVECTOR_SCHEMA_INCOMPATIBLE: "
            f"dense_embedding type {actual_type!r} != expected {expected_type!r}"
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
        raise PgVectorBootstrapSchemaError(
            "PGVECTOR_SCHEMA_INCOMPATIBLE: missing constraints "
            + ", ".join(missing)
        )

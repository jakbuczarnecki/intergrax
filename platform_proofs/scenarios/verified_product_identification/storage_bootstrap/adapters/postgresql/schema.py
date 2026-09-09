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

_IDENTIFIER_REQUIRED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("catalog_id", "text", "NO"),
    ("offer_id", "text", "NO"),
    ("source_revision_norm", "text", "NO"),
    ("source_revision", "text", "YES"),
    ("identifier_type", "text", "NO"),
    ("source_value", "text", "NO"),
    ("normalized_value", "text", "NO"),
    ("source_field", "text", "NO"),
)

_IDENTIFIER_REQUIRED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "vpi_dpi_source_identifier_pk",
    }
)

_IDENTIFIER_LOOKUP_INDEX_NAME = "vpi_product_identifiers_lookup_idx"

_LEXICAL_DOCUMENT_REQUIRED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("catalog_id", "text", "NO"),
    ("offer_id", "text", "NO"),
    ("source_revision_norm", "text", "NO"),
    ("source_revision", "text", "YES"),
    ("lexical_document", "text", "NO"),
    ("document_hash", "text", "NO"),
    ("document_length", "integer", "NO"),
    ("derivation_version", "text", "NO"),
)

_LEXICAL_DOCUMENT_REQUIRED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "vpi_lexical_document_pk",
    }
)

_LEXICAL_POSTING_REQUIRED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("term", "text", "NO"),
    ("catalog_id", "text", "NO"),
    ("offer_id", "text", "NO"),
    ("source_revision_norm", "text", "NO"),
    ("term_frequency", "integer", "NO"),
)

_LEXICAL_POSTING_REQUIRED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "vpi_lexical_posting_pk",
    }
)

_LEXICAL_POSTING_LOOKUP_INDEX_NAME = "vpi_lexical_posting_term_idx"

_LEXICAL_CORPUS_STATS_REQUIRED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("singleton_key", "text", "NO"),
    ("statistics_version", "text", "NO"),
    ("document_count", "bigint", "NO"),
    ("total_document_length", "bigint", "NO"),
    ("average_document_length", "double precision", "NO"),
)

_LEXICAL_CORPUS_STATS_REQUIRED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "vpi_lexical_corpus_stats_pk",
    }
)

_LEXICAL_TERM_STATS_REQUIRED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("term", "text", "NO"),
    ("document_frequency", "integer", "NO"),
)

_LEXICAL_TERM_STATS_REQUIRED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "vpi_lexical_term_stats_pk",
    }
)

LEXICAL_STATISTICS_VERSION = "v1"
LEXICAL_CORPUS_STATS_SINGLETON_KEY = "default"


@dataclass(frozen=True, slots=True)
class RelationalTableSpec:
    schema_name: str
    table_name: str


@dataclass(frozen=True, slots=True)
class IdentifierTableSpec:
    schema_name: str
    table_name: str


@dataclass(frozen=True, slots=True)
class LexicalDocumentTableSpec:
    schema_name: str
    table_name: str


@dataclass(frozen=True, slots=True)
class LexicalPostingTableSpec:
    schema_name: str
    table_name: str


@dataclass(frozen=True, slots=True)
class LexicalCorpusStatsTableSpec:
    schema_name: str
    table_name: str


@dataclass(frozen=True, slots=True)
class LexicalTermStatsTableSpec:
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


def create_identifier_table_ddl(spec: IdentifierTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = qualified_identifier_table(spec)
    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            catalog_id TEXT NOT NULL,
            offer_id TEXT NOT NULL,
            source_revision_norm TEXT NOT NULL DEFAULT '',
            source_revision TEXT,
            identifier_type TEXT NOT NULL,
            source_value TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            source_field TEXT NOT NULL,
            CONSTRAINT vpi_dpi_source_identifier_pk
                PRIMARY KEY (
                    catalog_id,
                    offer_id,
                    source_revision_norm,
                    identifier_type,
                    normalized_value,
                    source_field
                )
        )
        """
    ).format(table=qualified)


def create_identifier_lookup_index_ddl(spec: IdentifierTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = qualified_identifier_table(spec)
    return sql.SQL(
        """
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {table} (identifier_type, normalized_value)
        """
    ).format(
        index_name=sql.Identifier(_IDENTIFIER_LOOKUP_INDEX_NAME),
        table=qualified,
    )


def qualified_identifier_table(spec: IdentifierTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )


def identifier_insert_dml(spec: IdentifierTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        INSERT INTO {table} (
            catalog_id,
            offer_id,
            source_revision_norm,
            source_revision,
            identifier_type,
            source_value,
            normalized_value,
            source_field
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """
    ).format(table=qualified_identifier_table(spec))


def create_lexical_document_table_ddl(spec: LexicalDocumentTableSpec) -> Composable:
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
            lexical_document TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            document_length INTEGER NOT NULL,
            derivation_version TEXT NOT NULL,
            CONSTRAINT vpi_lexical_document_pk
                PRIMARY KEY (catalog_id, offer_id, source_revision_norm)
        )
        """
    ).format(table=qualified)


def create_lexical_posting_table_ddl(spec: LexicalPostingTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )
    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            term TEXT NOT NULL,
            catalog_id TEXT NOT NULL,
            offer_id TEXT NOT NULL,
            source_revision_norm TEXT NOT NULL DEFAULT '',
            term_frequency INTEGER NOT NULL,
            CONSTRAINT vpi_lexical_posting_pk
                PRIMARY KEY (
                    term,
                    catalog_id,
                    offer_id,
                    source_revision_norm
                )
        )
        """
    ).format(table=qualified)


def create_lexical_posting_lookup_index_ddl(spec: LexicalPostingTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )
    return sql.SQL(
        """
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {table} (term)
        """
    ).format(
        index_name=sql.Identifier(_LEXICAL_POSTING_LOOKUP_INDEX_NAME),
        table=qualified,
    )


def qualified_lexical_document_table(spec: LexicalDocumentTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )


def qualified_lexical_posting_table(spec: LexicalPostingTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )


def lexical_document_insert_dml(spec: LexicalDocumentTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        INSERT INTO {table} (
            catalog_id,
            offer_id,
            source_revision_norm,
            source_revision,
            lexical_document,
            document_hash,
            document_length,
            derivation_version
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """
    ).format(table=qualified_lexical_document_table(spec))


def lexical_posting_insert_dml(spec: LexicalPostingTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        INSERT INTO {table} (
            term,
            catalog_id,
            offer_id,
            source_revision_norm,
            term_frequency
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """
    ).format(table=qualified_lexical_posting_table(spec))


def create_lexical_corpus_stats_table_ddl(spec: LexicalCorpusStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )
    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            singleton_key TEXT NOT NULL DEFAULT 'default',
            statistics_version TEXT NOT NULL,
            document_count BIGINT NOT NULL,
            total_document_length BIGINT NOT NULL,
            average_document_length DOUBLE PRECISION NOT NULL,
            CONSTRAINT vpi_lexical_corpus_stats_pk PRIMARY KEY (singleton_key),
            CONSTRAINT vpi_lexical_corpus_stats_singleton_ck
                CHECK (singleton_key = 'default')
        )
        """
    ).format(table=qualified)


def create_lexical_term_stats_table_ddl(spec: LexicalTermStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    qualified = sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )
    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            term TEXT NOT NULL,
            document_frequency INTEGER NOT NULL,
            CONSTRAINT vpi_lexical_term_stats_pk PRIMARY KEY (term)
        )
        """
    ).format(table=qualified)


def qualified_lexical_corpus_stats_table(spec: LexicalCorpusStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )


def qualified_lexical_term_stats_table(spec: LexicalTermStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL("{}.{}").format(
        sql.Identifier(spec.schema_name),
        sql.Identifier(spec.table_name),
    )


def lexical_corpus_stats_lookup_dml(spec: LexicalCorpusStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        SELECT
            document_count,
            average_document_length
        FROM {table}
        WHERE singleton_key = %s
        """
    ).format(table=qualified_lexical_corpus_stats_table(spec))


def lexical_corpus_stats_increment_dml(spec: LexicalCorpusStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        INSERT INTO {table} (
            singleton_key,
            statistics_version,
            document_count,
            total_document_length,
            average_document_length
        )
        VALUES (%s, %s, 1, %s, %s::double precision)
        ON CONFLICT (singleton_key) DO UPDATE SET
            statistics_version = EXCLUDED.statistics_version,
            document_count = {table}.document_count + 1,
            total_document_length = {table}.total_document_length + EXCLUDED.total_document_length,
            average_document_length = (
                ({table}.total_document_length + EXCLUDED.total_document_length)::double precision
                / ({table}.document_count + 1)::double precision
            )
        """
    ).format(table=qualified_lexical_corpus_stats_table(spec))


def lexical_term_stats_increment_dml(spec: LexicalTermStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        INSERT INTO {table} (term, document_frequency)
        VALUES (%s, 1)
        ON CONFLICT (term) DO UPDATE SET
            document_frequency = {table}.document_frequency + 1
        """
    ).format(table=qualified_lexical_term_stats_table(spec))


def rebuild_lexical_corpus_stats_dml(
    document_spec: LexicalDocumentTableSpec,
    corpus_stats_spec: LexicalCorpusStatsTableSpec,
) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        INSERT INTO {corpus_stats} (
            singleton_key,
            statistics_version,
            document_count,
            total_document_length,
            average_document_length
        )
        SELECT
            %s,
            %s,
            COUNT(*)::bigint,
            COALESCE(SUM(document_length), 0)::bigint,
            COALESCE(AVG(document_length), 0)::double precision
        FROM {document}
        ON CONFLICT (singleton_key) DO UPDATE SET
            statistics_version = EXCLUDED.statistics_version,
            document_count = EXCLUDED.document_count,
            total_document_length = EXCLUDED.total_document_length,
            average_document_length = EXCLUDED.average_document_length
        """
    ).format(
        corpus_stats=qualified_lexical_corpus_stats_table(corpus_stats_spec),
        document=qualified_lexical_document_table(document_spec),
    )


def clear_lexical_term_stats_dml(term_stats_spec: LexicalTermStatsTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL("DELETE FROM {table}").format(
        table=qualified_lexical_term_stats_table(term_stats_spec)
    )


def insert_lexical_term_stats_from_postings_dml(
    posting_spec: LexicalPostingTableSpec,
    term_stats_spec: LexicalTermStatsTableSpec,
) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        INSERT INTO {term_stats} (term, document_frequency)
        SELECT
            term,
            COUNT(*)::integer
        FROM {posting}
        GROUP BY term
        """
    ).format(
        term_stats=qualified_lexical_term_stats_table(term_stats_spec),
        posting=qualified_lexical_posting_table(posting_spec),
    )


def rebuild_lexical_statistics(
    session: PostgreSQLSession,
    *,
    document_spec: LexicalDocumentTableSpec,
    posting_spec: LexicalPostingTableSpec,
    corpus_stats_spec: LexicalCorpusStatsTableSpec,
    term_stats_spec: LexicalTermStatsTableSpec,
    statistics_version: str = LEXICAL_STATISTICS_VERSION,
) -> None:
    session.execute(
        rebuild_lexical_corpus_stats_dml(document_spec, corpus_stats_spec),
        (LEXICAL_CORPUS_STATS_SINGLETON_KEY, statistics_version),
    )
    session.execute(clear_lexical_term_stats_dml(term_stats_spec))
    session.execute(
        insert_lexical_term_stats_from_postings_dml(posting_spec, term_stats_spec)
    )


def lexical_bm25_ranked_search_dml(
    document_spec: LexicalDocumentTableSpec,
    posting_spec: LexicalPostingTableSpec,
    corpus_stats_spec: LexicalCorpusStatsTableSpec,
    term_stats_spec: LexicalTermStatsTableSpec,
    *,
    k1: float,
    b: float,
) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        WITH query_terms AS (
            SELECT unnest(%s::text[]) AS term
        ),
        term_contributions AS (
            SELECT
                p.catalog_id,
                p.offer_id,
                p.source_revision_norm,
                d.source_revision,
                (
                    LN(
                        1.0 + (
                            cs.document_count::double precision
                            - ts.document_frequency::double precision
                            + 0.5
                        ) / (ts.document_frequency::double precision + 0.5)
                    )
                    * p.term_frequency::double precision
                    * ({k1} + 1.0)
                    / GREATEST(
                        p.term_frequency::double precision
                        + {k1} * (
                            1.0 - {b}
                            + {b} * d.document_length::double precision
                              / GREATEST(cs.average_document_length, 1.0)
                        ),
                        1e-9
                    )
                ) AS term_score
            FROM query_terms qt
            INNER JOIN {posting} p ON p.term = qt.term
            INNER JOIN {document} d
                ON d.catalog_id = p.catalog_id
               AND d.offer_id = p.offer_id
               AND d.source_revision_norm = p.source_revision_norm
            INNER JOIN {term_stats} ts ON ts.term = qt.term
            CROSS JOIN {corpus_stats} cs
            WHERE cs.singleton_key = %s
              AND cs.document_count > 0
              AND ts.document_frequency > 0
              AND p.term_frequency > 0
        )
        SELECT
            catalog_id,
            offer_id,
            source_revision_norm,
            source_revision,
            SUM(term_score) AS bm25_score
        FROM term_contributions
        GROUP BY catalog_id, offer_id, source_revision_norm, source_revision
        HAVING SUM(term_score) > 0.0
        ORDER BY
            bm25_score DESC,
            catalog_id ASC,
            offer_id ASC,
            source_revision_norm ASC
        LIMIT %s
        """
    ).format(
        posting=qualified_lexical_posting_table(posting_spec),
        document=qualified_lexical_document_table(document_spec),
        term_stats=qualified_lexical_term_stats_table(term_stats_spec),
        corpus_stats=qualified_lexical_corpus_stats_table(corpus_stats_spec),
        k1=sql.Literal(k1),
        b=sql.Literal(b),
    )


def lexical_posting_lookup_dml(spec: LexicalPostingTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        SELECT
            term,
            catalog_id,
            offer_id,
            source_revision_norm,
            term_frequency
        FROM {table}
        WHERE term = ANY(%s)
        """
    ).format(table=qualified_lexical_posting_table(spec))


def lexical_document_lookup_dml(spec: LexicalDocumentTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        SELECT
            catalog_id,
            offer_id,
            source_revision_norm,
            source_revision,
            document_length
        FROM {table}
        WHERE catalog_id = %s
          AND offer_id = %s
          AND source_revision_norm = %s
        """
    ).format(table=qualified_lexical_document_table(spec))


def verify_lexical_document_table_compatible(
    session: PostgreSQLSession,
    spec: LexicalDocumentTableSpec,
) -> None:
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
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: lexical document table missing"
        )

    actual_columns = {
        (str(row["column_name"]), str(row["data_type"]), str(row["is_nullable"]))
        for row in columns
    }
    for required_name, required_type, required_nullable in _LEXICAL_DOCUMENT_REQUIRED_COLUMNS:
        if (required_name, required_type, required_nullable) not in actual_columns:
            raise PostgreSqlBootstrapSchemaError(
                "POSTGRESQL_SCHEMA_INCOMPATIBLE: "
                f"missing or incompatible lexical document column {required_name}"
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
    missing = sorted(_LEXICAL_DOCUMENT_REQUIRED_CONSTRAINTS - present)
    if missing:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing lexical document constraints "
            + ", ".join(missing)
        )


def verify_lexical_corpus_stats_table_compatible(
    session: PostgreSQLSession,
    spec: LexicalCorpusStatsTableSpec,
) -> None:
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
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: lexical corpus stats table missing"
        )

    actual_columns = {
        (str(row["column_name"]), str(row["data_type"]), str(row["is_nullable"]))
        for row in columns
    }
    for required_name, required_type, required_nullable in _LEXICAL_CORPUS_STATS_REQUIRED_COLUMNS:
        if (required_name, required_type, required_nullable) not in actual_columns:
            raise PostgreSqlBootstrapSchemaError(
                "POSTGRESQL_SCHEMA_INCOMPATIBLE: "
                f"missing or incompatible lexical corpus stats column {required_name}"
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
    missing = sorted(_LEXICAL_CORPUS_STATS_REQUIRED_CONSTRAINTS - present)
    if missing:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing lexical corpus stats constraints "
            + ", ".join(missing)
        )


def verify_lexical_term_stats_table_compatible(
    session: PostgreSQLSession,
    spec: LexicalTermStatsTableSpec,
) -> None:
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
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: lexical term stats table missing"
        )

    actual_columns = {
        (str(row["column_name"]), str(row["data_type"]), str(row["is_nullable"]))
        for row in columns
    }
    for required_name, required_type, required_nullable in _LEXICAL_TERM_STATS_REQUIRED_COLUMNS:
        if (required_name, required_type, required_nullable) not in actual_columns:
            raise PostgreSqlBootstrapSchemaError(
                "POSTGRESQL_SCHEMA_INCOMPATIBLE: "
                f"missing or incompatible lexical term stats column {required_name}"
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
    missing = sorted(_LEXICAL_TERM_STATS_REQUIRED_CONSTRAINTS - present)
    if missing:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing lexical term stats constraints "
            + ", ".join(missing)
        )


def verify_lexical_posting_table_compatible(
    session: PostgreSQLSession,
    spec: LexicalPostingTableSpec,
) -> None:
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
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: lexical posting table missing"
        )

    actual_columns = {
        (str(row["column_name"]), str(row["data_type"]), str(row["is_nullable"]))
        for row in columns
    }
    for required_name, required_type, required_nullable in _LEXICAL_POSTING_REQUIRED_COLUMNS:
        if (required_name, required_type, required_nullable) not in actual_columns:
            raise PostgreSqlBootstrapSchemaError(
                "POSTGRESQL_SCHEMA_INCOMPATIBLE: "
                f"missing or incompatible lexical posting column {required_name}"
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
    missing = sorted(_LEXICAL_POSTING_REQUIRED_CONSTRAINTS - present)
    if missing:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing lexical posting constraints "
            + ", ".join(missing)
        )

    index_row = session.execute(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = %s
          AND tablename = %s
          AND indexname = %s
        """,
        (spec.schema_name, spec.table_name, _LEXICAL_POSTING_LOOKUP_INDEX_NAME),
    ).fetchone()
    if index_row is None:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing lexical posting lookup index "
            f"{_LEXICAL_POSTING_LOOKUP_INDEX_NAME}"
        )


def identifier_lookup_dml(spec: IdentifierTableSpec) -> Composable:
    _, _, _, sql = import_psycopg()
    return sql.SQL(
        """
        SELECT
            catalog_id,
            offer_id,
            source_revision_norm,
            source_revision,
            identifier_type,
            source_value,
            normalized_value,
            source_field
        FROM {table}
        WHERE identifier_type = %s
          AND normalized_value = %s
        ORDER BY
            catalog_id ASC,
            offer_id ASC,
            source_revision_norm ASC
        LIMIT %s
        """
    ).format(table=qualified_identifier_table(spec))


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


def verify_identifier_table_compatible(
    session: PostgreSQLSession,
    spec: IdentifierTableSpec,
) -> None:
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
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: identifier table missing"
        )

    actual_columns = {
        (str(row["column_name"]), str(row["data_type"]), str(row["is_nullable"]))
        for row in columns
    }
    for required_name, required_type, required_nullable in _IDENTIFIER_REQUIRED_COLUMNS:
        if (required_name, required_type, required_nullable) not in actual_columns:
            raise PostgreSqlBootstrapSchemaError(
                "POSTGRESQL_SCHEMA_INCOMPATIBLE: "
                f"missing or incompatible identifier column {required_name}"
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
    missing = sorted(_IDENTIFIER_REQUIRED_CONSTRAINTS - present)
    if missing:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing identifier constraints "
            + ", ".join(missing)
        )

    index_row = session.execute(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = %s
          AND tablename = %s
          AND indexname = %s
        """,
        (spec.schema_name, spec.table_name, _IDENTIFIER_LOOKUP_INDEX_NAME),
    ).fetchone()
    if index_row is None:
        raise PostgreSqlBootstrapSchemaError(
            "POSTGRESQL_SCHEMA_INCOMPATIBLE: missing identifier lookup index "
            f"{_IDENTIFIER_LOOKUP_INDEX_NAME}"
        )

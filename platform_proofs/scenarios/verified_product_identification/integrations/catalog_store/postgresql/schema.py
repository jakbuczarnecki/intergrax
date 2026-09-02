"""Scenario-owned PostgreSQL DDL for VPI catalog bootstrap."""

from __future__ import annotations

CATALOG_MANIFEST_DDL = """
CREATE TABLE IF NOT EXISTS vpi_catalog_manifest (
    manifest_id INTEGER PRIMARY KEY CHECK (manifest_id = 1),
    state TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    dataset_checksum TEXT NOT NULL,
    dataset_record_count BIGINT NOT NULL,
    search_representation_derivation_version TEXT NOT NULL,
    embedding_configuration_version TEXT NOT NULL,
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    catalog_schema_version TEXT NOT NULL,
    search_index_schema_version TEXT NOT NULL,
    bootstrap_implementation_version TEXT NOT NULL,
    catalog_id TEXT NOT NULL,
    source_revision TEXT,
    checkpoint_batch_ordinal INTEGER,
    checkpoint_rows_processed BIGINT NOT NULL DEFAULT 0,
    target_max_records BIGINT,
    catalog_source_offer_count BIGINT NOT NULL DEFAULT 0,
    catalog_identifier_count BIGINT NOT NULL DEFAULT 0,
    catalog_structured_attribute_count BIGINT NOT NULL DEFAULT 0,
    search_point_count BIGINT NOT NULL DEFAULT 0,
    failure_stage TEXT,
    failure_detail TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

SOURCE_OFFER_DDL = """
CREATE TABLE IF NOT EXISTS vpi_source_offer (
    catalog_id TEXT NOT NULL,
    offer_id TEXT NOT NULL,
    source_revision TEXT,
    record_json TEXT NOT NULL,
    derivation_version TEXT NOT NULL,
    ingestion_batch TEXT NOT NULL,
    global_row_index BIGINT NOT NULL,
    PRIMARY KEY (catalog_id, offer_id)
);
"""

IDENTIFIER_DDL = """
CREATE TABLE IF NOT EXISTS vpi_identifier (
    catalog_id TEXT NOT NULL,
    offer_id TEXT NOT NULL,
    identifier_type TEXT NOT NULL,
    source_value TEXT NOT NULL,
    lookup_value TEXT NOT NULL,
    source_field TEXT NOT NULL,
    source_revision TEXT,
    PRIMARY KEY (catalog_id, offer_id, identifier_type, lookup_value, source_field)
);
"""

STRUCTURED_ATTRIBUTE_DDL = """
CREATE TABLE IF NOT EXISTS vpi_structured_attribute (
    catalog_id TEXT NOT NULL,
    offer_id TEXT NOT NULL,
    attr_identity TEXT NOT NULL,
    canonical_key TEXT,
    source_key TEXT NOT NULL,
    source_value TEXT NOT NULL,
    normalized_text_value TEXT NOT NULL,
    typed_value_text TEXT,
    source_field TEXT NOT NULL,
    source_revision TEXT,
    PRIMARY KEY (catalog_id, offer_id, attr_identity)
);
"""

INDEX_DDL_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS vpi_identifier_lookup_idx "
    "ON vpi_identifier (identifier_type, lookup_value);",
    "CREATE INDEX IF NOT EXISTS vpi_structured_canonical_idx "
    "ON vpi_structured_attribute (canonical_key, normalized_text_value);",
    "CREATE INDEX IF NOT EXISTS vpi_source_offer_row_idx "
    "ON vpi_source_offer (global_row_index);",
)

SCHEMA_DDL_STATEMENTS: tuple[str, ...] = (
    CATALOG_MANIFEST_DDL,
    SOURCE_OFFER_DDL,
    IDENTIFIER_DDL,
    STRUCTURED_ATTRIBUTE_DDL,
    *INDEX_DDL_STATEMENTS,
)

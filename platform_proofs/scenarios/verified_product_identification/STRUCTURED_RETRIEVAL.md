# VPI structured retrieval (5C6D)

Canonical production structured retrieval uses the indexed adapter:

- `storage_bootstrap/adapters/postgresql/structured_search_adapter.py`
- composition: `retrieval/composition.py` → `build_structured_candidate_search`

## Backend

PostgreSQL derived-state attribute index:

- `vpi_structured_attribute` — one row per derived `StructuredAttribute`
- B-tree indexes:
  - `(canonical_key, normalized_text_value)` partial where `canonical_key IS NOT NULL`
  - `(source_key, normalized_text_value)`
- Optional CONTAINS index (when `pg_trgm` is preinstalled by infrastructure):
  - GIN `(normalized_text_value gin_trgm_ops)`

Rows are projected from Data Pack `record_json` via `derive_search_representation()` →
`StructuredSearchRepresentation` (same derivation as query-time normalization policy).

Primary key: `(catalog_id, offer_id, source_revision_norm, attr_identity)`.

## Query semantics

Multi-constraint recall: an offer is a candidate when it matches **at least one**
supplied constraint. Score is `matched_constraint_count / total_constraint_count`
(distinct satisfied query constraints, never duplicate DB rows).

Ordering: `matched_constraint_count DESC`, then `catalog_id`, `offer_id`,
`source_revision_norm`. Provider-side `LIMIT`.

### EQUALS

Deterministic normalized equality on indexed columns:

- attribute key matches `canonical_key` (preferred) or `source_key`
- value matches `normalized_text_value` exactly (same normalizer as index write)

### CONTAINS

When `pg_trgm` extension and GIN index are available:

- case-insensitive substring over `normalized_text_value` (`ILIKE '%value%'`)
- `"pro"` matches `"990 pro"`; does not change EQUALS semantics

When capability absent: query fails closed with `CatalogSearchFailure` (`INVALID_QUERY`).

No automatic `CREATE EXTENSION pg_trgm` during bootstrap.

## Legacy

`integrations/catalog_store/postgresql/catalog_search_adapter.py` → `PostgreSQLStructuredSearchAdapter`

**LEGACY / REFERENCE ONLY** (first constraint only, unindexed `ILIKE`). Not wired by
`build_structured_candidate_search`.

## Score semantics

`StructuredChannelScore` only — no global normalized score, percentage, or fusion.

# VPI Bootstrap Architecture

## Decision

VPI storage bootstrap is a **scenario-owned**, **provider-neutral** orchestration layer with reference adapters for PostgreSQL (catalog source truth) and Qdrant (derived lexical + dense search representation). Platform PostgreSQL session and Qdrant client boundaries are reused; no parallel provider frameworks.

## Operator flow

```text
dataset
→ validate configuration
→ validate dataset identity (FAST manifest or FULL checksum)
→ Gate 0 embedding probe (registry-resolved provider)
→ provider readiness (PostgreSQL, Qdrant)
→ bootstrap stores (idempotent DDL / collection prepare)
→ bounded ingest (streaming parquet → derive v2 → PG txn → embed → Qdrant upsert)
→ validate resulting state
→ READY
```

Entry point:

```bash
uv run python platform_proofs/scenarios/verified_product_identification/bootstrap.py --mode verify --max-records 1000
```

Environment loading uses `scripts/proof/intergrax_proof_environment.py` (process env > scenario `.env`).

## Architecture

```text
application/domain + derivation
        ↑
storage_bootstrap/contracts (ports, manifest, errors)
        ↑
storage_bootstrap/orchestration
        ↑
composition/bootstrap_runtime.py  ← only layer binding PostgreSQL + Qdrant
        ↑
integrations/catalog_store/postgresql
integrations/search_store/qdrant  → Intergrax vector index + VectorStore contracts
        ↑
Intergrax integration plugins (Qdrant provider owns vendor SDK)
```

VPI production code contains **zero** vendor SDK imports (`qdrant_client`, `psycopg`, embedding SDKs, etc.).
Provider selection (`qdrant`) is composition configuration; vendor implementation stays in `intergrax/integrations/providers/`.

## Contracts

- `CatalogBootstrapPort` — prepare, ingest_batch, validate, manifest I/O
- `SearchIndexBootstrapPort` — prepare, ingest_batch, validate, point count
- `EmbeddingExecutionPort` — Gate 0 probe + batch embedding for ingest (single provider instance per run)
- Typed `ValidationReport` / `BootstrapRunReport` (no bool-only validation)

## Reference adapters

**PostgreSQL** (`integrations/catalog_store/postgresql/`)

- Tables: `vpi_catalog_manifest`, `vpi_source_offer`, `vpi_identifier`, `vpi_structured_attribute`
- Idempotent upserts; one transaction per ingest batch
- Indexes: `(identifier_type, lookup_value)`, structured canonical lookup, `global_row_index`

**Qdrant** (`integrations/search_store/qdrant/`)

- Collection: dense (`dense`) + sparse (`sparse`) channels on same point
- Payload carries `SourceRecordRef` + derivation/embedding/dataset identity (no `record_json`)
- Deterministic logical point id: `vpi:{catalog_id}:{offer_id}:semantic:{derivation_version}`

## Manifest identity

Persisted in PostgreSQL `vpi_catalog_manifest` (single row). **Environment compatibility identity** requires matching:

- dataset checksum + record count
- derivation version, embedding provider/model/dimension
- catalog/search schema versions, bootstrap implementation version, catalog id

**Run target** (`target_max_records` / current invocation scope) is separate from compatibility identity. A prior verify run at 1000 rows does not make a later `--mode full` invocation READY at 1000; full continues ingest after the prior checkpoint when identity matches.

Mismatch → `VpiBootstrapCompatibilityError` (fail closed; no destructive rebuild).

## Idempotency & resume

- At-least-once batch ingest with deterministic keys / upserts
- Checkpoint: `checkpoint_rows_processed`, `checkpoint_batch_ordinal`
- Resume skips parquet rows before checkpoint; advances only after PG commit + Qdrant upsert + manifest write
- Manifest counters reflect **authoritative persisted totals** from adapters after each successful batch (retry-safe; no cumulative overcount on partial failure)
- verify → full: same identity + checkpoint below new requested target → continue ingest without rebuild
- verify → verify (same target): READY fast path with **current** Gate 0 probe (no duplicate ingest)
- requested target below existing checkpoint → fail closed (no silent scope shrink)

## READY semantics

All must pass:

- **Current-run** Gate 0 embedding probe (including READY fast path — no synthetic PASS)
- PostgreSQL schema + expected row counts for ingest scope
- Qdrant collection dimension + sparse channel compatibility + point count
- Manifest identity match + checkpoint complete for **current** requested target

Partial provider success → `FAILED`, never `READY`.

## Provider boundaries

- Orchestrator depends only on scenario ports (`CatalogBootstrapPort`, `SearchIndexBootstrapPort`, `EmbeddingExecutionPort`)
- Platform registry resolution lives in `composition/bootstrap_runtime.py` via `IntergraxEmbeddingBootstrapAdapter`
- Search bootstrap adapter uses public Intergrax contracts:
  - `VectorIndexAdministration` (control plane: probe, describe, prepare)
  - `VectorStore` (data plane: upsert via `add_records`)
- Qdrant `qdrant_client` imports exist only inside `intergrax/integrations/providers/vector_store/qdrant/`

## Failure model

`VpiBootstrapConfigurationError`, `VpiBootstrapCompatibilityError`, `VpiBootstrapProviderError`, `VpiBootstrapDataError` — chained from vendor errors at adapter boundary only.

## Provider swap

Implement `CatalogBootstrapPort` / `SearchIndexBootstrapPort` and wire in `composition/bootstrap_runtime.py` without orchestrator changes.

## Configuration

Scenario: `VPI_BOOTSTRAP_*`, `VPI_EMBEDDING_*`. Platform: `INTERGRAX_POSTGRESQL_*`, `INTERGRAX_QDRANT_*`.

## Bounded verification

`--mode verify` defaults to 1000 records unless `VPI_BOOTSTRAP_MAX_RECORDS` overrides. Full ingest (`--mode full`) uses unlimited when max not set — deferred to next qualification task.

## Next step

Full 3.77M ingest + reference BGE-M3 live qualification when environment supports Gate 0.

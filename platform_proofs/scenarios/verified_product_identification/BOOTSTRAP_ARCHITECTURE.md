# VPI Bootstrap Architecture

## Decision

VPI storage bootstrap is a **scenario-owned**, **provider-neutral** orchestration layer with reference adapters for PostgreSQL (catalog source truth) and Qdrant (derived lexical + dense search representation). Platform PostgreSQL session and Qdrant client boundaries are reused; no parallel provider frameworks.

**Canonical production flow (5C2):**

```text
WDC dataset
    ├── materialize_embeddings → READY embedding artifact
    └── catalog source truth
            ↓
storage bootstrap
   ├── PostgreSQL (from WDC derivation)
   └── Qdrant (from artifact vectors + lexical text)
```

Synchronous live embedding during storage bootstrap is **REMOVED**. Storage bootstrap requires a compatible READY artifact.

## Operator flow

```text
1. materialize_embeddings.py --max-records N
2. bootstrap.py --mode verify|full --max-records N
```

Storage bootstrap preflight:

```text
dataset identity
→ artifact manifest (state == READY)
→ artifact compatibility identity
→ artifact coverage >= requested target
→ artifact integrity (shard checksums)
→ provider readiness (PostgreSQL, Qdrant)
→ bootstrap stores (idempotent DDL / collection prepare)
→ bounded ingest (streaming WDC + aligned artifact rows → PG txn → Qdrant upsert)
→ validate resulting state
→ READY
```

Entry points:

```bash
uv run python platform_proofs/scenarios/verified_product_identification/materialize_embeddings.py --max-records 64
uv run python platform_proofs/scenarios/verified_product_identification/bootstrap.py --mode verify --max-records 64
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
composition/bootstrap_runtime.py  ← only layer binding PostgreSQL + Qdrant + Parquet artifact reader
        ↑
integrations/catalog_store/postgresql
integrations/search_store/platform_bootstrap_adapter  → Intergrax vector index + VectorStore contracts
embedding_materialization/stores/parquet (artifact reader only)
        ↑
Intergrax integration plugins (Qdrant provider owns vendor SDK)
```

```text
WDC
│
├── materialization → EmbeddingExecutionPort → READY artifact
│
└── catalog source truth
        ↓
storage bootstrap
   ├── CatalogBootstrapPort → PostgreSQL
   └── SearchIndexBootstrapPort → Qdrant (artifact dense + lexical)
```

VPI bootstrap orchestrator
        ↓
SearchIndexBootstrapPort
        ↓
PlatformSearchIndexBootstrapAdapter
        ↓
VectorIndexAdministration + VectorStore
        ↓
composition-selected Intergrax plugin
        ↓
Qdrant SDK

VPI production orchestration code contains **zero** vendor SDK imports (`qdrant_client`, `psycopg`, `pyarrow`, embedding SDKs, etc.)
and **zero** concrete vector-provider or embedding-provider implementation imports outside `composition/`.
Provider selection (`qdrant`) is composition configuration; vendor implementation stays in `intergrax/integrations/providers/`.

## Contracts

- `CatalogBootstrapPort` — prepare, ingest_batch, validate, manifest I/O
- `SearchIndexBootstrapPort` — prepare, ingest_batch, validate, point count
- `EmbeddingArtifactReaderPort` — read_manifest, iterate_shard_records, validate_identity (storage load input)
- `EmbeddingExecutionPort` — **materialization only** (Gate 0 probe + batch embedding)
- Typed `ValidationReport` / `BootstrapRunReport` (no bool-only validation)

## Reference adapters

**PostgreSQL** (`integrations/catalog_store/postgresql/`)

- Tables: `vpi_catalog_manifest`, `vpi_source_offer`, `vpi_identifier`, `vpi_structured_attribute`
- Idempotent upserts; one transaction per ingest batch
- Indexes: `(identifier_type, lookup_value)`, structured canonical lookup, `global_row_index`

**Search index** (`integrations/search_store/platform_bootstrap_adapter.py`)

- Provider-neutral adapter over `VectorIndexAdministration` + `VectorStore`
- Reference Qdrant wiring lives only in `composition/bootstrap_runtime.py`
- Payload carries `SourceRecordRef` + derivation/embedding/dataset identity (no `record_json`)
- Deterministic logical point id from artifact (`logical_point_id` field)

**Embedding artifact** (`embedding_materialization/stores/parquet/reader.py`)

- Reference `EmbeddingArtifactReaderPort` behind composition only
- Storage orchestrator never imports Parquet/PyArrow

## Manifest identity

Persisted in PostgreSQL `vpi_catalog_manifest` (single row). **Environment compatibility identity** requires matching:

- dataset checksum + record count
- derivation version, embedding provider/model/dimension (from validated artifact manifest)
- catalog/search schema versions, bootstrap implementation version, catalog id

**Run target** (`target_max_records` / current invocation scope) is separate from compatibility identity. A prior verify run at 1000 rows does not make a later `--mode full` invocation READY at 1000; full continues ingest after the prior checkpoint when identity matches.

Mismatch → `VpiBootstrapCompatibilityError` (fail closed; no destructive rebuild).

## Idempotency & resume

- At-least-once batch ingest with deterministic keys / upserts
- Checkpoint: `checkpoint_rows_processed`, `checkpoint_batch_ordinal`
- Resume skips WDC + artifact rows before checkpoint; advances only after PG commit + Qdrant upsert + manifest write
- Manifest counters reflect **authoritative persisted totals** from adapters after each successful batch (retry-safe; no cumulative overcount on partial failure)
- verify → full: same identity + checkpoint below new requested target → continue ingest without rebuild
- verify → verify (same target): READY fast path with **artifact identity validation** (no model load, no duplicate ingest)
- requested target below existing checkpoint → fail closed (no silent scope shrink)
- storage target must not exceed READY artifact `checkpoint_rows_materialized`

## READY semantics

All must pass:

- **Artifact input validation** (`embedding_artifact_ready`, `embedding_artifact_identity`) — including READY fast path
- PostgreSQL schema + expected row counts for ingest scope
- Search index dense dimension + sparse lexical channel compatibility + point count
- Manifest identity match + checkpoint complete for **current** requested target

Partial provider success → `FAILED`, never `READY`.

## Provider boundaries

- Orchestrator depends only on scenario ports (`CatalogBootstrapPort`, `SearchIndexBootstrapPort`, `EmbeddingArtifactReaderPort`)
- Reference composition wires `ParquetFilesystemArtifactReader` + PostgreSQL + Qdrant adapters
- **No** `IntergraxEmbeddingBootstrapAdapter` in storage composition
- Search bootstrap adapter uses public Intergrax contracts:
  - `VectorIndexAdministration` (control plane: probe, describe, prepare)
  - `VectorStore` (data plane: upsert via `add_records`)
- Qdrant `qdrant_client` imports exist only inside `intergrax/integrations/providers/vector_store/qdrant/`

## Failure model

`VpiBootstrapConfigurationError`, `VpiBootstrapCompatibilityError`, `VpiBootstrapProviderError`, `VpiBootstrapDataError` — chained from vendor errors at adapter boundary only.

## Provider swap

Implement `CatalogBootstrapPort` / `SearchIndexBootstrapPort` / `EmbeddingArtifactReaderPort` and wire in `composition/bootstrap_runtime.py` without orchestrator changes.

Future reference compositions may pair the same `PlatformSearchIndexBootstrapAdapter` with other platform plugins:

- Qdrant: `VectorIndexAdministration` + `VectorStore`
- Weaviate: `VectorIndexAdministration` + `VectorStore`
- PgVector: `VectorIndexAdministration` + `VectorStore`

## Configuration

Scenario: `VPI_BOOTSTRAP_*`, `VPI_EMBEDDING_*`, `VPI_EMBEDDING_ARTIFACT_PATH`. Platform: `INTERGRAX_POSTGRESQL_*`, `INTERGRAX_QDRANT_*`.

## Bounded verification

`--mode verify` defaults to 1000 records unless `VPI_BOOTSTRAP_MAX_RECORDS` overrides. Full ingest (`--mode full`) uses unlimited when max not set — deferred to next qualification task.

See [`EMBEDDING_MATERIALIZATION_ARCHITECTURE.md`](EMBEDDING_MATERIALIZATION_ARCHITECTURE.md) for the materialization stage.

## Next step

Full 3.77M ingest + reference BGE-M3 live qualification when environment supports bounded artifact reuse.

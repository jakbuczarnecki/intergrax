# VPI Embedding Materialization Architecture

## Decision

VPI embedding computation is separated from storage bootstrap via a **scenario-owned**, **restartable**, **provider-neutral** embedding artifact materialization plane. Expensive embedding work produces a versioned, sharded Parquet artifact that a later task (5C2) will load into PostgreSQL + Qdrant.

## Current vs target

**Current (qualification / MVP path):**

```text
dataset → derive → PostgreSQL → synchronous embed → Qdrant → checkpoint
```

**Target (this task creates artifact stage; 5C2 consumes it):**

```text
WDC dataset
    ↓
DerivedOfferSearchRepresentation
    ↓
EmbeddingMaterializationOrchestrator
    ↓
EmbeddingExecutionPort
    ↓
versioned/sharded embedding artifact (Parquet)
    ↓
[5C2] storage bootstrap → PostgreSQL + VectorStore
```

## Why the artifact exists

- Embedding computation is reusable across storage rebuilds
- Retries and restarts do not recompute committed vectors
- Storage providers can be rebuilt independently
- Embedding execution backend can change independently (HF CPU/GPU, vLLM, remote API)
- Full-scale processing (~3.77M × 1024 float32 ≈ 15.4 GB vectors) becomes operationally tractable via streaming shards

## Operator flow

```text
dataset identity
→ Gate 0 embedding probe
→ bounded derive + embed (streaming)
→ atomic shard commit
→ manifest checkpoint
→ validate
→ READY
```

Entry point:

```bash
uv run python platform_proofs/scenarios/verified_product_identification/materialize_embeddings.py --max-records 1000
```

Default `--max-records` is **1000** (bounded). Full 3.77M requires explicit intent.

## Architecture

```text
application/domain + derivation
        ↑
embedding_materialization/contracts (ports, manifest, errors)
        ↑
embedding_materialization/orchestration
        ↑
composition/materialization_runtime.py  ← binds Intergrax embedding + Parquet artifact store
        ↑
integrations/embedding/intergrax_adapter
embedding_materialization/stores/parquet
```

Materialization orchestrator depends only on:

- `EmbeddingExecutionPort`
- `EmbeddingArtifactWriterPort`

It does **not** depend on PostgreSQL, Qdrant, `SearchIndexBootstrapPort`, or `VectorStore`.

## Artifact identity

Immutable compatibility identity (fail closed on mismatch):

- `dataset_checksum`, `dataset_record_count`
- `search_representation_derivation_version`
- `embedding_configuration_version`
- `embedding_provider`, `embedding_model`, `embedding_dimension`
- `artifact_schema_version`, `catalog_id`

A short SHA256 fingerprint derived from identity is used in the default artifact directory name.

## Artifact format

- Schema version: `EMBEDDING_ARTIFACT_SCHEMA_VERSION = "v1"`
- Manifest: `manifest.json` with states `INITIALIZING | MATERIALIZING | VALIDATING | READY | FAILED`
- Shards: `part-000000.parquet`, contiguous global row ranges, no gaps/overlaps
- Vectors: `float32` fixed-size lists in PyArrow Parquet
- Each row stores: `global_row_index`, `logical_point_id`, source ref, `derivation_version`, `semantic_text`, `lexical_text`, embedding identity, `dense_embedding`

The artifact is a **materialized search representation + dense embedding** bundle, not a naked vector dump.

## Sharding and atomic commit

1. Write `part-NNNNNN.parquet.tmp`
2. Validate row alignment and checksum
3. Atomic rename to `part-NNNNNN.parquet`
4. Advance manifest checkpoint

**Crash reconciliation:** if rename succeeded but manifest checkpoint did not advance, restart validates the orphan shard against the expected next row range and adopts it without re-embedding.

## Checkpoint / resume

- `checkpoint_rows_materialized`, `checkpoint_shard_ordinal`, typed `committed_shards` descriptors (ordinal, row range, SHA256)
- Same-target restart: validate only, **zero** embedding calls
- Target extension: materialize delta rows only
- Requested target below checkpoint: fail closed

## Configuration

| Variable | Purpose |
|----------|---------|
| `VPI_EMBEDDING_ARTIFACT_PATH` | Root directory for fingerprinted artifacts |
| `VPI_EMBEDDING_ARTIFACT_SHARD_SIZE` | Records per shard (default 10_000) |
| `VPI_EMBEDDING_MATERIALIZATION_BATCH_SIZE` | Embedding batch size (default 64) |
| `VPI_EMBEDDING_MATERIALIZATION_MAX_RECORDS` | Default CLI bound (default 1000) |
| `VPI_EMBEDDING_PROVIDER/MODEL/DIMENSION` | Reused from existing embedding config |

## Provider neutrality

Materialization code contains **zero** vendor SDK imports. GPU/device selection belongs to the Intergrax embedding provider plugin configuration.

## Next step (5C2)

Refactor storage bootstrap to consume the READY artifact instead of live synchronous embedding. Remove duplicate long-term embedding path after 5C2.

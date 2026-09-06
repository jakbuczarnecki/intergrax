# VPI Data Pack v1 contract

Canonical generated root:

```text
platform_proofs/scenarios/verified_product_identification/dataset/generated/data_pack/
```

Proof-50 artifacts:

```text
platform_proofs/scenarios/verified_product_identification/dataset/generated/data_pack/proof-50/
```

## Format identity

| Concept | Value |
|---|---|
| Data pack format version | `vpi.data_pack/1.0.0` |
| Relational schema version | `vpi.relational/1.0.0` |
| Embedding schema version | `vpi.embedding/1.0.0` |
| Parquet file format | `parquet` |

Format version, schema versions, and semantic content identity are separate fields.

## Field classification

| Field | Class |
|---|---|
| `record_json` | SOURCE TRUTH — exact selected WDC offer JSON bytes as read from the canonical dataset row (UTF-8, no re-serialization) |
| `semantic_text` | DERIVED — search representation semantic text |
| `semantic_text_hash` | DERIVED integrity identity (`sha256` over UTF-8 semantic text) |
| `title`, `brand`, `category`, `description` | DERIVED convenience denormalization |
| `has_identifiers`, `has_spec_table`, `has_structured_attributes` | DERIVED indexing flags |
| `dense_embedding` | DERIVED — provider output for frozen model identity |
| `build_execution_provenance` | EXECUTION metadata only — excluded from content identity |

## Model identity

Canonical VPI embedding:

| Property | Value |
|---|---|
| provider | `hf` |
| model | `BAAI/bge-m3` |
| dimension | `1024` |
| revision | immutable Hugging Face commit SHA resolved at build time |

`EmbeddingPackIdentity` stores `model_revision` and optional `artifact_fingerprint` (config-file fingerprint from local HF snapshot). Canonical builds fail closed when revision cannot be resolved.

## Content identity vs artifact binary identity

- **Content identity** (`content_identity`): deterministic SHA-256 over semantic inputs (dataset checksum, derivation version, semantic text version, embedding provider/model/revision, dimension, schema versions). Excludes timestamps and binary shard checksums.
- **Artifact binary identity**: SHA-256 entries in `checksums/SHA256SUMS` identify the exact published bytes. GPU embedding may produce slightly different floats across hardware; published pack checksums identify the released artifact.

## Shard contract

- Naming: `part-000001.parquet`, `part-000002.parquet`, …
- Ordinal starts at 1, zero-padded width 6
- READY packs: no gaps in ordinals 1..N
- Each shard descriptor: `ordinal`, `relative_path`, `record_count`, `sha256`, `source_ref_count`, `schema_version`
- Shard index: typed `ShardIndex` in `indexes/shards.json`

### Shard pairing invariant

```text
set(relational_shard_N.source_refs) == set(embedding_shard_N.source_refs)
```

Enforced via matching `record_count` and `source_ref_count` per ordinal plus cross-artifact ref validation.

### Record ordering

Records within each shard are ordered by `global_row_index` ascending (deterministic selected dataset order).

## Vector serialization (Parquet)

| Property | Contract |
|---|---|
| dtype | `float32` |
| dimension | fixed per pack (`embedding_dimension`) |
| ordering | index order within semantic text batch |
| nullability | not nullable |
| values | all finite |

## Checksum contract

- Algorithm: SHA-256
- File: `checksums/SHA256SUMS`
- Paths: relative to data pack root (no absolute filesystem paths)

## READY semantics

| Status | Meaning |
|---|---|
| `DataPackManifest.status = READY` | Internal artifact integrity gates passed (manifest, typed shard index, checksums, cross-artifact identity) |
| `DataPackProofReport.status = READY` | External storage/retrieval proof passed (load, channels, mapping, negative query) |

Distributed final manifests should be `READY` only. Builder runtime state (`BUILDING`, `VALIDATING`, `FAILED`) is builder-local.

## Compatibility rejection

`validate_data_pack_compatibility(...)` rejects before storage ingest:

- unsupported data pack format version
- relational / embedding schema version mismatch
- derivation or semantic text version mismatch
- embedding provider, model, revision, or dimension mismatch
- dataset checksum mismatch
- content identity mismatch
- shard ordinal gaps, duplicates, or pairing mismatch
- checksum / shard SHA-256 mismatch

Pre-freeze proof artifacts with `model_revision = null` are development evidence only, not distributable v1.

## Provider neutrality

Data pack contracts contain no PostgreSQL schema names, Qdrant collection names, localhost URLs, credentials, or provider deployment IDs. Loader composition supplies those at runtime.

## License / distribution

Redistribution of prepared WDC-derived data packs and embedding artifacts requires separate license/distribution review. This repository does not grant redistribution permission.

## Rebuild proof-50

```powershell
$env:VPI_EMBEDDING_DEVICE="cuda"
.tmp/session/vpi-5c4a2/cuda-venv/Scripts/python.exe `
  platform_proofs/scenarios/verified_product_identification/dataset/run_proof_50.py
```

Generated parquet, vectors, manifests, and evidence under `generated/data_pack/**` are gitignored.

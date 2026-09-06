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
- Each shard descriptor: `ordinal`, `relative_path`, `record_count`, `sha256`, `source_ref_count`, `source_ref_set_sha256`, `schema_version`
- Shard index: typed `ShardIndex` in `indexes/shards.json`

### Shard pairing invariant

```text
set(relational_shard_N.source_refs) == set(embedding_shard_N.source_refs)
```

Proven per ordinal via matching:

- `record_count`
- `source_ref_count` (must equal `record_count` for READY shards)
- `source_ref_set_sha256` — deterministic SHA-256 over the shard's `SourceRecordRef` identities (encoding version `vpi.source-ref/1`)

#### `source_ref_set_sha256` v1 algorithm

1. For each `SourceRecordRef`, compute canonical binary identity bytes via length-prefixed UTF-8 encoding:
   - `catalog_id`: 4-byte big-endian UTF-8 byte length + UTF-8 bytes
   - `offer_id`: 4-byte big-endian UTF-8 byte length + UTF-8 bytes
   - `source_revision`: 1-byte presence flag (`0x00` = absent/`None`, `0x01` = present) and, when present, 4-byte big-endian UTF-8 byte length + UTF-8 bytes
2. Sort encoded identities lexicographically by byte representation (order-independent set semantics).
3. For each sorted encoded identity, update SHA-256 with 4-byte big-endian record length + record bytes.
4. Digest is duplicate-sensitive: repeated identities change the digest; READY shard validation requires unique refs per shard.

```text
relational_N.source_ref_set_sha256 == embedding_N.source_ref_set_sha256
```

Cross-artifact ref validation and file-level digest recomputation provide additional fail-closed checks when `pack_root` is present.

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

## Resumable builder (VPI-IMPLEMENTATION-5C4D3)

Production-intended multi-shard builder with crash-safe resume. Builder state is **not** stored in `DataPackManifest.status`.

### Layout

```text
generated/data_pack/<build>/
  manifest/manifest.json          # READY only after full finalization
  relational/part-000001.parquet
  embeddings/part-000001.parquet
  indexes/shards.json             # written at finalization
  checksums/SHA256SUMS            # written at finalization
  state/build-state.json          # builder-local authority (not distributable)
```

### Shard atomicity

- Resume unit = one shard ordinal
- Non-READY shards are rebuilt from scratch (no record-level resume in v1)
- READY shards are immutable; validated before skip on resume
- Corrupt READY shard → fail closed (`CORRUPT_READY_SHARD`), no silent rebuild
- Temp shards use `.parquet.tmp`; serialized temp artifacts are read back and validated before atomic rename to final `.parquet`
- Final filename means validated immutable shard, not artifact awaiting validation
- Crash between relational/embedding renames leaves shard non-READY; resume removes partial finals/temps and rebuilds
- Both finals may exist while state is still VALIDATING; resume does not auto-adopt them

### Dataset reader

- Row-group-aware `read_range(start, end)` reads only intersecting Parquet row groups (no O(N × shard_count) prefix rescan)
- Metadata index is resident; row-group payloads load on demand per requested range

### Shard lifecycle (builder-local)

```text
PENDING → DERIVING → EMBEDDING → WRITING → VALIDATING → READY
```

Incomplete outputs use `.parquet.tmp` suffix; never adopted as READY. Orphan `.tmp` or partial finals for non-READY shards are removed on resume.

### Content identity vs shard layout

`content_identity` (frozen v1) excludes `shard_size` and binary shard checksums. Different shard sizes produce different binary releases (distinct `SHA256SUMS`) but the same semantic `content_identity` when dataset, derivation, and model identity match.

### CLI

```powershell
uv run --group platform-proofs-vpi-dataset python `
  platform_proofs/scenarios/verified_product_identification/dataset/run_data_pack_build.py `
  --output-root platform_proofs/scenarios/verified_product_identification/dataset/generated/data_pack/canonical-v1 `
  --shard-size 25000 `
  --resume
```

| Flag | Purpose |
|---|---|
| `--shard-size` | Records per shard (default `25000`) |
| `--resume` | Required when `state/build-state.json` exists |
| `--start-fresh` | Clear scenario-owned build subtree only |
| `--max-shards` / `--max-records` | Qualification hooks; partial build is **not** distributable |
| `--stop-after-shard` | Graceful stop after N shards |

Partial builds do not emit READY manifest, `shards.json`, or `SHA256SUMS`.

### Corruption and recovery guarantees

| State class | Resume behavior |
|---|---|
| NON-READY (`PENDING`…`VALIDATING`) | Discard incomplete outputs for the current shard; rebuild from shard start |
| READY | Validate integrity metadata + on-disk artifacts; skip when valid |
| Corrupt READY | Fail closed (`VpiDataPackReadyShardCorruptionError`); **no** silent rebuild or repair |

Representative fail-closed conditions:

- missing or SHA-mismatched READY shard file
- `source_ref_set_sha256` mismatch between metadata and Parquet contents
- relational/embedding pair identity mismatch (two valid files with different ref sets)
- malformed or unsupported `state/build-state.json`
- `content_identity`, `shard_size`, or `expected_record_count` mismatch on resume

Qualification: `tests/unit/.../test_vpi_data_pack_resume_corruption.py`; real CUDA runner:
`platform_proofs/scenarios/verified_product_identification/dataset/qualification/run_cuda_resume_corruption_qualification.py`.

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

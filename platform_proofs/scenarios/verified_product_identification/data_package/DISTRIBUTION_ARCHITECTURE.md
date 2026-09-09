# VPI Portable Data Pack Distribution Architecture

**Task:** VPI-IMPLEMENTATION-5C4H-A  
**Publication:** blocked until 5C4H-B redistribution qualification

## Flow

```text
FINALIZED Data Pack + 5C4G PASS
        ↓
ProofDataPackageDescriptor (committed trust anchor)
        ↓
immutable per-file distribution objects (FILE_PER_SHARD)
        ↓
DataPackageTransportPort (HTTPS or local mirror)
        ↓
DataPackageCache (SHA256 content-addressed)
        ↓
DataPackageInstaller (resume + verify + atomic publish)
        ↓
local verified Data Pack
        ↓
future storage bootstrap (PostgreSQL/Qdrant/pgvector — consumer choice)
```

## Three identities

| Identity | Owner | Examples |
|----------|-------|----------|
| Data Pack semantic | VPI manifest | dataset checksum, embedding model, dimension, policy |
| Distribution package | `ProofDataPackageDescriptor` | `package_id`, `package_version`, file SHA256 list |
| Location | install request / env | HTTPS base URI, local mirror path |

Location never changes package identity. The same descriptor may be mirrored on S3, R2, GCS, Azure Blob, B2, or HTTP without version or checksum changes.

## Package granularity

**Decision:** distribute original immutable files (one descriptor entry per shard), not a monolithic archive.

Rationale: resumable per-shard downloads, corruption recovery without full re-download, cache reuse across package versions, mirror-friendly object layout, and reuse of `DataPackageInstaller` without extraction semantics.

## Package content policy

| Path | Included | Role |
|------|----------|------|
| `manifest/manifest.json` | yes | MANIFEST |
| `relational/*.parquet` | yes | RELATIONAL_SHARD |
| `embeddings/*.parquet` | yes | EMBEDDING_SHARD |
| `indexes/shards.json` | yes | SHARD_INDEX |
| `checksums/SHA256SUMS` | yes | CHECKSUMS |
| `evidence/proof-report.json` | yes | PROOF_REPORT |
| `state/build-state.json` | **no** | operational resumable-build provenance only |

## Descriptor generation preconditions

`build_data_pack_descriptor` refuses unless:

- validation report `verdict == PASS`
- `finalized_artifact_valid == true`
- manifest status `READY`
- required distributable files exist
- SHA256SUMS verifies on-disk bytes

Never generate from RUNNING, partial, interrupted, or unvalidated artifacts.

## Trust model

- Package SHA256: transport integrity (`ProofDataPackageDescriptor`)
- Data Pack manifest + 5C4G: semantic integrity at publish time
- Post-install validation: descriptor byte verification, `SHA256SUMS`, manifest `READY`, shard index consistency (no `build-state.json` required)
- Cryptographic signing: **not implemented** (`ARTIFACT_SIGNATURE_NOT_IMPLEMENTED`)

## Redistribution gate

Current status: `REDISTRIBUTION_REVIEW_REQUIRED`. Public HTTPS publication is blocked. Local mirror install is supported for development and offline tests.

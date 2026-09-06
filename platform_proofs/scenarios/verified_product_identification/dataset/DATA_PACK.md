# Proof-50 data pack

Canonical generated root:

```text
platform_proofs/scenarios/verified_product_identification/dataset/generated/data_pack/proof-50/
```

## What proof-50 validates

Proof-50 is the safety gate before the full 3.77M builder. It proves:

- deterministic 50-record WDC sample selection
- canonical relational + embedding parquet artifacts
- real BGE-M3 (`BAAI/bge-m3`, 1024D) embeddings via CUDA
- cross-artifact `source_record_ref` and `semantic_text_hash` equality
- checksum integrity
- provider-neutral load into PostgreSQL + Qdrant without re-embedding
- real retrieval across exact, lexical, structured, and vector channels
- source identity mapping from retrieval candidates back to relational truth

## Rebuild

```bash
$env:VPI_EMBEDDING_DEVICE="cuda"
.tmp/session/vpi-5c4a2/cuda-venv/Scripts/python.exe `
  platform_proofs/scenarios/verified_product_identification/dataset/run_proof_50.py
```

Isolated reference storage namespace:

- PostgreSQL schema: `vpi_proof_5c4d1`
- Qdrant collection: `vpi_offers_proof_5c4d1`

## Pass gate

All of the following must pass before VPI-IMPLEMENTATION-5C4D2 (contract freeze):

- 50 relational rows and 50 embedding rows
- exact `source_record_ref` set equality
- semantic text hash validation
- checksum validation
- relational + vector load with zero embedding provider calls during load (storage loader architecture; not a measured provider call counter unless instrumented)
- retrieval + mapping proof
- provider-neutral contracts (no PostgreSQL/Qdrant imports in `dataset/data_pack/contracts/`)

## READY semantics

`DataPackManifest.status = READY` means the generated data-pack artifact is internally complete and valid (manifest, shards, checksums, cross-artifact identity).

`DataPackProofReport.status = READY` means the artifact additionally passed the external storage/retrieval proof (load, retrieval channels, mapping, negative query gate).

These statuses are intentionally separate; manifest READY does not imply proof READY.

Generated parquet, vectors, manifests, and evidence under `generated/data_pack/**` are gitignored.

# VPI Dataset Reproducibility

**Scenario:** Verified Product Identification  
**Package identity:** `verified-product-identification` v1.0.0  
**Task:** VPI-IMPLEMENTATION-5C3

## 1. Purpose

This document records how the VPI canonical dataset and embedding artifact are produced, verified, packaged, and consumed. It separates the **normal Quick Start** (install prebuilt Data Pack) from the **reproducible-from-source** path (WDC → selection → embeddings → bootstrap).

## 2. Original dataset

- **Corpus:** Web Data Commons Large Scale Product Corpus V2  
- **File:** `offers_corpus_all_v2_non_norm` (NDJSON)  
- **Source URL:** https://webdatacommons.org/structureddata/pds/large-scale-product-corpora/v2/  
- **Source record count (VPI build):** 26,507,210

## 3. Raw acquisition

Operators download the WDC non-normalized offers file manually and place it at:

`dataset/raw/nonnormalized_offersV2` (gitignored)

The data package installer does **not** contact WDC. Runtime bootstrap does **not** contact WDC.

## 4. Original record structure

Each source line is a JSON object with heterogeneous fields (`id`, `cluster_id`, `identifiers`, `title`, `keyValuePairs`, `specTableContent`, etc.). The builder preserves each selected record losslessly as UTF-8 JSON in Parquet.

## 5. Selection rationale

Product identification requires structured evidence beyond title similarity. Records with `keyValuePairs` or `specTableContent` provide attribute and specification signal for identity verification.

## 6. Exact deterministic selection

**Rule:** `keyValuePairs != null OR specTableContent != null`

Implemented in `dataset/build_wdc_dataset.py` (`record_is_selected`). No randomness.

## 7. Cleaning / normalization

- Malformed JSON lines increment `malformed_record_count` and are skipped.  
- Valid JSON is normalized to stable key ordering for nested objects when written to Parquet.  
- No semantic rewriting of product attributes.

## 8. Selected corpus statistics

| Metric | Value |
|--------|-------|
| Source records | 26,507,210 |
| Selected records | 3,770,377 |
| Rejected | 22,736,833 |
| Malformed | 0 |
| Records with keyValuePairs | 2,492,991 |
| Records with specTableContent | 3,770,377 |
| Records with both | 2,492,991 |

Canonical manifest: `dataset/processed/selected_offers_manifest.json`

## 9. Derived Search Representation

Catalog rows are transformed into lexical + semantic search text via `application/catalog/derive_search_representation.py`.

**Derivation version:** `v2` (`SEARCH_REPRESENTATION_DERIVATION_VERSION`)

## 10. Identifier handling

Identifiers from WDC `identifiers` arrays are normalized for exact-match retrieval channels during catalog ingest (GTIN, MPN, SKU fragments).

## 11. Structured attributes

`keyValuePairs` and `specTableContent` feed structured attribute retrieval and evidence extraction during verification.

## 12. Lexical representation

Lexical text is derived from title, brand, category, identifiers, and selected attribute fields per derivation v2 rules.

## 13. Semantic representation

Dense vectors are computed from the semantic text channel using the configured embedding model (reference: BGE-M3).

## 14. Embedding configuration

| Field | Reference value |
|-------|-----------------|
| Provider | `hf` |
| Model | `BAAI/bge-m3` |
| Dimension | 1024 |
| Configuration version | `v1` |

Environment overrides: `VPI_EMBEDDING_*` (see `application/config/embedding_configuration.py`).

## 15. Why BGE-M3 reference model was selected

Documented in [EMBEDDING_PROVIDER_DECISION.md](EMBEDDING_PROVIDER_DECISION.md): multilingual dense retrieval quality at catalog scale with self-hosted execution path.

## 16. Embedding materialization

`materialize_embeddings.py` streams the canonical dataset, executes `EmbeddingExecutionPort`, and writes Parquet shards + `manifest.json` under the artifact root.

**Artifact schema version:** `v1`  
**State machine:** INITIALIZING → MATERIALIZING → VALIDATING → READY

## 17. Sharding

Shards are Parquet files (`part-NNNNNN.parquet`) with manifest-recorded ordinals, row ranges, record counts, and per-shard SHA256.

## 18. Checksums and identity

| Layer | Authority |
|-------|-----------|
| Package descriptor (`intergrax.proof_data_package.v1`) | Transport integrity for each distributed file |
| Dataset manifest | `output_sha256`, `selected_record_count`, builder version |
| Embedding artifact manifest | Dataset linkage + embedding identity + shard checksums |
| VPI semantic validator | Cross-manifest compatibility before bootstrap |

**Canonical dataset SHA256:** `fc1268a9c4b3e37325919cd127912a67db0a0b6d1943229a2026d9fedff1d998`

## 19. Restart / resume

- **Package install:** per-file `.part` partials; HTTP Range resume when supported (`intergrax.proof_data`).  
- **Embedding materialization:** checkpointed shard ordinals in artifact manifest.  
- **Storage bootstrap:** idempotent provider preparation; bounded ingest batches.

## 20. VPI Data Pack construction

**Build (local trusted):**

1. Produce `dataset/selected_offers.parquet` + `dataset/manifest.json`  
2. Materialize `embeddings/` artifact to READY  
3. Write `provenance.json`  
4. Run `data_package/build_descriptor.py` to emit immutable `package.json` with file SHA256 entries

**Install:** `setup_data.py` — no torch/BGE/Qdrant/PostgreSQL imports.

**Publish:** future HTTPS upload; blocked until [DATASET_DISTRIBUTION_REVIEW.md](DATASET_DISTRIBUTION_REVIEW.md) approval.

## 21. Storage bootstrap

`bootstrap.py` loads PostgreSQL catalog source truth from the dataset and Qdrant search representation from the READY embedding artifact. Live embedding during bootstrap is removed (5C2).

## 22. PostgreSQL role

Canonical catalog source truth: offer JSON, identifiers, derived search fields, ingest manifests.

## 23. Qdrant role

Derived search index: dense vectors + lexical payload for hybrid retrieval. Not canonical truth.

## 24. Validation

Install-time: package descriptor checksums → VPI semantic validator (dataset + embedding manifest compatibility).

Bootstrap-time: dataset identity, artifact READY state, shard checksums, coverage ≥ requested target.

## 25. Reproducing the build yourself

```bash
uv sync --group platform-proofs-vpi-dataset
# place WDC raw file, then:
uv run --group platform-proofs-vpi-dataset python dataset/build_wdc_dataset.py
uv run python materialize_embeddings.py --max-records <N>
uv run python data_package/build_descriptor.py  # when packaging
```

Expect ~30+ minutes for dataset build and substantially longer for full 3.77M embedding materialization.

## 26. Quick Start using prebuilt package

```bash
uv run python setup_data.py
uv run python bootstrap.py --mode verify --max-records 64
```

See [README.md](README.md) Quick Start section.

## 27. Licensing / redistribution status

**Status:** `REDISTRIBUTION_REVIEW_REQUIRED` — public package publication blocked. See [DATASET_DISTRIBUTION_REVIEW.md](DATASET_DISTRIBUTION_REVIEW.md). No legal conclusion is encoded in code.

## 28. Known limitations

- Hugging Face model revision/hash is not pinned in current embedding configuration; provenance records `embedding_model_revision: null`.  
- Full production Data Pack embedding shards are not built or published in 5C3.  
- Committed `data_package/v1/package.json` template lists dataset-side files; embedding shards are added at package build time.  
- `proof.json` v3 is unchanged; data packages use separate `intergrax.proof_data_package.v1` descriptors.  
- Package signing (publisher authenticity) is not implemented; SHA256 only.

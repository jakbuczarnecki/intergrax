# VPI Dataset Distribution Review

**Status:** UNRESOLVED — public publication blocked  
**Review date:** 2026-09-04  
**Task:** VPI-IMPLEMENTATION-5C3

## Source corpus

| Field | Value |
|-------|-------|
| Name | Web Data Commons Large Scale Product Corpus V2 (non-normalized offers) |
| Source identifier | `offers_corpus_all_v2_non_norm` |
| Source URL | https://webdatacommons.org/structureddata/pds/large-scale-product-corpora/v2/ |
| Raw records in VPI build | 26,507,210 |

## Derived VPI selected corpus

| Field | Value |
|-------|-------|
| Selection rule | `keyValuePairs != null OR specTableContent != null` |
| Selected records | 3,770,377 |
| Processed dataset SHA256 | `fc1268a9c4b3e37325919cd127912a67db0a0b6d1943229a2026d9fedff1d998` |
| Approximate Parquet size | 1.84 GiB (1,838,502,691 bytes per manifest) |

## Attribution / license evidence

| Item | Status |
|------|--------|
| Authoritative license URL for redistribution of derived selected corpus | **Not confirmed in this task** |
| Attribution requirements for public republication | **Not confirmed in this task** |
| Embedding model (BAAI/bge-m3) redistribution of computed vectors | **Not confirmed in this task** |

## Redistribution decision

| Asset | Status |
|-------|--------|
| Raw WDC download | Operator responsibility; not redistributed by Intergrax package |
| Derived selected Parquet | `REDISTRIBUTION_REVIEW_REQUIRED` |
| Precomputed embedding Parquet shards | `REDISTRIBUTION_REVIEW_REQUIRED` |
| Combined VPI Data Pack | `REDISTRIBUTION_REVIEW_REQUIRED` — **public HTTPS publication blocked** |

## Code publication gate

Committed package descriptor (`data_package/v1/package.json`) uses `redistribution_status: REDISTRIBUTION_REVIEW_REQUIRED`. No public distribution URL is configured by default (`VPI_DATA_PACKAGE_BASE_URL` unset).

## Decision

**BLOCKED** until product/legal review records authoritative evidence for derived-data and embedding redistribution. This task delivers local/test install mechanics only.

## Next review actions

1. Obtain authoritative WDC / DWS Group redistribution terms for derived subset publication.
2. Record required attribution text in this document.
3. Confirm Hugging Face model license implications for redistributing precomputed embeddings.
4. After approval, publish immutable package version to chosen HTTPS object storage and pin `VPI_DATA_PACKAGE_BASE_URL`.

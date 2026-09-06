# VPI dataset reproducibility lineage (frozen)

This document records the frozen semantic lineage for VPI Data Pack v1. It does not yet describe the resumable 3.77M builder (VPI-IMPLEMENTATION-5C4D3).

## Lineage chain

```text
WDC source corpus
  → canonical selected dataset (processed/selected_offers.parquet)
  → derivation version (search representation)
  → semantic text version
  → BAAI/bge-m3 immutable HF revision
  → VPI Data Pack v1 artifacts
```

## Frozen identities

| Stage | Identity |
|---|---|
| Source dataset | `processed/selected_offers_manifest.json` (`output_sha256`) |
| Derivation | `SEARCH_REPRESENTATION_DERIVATION_VERSION` |
| Semantic text | same as derivation version for v1 |
| Embedding provider | `hf` |
| Embedding model | `BAAI/bge-m3` |
| Embedding revision | Hugging Face commit SHA resolved at canonical build (local cache preferred, Hub fallback) |
| Embedding dimension | `1024` |
| Data pack format | `vpi.data_pack/1.0.0` |
| Relational schema | `vpi.relational/1.0.0` |
| Embedding schema | `vpi.embedding/1.0.0` |

## Pre-freeze artifacts

Proof artifacts generated before contract freeze (null `model_revision`) are retained only as development evidence. They are not compatible with frozen v1 canonical validation.

## License / distribution

Prepared WDC-derived data packs and embedding artifacts require separate license/distribution review before any external redistribution.

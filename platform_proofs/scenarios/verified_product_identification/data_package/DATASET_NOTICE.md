# VPI Data Pack — Distribution Notice Template

> **Template only.** Do not treat this file as a license for the VPI Data Pack. Public redistribution remains blocked until `redistribution-review.json` records `PUBLIC_REDISTRIBUTION_APPROVED` for the relevant asset.

## Source corpus attribution (recommended provenance)

| Field | Value |
|-------|-------|
| Source name | Web Data Commons Large Scale Product Corpus V2 |
| Source dataset | `offers_corpus_all_v2_non_norm` |
| Source URL | https://webdatacommons.org/largescaleproductcorpus/v2/index.html |
| Source terms reference | https://commoncrawl.org/terms-of-use/ |

## Transformation description

Intergrax Verified Product Identification (VPI) applies a deterministic subset rule (`keyValuePairs != null OR specTableContent != null`), canonical relational normalization, semantic text derivation, and BGE-M3 dense embedding generation. The resulting canonical portable Data Pack is a derived artifact; it is **not** a re-licensing of upstream Web Data Commons or Common Crawl content.

## Model attribution

| Field | Value |
|-------|-------|
| Model | `BAAI/bge-m3` |
| Model URL | https://huggingface.co/BAAI/bge-m3 |
| Model license | MIT (see https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE) |
| Model weights in package | **No** — only precomputed embedding vectors |

## Third-party rights disclaimer

Source product text, titles, descriptions, brands, and specifications originate from third-party websites extracted via Common Crawl and Web Data Commons pipelines. **Those third-party rights may remain restricted** even when data is publicly downloadable from upstream hosts. This notice does not grant republication permission.

## Intergrax provenance

| Field | Value |
|-------|-------|
| Package id | `verified-product-identification` |
| Qualification record | `data_package/v1/redistribution-review.json` |
| Review task | VPI-IMPLEMENTATION-5C4H-B |

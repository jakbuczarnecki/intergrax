# VPI Dataset Distribution Review

**Status:** `REDISTRIBUTION_REVIEW_REQUIRED` — public publication blocked  
**Review date:** 2026-09-09  
**Task:** VPI-IMPLEMENTATION-5C4H-B  
**Prior review:** VPI-IMPLEMENTATION-5C3 (historical unresolved notes retained below)

> **Not legal advice.** This document records evidence-backed qualification for engineering gates only. Code enforces the recorded decision; it cannot manufacture a legal right.

Machine-readable qualification: [`data_package/v1/redistribution-review.json`](data_package/v1/redistribution-review.json)

## 1. Scope

Qualification covers whether Intergrax may **publicly redistribute** the VPI canonical portable Data Pack and its components. Separate decisions are recorded for:

| Asset | Question |
|-------|----------|
| A | WDC-derived relational records |
| B | Precomputed BAAI/bge-m3 embedding vectors |
| C | Combined canonical VPI Data Pack |
| D | Package descriptor / checksums / metadata only |

Local install, deterministic rebuild, and internal development remain supported. **No public upload, release, or HTTPS distribution URL** is authorized by this review.

## 2. Exact artifact identity

| Field | Value |
|-------|-------|
| Canonical processed dataset SHA256 | `fc1268a9c4b3e37325919cd127912a67db0a0b6d1943229a2026d9fedff1d998` |
| Package id | `verified-product-identification` |
| Package version | `1.0.0` |
| Builder | `verified_product_identification_wdc_builder/1.0.0` |
| Embedding model | `BAAI/bge-m3` (revision `5617a9f61b028005a4858fdac845db406aefb181`) |
| Embedding dimension | 1024 |
| Model weights in package | **No** |

## 3. Source corpus

| Field | Value |
|-------|-------|
| Name | Web Data Commons Large Scale Product Corpus V2 (non-normalized offers) |
| Source identifier | `offers_corpus_all_v2_non_norm` |
| Source URL | https://webdatacommons.org/largescaleproductcorpus/v2/index.html |
| Alternate catalog URL | https://webdatacommons.org/structureddata/pds/large-scale-product-corpora/v2/ |
| Raw records in VPI build | 26,507,210 |
| Selection rule | `keyValuePairs != null OR specTableContent != null` |
| Selected records | 3,770,377 |

**Lineage:** WDC states Product Corpus V2 was derived from the November 2017 WDC schema.org extraction of Common Crawl structured data (see WDC LSPM V2 page).

## 4. WDC authoritative evidence

| source_id | Publisher | Finding |
|-----------|-----------|---------|
| `wdc-main-license` | WDC / DWS Group | Site-wide **License** section licenses only the **extraction framework** under Apache. It does **not** license extracted dataset contents for redistribution. |
| `wdc-lspm-v2-page` | WDC / DWS Group | Corpus V2 is offered for public download for research. No explicit third-party republication license for derived subsets. |
| `wdc-product-corpus-lineage` | WDC / DWS Group | Corpus derived from Common Crawl–based schema.org extraction. |

**Critical separation:** Apache license on WDC extraction **software** ≠ Apache license on Product Corpus **data**.

## 5. Common Crawl evidence

| source_id | Publisher | Finding |
|-----------|-----------|---------|
| `common-crawl-tou` | Common Crawl Foundation (last updated 2024-03-07) | Limited, non-transferable, non-sublicensable license to access/use the Service. Crawled content remains subject to **separate third-party terms**. Users must respect third-party copyrights. CC **recommends legal counsel before commercial use**. AI/ML use triggers explicit indemnification obligations. **No affirmative grant** of third-party republication rights over extracted product text. |

## 6. Third-party content analysis

**Corpus-level conclusion:** `THIRD_PARTY_CONTENT_RIGHTS_UNRESOLVED`

WDC/Common Crawl terms provide access while **original third-party rights in crawled websites remain intact**. No authoritative source grants Intergrax republication rights over extracted product descriptions from ~79k source websites.

### VPI relational field classification (schema/contracts)

| Classification | Fields |
|----------------|--------|
| `COPIED_SOURCE_CONTENT` | `title`, `brand`, `category`, `description`, `semantic_text`, `record_json` (contains source offer text, identifiers, spec content) |
| `DERIVED_METADATA` | `semantic_text_hash`, `derivation_version`, `global_row_index`, `has_identifiers`, `has_spec_table`, `has_structured_attributes` |
| `TECHNICAL_PROVENANCE` | `source_ref` (stable source keys; not expressive republication of full pages) |

Distributing relational Parquet **materially republishes copied source content**, not merely technical provenance.

## 7. BGE-M3 license evidence

| source_id | Publisher | Finding |
|-----------|-----------|---------|
| `bge-m3-hf-metadata` | BAAI via Hugging Face | Model `BAAI/bge-m3` at revision `5617a9f61b028005a4858fdac845db406aefb181` declares `license:mit`. |
| `bge-m3-flagembedding-license` | FlagOpen FlagEmbedding | Upstream `LICENSE` file is MIT. |

**VPI does not distribute model weights.** Only precomputed dense embedding vectors are stored in embedding Parquet shards.

## 8. Generated-output analysis

| Gate | Status | Rationale |
|------|--------|-----------|
| `MODEL_LICENSE_GATE` | **PASSED** | MIT confirmed for model software/weights; weights not redistributed by VPI. |
| `INPUT_DATA_RIGHTS_GATE` | **FAILED** | Embeddings are computed from WDC-derived semantic text whose republication rights are unresolved. |
| `OUTPUT_RESTRICTION_GATE` | **UNCLEAR** | No authoritative MIT or BAAI term found that explicitly restricts redistribution of generated dense vectors; insufficient alone to approve without input-rights gate. |

**Rule applied:** MIT model license does **not** imply embedding dataset is automatically safe for public redistribution.

## 9. Asset decision matrix

| Asset | Status | Blocking uncertainty | Recommended action |
|-------|--------|----------------------|--------------------|
| Raw WDC corpus | `INTERNAL_USE_ONLY` | N/A (excluded) | Operator obtains upstream; Intergrax does not redistribute raw corpus. |
| VPI relational Parquet | `REDISTRIBUTION_REVIEW_REQUIRED` | `THIRD_PARTY_CONTENT_RIGHTS_UNRESOLVED` | Block public publication pending WDC/counsel clarification. |
| VPI embedding Parquet | `REDISTRIBUTION_REVIEW_REQUIRED` | `INPUT_DATA_RIGHTS_GATE` failed | Block public publication; model MIT alone insufficient. |
| Combined VPI Data Pack | `REDISTRIBUTION_REVIEW_REQUIRED` | Combined gate fail-closed | No public HTTPS publication. |
| Package metadata | `REDISTRIBUTION_REVIEW_REQUIRED` | Conservative metadata gate | Do not treat descriptor-only publication as approved by default. |

## 10. Attribution requirements

| Kind | Status |
|------|--------|
| Legally required attribution text | **Not confirmed** — no authoritative obligation text identified for derived republication. |
| Recommended provenance (good practice) | Recorded in `redistribution-review.json` → `recommended_attribution` and [`data_package/DATASET_NOTICE.md`](data_package/DATASET_NOTICE.md) template. |

Recommended fields include WDC corpus name/URL, Common Crawl ToU reference, `BAAI/bge-m3` model attribution, and Intergrax transformation notice. These are **not labeled legally required** without Tier-1 evidence.

## 11. Remaining uncertainties

1. Whether WDC/DWS Group grants (or can grant) third-party public redistribution of derived Product Corpus V2 subsets.
2. Whether precomputed embeddings are treated as independent derivative works or as republishing restricted input content.
3. Whether metadata-only publication is permissible without a corpus redistribution grant.

## 12. Final publication verdict

| Check | Result |
|-------|--------|
| Public redistribution authorized | **NO** |
| Package `redistribution_status` | `REDISTRIBUTION_REVIEW_REQUIRED` |
| Public URL configured | **NO** (`VPI_DATA_PACKAGE_BASE_URL` unset) |
| Artifact uploaded | **NO** |
| Publication gate | **Fail-closed** — see `data_package/publication.py` + `redistribution-review.json` |

## 13. Review provenance

| Field | Value |
|-------|-------|
| Review id | `vpi-redistribution-review-2026-09-09` |
| Reviewer | VPI-IMPLEMENTATION-5C4H-B |
| Evidence tier policy | Tier 1–2 only for positive authorization; Tier 3 discovery only |
| Code contract | `data_package/redistribution_qualification.py` |

## Legal escalation package

**Contact parties (do not send automatically):**

- Web Data Commons maintainers / University of Mannheim Data and Web Science Group (WDC Google Group)
- Common Crawl Foundation (for ToU scope questions)
- Professional legal counsel

**Exact unresolved question:**

> Does Web Data Commons / the University of Mannheim Data and Web Science Group authorize third-party public redistribution of a derived subset of Large Scale Product Corpus V2 (`offers_corpus_all_v2_non_norm`), including relational fields (title, description, brand, specification text) and/or precomputed BAAI/bge-m3 embedding vectors generated from that subset?

**Proposed asset:** VPI canonical portable Data Pack (relational + embedding Parquet, manifest, checksums, proof evidence, package descriptor).

**Transformations performed:** Subset selection (`keyValuePairs != null OR specTableContent != null`); canonical relational normalization; semantic text derivation; deterministic BGE-M3 dense embedding generation; Data Pack sharding and validation.

## Fallback contingency (not implemented)

| Fallback | Viability |
|----------|-----------|
| `EMBEDDINGS_ONLY` distribution | **UNRESOLVED** — technically useful for some consumers but `INPUT_DATA_RIGHTS_GATE` remains failed. |
| Reproducible local build | **YES** — operator downloads source under upstream terms → Intergrax deterministic processing → local artifact. |

## Historical context (5C3)

Prior task VPI-IMPLEMENTATION-5C3 established local/test install mechanics and blocked public publication pending license review. This task replaces unresolved placeholders with frozen Tier-1/Tier-2 evidence records and typed enforcement. Default gate remains unchanged: **`REDISTRIBUTION_REVIEW_REQUIRED`**.

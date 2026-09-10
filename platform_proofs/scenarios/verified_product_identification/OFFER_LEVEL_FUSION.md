# VPI offer-level candidate fusion (5C7)

## Purpose

5C7 answers only: **which source offers should downstream identity reasoning inspect first?**

It does **not** verify products. A top fused rank is **not** a verified product.

## Input and output

- **Input:** `MultiChannelCandidateCollection` with `ProductCandidate` rows from successful retrieval channels (`EXACT`, `LEXICAL`, `STRUCTURED`, `VECTOR`).
- **Output:** one deterministic ranked list of offer-level `FusedOfferCandidate` rows with preserved per-channel evidence.

## Canonical strategy

Baseline strategy: **Reciprocal Rank Fusion (RRF)** via `ReciprocalRankFusionStrategy`.

Zero-based channel rank adjustment is explicit:

```text
contribution = 1 / (rrf_k + rank + 1)
```

Examples with `rrf_k = 60`:

| channel rank | contribution |
|---|---|
| 0 | `1/61` |
| 1 | `1/62` |

`rrf_k = 60` is the standard conservative RRF baseline — not a VPI-tuned business constant.

## Score semantics

- **Fusion score:** rank aggregation priority only. Not confidence, probability, or verification score.
- **Raw channel scores are never summed or normalized.** BM25, structured constraint ratios, cosine similarity, and exact identifier evidence remain typed evidence attached to each channel row.
- **No channel weights** in the baseline strategy.

## Deduplication identity

Offers merge across channels using canonical `SourceRecordRef` identity:

- `catalog_id`
- `offer_id`
- `source_revision` (including `None` vs explicit revision)

Same `offer_id` in different catalogs remain distinct offers.

## Tie-break (deterministic)

After `fusion_score DESC`:

1. `supporting_channel_count DESC`
2. best channel rank ASC
3. `catalog_id ASC`, then `offer_id ASC`, then normalized `source_revision ASC`

Physical input batch order does not affect ranking.

## Explainability

Each fused offer exposes `OfferChannelEvidence`:

- channel
- original channel rank
- original typed channel score
- reciprocal rank contribution

Evidence order within one offer: `EXACT`, `LEXICAL`, `STRUCTURED`, `VECTOR`.

## Boundaries

- No provider imports (PostgreSQL, Qdrant, pgvector, embedding runtimes).
- No LLM calls.
- No product identity clustering and no `cluster_id` usage.
- Duplicate same source offer within one channel is a contract violation (fail closed).
- Empty input returns an empty fused collection — not `NO_MATCH`.

## Composition

```python
from platform_proofs.scenarios.verified_product_identification.application.fusion import (
    OfferCandidateFusionRequest,
    build_offer_candidate_fusion,
)

service = build_offer_candidate_fusion()
result = service.fuse(OfferCandidateFusionRequest(candidates=collection, limit=20))
```

Alternate strategies can be injected through `OfferCandidateFusionService(strategy=...)`.

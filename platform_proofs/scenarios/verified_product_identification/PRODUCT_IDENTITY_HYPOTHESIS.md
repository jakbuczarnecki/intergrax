# VPI product identity hypothesis (5C8)

## Purpose

5C8 answers: **which fused offers may represent the same real-world product, and what evidence supports or contradicts that hypothesis?**

It does **not** verify products. Output is:

- hypothesis + evidence + contradictions

Not:

- `VERIFIED` / `AMBIGUOUS` / `NO_MATCH`

## Core invariant

```text
IDENTITY HYPOTHESIS != VERIFIED PRODUCT
```

An offer candidate is one source offer. An identity hypothesis is one possible real product represented by one or more source offers. These are different abstraction levels.

## Input and output

- **Input:** `FusedOfferCandidateCollection` from 5C7 (bounded, default `max_candidates=20`).
- **Output:** `ProductIdentityHypothesisCollection` with typed `IdentityEvidence` and `IdentityContradiction` rows.

## Evidence model

Evidence is typed and provenance-bearing:

- `EXACT_IDENTIFIER_MATCH` — GTIN / SKU / product_id intersection (strong)
- `MODEL_NUMBER_MATCH` — normalized MPN intersection (strong)
- `BRAND_MATCH` — normalized brand equality when both sides present (strong)
- `STRUCTURED_ATTRIBUTE_MATCH` — same canonical key + normalized value (strong)
- `TITLE_TOKEN_SUPPORT` — lexical channel recall context only (weak)
- `SEMANTIC_SUPPORT` — vector channel recall context only (weak)

Weak evidence is recorded but **never** forces grouping.

## Contradiction model

Contradictions are discrete, first-class, and high priority for downstream verification:

- `IDENTIFIER_CONFLICT`
- `MODEL_NUMBER_CONFLICT`
- `BRAND_CONFLICT`
- `STRUCTURED_ATTRIBUTE_CONFLICT`

Missing attribute values are **unknown**, not conflicting.

## Grouping safety

Canonical strategy: `DeterministicEvidenceIdentityHypothesisStrategy`.

- Pairwise evidence over the bounded fused set only (`O(N²)`).
- Complete-link compatibility: every member must be eligible with **all** current members.
- No single-link transitive clustering.
- Example blocked case: `A~B`, `B~C`, `A!~C` does **not** merge all three.

Eligible strong support (discrete rules):

1. matching strong identifier, or
2. matching MPN + compatible brand (match or missing brand), or
3. two or more matching structured identity attributes.

## Hypothesis identity

`hypothesis_id` is a deterministic SHA-256 digest over sorted `SourceRecordRef` members (`application/domain/source_identity.py`). No random UUIDs, no `hash()`, no `cluster_id`.

## Ordering

Hypothesis order is deterministic and **not** identity confidence:

1. best (minimum) fused rank among members ASC
2. member count DESC
3. `hypothesis_id` ASC

## Explicit non-goals

- no `cluster_id` in identity logic
- no LLM calls
- no new embedding calls
- no `identity_confidence: float`
- no verification verdict labels
- no dataset / storage_bootstrap / provider imports in the identity application package

## Service boundary

```python
service = build_product_identity_hypothesis_service(source_port=source_port)
result = service.form_hypotheses(
    ProductIdentityHypothesisRequest(fused_candidates=fused_collection)
)
```

`SourceRecordFetchPort` loads each unique `SourceRecordRef` at most once per operation. Missing source records fail explicitly via `IdentityEvidenceUnavailableError`.

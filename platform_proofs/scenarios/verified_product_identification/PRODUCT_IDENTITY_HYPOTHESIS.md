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

## Identifier scope (retrieval vs identity)

Exact retrieval and cross-offer product identity use different comparability rules:

| Identifier | Exact retrieval (5C6B) | Cross-offer identity scope |
|---|---|---|
| GTIN | supported | **global** — strong support or contradiction when both sides explicit |
| MPN | supported | **manufacturer-scoped** — requires compatible brand/manufacturer context |
| SKU | supported | **source-local** — never compared across catalogs; same-catalog only |
| PRODUCT_ID | supported | **source-local / conservative** — excluded from cross-catalog identity unless an explicit namespace contract exists (none in VPI today) |

**Invariant:** an identifier used for exact lookup is not automatically globally comparable for product identity.

Missing identifier values on one or both sides are **unknown**, not conflicting — especially for source-local families where absence of intersection must not imply contradiction across catalogs.

## Evidence model

Evidence is typed and provenance-bearing:

- `EXACT_IDENTIFIER_MATCH` — global GTIN intersection, or same-catalog SKU intersection (strong)
- `MODEL_NUMBER_MATCH` — normalized MPN intersection with manufacturer context (strong; grouping still requires compatible brand)
- `BRAND_MATCH` — normalized brand equality when both sides present (strong)
- `STRUCTURED_ATTRIBUTE_MATCH` — same canonical key + normalized value (strong)
- `TITLE_TOKEN_SUPPORT` — lexical channel recall context only (weak)
- `SEMANTIC_SUPPORT` — vector channel recall context only (weak)

Weak evidence is recorded but **never** forces grouping.

## Contradiction model

Contradictions are discrete, first-class, and provenance-bearing:

- `IDENTIFIER_CONFLICT`
- `MODEL_NUMBER_CONFLICT`
- `BRAND_CONFLICT`
- `STRUCTURED_ATTRIBUTE_CONFLICT`

Missing attribute values are **unknown**, not conflicting.

### Contradiction precedence (authority vs traceability)

**A contradiction existing does not mean it is authoritative enough to veto grouping.**

Grouping eligibility uses **blocking** contradictions only. Nonblocking contradictions remain in the hypothesis for traceability.

| Contradiction | Grouping authority |
|---|---|
| GTIN `IDENTIFIER_CONFLICT` | **blocking** |
| MPN `MODEL_NUMBER_CONFLICT` (compatible manufacturer context) | **blocking** |
| `BRAND_CONFLICT` | **blocking** |
| `STRUCTURED_ATTRIBUTE_CONFLICT` | **blocking** (unchanged 5C8 semantics) |
| same-catalog SKU `IDENTIFIER_CONFLICT` | **nonblocking** — source-local inconsistency |
| `PRODUCT_ID` conflict | **nonblocking** / not emitted cross-catalog |

Derived policy: identifier-family scope from `identity_scope_for_identifier_type()` — `SOURCE_LOCAL` contradictions are recorded but do not outrank global or manufacturer-scoped identity evidence.

`OfferPairIdentityAssessment.has_contradiction` remains truthful for any contradiction. Use `has_blocking_contradiction` (or `is_blocking_identity_contradiction(...)`) for grouping policy.

## Grouping safety

Canonical strategy: `DeterministicEvidenceIdentityHypothesisStrategy`.

- Pairwise evidence over the bounded fused set only (`O(N²)`).
- Complete-link compatibility: every member must be eligible with **all** current members.
- No single-link transitive clustering.
- Example blocked case: `A~B`, `B~C`, `A!~C` does **not** merge all three.

Eligible strong support (discrete rules):

1. matching global GTIN, or
2. matching MPN + compatible brand (match or missing brand), or
3. two or more matching structured identity attributes.

Source-local identifier mismatch (SKU / PRODUCT_ID across catalogs) does **not** block valid GTIN or manufacturer-scoped MPN grouping. Same-catalog SKU mismatch is recorded as a nonblocking contradiction and must not veto stronger global or manufacturer-scoped identity evidence.

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

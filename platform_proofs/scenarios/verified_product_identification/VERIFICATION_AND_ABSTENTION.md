# Verification and abstention (5C10)

5C10 is the **first** application layer that may emit terminal product-identification outcomes:

- `VERIFIED`
- `AMBIGUOUS`
- `INSUFFICIENT_INFORMATION`
- `NO_MATCH`

## Top-ranked ≠ verified

5C9 ranking (`reranked_position`, fusion rank, BM25, vector cosine, channel counts) selects **evaluation order only**. It is **not** verification evidence and must never imply `VERIFIED`.

## Verification vs ranking

| Layer | Question |
|-------|----------|
| 5C9 identity evaluation | Which hypotheses deserve attention first; evidence/contradiction profile |
| 5C10 verification | Whether one identity is uniquely defensible under request constraints |

## Constraint semantics

Each required hard constraint is classified **exactly one** of:

- `SUPPORTED` — catalog evidence matches the requested value
- `CONTRADICTED` — catalog evidence conflicts with the request
- `MISSING` — neither support nor contradiction is available (not support)

Negative constraints exclude values (e.g. user says not SATA; catalog shows SATA → contradiction).

Soft preferences may influence upstream ranking and diagnostics but **never** prove identity.

## `VERIFIED`

Exactly one hypothesis is materially viable:

- no internal **blocking** contradiction
- all required constraints `SUPPORTED`
- no violated negative constraints
- identity evidence materially sufficient (GTIN/MPN authority from 5C8/5C9 — not lexical/vector/brand-only)
- no competing viable identity remains

## `AMBIGUOUS`

At least two hypotheses are materially viable and evidence cannot distinguish them. **Not** score-margin ambiguity.

## `INSUFFICIENT_INFORMATION`

Required distinguishing knowledge is missing (user request and/or catalog fields). Distinct from rejection.

## `NO_MATCH`

Positive rejection: every evaluated hypothesis is materially contradicted, or typed rejection evidence is supplied for an empty ranked set. Empty retrieval alone is **not** `NO_MATCH`.

## Failures vs business outcomes

Infrastructure/catalog provider failures propagate via `CatalogSearchFailure` on `ProductIdentificationVerificationOutcome` — they do **not** map to the four business outcomes.

## No confidence / no LLM

Deterministic rule policy only: no confidence scores, thresholds, weighted sums, LLM, embeddings, or source refetch in 5C10.

## Provenance

Support and contradiction rows reference existing `IdentityEvidence` / `IdentityContradiction` provenance from 5C8/5C9.

## 5C11 handoff

`missing_requirements` exposes typed clarification needs; 5C10 does **not** generate natural-language questions.

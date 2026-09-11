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

## Direct query-to-source evidence

5C8 projects **`SourceIdentityFact`** rows onto each `ProductIdentityHypothesis` (`source_identity_facts`): one immutable fact per source offer (identifiers, structured attributes, brand) with `SourceRecordRef`, normalized value, and provenance.

5C10 compares the user request to those facts:

- `ProductIdentificationQueryContext.requested_identifiers` are operational (GTIN/MPN support, contradiction, or missing).
- Required and negative constraints prefer direct per-offer facts; pairwise `IdentityEvidence` remains a fallback when facts were not projected.

**Cross-offer pair evidence** (GTIN match between two offers) is **corroboration**, not a prerequisite for singleton verification. A single-offer hypothesis with matching requested GTIN and direct attribute facts can reach `VERIFIED` without internal pairs.

## Constraint semantics

Each required hard constraint is classified **exactly one** of:

- `SUPPORTED` — catalog evidence matches the requested value
- `CONTRADICTED` — catalog evidence conflicts with the request
- `MISSING` — neither support nor contradiction is available (not support)

Negative constraints exclude values (e.g. user says not SATA; catalog shows SATA → contradiction).

Soft preferences may influence upstream ranking and diagnostics but **never** prove identity.

## Unique defensible identity (`VERIFIED`)

Exactly one hypothesis is in state **`SUPPORTED`**:

- no internal **blocking** contradiction
- all required constraints `SUPPORTED`
- no violated negative constraints
- identity evidence materially sufficient (direct requested identifiers and/or authoritative pair corroboration — not lexical/vector/brand-only)
- **every other hypothesis is materially eliminated (`CONTRADICTED`) or not an unresolved competing identity**

An **`INCOMPLETE`** competitor that already shows partial identity support or satisfied required fields (but lacks a material discriminator) prevents `VERIFIED` → `INSUFFICIENT_INFORMATION` with `UNRESOLVED_COMPETING_IDENTITY`. Weak non-competing incompletes do not block a uniquely supported alternative.

## `AMBIGUOUS`

At least two hypotheses are **`SUPPORTED`**. **Not** score-margin ambiguity. Unresolved multi-way identity uses typed `competing_identity` missing requirements — not a fabricated `variant` attribute.

## `INSUFFICIENT_INFORMATION`

Required distinguishing knowledge is missing (user request and/or catalog fields), or an unresolved competing identity remains. Distinct from rejection.

## `NO_MATCH`

Positive rejection: every evaluated hypothesis is **`CONTRADICTED`**, or typed rejection evidence is supplied for an empty ranked set. A mix of `CONTRADICTED` and `INCOMPLETE` is **not** `NO_MATCH`. Empty retrieval alone is **not** `NO_MATCH`.

## Failures vs business outcomes

Infrastructure/catalog provider failures propagate via `CatalogSearchFailure` on `ProductIdentificationVerificationOutcome` — they do **not** map to the four business outcomes.

## No confidence / no LLM

Deterministic rule policy only: no confidence scores, thresholds, weighted sums, LLM, embeddings, or source refetch in 5C10.

## Provenance

Support and contradiction decisions retain catalog provenance via `SourceIdentityFact` and, when used, existing `IdentityEvidence` / `IdentityContradiction` rows from 5C8/5C9.

## 5C11 handoff

`missing_requirements` exposes typed clarification needs; 5C10 does **not** generate natural-language questions.

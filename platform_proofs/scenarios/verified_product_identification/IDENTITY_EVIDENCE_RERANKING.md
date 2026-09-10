# VPI identity evidence reranking (5C9)

## Purpose

5C9 answers: **which identity hypotheses deserve downstream verification first, and what evidence/contradiction state does each hypothesis have?**

5C9 does **not** answer which product is verified. Top-ranked hypothesis ≠ verified product. Reranking ≠ verification.

## Input / output

- **Input:** `ProductIdentityHypothesisCollection` from 5C8.
- **Output:** `RankedIdentityHypothesisCollection` of `EvaluatedIdentityHypothesis` rows with contiguous `reranked_position` values starting at zero.

## Contradiction scope

Each `IdentityContradiction` is classified at the evaluation boundary:

| Scope | Rule |
|---|---|
| **INTERNAL** | Both `source_refs` are hypothesis members |
| **EXTERNAL** | Exactly one `source_ref` is a member |
| **INVALID** | Neither ref is a member — fail closed |

Blocking vs nonblocking authority is reused from 5C8 via `is_blocking_identity_contradiction(...)`.

- **Internal blocking** outranks all positive identity evidence.
- **Internal nonblocking** (e.g. same-catalog SKU mismatch) is preserved and counted, but does not veto stronger global/manufacturer identity evidence.
- **External** contradictions are separation evidence only — they do not veto the hypothesis and are not added to internal blocking counts.

## Evidence profile

Internal evidence only contributes to ranking. Profiles use normalized pair coverage and distinct structured keys — never raw `len(evidence)` or member-count bias.

| Tier | Evidence |
|---|---|
| 1 | Global GTIN `EXACT_IDENTIFIER_MATCH` (STRONG) |
| 2 | Manufacturer-scoped `MODEL_NUMBER_MATCH` / MPN (STRONG) |
| 3 | `STRUCTURED_ATTRIBUTE_MATCH` breadth (distinct keys) |
| Context | `TITLE_TOKEN_SUPPORT`, `SEMANTIC_SUPPORT` (weak; not viability evidence) |

Brand match alone is supporting context, not product-specific identity.

## Lexicographic ranking policy

No weighted sum. No confidence / probability / scalar identity score.

Frozen ordering key (`DeterministicEvidenceIdentityRankingStrategy`):

1. Internal blocking contradiction present: **NO** before **YES**
2. Global GTIN internal pair coverage: higher supported, then higher possible
3. Manufacturer MPN internal pair coverage: higher supported, then higher possible
4. Structured identity breadth: more distinct internal structured keys first
5. Internal nonblocking contradiction burden: fewer first
6. Best member `fused_rank` among hypothesis members: lower first
7. Member count: lower first (late deterministic tie-break only)
8. `hypothesis_id`: ascending final tie-break

`fusion_score` is not used. `fused_rank` is late context only.

## Singleton semantics

A one-member hypothesis has `possible_internal_pairs = 0`. Lack of cross-offer support is not negative evidence.

## Service boundary

```python
service = build_identity_hypothesis_evaluation_service()
result = service.evaluate(
    IdentityHypothesisEvaluationRequest(hypotheses=collection)
)
```

`IdentityHypothesisRankingStrategy` is injectable via constructor — no registry or dynamic discovery.

## Constraints

- Pure in-memory over 5C8 output — no source fetch, LLM, embeddings, or providers.
- Immutable public contracts (`@dataclass(frozen=True, slots=True)`), tuple collections.
- No terminal verdict enums (`VERIFIED`, `AMBIGUOUS`, `NO_MATCH`, …).

## Handoff

5C10 consumes ranked evaluated hypotheses for terminal verification and abstention decisions.

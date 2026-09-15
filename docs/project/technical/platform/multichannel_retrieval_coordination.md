# Generic multi-channel retrieval coordination

Platform capability: `intergrax.rag.retrieval.multichannel`.

**Canonical decision:** [ADR-RAG-001](../adr/entries/2026-09-13/ADR-RAG-001.md) (Accepted).

## Purpose

Coordinate independent retrieval **channel operations** in a deterministic order. Each channel returns a typed `RetrievalChannelOutcome` (succeeded, skipped, or failed). The coordinator aggregates outcomes without interpreting domain results.

## Ownership

| Layer | Owns |
| --- | --- |
| **Platform** | Channel key value object, status enum, failure DTO (`failure_code` canonical trimmed identity; `message` human-readable), outcome invariants, execution plan validation (duplicate keys), coordinator protocol, default sequential implementation |
| **Scenario** | Channel semantics, enable/skip policy, query construction, domain `TResult`, provider adapters, fusion, fatal-vs-tolerant failure policy |

## Failure semantics

A channel `FAILED` outcome does **not** stop later channels. The scenario decides whether any failure is terminal.

Each operation's declared `channel_key` is authoritative for the plan. `execute()` must return an outcome whose `channel_key` matches; mismatch raises `MultiChannelRetrievalContractError` (contract violation, not a channel `FAILED` outcome).

## Extension

Depend on `MultiChannelRetrievalCoordinator[TResult]` and inject an implementation (default: `SequentialMultiChannelRetrievalCoordinator`). No plugin registry is required for v1.

## Rank fusion (P1C / P1C-D1)

**Canonical decision:** [ADR-RAG-002](../adr/entries/2026-09-15/ADR-RAG-002.md) (Accepted design). Supplements [ADR-RAG-001](../adr/entries/2026-09-13/ADR-RAG-001.md).

Platform module: `intergrax.rag.retrieval.fusion`.

```text
retrievers / channel ops
  → RankedRetrievalChannel (per channel)
  → RankFusionStrategyPort (selected in composition / bootstrap — not inside consumer defaults)
  → FusedRankedCandidate (+ channel evidence)
  → (optional) PostFusionScoringStrategyPort — e.g. LexicalHybrid legacy alpha blend
  → consumer / scenario policy
```

P1C initial implementation (`3bf79bb…`) is **not** audit-complete; P1C-R1 applies ADR-RAG-002.

Key semantics (summary — full matrices in ADR):

- **Rank authority:** list index is authoritative; `candidate.rank` must match index (Model C).
- **Duplicates in channel / duplicate channel_key / payload mismatch:** contract violations (fail-fast).
- **RRF math:** public `reciprocal_rank_contribution`; strategy `reciprocal-rank-fusion.v1`.
- **VPI offer fusion:** domain plugin on `OfferCandidateFusionPort`; shares math primitive only.
- **Fusion ≠ reranking:** `RRFReranker` out of scope for P1C-D1.

## Non-goals (v1)
- Retries, concurrency, or provider execution inside the coordinator
- Product/catalog DTOs or VPI channel enums
- Diagnostic spine integration (**P1B**, not part of P1A)

## Integration status

| Item | State |
| --- | --- |
| P1A platform contract + sequential coordinator | **IMPLEMENTED** |
| P1A-R1 identity / `failure_code` hardening | **IMPLEMENTED** |
| VPI adoption (`MultiChannelRetrievalService` → coordinator) | **PENDING** |
| Execution Engine wiring | **PENDING** |
| Governance integration | **PENDING** (evaluate at E2E boundary) |
| Diagnostic projection | **PLATFORM CONTRACT IMPLEMENTED** (P1B); VPI adapter implemented; runtime bind pending |

VPI remains the **driver** of this platform evolution; the business requirement is unchanged.

## Architecture placement

```text
Execution Engine / runtime composition  →  scenario execution  →  platform multichannel coordinator  →  scenario policy/plugins  →  provider adapters
```

P1A delivers the coordinator layer only; upstream/downstream wiring is not complete.

## Empty execution plan

An empty `operations` tuple is valid and yields an empty `MultiChannelRetrievalResult`.

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

## Rank fusion (P1C)

Canonical platform module: `intergrax.rag.retrieval.fusion`.

```text
Ranked candidate lists per channel
  → RankFusionStrategyPort
  → default ReciprocalRankFusionStrategy (RRF)
  → fused ranked candidates (+ optional provenance)
```

RRF formula (0-based channel rank): `score(d) += 1 / (rrf_k + rank + 1)` per channel.
Typed config: `RankFusionConfiguration` (`rrf_k`, default `60`). Scenario offer-grain fusion
remains a VPI plugin; it reuses `reciprocal_rank_contribution` from this module.

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

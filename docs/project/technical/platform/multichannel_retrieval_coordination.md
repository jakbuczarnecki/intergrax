# Generic multi-channel retrieval coordination

Platform capability: `intergrax.rag.retrieval.multichannel`.

## Purpose

Coordinate independent retrieval **channel operations** in a deterministic order. Each channel returns a typed `RetrievalChannelOutcome` (succeeded, skipped, or failed). The coordinator aggregates outcomes without interpreting domain results.

## Ownership

| Layer | Owns |
| --- | --- |
| **Platform** | Channel key value object, status enum, failure DTO, outcome invariants, execution plan validation (duplicate keys), coordinator protocol, default sequential implementation |
| **Scenario** | Channel semantics, enable/skip policy, query construction, domain `TResult`, provider adapters, fusion, fatal-vs-tolerant failure policy |

## Failure semantics

A channel `FAILED` outcome does **not** stop later channels. The scenario decides whether any failure is terminal.

## Extension

Depend on `MultiChannelRetrievalCoordinator[TResult]` and inject an implementation (default: `SequentialMultiChannelRetrievalCoordinator`). No plugin registry is required for v1.

## Non-goals (v1)

- Rank fusion (RRF) — separate capability
- Retries, concurrency, or provider execution inside the coordinator
- Product/catalog DTOs or VPI channel enums
- Diagnostic spine integration (planned follow-up)

## Empty execution plan

An empty `operations` tuple is valid and yields an empty `MultiChannelRetrievalResult`.

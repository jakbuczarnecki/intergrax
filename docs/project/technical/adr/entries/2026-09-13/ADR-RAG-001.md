# ADR-RAG-001: Generic Multi-Channel Retrieval Coordination

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-09-13 |
| **Deciders** | Platform / VPI architecture review |
| **Related** | VPI `VPI_PLATFORM_CAPABILITY_DECISION_REVIEW.md` · `intergrax.rag.retrieval.multichannel` |

## Context

Verified Product Identification (VPI) coordinates several independent catalog retrieval channels (exact identifier, lexical, structured, vector) per identification request. The **orchestration envelope** (ordered execution, per-channel status, typed failure, aggregation) is reusable outside product identification. Catalog query shapes, channel policy, and candidate fusion remain scenario concerns.

Existing platform pieces serve different roles:

- `RetrievalService` + `RetrieverRegistry` — single-path document RAG retrieval
- `hybrid_retrieval_orchestrator` / `execute_hybrid_retrieval` — fixed vector/keyword/graph fusion for memory-depth RAG

None provide generic N-channel execution with scenario-supplied operations and domain-neutral outcomes.

## Decision

Introduce `intergrax.rag.retrieval.multichannel` with:

- Immutable contracts: `RetrievalChannelKey`, `RetrievalChannelStatus`, `RetrievalChannelFailure`, `RetrievalChannelOutcome[T]`, `MultiChannelRetrievalResult[T]`
- `RetrievalChannelOperation[T]` and `MultiChannelRetrievalCoordinator[T]` protocols
- `SequentialMultiChannelRetrievalCoordinator` as the default deterministic implementation

**Scenario owns:** channel semantics, policy, query DTOs, business result types, provider adapters, fusion.

**Platform owns:** generic orchestration only.

## Non-goals

- Replacing or changing `RetrievalService` behavior
- Product/catalog semantics in platform code
- Rank fusion inside the coordinator
- Retries, concurrency, or provider execution in the coordinator
- Decision System or diagnostic projection (follow-up tasks)

## Consequences

### Positive

- Scenarios plug channel operations through public protocols without altering platform core orchestration
- VPI can adopt the coordinator in a later migration while keeping catalog ports and DTOs

### Negative

- Additional API surface to maintain alongside existing RAG retrieval paths until scenarios migrate

## Compliance

- Tier boundaries preserved (no `platform_proofs` imports in platform module)
- VPI runtime unchanged in P1A; adoption pending

## Implementation notes

- Code: `intergrax/rag/retrieval/multichannel/`
- Tests: `tests/unit/rag/retrieval/test_multichannel_retrieval_coordinator.py`
- Doc: `docs/project/technical/platform/multichannel_retrieval_coordination.md`

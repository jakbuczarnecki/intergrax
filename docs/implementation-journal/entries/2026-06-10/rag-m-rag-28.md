---
id: IJ-2026-06-10-037
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.28
  - GAP-RAG-10
  - GAP-RAG-11
  - GAP-RAG-12
status: completed
commit: pending
adr: none — resilience wiring on existing RetrieverEngine; no contract boundary change
---

# Retriever fallback chain + structured retrieval errors (M-RAG.28)

## Operator request

Execute next Wave 3 step: retriever resilience after M-RAG.27 OTel spans.

## Summary

- `RetrievalError` + `RetrievalErrorKind` in `retrieval/retrieval_errors.py`
- `RetrieverEngine` retry default raised to 2 (aligned with `EmbeddingEngine`)
- Canonical fallback chain `fusion` → `hybrid` → `vector_similarity` via `retrievers/resilience/retriever_fallback.py`
- Optional `RetrieverVectorCircuitBreaker` wrapping vector-backend calls
- `RetrieverExecutionMetadata` propagated to `RetrievalTrace` (`fallback_applied`, `attempted_retriever_ids`)

## Project impact

Retriever failures degrade gracefully through a typed fallback chain instead of hard errors, with structured `RetrievalError` diagnostics for operators and closed GAP-RAG-10/11/12.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-10..12 closed |
| Plan | `docs/plan/RAG.md` M-RAG.28 **Done** |

## Changed artifacts

- `intergrax/rag/retrieval/retrieval_errors.py` — `RetrievalError`, `RetrievalErrorKind`
- `intergrax/rag/retrievers/resilience/retriever_fallback.py` — canonical fallback chain
- `intergrax/rag/retrievers/retriever_engine.py` — retry default and circuit breaker hook

## Verification

```bash
uv run pytest tests/unit/rag/retrievers/test_retriever_fallback_chain.py tests/unit/rag/retrievers/test_retriever_engine_resilience.py tests/unit/rag/retrieval/test_retrieval_service_fallback_trace.py -m gate -q
```

## Risks and follow-ups

- Circuit breaker thresholds may need per-backend tuning when live vector stores are under load.
- Next Wave 3 item: **M-RAG.29** formal citation model.

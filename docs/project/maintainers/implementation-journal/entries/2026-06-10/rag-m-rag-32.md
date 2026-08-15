---
id: IJ-2026-06-10-041
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.32
  - GAP-RAG-16
status: completed
commit: pending
adr: none — optional LLM routing behind profile flag; heuristic fallback preserved
---

# Optional LLM QueryRouter tier classifier (M-RAG.32)

## Operator request

Execute first remaining Wave 3 item: LLM tier routing for `QueryRouter`.

## Summary

- `RagProfile.llm_route_enabled` (default off; `INTERGRAX_RAG_LLM_ROUTE_ENABLED`)
- `routing/llm_tier_classifier.py` — `classify_route_tier_with_llm()` with heuristic fallback
- `QueryRouter` accepts optional `LLMAdapter`; `RetrievalService` passes `llm_for_routing` (or `llm_for_agentic`)
- `RetrievalTrace.route_classifier` + `rag.retrieve` diagnostics `route_classifier`

## Project impact

Operators can opt into LLM-assisted query routing while preserving heuristic fallback, improving deep-tier retrieval without breaking default profiles.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/RAG.md` GAP-RAG-16 closed |
| Plan | `docs/project/maintainers/plans/RAG.md` M-RAG.32 **Done** |

## Changed artifacts

- `intergrax/rag/routing/llm_tier_classifier.py` — LLM tier classifier with fallback
- `intergrax/rag/routing/query_router.py` — optional `LLMAdapter` integration
- `intergrax/rag/profiles/rag_profile.py` — `llm_route_enabled` profile flag

## Verification

```bash
uv run pytest tests/unit/rag/routing/test_query_router_llm_tier.py -m gate -q
uv run pytest tests/unit/rag/ -m gate -q
```

## Risks and follow-ups

- LLM routing adds latency and cost when enabled; remains off by default.
- Next item: **M-RAG.34** agentic loop per-iteration retriever override.

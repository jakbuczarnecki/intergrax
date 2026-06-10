---
id: IJ-2026-06-10-012
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.32
  - GAP-RAG-16
status: completed
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

## Verification

```bash
uv run pytest tests/unit/rag/routing/test_query_router_llm_tier.py -m gate -q
uv run pytest tests/unit/rag/ -m gate -q
```

## Next step

**M-RAG.34** — agentic loop per-iteration retriever override.

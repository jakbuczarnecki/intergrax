---
id: IJ-2026-06-10-013
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.34
  - GAP-RAG-19
status: completed
adr: none — schedule and latency budget are profile/trace extensions on existing agentic loop
---

# Agentic loop per-iteration retriever override + latency budget trace (M-RAG.34)

## Operator request

Execute next M-RAG-DEPTH item after M-RAG.32.

## Summary

- `RagProfile.agentic_iteration_retriever_ids` (env `INTERGRAX_RAG_AGENTIC_ITERATION_RETRIEVERS`)
- `RagProfile.agentic_max_total_latency_ms` (env `INTERGRAX_RAG_AGENTIC_MAX_TOTAL_LATENCY_MS`)
- `retrieval/agentic_policy.py` — schedule resolution + budget check
- `AgenticRetrievalLoop` records per-iteration retriever/latency, refine call count, budget metadata on `RetrievalTrace`
- Stop reason `latency_budget` when cumulative loop latency exceeds profile budget

## Verification

```bash
uv run pytest tests/unit/rag/retrieval/test_agentic_loop_iteration_trace.py -m gate -q
uv run pytest tests/unit/rag/ -m gate -q
```

## Next step

**M-RAG.37** — semantic chunking ingest size guard (or **M-RAG.36** RAG load/soak gate).

---
id: IJ-2026-06-10-015
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.36
  - GAP-RAG-21
status: completed
adr: none — soak contract on existing RetrievalService; mirrors M-RAG.30 vector-store pattern
---

# RAG concurrent retrieve load/soak gate (M-RAG.36)

## Operator request

Execute final M-RAG-DEPTH item: RAG load/soak gate with concurrent retrieve SLO.

## Summary

- `evaluation/load_soak.py` — `run_retrieval_load_soak()` with `LoadSoakConfig` (workers, p95 latency, recall floor)
- Golden retrieval corpus merge via `build_soak_retrieval_service()` + `soak_queries_from_golden_cases()`
- Gate tests: `test_rag_load_soak_gate.py` (pass + latency budget failure)
- CI: `.github/workflows/rag-guard.yml` runs `pytest -m gate` on RAG paths

## Verification

```bash
uv run pytest tests/unit/rag/evaluation/test_rag_load_soak_gate.py -m gate -q
uv run pytest tests/unit/rag/ -m gate -q
```

## Milestone

**Phase M-RAG-DEPTH complete** (M-RAG.23–M-RAG.37 all Done).

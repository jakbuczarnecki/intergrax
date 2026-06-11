---
id: IJ-2026-06-10-006
date: 2026-06-10
tiers:
  - tier-0
  - tier-3
scope: RAG
plan_ref:
  - M-RAG.33
  - GAP-RAG-18
status: completed
adr: none — presets and wiring on existing GraphRAG stack
---

# GraphRAG Tier-3 production profile contract (M-RAG.33)

## Operator request

Continue M-RAG-DEPTH Wave 2: separate harness vs production GraphRAG presets and require neo4j on product hosts.

## Summary

Documented `production_rag_profile()` as harness/lab-only (in-memory graph). Added `production_graph_rag_profile()`, `validate_graph_rag_production_wiring()`, and `is_harness_graph_rag_profile()`. Product hosts via `resolve_rag_profile_for_environment` now apply neo4j backend and validate `IntegrationProfile.graph_store` slug. `create_default_rag_stack` passes integration graph store instance into `create_rag_graph_store`.

## Verification

```bash
uv run pytest tests/unit/rag/profiles/test_production_graph_rag_profile.py tests/unit/rag/graph/test_graph_rag_neo4j_prod_contract.py -m gate -q
```

Result: 8 passed.

## Next step

**M-RAG.35** — cross-backend tenant isolation contract tests.

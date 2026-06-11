---
id: IJ-2026-06-10-042
date: 2026-06-10
tiers:
  - tier-0
  - tier-3
scope: RAG
plan_ref:
  - M-RAG.33
  - GAP-RAG-18
status: completed
commit: pending
adr: none — presets and wiring on existing GraphRAG stack
---

# GraphRAG Tier-3 production profile contract (M-RAG.33)

## Operator request

Continue M-RAG-DEPTH Wave 2: separate harness vs production GraphRAG presets and require neo4j on product hosts.

## Summary

Documented `production_rag_profile()` as harness/lab-only (in-memory graph). Added `production_graph_rag_profile()`, `validate_graph_rag_production_wiring()`, and `is_harness_graph_rag_profile()`. Product hosts via `resolve_rag_profile_for_environment` now apply neo4j backend and validate `IntegrationProfile.graph_store` slug. `create_default_rag_stack` passes integration graph store instance into `create_rag_graph_store`.

## Project impact

Tier-3 product hosts cannot accidentally deploy in-memory GraphRAG; production wiring requires neo4j integration validation at bootstrap.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-18 closed |
| Plan | `docs/plan/RAG.md` M-RAG.33 **Done** |

## Changed artifacts

- `intergrax/rag/profiles/production_graph_rag_profile.py` — production presets and validation
- `intergrax/rag/bootstrap/rag_stack_bootstrap.py` — graph store wiring
- `tests/unit/rag/profiles/test_production_graph_rag_profile.py` — gate coverage

## Verification

```bash
uv run pytest tests/unit/rag/profiles/test_production_graph_rag_profile.py tests/unit/rag/graph/test_graph_rag_neo4j_prod_contract.py -m gate -q
```

Result: 8 passed.

## Risks and follow-ups

- Neo4j ops readiness remains an operator prerequisite for GraphRAG production profiles.
- Next Wave 2 item: **M-RAG.35** cross-backend tenant isolation contract tests.

---
id: IJ-2026-06-10-001
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.23
  - AUDIT-IDEAL-14.3
  - M-RAG.6
status: completed
commit: 94bea682
adr: none — profile wiring to existing retriever bootstrap; no new platform contract
---

# Wire RagProfile.query_expansion to deep-tier multiquery retrieval

## Operator request

Close the P0 RAG audit finding: `RagProfile.query_expansion` and `INTERGRAX_RAG_QUERY_EXPANSION` were defined but not connected to the retrieval path, leaving multi-query expansion dead configuration on production profiles.

## Summary

Injected `query_expander_from_profile()` into retriever bootstrap and routed deep-tier retrieval through `MultiQueryRetriever` when `query_expansion != off`. Updated `rag_stack_bootstrap`, `retriever_bootstrap`, and related resolve paths. Added gate test `test_rag_profile_query_expansion_wiring.py`. Closed GAP-RAG-01, GAP-RAG-17, GAP-RAG-23 in architecture register.

## Project impact

Tier-3 hosts can now enable LLM query expansion via `RagProfile` or environment without custom retriever wiring. Deep-tier retrieval behaviour matches documented profile semantics — a prerequisite for M-RAG-DEPTH Wave 2–3 hardening and trustworthy RAG audits.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` §Engine depth audit register |
| Plan | `docs/plan/RAG.md` Wave 1 step 1.1 **M-RAG.23 Done** |
| Audit | AUDIT-IDEAL-14.3, GAP-RAG-01/17/23 |

## Changed artifacts

- `intergrax/rag/retrievers/bootstrap/retriever_bootstrap.py` — deep tier → multiquery when expansion enabled
- `intergrax/rag/bootstrap/rag_stack_bootstrap.py` — stack wiring
- `tests/unit/rag/profiles/test_rag_profile_query_expansion_wiring.py` — regression gate
- `docs/architecture/RAG.md`, `docs/plan/RAG.md` — gap closure

## Verification

```bash
uv run pytest tests/unit/rag/profiles/test_rag_profile_query_expansion_wiring.py -q
```

Result: pass (gate coverage for profile → retriever path).

## Risks and follow-ups

- Wave 2 items (M-RAG.24 dual-index, M-RAG.25 poisoning on catalog path) remain **Planned**.
- Next: M-RAG.24 or M-RAG.25 per `docs/plan/RAG.md` Wave 2 priority.

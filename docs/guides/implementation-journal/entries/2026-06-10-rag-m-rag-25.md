---
id: IJ-2026-06-10-002
date: 2026-06-10
tiers:
  - tier-0
  - tier-3
scope: RAG
plan_ref:
  - M-RAG.25
  - AUDIT-IDEAL-14.5
  - GAP-RAG-04
status: completed
adr: none — reuses existing `filter_retrieved_chunks_for_poisoning` from V-REM-SEC.2
---

# Catalog retrieval poisoning defense on rag.retrieve (M-RAG.25)

## Operator request

Close Wave 2 security gap: retrieval poisoning defense was Nexus-only (`RagStep`); direct `rag.retrieve` catalog path bypassed trust-score quarantine.

## Summary

Wired `filter_retrieved_chunks_for_poisoning` into `perform_rag_retrieve` when `ToolWiringContext.security_profile.retrieval_poisoning_defense_enabled` is true. Added `security_profile` slot to `ToolWiringContext` and Tier-3 `build_application_tool_wiring` / `wire_application_environment` propagation from `ApplicationEnvironmentProfile`.

## Project impact

Catalog `rag.retrieve` and Nexus `RagStep` now share the same poisoning middleware when product hosts enable the security profile — required for untrusted-surface retrieval without forcing Nexus-only paths.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-04 closed |
| Plan | `docs/plan/RAG.md` M-RAG.25 **Done**, AUDIT-IDEAL-14.5 **Done** |

## Changed artifacts

- `intergrax/tools/providers/rag/service.py` — poisoning filter after retrieve
- `intergrax/tools/registry/wiring.py` — `security_profile` on `ToolWiringContext`
- `intergrax/applications/_shared/tool_wiring.py`, `environment_wiring.py` — profile propagation
- `tests/unit/tools/providers/rag/test_rag_retrieve.py` — quarantine / skip / all-quarantined cases

## Verification

```bash
uv run pytest tests/unit/tools/providers/rag/test_rag_retrieve.py -q
```

Result: 8 passed.

## Risks and follow-ups

- Hosts must wire `security_profile` on `ToolWiringContext` (automatic via `wire_application_environment`).
- Next Wave 2 item: **M-RAG.24** (DualIndex + hierarchical bootstrap).

---
id: IJ-2026-06-18-021
date: 2026-06-18
tiers:
  - tier-0
scope: RAG
plan_ref:
  - RAG-MAINT-01
  - RAG-MAINT-02
  - RAG-MAINT-03
  - RAG-MAINT-04
status: completed
commit: pending
adr: none — maintenance gates and audit prompt sync; no contract change
---

# RAG-MAINT-01..04 — audit maintenance implementation

## Operator request

Implement all Planned §6.1av RAG maintenance tasks (layer 12).

## Summary

Added `check_rag_maturity_labels.py` gate wired into rag-guard for STABLE/beta vector-store manifest honesty. Added nightly `rag_load_soak_report.py` with JSON artifact export. Regenerated audit prompt with LC-closed GAP-RAG register. Documented M-RAG.58 Frozen owner cross-ref to AHI-MAINT-04.

## Project impact

RAG ops promotion criteria and nightly soak depth are CI-enforced; audit prompt matches layer completion truth.

## Traceability

| Link | Target |
|------|--------|
| Plan | `docs/plan/RAG.md` §6.1av |
| Gate | `scripts/maintenance/check_rag_maturity_labels.py` |
| Nightly | `scripts/release/rag_load_soak_report.py` |

## Verification

```bash
uv run python scripts/maintenance/check_rag_maturity_labels.py
uv run python scripts/release/rag_load_soak_report.py
uv run python scripts/audit/generate_domain_audit_prompts.py
```

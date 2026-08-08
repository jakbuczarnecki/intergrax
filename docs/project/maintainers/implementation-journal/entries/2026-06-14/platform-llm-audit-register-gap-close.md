---
id: IJ-2026-06-14-007
date: 2026-06-14
tiers:
  - tier-0
scope: LLM_ADAPTERS
plan_ref:
  - M-LLM-X.8.1
  - LLM-AUDIT-14
  - LLM-AUDIT-15
  - AUDIT-IDEAL-6.1
  - AUDIT-IDEAL-6.2
  - AUDIT-IDEAL-6.7
status: completed
commit: pending
adr: none — documentation-only sync; no contract change
---

# LLM — audit register gap close and M-LLM-R conformance matrix

## Operator request

Close remaining documentation gaps identified in post-audit LLM verification (LLM-AUDIT-14/15, AUDIT-IDEAL 6.x table, M-LLM-R conformance, anti-patterns) without code changes.

## Summary

Extended `docs/project/architecture/LLM_ADAPTERS.md` with AUDIT-IDEAL §6 cross-ref table, LLM-AUDIT-14/15 register rows, 16-dimension M-LLM-R conformance matrix, structured output (6.1 Done) subsection, CI vendor-import guard, anti-patterns table, and clarified preflight vs history-layer token gaps.

Synced traceability in `docs/project/maintainers/plans/LLM_ADAPTERS.md` and register IDs in `docs/project/maintainers/audit/LLM_ADAPTERS.md`.

## Project impact

Architecture canon now fully mirrors the audit instruction gap list (15 LLM-AUDIT IDs + AUDIT-IDEAL 6.1–6.7). M-LLM-X code waves can proceed with unambiguous traceability.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/LLM_ADAPTERS.md` — §Audit register, §Anti-patterns, §M-LLM-R conformance |
| Plan | `docs/project/maintainers/plans/LLM_ADAPTERS.md` — M-LLM-X traceability |
| Audit instruction | `docs/project/maintainers/audit/LLM_ADAPTERS.md` — known gaps table |

## Changed artifacts

- `docs/project/architecture/LLM_ADAPTERS.md`
- `docs/project/maintainers/plans/LLM_ADAPTERS.md`
- `docs/project/maintainers/audit/LLM_ADAPTERS.md`

## Verification

```bash
python scripts/audit/check_docs_domain_pairs.py
python scripts/maintenance/check_harness_adr.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass (doc-only iteration).

## Risks and follow-ups

- M-LLM-X.1–X.8 code remains Planned — register rows document open gaps until implementation closes them.
- `validate_runtime()` for LLM profile (LLM-AUDIT-8) still pending X.7.

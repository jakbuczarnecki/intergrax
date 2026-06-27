---
id: IJ-2026-06-14-006
date: 2026-06-14
tiers:
  - tier-0
  - tier-1
scope: MEMORY
plan_ref:
  - MEM-VEC-0.1
  - MEM-VEC-doc-sync
status: completed
commit: pending
adr: no ADR needed — documentation sync only; canon aligned to MEM-VEC/MEM-DEPTH Done state
---

# MEMORY — post-verification architecture and audit instruction sync

## Operator request

Verify consistency between memory layer audit findings and architecture canon; complete documentation gaps.

## Summary

Synchronized `docs/architecture/MEMORY.md` after MEM-VEC closeout: removed stale "target/partial/not shipped" labels, updated MemoryKind enum, persistence matrix (Mongo session), market parity §14, read-path naming (`run_session_semantic_recall_context`, CE `SessionSemanticRecallProvider`), §7.3 budgeting vs ADR-MEM-001, and added §17 audit register (MEM-AUDIT-1…7).

Updated `docs/plan/MEMORY.md` AUDIT-IDEAL header, 16.x CE ownership notes, MemoryKind as-built text, session Mongo row. Refreshed `docs/audit/MEMORY.md` gaps table and code paths. Updated audit map §15 for MEM-VEC status.

## Project impact

Auditors and implementers have a single consistent MEMORY canon matching MEM-VEC-1/2 code reality; remaining backlog (MEM-VEC-3, MEM-DEPTH-5.2) is explicit in audit register.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/MEMORY.md` §17 audit register |
| Plan | `docs/plan/MEMORY.md` — AUDIT-IDEAL + MEM-VEC |
| Audit prompt | `docs/audit/MEMORY.md` |
| ADR | ADR-MEM-001, ADR-MEM-002 (unchanged) |

## Changed artifacts

- `docs/architecture/MEMORY.md`
- `docs/plan/MEMORY.md`
- `docs/audit/MEMORY.md`
- `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` §15

## Verification

```bash
python scripts/audit/check_docs_domain_pairs.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass (doc-only iteration).

## Risks and follow-ups

- MEM-VEC-3 plugin EP + skill runtime still Planned.
- Journal IJ-2026-06-14-003 commit hash still pending on code iteration entry.

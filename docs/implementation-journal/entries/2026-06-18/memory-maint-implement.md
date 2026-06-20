---
id: IJ-2026-06-18-022
date: 2026-06-18
tiers:
  - tier-0
scope: MEMORY
plan_ref:
  - MEM-MAINT-01
  - MEM-MAINT-02
  - MEM-MAINT-03
  - MEM-MAINT-04
status: completed
commit: pending
adr: none — depth helpers and tests; no contract change
---

# MEM-MAINT-01..04 — audit maintenance implementation

## Operator request

Implement all Planned §6.1av Memory maintenance tasks (layer 13).

## Summary

Added cognitive store mapping for procedural/org kinds, org memory maturity checklist, temporal validity tests, and explicit LangMem/Zep parity backlog boundary. Audit prompt synced to LC closeout.

## Verification

```bash
uv run pytest tests/unit/memory/test_memory_maint_depth.py -q
```

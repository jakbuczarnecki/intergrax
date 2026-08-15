---
id: IJ-2026-06-18-024
date: 2026-06-18
tiers:
  - tier-0
scope: MODALITY
plan_ref:
  - MOD-MAINT-01
  - MOD-MAINT-02
  - MOD-MAINT-03
  - MOD-MAINT-04
status: completed
commit: pending
adr: none — test/fixture hygiene and ops docs
---

# MOD-MAINT-01..04 — audit maintenance implementation

## Summary

Fixed OpenCV availability round-trip probe, shared vision golden fixture conftest, per-test skipif for celery registry test, three-plane ops runbook in architecture, and MOD-MAINT-04 remote serving backlog row.

## Verification

```bash
uv run pytest tests/unit/model_inference/ -q
```

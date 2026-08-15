---
id: IJ-2026-06-18-023
date: 2026-06-18
tiers:
  - tier-0
  - tier-1
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-MAINT-01
  - CE-MAINT-02
  - CE-MAINT-03
  - CE-MAINT-04
status: completed
commit: pending
adr: none — observability payload extension; backward compatible
---

# CE-MAINT-01..04 — audit maintenance implementation

## Summary

Wired OTel SDK spans on context assemble path, added fragment cost fields to CONTEXT_ASSEMBLED v2 payload, preset regression baseline tests, and audit prompt LC sync with GAP-CTX-12 Frozen cross-ref to AHI.

## Verification

```bash
uv run pytest tests/unit/context/test_ce_maint_observability.py tests/unit/context/test_ce_preset_regression_baselines.py -q
```

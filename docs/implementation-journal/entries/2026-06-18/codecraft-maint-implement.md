---
id: IJ-2026-06-18-032
date: 2026-06-18
tiers:
  - tier-0
  - tier-1
scope: CODE_CRAFT
plan_ref:
  - ECC-MAINT-02
  - ECC-MAINT-03
  - ECC-MAINT-04
status: completed
commit: pending
adr: none — wiring and observability depth; no new platform contract
---

# ECC-MAINT-02..04 — audit maintenance implementation

## Operator request

Close remaining Planned §6.1av Code Craft tasks for domains 1–11 depth backlog.

## Summary

Wired `codegen_llm_profile` / `codegen_llm_profile_ref` through `codegen_llm_resolver` and host wiring. Added `container` isolation tier routing in `sandbox_resolver`. Shipped §10.2 metrics via `CodeCraftMetricsSnapshot` and `codecraft.metrics_snapshot` trace step.

## Changed artifacts

- `intergrax/codecraft/llm_codegen_adapter.py`, `codegen_llm_resolver.py`
- `intergrax/runtime/codecraft/trace.py`, `sandbox_resolver.py`
- `intergrax/applications/_shared/codecraft_wiring.py`, `environment_wiring.py`
- `tests/unit/runtime/codecraft/test_ecc_maint_depth.py`

## Verification

```bash
uv run pytest tests/unit/runtime/codecraft/test_ecc_maint_depth.py -q
```

## Risks

Container tier without hosted sandbox falls back to local sandbox (documented dev-only path).

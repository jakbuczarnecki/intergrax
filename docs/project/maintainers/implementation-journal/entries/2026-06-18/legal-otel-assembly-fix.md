---
id: IJ-2026-06-18-037
date: 2026-06-18
tiers:
  - tier-0
  - tier-3
scope: INTEGRATIONS, legal_application
plan_ref:
  - INT-MAINT-03
status: completed
commit: pending
adr: none — align legal product integration preset with product observability profile
---

# Legal host OTEL assembly fix

## Operator request

Fix failing gate test `test_legal_backend_chat_with_unified_task_runner` and sync audit documentation.

## Summary

`ApplicationEnvironmentProfile.product_defaults()` enables `otel_enabled` via `production_slo()`, but `IntegrationProfile.legal_product()` lacked `observability_backend`. Added OTEL backend (matching `research_product` pattern). Added regression test `test_legal_environment_observability_assembly_valid`.

## Changed artifacts

- `intergrax/integrations/registry/profile.py` — `legal_product()`
- `tests/unit/applications/test_legal_manifest_wiring.py`
- `docs/audit_results/2026-06-18/RUN_SUMMARY.md`
- `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — §6.1av cross-domain status

## Verification

```bash
uv run pytest -m "gate and not no_ci" -q
```

Result: **1495 passed** (2026-06-18).

## Risks

None — legal host now satisfies existing observability assembly invariant.

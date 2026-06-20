---
id: IJ-2026-06-11-032
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-PROD-7
status: completed
commit: pending
adr: none — CI gate on existing CostProfile + AgentBudgetSlice contracts
---

# Sprint 7 APP-PROD-7 — STRICT product budget enforcement gate

## Operator request

Continue Tier-3 application architecture sprint queue with APP-PROD-7: enforce COST profile and per-agent `budget_slice` on STRICT product hosts after ACP-TOK completion.

## Summary

Added `check_budget_enforcement.py` and wired it into `check_application_production_gates.py`. Introduced `budget_wiring.py` helpers (`product_budget_reaction`, `product_agent_budget_slice`) and `product_manifest_registry.py` for canonical product manifests. `product_defaults` now seeds `budget_reaction`. All four product hosts (legal, research, dispute_sim, local_workspace) declare HARD `budget_slice` on every roster agent.

## Project impact

STRICT product hosts cannot ship without environment token ceiling, reaction policy, and per-agent HARD caps — closing architecture §43 host gate row. Unblocks production claims for mutating product environments with kernel-enforced budgets.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §43 · §40.2 APP-PROD-7 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` §6.2y step 4 |
| Cross-plan | ACP-TOK-* (enforcement runtime) |

## Changed artifacts

- `scripts/check_budget_enforcement.py` — APP-PROD-7 CI gate (new)
- `scripts/check_application_production_gates.py` — aggregate includes budget check
- `intergrax/applications/_shared/budget_wiring.py` — helpers + conformance validator (new)
- `intergrax/applications/_shared/product_manifest_registry.py` — product manifest index (new)
- `intergrax/applications/contracts/environment_profile.py` — `budget_reaction` in `product_defaults`
- `applications/*/manifest.py` — `budget_slice` on product roster agents
- `tests/unit/scripts/test_check_budget_enforcement.py`
- `tests/unit/applications/test_budget_enforcement_conformance.py`

## Verification

```bash
uv run python scripts/check_budget_enforcement.py
uv run python scripts/check_application_production_gates.py
uv run pytest tests/unit/scripts/test_check_budget_enforcement.py tests/unit/applications/test_budget_enforcement_conformance.py -m gate -q
```

Result: pass (8 tests).

## Risks and follow-ups

- APP-CON-5: hook timeout and error→BLOCK middleware.
- APP-CON-6: `RunArtifactBundle` on `ApplicationRunSummary.metadata`.

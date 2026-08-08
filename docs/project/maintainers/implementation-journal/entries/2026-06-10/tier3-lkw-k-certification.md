---
id: IJ-2026-06-10-007
date: 2026-06-10
tiers:
  - tier-2
  - tier-3
scope: agents/vendor_discovery, applications/local_workspace_application, intergrax/applications/_shared
plan_ref:
  - AUDIT-IDEAL-28.4
  - AUDIT-IDEAL-5.3
  - K.1
  - K.2
status: completed
commit: d07e4d62
adr: none — product certification wiring; platform contracts unchanged
---

# LKW hybrid daemon, product observability dashboard, and K.1/K.2 business agent certification

## Operator request

Close AUDIT-IDEAL Band 3 product-facing items: production governance dashboard, Local Knowledge Workspace hybrid daemon, and certify first business agents (K.1/K.2) on the harness reference path.

## Summary

Added `vendor_discovery` Tier-2 agent scaffold with contract, pipeline, and tests. Wired Tier-3 shared modules: `business_agent_certification.py`, `lkw_hybrid_daemon.py`, product observability dashboard routes/wiring. Updated AUDIT-IDEAL and TIER3 plan registers.

## Project impact

Intergrax has an end-to-end **product validation path**: certified business agents on a reference Tier-3 host with observability dashboard and hybrid daemon — proving Harness composition beyond platform-only gates.

## Traceability

| Link | Target |
|------|--------|
| Agent | `agents/vendor_discovery/` |
| Application | `applications/local_workspace_application/` |
| Plan | `docs/project/maintainers/plans/AUDIT_IDEAL_2026.md`, `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` |
| Agents roster | `agents/README.md` |

## Changed artifacts

- `agents/vendor_discovery/` — new agent
- `intergrax/applications/_shared/lkw_hybrid_daemon*.py`
- `intergrax/applications/_shared/business_agent_certification.py`
- `intergrax/applications/_shared/product_observability_dashboard_*.py`

## Verification

```bash
uv run pytest agents/vendor_discovery/tests/ -q
```

Result: pass.

## Risks and follow-ups

- K.1/K.2 certification is harness-validation baseline; additional business agents remain §6.3 end-of-plan unless reprioritized.

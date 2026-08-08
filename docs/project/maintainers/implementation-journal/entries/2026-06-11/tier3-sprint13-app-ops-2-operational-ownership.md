---
id: IJ-2026-06-11-038
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-OPS-2
status: completed
commit: pending
adr: none — formalizes existing informal ARCHITECTURE.md ownership; no new runtime primitive
---

# Sprint 13 — ApplicationOperationalOwnership on product manifests

## Operator request

Continue Tier-3 application architecture sprint queue: APP-OPS-2 — operational ownership schema on `ApplicationManifest` with APP-PROD CI gate.

## Summary

- `operational_ownership.py` — `ApplicationOperationalOwnership`, owner/maintainer/escalation models.
- `ApplicationManifest.ownership` optional field; required for PRODUCT profile via gate.
- `ownership_wiring.py` — `standard_product_operational_ownership`, `evaluate_application_ownership`, `check_manifest_operational_ownership`.
- Reference product manifests (legal, research, dispute_sim, local_workspace) declare ownership.
- `scripts/maintenance/check_application_ownership.py` wired into `check_application_production_gates.py`.

## Project impact

Product hosts now carry typed ops contacts for incident routing, HITL/budget escalation, and deploy review — symmetric to agent `ProductionOwnerMetadata` at environment scope.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §50.2 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` APP-OPS-2 · §6.2y step 10 |

## Changed artifacts

- `intergrax/applications/contracts/operational_ownership.py`
- `intergrax/applications/contracts/manifest.py`
- `intergrax/applications/_shared/ownership_wiring.py`
- `applications/*/manifest.py` (×4 product hosts)
- `scripts/maintenance/check_application_ownership.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_operational_ownership_gate.py \
  tests/unit/scripts/test_check_application_ownership.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- Ownership defaults use platform team placeholders — product teams should customize per deployment.
- APP-OPS-3 health score next in §6.2y queue.

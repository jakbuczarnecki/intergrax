# T-IGX-097-B — EC-ACCELERATION-02 Package C (EQO-CL-01 / EC-F12 GCA strict smoke)

## Scope

PRE_FCS_CONSOLIDATED_DEFECT_SETTLEMENT EC-ACCELERATION-02 Package C: replace incomplete direct-factory GCA host smoke with revision-bound strict production activation flow; preserve STRICT authority without LAB fallback.

## Plan digest

Plan digest: n/a (EC-ACCELERATION-02 operator batch Package C)

## Approved

Approved: operator-delegate @ 2026-11-28T16:05:00+01:00

Delegation: EC-ACCELERATION-02 operator execution authority (Maximum Safe Batch 2.0 Package C)

## Write set

```text
applications/governed_contractor_application/tests/host/**
tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py
governance/tickets/T-IGX-097-B.md
```

## Test contract

GCA smoke 3/3 PASS; adjacent D4 strict launcher regressions; `uv run ruff check` on write set; `git diff --check`; `uv run python -m scripts.governance.igx_gov ticket`.

## Out of scope

Weakening `resolve_roster_agent_contract`, synthetic authority, production factory semantics changes.

## Checkpoint

- 2026-11-28: EC-F12 BLOCKED — `ProductionRegistryProjectionInputError: runtime revision application_id mismatch with projection scope` when binding real `build_governed_contractor_manifest()` to revision-qualified bundle; needs owner revision-bound GCA roster/artifact qualification fixture (distinct from AC3 `_ac3_revision_bound_strict_external_work_host_manifest`).

## Lifecycle state

in_progress
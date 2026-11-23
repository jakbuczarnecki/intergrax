# T-IGX-070-C — SFX R2 canonical overlay adoption (Option B, docs/governance only)

## Scope

Bounded adoption campaign **IGX-SFX-R2-CANONICAL-ADOPTION-001**: merge R2 overlay into `SFX-MASTER-PLAN.md`, synchronize R2 publication SSOT, record enterprise-quality posture on documentation merges. **Does not** authorize SFX-P5+, delegated multi-submit, SEF CON-N-01 change, or production admission.

**Inputs:** R2 package on `development` @ `7ddd53266d32ce47f059456e110e414b92248453` (PR #756–#757).

Plan: [`PLAN_T-IGX-070-C.md`](plans/PLAN_T-IGX-070-C.md).

Plan digest: 5b15f41e141c81c2de980d8e4312f3d3039c483a0a1c316e0e7c491a5928f19f

## Baseline

| Field | Value |
| --- | --- |
| Binding plan revision | **R1** (operator-approved 2026-11-12) |
| R2 publication merge | `7ddd53266d32ce47f059456e110e414b92248453` |
| T-IGX-070-B | `plan_approved` (corrective publication) |
| T-IGX-070-A | `plan_pending` (not inferred from 070-B) |

## Out of scope

`agents/scenario_factory/` implementation; Tier-3 product features; ARCH-FREEZE promotion; clearing unrelated `governance-fast` / `ent-audit-fast` failures.

## Test contract

```text
uv run python -c "import json,glob; [json.load(open(p)) for p in glob.glob('docs/project/maintainers/qualification/scenario_factory_x/design/schemas/proposed/*.json')]"
```

## Lifecycle state

plan_pending

## Deliverables

- `governance/tickets/plans/SFX-MASTER-PLAN.md` §18 overlay
- Updated R2 publication dossiers + hub index
- [`OD-SFX-R2-CANONICAL-ADOPTION.md`](../docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/OD-SFX-R2-CANONICAL-ADOPTION.md)

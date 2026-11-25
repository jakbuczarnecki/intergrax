# T-IGX-070-C — SFX R2 canonical overlay adoption (Option B, docs/governance only)

## Scope

Bounded adoption campaign **IGX-SFX-R2-CANONICAL-ADOPTION-001**: merge R2 overlay into `SFX-MASTER-PLAN.md`, synchronize R2 publication SSOT, record enterprise-quality posture on documentation merges. **Does not** authorize SFX-P5+, delegated multi-submit, SEF CON-N-01 change, or production admission.

**Inputs:** R2 package on `development` @ `7ddd53266d32ce47f059456e110e414b92248453` (PR #756–#757).

Plan: [`PLAN_T-IGX-070-C.md`](plans/PLAN_T-IGX-070-C.md).

Plan digest: 7c039ad2c00db38b6a1e0ac8f0feca204cfd39c4805efc584c4aaa92752e9925

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

## Approved

Approved: operator-delegate @ 2026-11-23T16:18:52+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-R2-070-C-PLAN-APPROVAL.md@58683f18aa029f85b487d2ae1f5df6228b3dfc68
Delegate session: local_491df1e7-9f89-43d0-a7d8-ffdbbfb9cab4

## Lifecycle state

in_progress

## Closure batch (F-GOV-01)

| Field | Value |
| --- | --- |
| G6 corrective publication | `7c16d9b5488234475e80a2e758158decca5e5ac5` |
| F-GOV-01 disposition | [`OD-SFX-R2-GOV-01.md`](../docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/OD-SFX-R2-GOV-01.md) |
| G6 delta | **ACCEPT** (auditor `local_039b59d4-5f1c-4360-8fd7-28315b244478`) |
| `G6_GOVERNANCE_VERDICT` | **NONCOMPLIANT** (preserved) |

## Checkpoint

```text
verify_plan_approval: PASS (2026-11-23, .venv python)
test contract (6 JSON schemas): PASS (2026-11-23, .venv python)
verify_ticket_lifecycle: pending after ticket edit
igx_gov ticket: FAIL/BLOCKED — push-fast exit=2, tiktoken pyd install Odmowa dostępu (os error 5); not claimed PASS
```

## Deliverables

- `governance/tickets/plans/SFX-MASTER-PLAN.md` §18 overlay
- Updated R2 publication dossiers + hub index
- [`OD-SFX-R2-CANONICAL-ADOPTION.md`](../docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/OD-SFX-R2-CANONICAL-ADOPTION.md)
- [`OD-SFX-R2-GOV-01.md`](../docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/OD-SFX-R2-GOV-01.md) (F-GOV-01 disposition)

## Implementer session

Implementer session: local_5e7f04c8-9480-4633-8618-f00a35f402ee

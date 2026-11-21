# EQO-CLOSURE v2 — execution artifacts

Master plan (read-only reference): `eqo_closure_master_7499b85e.plan.md` (Cursor plans store).

Pins (plan §1):

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

Write set for this band: **only** files under this directory (`eqo_closure/`). No product code, no FRZ, no roadmap edits.

W0 publication (EQO-W0-01..05):

- `EQO_PIN_BIND.json`, `EQO_ARTIFACT_RECOVERY_REGISTRY.jsonl`, `pr680_governance_fast.md`
- `EQO_CURRENT_STATE_INVENTORY.md`, `EQO_CANONICAL_SOURCE_REGISTRY.md`
- `EQO_A_REM_MASTER_FINDING_MATRIX.jsonl`, `EQO_A_REM_CLOSURE_GAP_REPORT.md`, `EQO_A_REM_REMAINING_EXECUTION_ROADMAP.md`
- `EQO_ARCH_FREEZE_IMPACT_MATRIX.jsonl`, `EQO_MASTER_CLOSURE_MATRIX.md`
- `EQO_OPERATOR_DECISION_PACKET.md`, `EQO_EXECUTION_AGENT_INSTRUCTION.md`, `EQO_MASTER_COMPLETION_PLAN.md`

Machine registries:

- `EQO_MASTER_COMPLETION_WBS.json` — task registry (plan §4).
- `EQO_MASTER_DEPENDENCY_DAG.json` — dependency edges (plan §5).
- `EQO_B2_DISPOSITION.jsonl` — B-2 finding → task mapping (W4-01).
- Campaign executable outlines: `EQO_*_PLAN.md`.
- Matrices: `EQO_B1_*.jsonl`, `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl`.

Terminal states (plan): `EQO_CLOSED_INDEPENDENTLY_VERIFIED` or `EQO_NOT_CLOSED_EXACT_BLOCKERS` — recorded in closure matrix when W9 runs; not asserted by this scaffold.

## W0 publish band (completed)

- `EQO_PIN_BIND.json`, inventory, source registry, recovery registry
- Master finding matrix, impact matrix, gap report, W2 roadmap
- `pr680_governance_fast.md` (W0-03)
- Closure matrix, operator packet, execution instruction, master index
- `EQO_C_DRIFT.jsonl` — alias of historical matrix until C-03 execution fills drift_class

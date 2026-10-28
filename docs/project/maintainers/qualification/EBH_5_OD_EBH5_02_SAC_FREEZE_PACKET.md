# OD-EBH5-02 — SAC FREEZE READY FOR OPERATOR DECISION

| Field | Value |
| --- | --- |
| **Decision ID** | OD-EBH5-02 |
| **Action** | Approve **SAC-EBH-5-v1** freeze |
| **Status** | **READY FOR OPERATOR DECISION** (not approved) |
| **Blocks** | OD-EBH5-08 parent CLOSED |

## Package contents

1. [`SAC_EBH_5_v1_PROPOSAL.md`](SAC_EBH_5_v1_PROPOSAL.md) — contract scope and freeze checklist.
2. [`EBH_5_FINAL_PARENT_CLOSURE_CANDIDATE.md`](EBH_5_FINAL_PARENT_CLOSURE_CANDIDATE.md) — terminal technical reconciliation.
3. [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) — SHA ladder and test provenance.
4. Fixed substrate ratifications — see [`EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md`](EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md) + SSOT `_fixed_substrate_registry.py`.

## Sub-decisions in same packet (operator-only)

| Sub-ID | Topic | SSOT status |
| --- | --- | --- |
| OD-EBH5-02-A | SAC-EBH-5-v1 freeze | **OPEN** |
| OD-EBH5-04 reaffirm | Runtime SQLite FIXED bundle | **PROPOSED — OPERATOR PENDING** |
| OD-EBH5-04 lab | Lab adaptive SQLite | **PROPOSED — OPERATOR PENDING** |
| OD-EBH5-04 scoped | Single-node triad | **RATIFIED — SCOPED PROD-Q** (reaffirmation optional) |

## Required operator record (not pre-filled)

```text
OPERATOR APPROVED — EBH-5 SAC FREEZE
```

Record in [`EBH_5_OPERATOR_DECISION_REGISTER.md`](EBH_5_OPERATOR_DECISION_REGISTER.md) with date and operator identity.

## Agent boundary

Cursor **must not** insert operator approval text or mark SAC **FROZEN** without explicit operator message authorizing OD-EBH5-02.

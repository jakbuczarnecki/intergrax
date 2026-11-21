# EQO-X-07 — EDER-N-0002 E-03 route memo

**Register row:** `EDER-N-0002` (Q-WA-07, post-ingest)  
**Owner (proposed):** DUP-X (`owner_status=SINGLE_PROPOSED`, `owner_vocabulary_status=CONFIRMED`)  
**Task:** record E-03 routing only — no `register.jsonl` edit, no enum re-identification on this pass.

## Pins (`EQO_PIN_BIND.json`)

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Register fields (read-only @ `4af9a00a0`)

| Field | Value |
|-------|-------|
| `register_class` | `DEBT` |
| `next_track` | `DECISION` |
| `frz_arch_freeze_impact` | `UNKNOWN` |
| `lifecycle_state` | `INGESTED` |
| `disposition` (EDER) | `DECISION_PENDING` |

## Scope note (read-only)

Row text: duplicate orchestration enums in Tier-1 runtime and Tier-3 application wiring left in place by T-IGX-043-G.  
This memo does **not** name enums or re-diff tiers at the pins (per plan §EQO-X-07 — owner analysis in E-03, not EQO guess).

## E-03 route (`EQO_EDER_COMPLETE_DISPOSITION_PLAN.md`)

| Mechanism | Routing |
|-----------|---------|
| Owner batch | `SINGLE_PROPOSED` vs DP-47 — one OD-EQO-09 confirm |
| EQO matrix disposition | `WAIT_FOR_OPERATOR` until TRIAGED |
| Impact transition | `UNKNOWN` → DUP-X / Operator via E-03 (canonical definition or explicit mirror intent) |
| Signed DUP-X records | Historical — not edited in this workstream |

## Disposition (interim)

`NOT_CLOSED` — conscious residual after 043-G; full disposition owned by DUP-X + Operator package when TRIAGED.

**Audit:** session E pending.
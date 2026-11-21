# EQO-X-08 — impact BLOCKS verification read (five findings)

**Task:** `EQO-X-08` · **Acceptance:** five `OWNER_DECISION` finding IDs dispositioned (read-only verification)

## Five IDs (`EQO_B2_DISPOSITION.jsonl`)

| Finding ID | Severity | Owner (disposition) |
|------------|----------|---------------------|
| EQOA-A1-017 | MEDIUM | Operator (roadmap status) / TOOL-LIFE-X stage owner |
| EQOA-A2-017 | INFO | PDC-X |
| EQOA-A3-021 | MEDIUM | ARCH-FREEZE / PDC-X |
| EQOA-A7-005 | INFO | ARCH-FREEZE |
| EQOA-A7-029 | INFO | ARCH-FREEZE |

## Impact cross-check (`EQO_ARCH_FREEZE_IMPACT_MATRIX.jsonl`)

Read-only pass @ DOC_PIN: all five IDs retain `eqo_impact_class` **`BLOCKS_ARCH_FREEZE`** and `current_status` **`BLOCKED`**.

| Finding | `dossier_impact` | `eqo_impact_class` | `current_status` |
|---------|------------------|--------------------|------------------|
| A1-017 | BLOCKS | BLOCKS_ARCH_FREEZE | BLOCKED |
| A2-017 | BLOCKS | BLOCKS_ARCH_FREEZE | BLOCKED |
| A3-021 | BLOCKS | BLOCKS_ARCH_FREEZE | BLOCKED |
| A7-005 | BLOCKS | BLOCKS_ARCH_FREEZE | BLOCKED |
| A7-029 | BLOCKS | BLOCKS_ARCH_FREEZE | BLOCKED |

EQO records the read only — **no** G-band downgrade without operator OD packet (`OWNER_DECISION` route per `EQO-W4-03` batching).

## Interim disposition

`WAIT_FOR_OPERATOR` — batch OWNER_DECISION per `EQO_B2_REMAINING_INVARIANTS_PLAN.md` (W4-03 addendum), not per-finding chat.

**Audit:** session E pending.

## Wave-4 refresh

Five IDs unchanged `BLOCKS_ARCH_FREEZE` / `BLOCKED`; owner batch remains `W4-03_OWNER_DECISION_BATCH.md` (not reopened in wave-4).

**Recorded:** 2026-11-21 · lane `eqo/wave-4/x`
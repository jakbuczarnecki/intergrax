# EQO-W4-03 — OWNER_DECISION batch draft (no operator dialog)

**Parent:** `EQO-W4-03` · **Source reads:** `eqo_closure/x/X-08/BLOCKS_IMPACT_READ.md` @ DOC_PIN  
**Disposition registry:** `EQO_B2_DISPOSITION.jsonl` (69 rows with `next_action_source=OWNER_DECISION` excluding the five X-08 IDs handled below)  
**Status:** `WAIT_FOR_OPERATOR` — draft only; no OD addendum append in this wave.

## Pins

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## X-08 five IDs (BLOCKS_ARCH_FREEZE verified)

Cross-read @ DOC_PIN confirms all five retain `eqo_impact_class=BLOCKS_ARCH_FREEZE` and `current_status=BLOCKED` in `EQO_ARCH_FREEZE_IMPACT_MATRIX.jsonl`.

| Finding ID | Severity | Owner (disposition) | Proposed batch action |
|------------|----------|---------------------|------------------------|
| EQOA-A1-017 | MEDIUM | Operator (roadmap status) / TOOL-LIFE-X | OD batch item — no G-band downgrade without operator packet |
| EQOA-A2-017 | INFO | PDC-X | OD batch item |
| EQOA-A3-021 | MEDIUM | ARCH-FREEZE / PDC-X | OD batch item |
| EQOA-A7-005 | INFO | ARCH-FREEZE | OD batch item |
| EQOA-A7-029 | INFO | ARCH-FREEZE | OD batch item |

**Interim:** `WAIT_FOR_OPERATOR` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md` — single OD-EQO addendum, not per-finding chat.

## Remaining W4-03 rows (64)

Machine count: **69** `OWNER_DECISION` assignments in B-2 disposition minus **5** X-08 = **64** rows batched into `EQO-W4-03` without row-level operator dialog in this implementer session.

Steward rule: one consolidated operator decision packet referencing severity histogram (MEDIUM/LOW/INFO) and owner labels from `EQO_B2_DISPOSITION.jsonl`; **no** independence claim; **no** FRZ promotion.

## Acceptance (when operator acts)

- Operator publishes OD addendum (existing `EQO_OPERATOR_DECISION_PACKET.md` route or successor ticket).
- Matrix row C18/C06 may move from `PARTIALLY_MET` only after recorded decisions — not in wave-3.

## Audit

Session E pending for OWNER_DECISION reads; implementer does not PASS.

**Ref:** `governance/tickets/T-IGX-046-F.md`
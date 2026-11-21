# EQO-W4-03 — operator OWNER_DECISION batch (English steward memo)

**Audience:** Repository operator / steward  
**Status:** `WAIT_FOR_OPERATOR` — proposal only; no OD append in wave-7.

## Summary

EQO batches **69** findings with `next_action_source=OWNER_DECISION` from `EQO_B2_DISPOSITION.jsonl` into **one** consolidated decision packet. Five additional BLOCKS_ARCH_FREEZE IDs (X-08 set) are listed in the batch draft but verified separately @ DOC_PIN.

## Pins

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Steward ask (single packet)

1. Confirm or amend the **batch route** — one OD-EQO addendum referencing severity histogram (MEDIUM / LOW / INFO) and owner labels from disposition, not 69 separate threads.  
2. For X-08 five IDs (`EQOA-A1-017`, `A2-017`, `A3-021`, `A7-005`, `A7-029`): confirm **no G-band downgrade** without explicit operator text (see `x/X-08/BLOCKS_IMPACT_READ.md`).  
3. Record decisions in the existing operator packet route (`EQO_OPERATOR_DECISION_PACKET.md` or successor ticket) — EQO does not self-apply.

## Explicit non-claims

- No implementer PASS.  
- No FRZ promotion.  
- No row-level owner confirmation invented in EQO docs.

## Machine references

- Draft enumeration: `W4-03_OWNER_DECISION_BATCH.md`  
- Impact matrix: `EQO_ARCH_FREEZE_IMPACT_MATRIX.jsonl`  
- B-2 plan: `EQO_B2_REMAINING_INVARIANTS_PLAN.md` §EQO-W4-03

**Ref:** `governance/tickets/T-IGX-046-F.md`
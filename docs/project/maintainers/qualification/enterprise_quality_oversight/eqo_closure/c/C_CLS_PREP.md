# EQO-C-CLS — prep checklist (wave-7)

**Parent:** `EQO-C-CLS` · **Plan:** `EQO_C_REGRESSION_EXECUTION_PLAN.md` §Acceptance  
**Status:** `NOT_READY` — prep only; no campaign close claim.

## Pins

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Task gates

| Task | Wave-7 prep read |
|------|------------------|
| **C-01** | `DISPOSITIONED_NOT_CLOSED` — `c/A3-004/CLOSURE.md` |
| **C-02** | `DISPOSITIONED_NOT_CLOSED` — `c/A6-004/CLOSURE.md` |
| **C-03** | 18 layers classified — no `UNKNOWN` in `EQO_C_DRIFT.jsonl` |
| **C-04** | `BLOCKED` — `runs/c/C04_STATUS.md` + manifest |
| **C-05** | `WAIT_FOR_INTEGRATION` (FCS-bound) |

## C-CLS acceptance (plan)

Both Campaign C findings **dispositioned** — `MET` at implementer band (C-01/C-02 closure memos).  
C-03 complete — `MET`.  
C-04 **not** green — `NOT_MET` → **C-CLS blocked**.

## Next operator / train actions

1. Resolve or narrow C-04 replay slice (product path or smaller pytest surface).  
2. Schedule session E for C-01/C-02 reads.  
3. Re-open C-CLS only after C-04 unblocks or steward records exact blocker per `FULL_REQUAL_REQUIRED` row.

**Ref:** `governance/tickets/T-IGX-046-F.md`
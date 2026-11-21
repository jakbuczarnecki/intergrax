# EQO-W4-CLS — prep checklist (wave-7)

**Parent:** `EQO-W4-CLS` · **Plan:** `EQO_B2_REMAINING_INVARIANTS_PLAN.md`  
**Status:** `PARTIAL` — all B-2 batch tasks **started**; row-level closure not asserted.

## Batch start inventory

| Task | Rows | Artifact | Wave-7 state |
|------|-----:|----------|--------------|
| EQO-W4-02 | 95 | `w4/W4-02_TRANCHE_*.md` (01–07) | Queue complete outside Campaign B |
| EQO-W4-03 | 69 | `W4-03_OWNER_DECISION_BATCH.md` + English memo | `WAIT_FOR_OPERATOR` |
| EQO-W4-04 | 5 | `W4-04_REQUALIFY_BATCH.md` | Drafted wave-4 |
| EQO-W4-05 | 5 | `W4-05_NONE_JUSTIFICATION.md` | Drafted wave-4 |
| EQO-X-08 | 5 | `x/X-08/BLOCKS_IMPACT_READ.md` | Read @ DOC_PIN |

**Registry:** `EQO_B2_DISPOSITION.meta.json` — `verification: PASS`, 186 B-2 rows.

## W4-CLS rule (plan)

Every B-2 row has disposition (not necessarily `CLOSED_WITH_EVIDENCE`). Wave-7 prep confirms **assignment coverage** and **batch artefacts present**; 69 OWNER_DECISION rows remain steward-blocked until operator OD addendum.

## PT-Q02 observation

EQO records batch readiness only — does **not** execute ARCH-FREEZE PT-Q02.

## Blockers to full W4-CLS

- W4-03 operator packet (64 + 5 X-08 IDs batched).  
- W4-02 open doc-fix rows without owner PR or explicit `BLOCKED` line (per-tranche memos).  
- No independence or FRZ promotion in this wave.

**Ref:** `governance/tickets/T-IGX-046-F.md`
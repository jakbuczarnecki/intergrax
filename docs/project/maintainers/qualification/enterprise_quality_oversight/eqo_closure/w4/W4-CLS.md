# EQO-W4-CLS — B-2 disposition completeness (wave-11)

**Task ID:** `EQO-W4-CLS` · **Plan:** `EQO_B2_REMAINING_INVARIANTS_PLAN.md`  
**Disposition:** `PARTIAL` — registry **complete**; row-level **closure** not asserted.  
**Pin:** DOC_PIN `8cee3c5a913f234e8fb65053b6ab4f34ed78249b`

## Source artefacts

| Artefact | Role |
|----------|------|
| `EQO_B2_DISPOSITION.jsonl` | 186 B-2 row assignments |
| `EQO_B2_DISPOSITION.meta.json` | W4-01 verification snapshot |
| `w4/W4-CLS_PREP.md` | Wave-7 prep inventory |
| Batch memos | `W4-02_TRANCHE_*.md`, `W4-03_*`, `W4-04_*`, `W4-05_*`, `x/X-08/BLOCKS_IMPACT_READ.md` |

## Mechanical completeness (wave-11 read)

| Check | Result |
|-------|--------|
| Row count vs meta | `186` / `186` (`delta: 0`) |
| `verification` field | `PASS` |
| Missing `assigned_task_id` | `0` |
| Unique `finding_id` | `186` |

## Assignment histogram (registry)

| `assigned_task_id` | Rows |
|--------------------|-----:|
| EQO-W4-02 | 95 |
| EQO-W4-03 | 69 |
| EQO-W4-04 | 5 |
| EQO-W4-05 | 5 |
| EQO-X-08 | 5 |
| EQO-W2-02 | 2 |
| EQO-X-02 | 2 |
| EQO-X-03 | 1 |
| EQO-X-04 | 1 |
| EQO-X-05 | 1 |

## `next_action_source` (all rows dispositioned)

| Source | Count |
|--------|------:|
| DOC_FIX_BY_OWNER | 95 |
| OWNER_DECISION | 79 |
| REQUALIFY | 7 |
| NONE | 5 |

**Wave-11 read:** every open Campaign B-2 finding has a machine row with `assigned_task_id` and `next_action_source` — plan rule “every B-2 row has disposition” is **MET** at registry level.

## Batch artefact coverage

| Task | Wave-11 state |
|------|----------------|
| EQO-W4-02 (95) | Tranches 01–07 present — queue complete outside Campaign B |
| EQO-W4-03 (69) | `W4-03_OWNER_DECISION_BATCH.md` + English steward memo — `WAIT_FOR_OPERATOR` |
| EQO-W4-04 (5) | `W4-04_REQUALIFY_BATCH.md` |
| EQO-W4-05 (5) | `W4-05_NONE_JUSTIFICATION.md` |
| EQO-X-08 (5) | `x/X-08/BLOCKS_IMPACT_READ.md` @ DOC_PIN |

## Terminal W4-CLS gates (not met)

| Blocker | Route |
|---------|--------|
| 69 + batched OWNER_DECISION rows | Operator OD packet — no EQO self-apply |
| DOC_FIX_BY_OWNER rows without owner PR | Per-tranche `W4-02` memos — doc fix or explicit `BLOCKED` line |
| REQUALIFY / NONE rows | Evidence or `ACCEPTED_SCOPED_LIMITATION` via operator only |

**Aggregate:** W4-CLS **disposition completeness** is published; **EQO_CLOSED** is **not** claimed. EQO does not run ARCH-FREEZE PT-Q02.

## Limits and non-claims

- Registry read only — no edits to `EQO_B2_DISPOSITION.jsonl` in wave-11.
- No independence claim, no FRZ promotion, no person assignment invented in EQO.

**Ref:** `governance/tickets/T-IGX-046-F.md`
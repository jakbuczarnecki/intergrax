# EQO B-2 — Remaining findings disposition

**Parent:** `EQO-W4` · **Source:** `origin/development` → `campaign_a/findings/FINDINGS.jsonl` at `DOC_PIN`.

## Scope equation

```
OPEN (239) − CAMPAIGN_B (51) − CAMPAIGN_C (2) = 186
```

Excluded from B-2 file (handled elsewhere):

- **51** → `EQO-W3` workstreams (B-1).
- **2** → `EQO-C-01` (A3-004), `EQO-C-02` (A6-004).
- **12** → `EQO-X-*` / `EQO-W2-02` (see plan §4.4); listed in disposition, not double-counted as W4 batches.

## Machine registry

| Artifact | Task | Acceptance |
|----------|------|------------|
| `EQO_B2_DISPOSITION.jsonl` | W4-01 | 186 rows; each `assigned_task_id` unique owner task |
| `EQO_B2_DISPOSITION.meta.json` | W4-01 | `verification: PASS`, histogram |

## Batch execution (parallel with W3)

| Task | Trigger (`next_action` or rule) | Executor action |
|------|-----------------------------------|-----------------|
| **EQO-W4-02** | `DOC_FIX_BY_OWNER` (95 rows in full register; subset in B-2) | Doc PR in existing docs ticket **or** `BLOCKED` + missing owner name |
| **EQO-W4-03** | `OWNER_DECISION` (not X-08) | Append to **single** OD-EQO addendum; `WAIT_FOR_OPERATOR` |
| **EQO-W4-04** | `REQUALIFY` | Link row to `EQO-C-03` layer ID + `EQO_C_HISTORICAL_*` |
| **EQO-W4-05** | `NONE` (5) | Matrix justification **or** steward `next_action` change via EDER |
| **EQO-W4-CLS** | all batches started | Every B-2 row has disposition (not necessarily `CLOSED_WITH_EVIDENCE`) |

## Assignment histogram (W4-01 generated)

| assigned_task_id | Rows |
|------------------|-----:|
| EQO-W4-02 | 95 |
| EQO-W4-03 | 69 |
| EQO-W4-04 | 5 |
| EQO-W4-05 | 5 |
| EQO-X-08 | 5 |
| EQO-X-02 | 2 |
| EQO-W2-02 | 2 |
| EQO-X-03 | 1 |
| EQO-X-04 | 1 |
| EQO-X-05 | 1 |

## Closure rule

`ACCEPTED_SCOPED_LIMITATION` only with existing OD or new OD in the Operator package. Silence does **not** clear a row. `EQO-W4-CLS` feeds ARCH-FREEZE PT-Q02 gate observation (EQO does not run PT-Q02).
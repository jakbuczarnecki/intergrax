# EQO-E-02 — Mechanical triage report (register @ DOC_PIN)

**Source blob:** `docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl` @ `8cee3c5a913f234e8fb65053b6ab4f34ed78249b`  
**Reconciliation:** `rows_total` **1073** (1069 ingest + 4 `EDER-N-0001..0004`) — see `reconciliation.json` (EQO-E-01).  
**Action:** No in-repo `register.jsonl` at qualification branch HEAD; **mechanical rewrite deferred** — this report records counts and blockers only.

## Counts (read-only scan)

| Metric | Value |
|--------|------:|
| Rows | 1073 |
| `lifecycle_state=INGESTED` | 1073 |
| `severity=UNRATED` (or empty) | 688 |
| `next_track=UNCLASSIFIED` | 428 |
| `next_track=DOC_PATCH` | 222 |
| `next_track=ACCEPTED_LIMITATION` | 199 |
| `next_track=DECISION` | 132 |
| `next_track=VERIFY_TEST` | 41 |
| `next_track=CODE_FIX` | 31 |
| `next_track=REQUALIFY` | 20 |

## EQO-E-02 rules — applicability

| Rule | Status |
|------|--------|
| (1) Copy severity from finding when UNRATED | **BLOCKED** — needs steward write on `register.jsonl` @ DOC_PIN ancestry PR |
| (2) Map `next_action` → `next_track` | **PARTIAL** — 428 still `UNCLASSIFIED` (finding lacks mapping) |
| (3) Path byte compare TRAIN vs DOC | **NOT_RUN** — requires batch job + checkout of register write set |
| (4) Duplicate `SUPERSEDED` | **NOT_RUN** |

## Blocker

Publish `register.jsonl` updates only through an approved docs ticket on `development` (steward band B/F). EQO qualification band **does not** bulk-edit 1073 rows without steward script and independent audit sampling (E-04).

## Closure criterion (E-02)

Mechanical pass reduces `UNRATED` to rows **without** finding severity only, and `UNCLASSIFIED` to rows **without** `next_action` mapping — target counts to be recorded after steward run.
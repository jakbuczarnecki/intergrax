# STATE-X-R2 — Decision / Attempt / Lineage Concurrency & Replay Closure

**Parent:** STATE-X — Persistence, State & Recovery Certification

**Accepted children:** STATE-X-P0, STATE-X-R1

**START_HEAD:** `272f0af9c0c348febb22b3aaef5769989d454e5b`

**Status:** READY FOR AUDIT (pending independent GitHub SHA audit)

---

## Production delta (expected 0–2 files)

| File | Change |
|---|---|
| `intergrax/runtime/execution/decision_checkpoint_persistence.py` | Protocol `materialized_revision(key)` |
| `intergrax/runtime/execution/decision_recovery.py` | Terminal snapshot save uses CAS (`expected_revision`) |

Attempt / lineage production semantics unchanged on current-head evidence.

---

## Decision CAS closure

| Case | Evidence |
|---|---|
| Port declares `materialized_revision` | R2-Q01 |
| Initial CAS `expected_revision=0` → rev 1 | R2-Q02 (memory + SQLite) |
| Update CAS `expected_revision=1` → rev 2 | R2-Q03 |
| Stale concurrent writer | R2-Q04 |
| Production blind `save_decision_checkpoint` | R2-Q05 (= 0) |
| Finalization dominates snapshot | R2-Q06 |
| CAS loss after finalization commit | R2-Q07 |
| Event vs snapshot conflict types | R2-Q08 |
| Tenant key isolation | R2-Q09 |

---

## Attempt / lineage / retry

Mechanical proof: `tests/qualification/state_x/test_state_x_r2_decision_attempt_lineage.py` (R2-Q10–Q24).

**Fork disposition:** FRZ-REC-05 — N/A-WITH-EVIDENCE for R2 families (retry keeps `RunId`, new `AttemptId`; no canonical fork API in scope).

---

## Ownership graph (unchanged)

| Concern | Owner |
|---|---|
| Authoritative decision history | `DecisionEventAppendPort` |
| Materialized decision snapshot | `DecisionCheckpointPersistence` |
| Attempt transitions | `AttemptLifecycleService` |
| Lineage facts | `ExecutionLineagePersistence` |
| Retry orchestration | `ExecutionAttemptRetryService` |

---

## FRZ evidence (no PASS promotion)

Mapped to R2 qualification tests and listed replay suites; certification evidence only.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

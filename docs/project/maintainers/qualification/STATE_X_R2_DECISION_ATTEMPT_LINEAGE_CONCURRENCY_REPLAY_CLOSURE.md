# STATE-X-R2 — Decision / Attempt / Lineage Concurrency & Replay Closure

**Parent:** STATE-X — Persistence, State & Recovery Certification

**Accepted children:** STATE-X-P0, STATE-X-R1

**START_HEAD:** `272f0af9c0c348febb22b3aaef5769989d454e5b`

**Status:** READY FOR AUDIT (pending independent GitHub SHA audit)

---

## Production delta (expected 0–2 files)

| File | Change |
|---|---|
| `intergrax/runtime/execution/decision_checkpoint_persistence.py` | `MaterializedDecisionCheckpoint`, `load_materialized` |
| `intergrax/runtime/execution/decision_recovery.py` | Terminal save binds CAS to read token |

Attempt / lineage production semantics unchanged on current-head evidence.

---

## Decision CAS closure

| Case | Evidence |
|---|---|
| Typed materialized load envelope | R2-R1-Q01 |
| Initial CAS `expected_revision=0` → rev 1 | R2-Q02 (memory + SQLite) |
| Update CAS `expected_revision=1` → rev 2 | R2-Q03 |
| Stale concurrent writer | R2-Q04 |
| Production blind `save_decision_checkpoint` | R2-Q05 (= 0) |
| Finalization dominates snapshot | R2-Q06 |
| CAS loss after finalization commit | R2-Q07 |
| Event vs snapshot conflict types | R2-Q08 |
| Tenant key isolation | R2-Q09 |

---

## R2-R1 — read-token CAS closure (corrective)

| Invariant | Evidence |
|---|---|
| Snapshot + `snapshot_revision` loaded together (`load_materialized`) | R2-R1-Q02–Q06 |
| Standalone decision `materialized_revision` removed | R2-R1-Q07 |
| Stale writer rejected before finalization commit | R2-R1-Q08–Q09 |
| Race after finalization commit → CAS fail, finalization kept | R2-R1-Q10 / R2-Q07 |
| First-write `expected_snapshot_revision=0` | R2-R1-Q11 |
| Token/key binding (no laundering) | R2-R1-Q12 |

**Status:** READY FOR AUDIT (pending independent GitHub SHA audit)

---

## R2-R1-R1 — token non-detachment closure

| Invariant | Evidence |
|---|---|
| Raw `expected_snapshot_revision` removed from semantic terminal API | R2-R1-R1-Q02–Q03 |
| Typed `ExpectedDecisionSnapshotAbsence` / `ExistingMaterializedDecisionSnapshot` | R2-R1-R1-Q01 |
| Create-first via explicit absence expectation | R2-R1-R1-Q04–Q05 |
| Existing write requires full materialized envelope | R2-R1-R1-Q06 |
| Stale existing expectation before finalization | R2-R1-R1-Q07 |
| Same-key token laundering rejected | R2-R1-R1-Q08 |
| Semantic revision-state mismatch rejected | R2-R1-R1-Q09 |
| Conflicting authoritative outcome laundering rejected | R2-R1-R1-Q10 |
| Cross-tenant / cross-key expectation rejected | R2-R1-R1-Q11 |
| Race after finalization: finalization kept, snapshot CAS stale | R2-R1-R1-Q12 |
| No raw revision terminal callsites | R2-R1-R1-Q13 |
| Low-level `save_decision_checkpoint` CAS preserved | R2-R1-R1-Q14 |

**Status:** READY FOR AUDIT (pending independent GitHub SHA audit)

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

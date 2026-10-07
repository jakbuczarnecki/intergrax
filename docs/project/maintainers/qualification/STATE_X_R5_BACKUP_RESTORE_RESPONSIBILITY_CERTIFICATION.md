# STATE-X-R5 — Backup / Restore Responsibility & Recovery Consistency

| Field | Value |
| --- | --- |
| STATE_X_R5_PRE_AUDIT_HEAD | `61faf8f317125b996526ceceaf5754b9c28073d6` |
| R4 accepted closure SHA | `61faf8f317125b996526ceceaf5754b9c28073d6` |
| R5 structural baseline SHA | `3e12503b8677e06d4a032983eb1d28d35761fb4c` |
| Scope | FRZ-REC-08 — backup/restore responsibility, physical units, semantic recovery sets, restore validation |
| Production delta | **0** |
| R5 accepted closure SHA | `bcd8157065cc649412b64e9d6ada34be92d4b6a3` |
| Status (Cursor) | **STATE-X-R5 CLOSED** @ accepted SHA; **STATE-X-R6** is next mandatory child |

## R4 reconciliation

Independent audit accepted @ `61faf8f317125b996526ceceaf5754b9c28073d6`:

- **STATE-X-R4-R1-Q1** — CLOSED  
- **STATE-X-R4-R1** — CLOSED  
- **STATE-X-R4** — CLOSED  

Bookkeeping updated in roadmap, freeze checklist, and [`STATE_X_R4_TASK_CHECKPOINT_RESTORE_INTEGRITY_CERTIFICATION.md`](STATE_X_R4_TASK_CHECKPOINT_RESTORE_INTEGRITY_CERTIFICATION.md). R4-R1-Q1 evidence matrix numbering corrected vs DG-001 (not R4-Q47).

## Responsibility model

| Layer | Owner |
| --- | --- |
| Physical snapshot / restore transport | Backend / operator |
| Semantic acceptance after restore | Integrax (provider load + family validators + recovery gates) |
| Semantic truth ownership | Unchanged per SX-F01..F15 inventory |

No platform generic `backup()` / `restore()` APIs added to persistence contracts.

## Matrices (mechanical SSOT)

- Family matrix: `tests/qualification/state_x/_r5_backup_restore_support.py` → `R5_BACKUP_RESTORE_FAMILY_MATRIX`  
- Physical backup units: `PHYSICAL_BACKUP_UNIT_MATRIX`  
- Semantic recovery flows: `SEMANTIC_RECOVERY_FLOW_MATRIX`  
- FRZ-REC-08 structural completeness: `assert_frz_rec_08_r5_completeness()`  
- FRZ-REC-08 behavioral evidence registry: `assert_frz_rec_08_behavioral_evidence_complete()` → `R5_FRZ_REC_08_BEHAVIORAL_EVIDENCE`

## SQLite physical grouping

`SQLiteTaskCheckpointStore` = one file: `task_checkpoints`, `scheduled_resumes`, `scheduler_ledger`, `task_execution_terminal` (F01/F02/F13/F05 capability).

`SQLiteRuntimePersistenceBundle` = **multiple independent DB paths** — not one atomic cross-file backup unit; platform defines fail-closed semantics on skew (behaviorally qualified in R5-Q1).

## SX-F08 (RunTrace) — semantic interpretation

Enum label `NOT_DURABLE_REBUILDABLE` remains historical/coarse inventory classification.

**R5 interpretation for SX-F08:**

- Not required as canonical execution-recovery truth.  
- Loss may reduce observability only; it does not change checkpoint/terminal/idempotency truth.  
- **Automatic reconstruction is NOT certified by R5** (no qualified rebuild path on HEAD).

## FRZ mapping

| Criterion | R5 disposition |
| --- | --- |
| **FRZ-REC-08** | READY FOR INDEPENDENT CLOSURE REVIEW (structural + behavioral gates on HEAD) |
| **FRZ-REC-05** | **OPEN** — not claimed in R5 |
| FRZ-REC-01..04, 06, 07, 09, 10 | Supporting replay only |
| FRZ-STA-01..03, 06, 07, 08 | Supporting |
| FRZ-TEN-04, FRZ-TEN-08 | Scoped tenant evidence |

## R5-Q1 cross-store restore skew (behavioral)

| Scenario | Evidence class | Test |
| --- | --- | --- |
| Old checkpoint + newer terminal (separate stores) | **BEHAVIORAL** | `test_r5_q1_old_checkpoint_newer_terminal_restore_skew_fails_closed` |
| Corrupt terminal authority ≠ absence | **BEHAVIORAL** | `test_r5_q1_terminal_backend_corruption_is_not_treated_as_absence` |
| Old compensation queue + newer idempotency (SQLite restore) | **BEHAVIORAL** | `test_r5_q1_old_compensation_queue_newer_idempotency_no_duplicate_effect` |
| RunTrace non-authoritative; no rebuild guarantee | **BEHAVIORAL** + **STRUCTURAL** | `test_r5_q1_run_trace_loss_is_non_authoritative_not_claimed_rebuildable` |
| Compensation absence cannot infer COMPLETED | **RESPONSIBILITY CONTRACT** + **BEHAVIORAL** skew test above | SX-F11 matrix + Q1 scenario B |

### Scenario A timeline (checkpoint ↔ terminal)

```text
T1 — SQLiteTaskCheckpointStore: resumable checkpoint (WAITING_FOR_HUMAN)
T2 — InMemoryExecutionTerminalStore (separate): COMPLETED for same tenant/task/run
Recovery — LongRunningCoordinator.restore_if_resuming(..., execution_terminal=...)
RESULT — CheckpointResumeValidationError; live Task unchanged; no execution
```

### Scenario B timeline (compensation ↔ idempotency)

```text
T1 — compensation.db snapshot: job J = PENDING
T2 — canonical side-effect execute; idempotency COMPLETED; handler.calls = 1
T3 — restore compensation.db from T1; idempotency.db remains T2
T4 — drain_pending_compensation_jobs(...)
RESULT — handler.calls = 1; queue not left silently PENDING for re-execution
```

## R5-Q01..Q40 evidence

| Q | Evidence class | Evidence |
| --- | --- | --- |
| Q01–Q09 | STRUCTURAL | `test_r5_q01` … `test_r5_q09` |
| Q10–Q13 | BEHAVIORAL | SQLite operator file restore |
| Q14 | STRUCTURAL | bundle fail-closed text guard (+ Q1 behavioral skew) |
| Q15–Q18 | STRUCTURAL / replay | terminal/lineage/attempt/budget |
| Q19 | BEHAVIORAL | executes `test_r3_r2_q32_…` on HEAD |
| Q20–Q21 | STRUCTURAL + CONTRACT | compensation responsibility |
| Q22–Q24 | STRUCTURAL / replay | HITL/agent |
| Q25–Q27 | STRUCTURAL | F07/F08/F15 classification |
| Q28–Q30 | BEHAVIORAL | corruption + tenant |
| Q31–Q34 | BEHAVIORAL / STRUCTURAL | identity/authority |
| Q35–Q36 | STRUCTURAL | import gates |
| Q37–Q41 | STRUCTURAL | tenant audit + FRZ-REC-08 gates |
| **Q1 (R5-Q1 child)** | **BEHAVIORAL** | `_r5_q1_cross_store_restore_tests.py` |

## FRZ-REC-08 evidence table (§50)

| Requirement | Evidence class | Exact evidence |
| --- | --- | --- |
| responsibility complete | STRUCTURAL | `assert_frz_rec_08_r5_completeness()` |
| physical unit complete | STRUCTURAL | `PHYSICAL_BACKUP_UNIT_MATRIX` |
| restore roundtrip | BEHAVIORAL | Q10–Q13 |
| cross-store skew | BEHAVIORAL | R5-Q1 scenario A/B |
| corrupt state | BEHAVIORAL | Q28, Q1 corrupt terminal |
| authority preservation | BEHAVIORAL | Q32 + R3/R4 replay |
| tenant isolation | BEHAVIORAL | Q30–Q31 + R4 replay |
| provider responsibility | RESPONSIBILITY CONTRACT | family matrix |
| projection semantics (F08) | STRUCTURAL + BEHAVIORAL | inventory + Q1 F08 test |

## Regression replay (current HEAD)

```text
uv run --with cryptography pytest tests/qualification/state_x -k "r5_q1" -p no:xdist -q
uv run --with cryptography pytest tests/qualification/state_x/_r3_r2_qualification_tests.py -k "q32 or q30 or q31" -p no:xdist -q
uv run --with cryptography pytest tests/qualification/state_x/test_state_x_r4_task_checkpoint_restore.py tests/qualification/state_x/_r4_r1_restore_consumer_convergence_tests.py -p no:xdist -q
uv run --with cryptography pytest tests/qualification/state_x/test_state_x_r5_backup_restore.py -p no:xdist -q
uv run --with cryptography pytest tests/qualification/state_x -p no:xdist -q
```

(Exact pass/skip counts recorded in session log under `.tmp/session/state-x-r5-q1/`.)

## Tenant Isolation Audit

```text
tenant scope applicable: YES
canonical tenant identity: per-family tenant key from STATE_X_FAMILY_INVENTORY / R5 matrix tenant_behavior
tenant owner: each semantic family contract; backup operator is not tenant owner
propagation path: physical backend restore → provider load → family validator → cross-family recovery gate → execution/recovery consumer
state isolation: restored data remains partitioned by original tenant identity
provider/config isolation: backup unit preserves backend tenant partition/key semantics
evidence/trace isolation: restored evidence remains tenant-bound and cannot become execution authority
async/recovery continuity: restore preserves validated tenant across resume/retry/recovery
cross-tenant path: restored tenant-A truth cannot be consumed as tenant-B truth (R5-Q30/Q31, R4 replay, R5-Q1 skew)
fail-closed behavior: mismatch/corruption/incomplete required state is rejected
adversarial evidence: test_r5_q28..q31, R4/R3 replays, R5-Q1 behavioral tests
result: PASS
```

## Enterprise audit matrix (§51)

| Area | Result |
| --- | --- |
| R5 structural matrix | PASS |
| physical backup units | PASS |
| semantic recovery sets | PASS |
| checkpoint-terminal skew runtime proof | PASS |
| no terminal resurrection | PASS |
| compensation-idempotency skew runtime proof | PASS |
| no duplicate external effect | PASS |
| missing compensation not silent completion | PASS |
| corrupt restore fail-closed | PASS |
| restored HumanDecision no authority mint | PASS (R3/R4 replay) |
| AgentCheckpoint identity continuity | PASS (R3-R5 replay) |
| tenant isolation | PASS |
| authority non-expansion | PASS |
| identity preservation | PASS |
| F08 non-authoritative | PASS |
| F08 no unsupported rebuild promise | PASS |
| no backup architecture expansion | PASS |
| FRZ-REC-08 evidence completeness | PASS |
| regression protection | PASS |

## Findings

| Classification | Count |
| --- | --- |
| IN-SCOPE BLOCKER | 0 |
| TRACKED FREEZE DEBT | unchanged (FRZ-REC-05 next child) |
| ENVIRONMENT/TEST ISSUE | Redis idempotency skips in R3-R2 (pre-existing) |

## R5-Q1 closure @ `bcd8157065cc649412b64e9d6ada34be92d4b6a3`

Independent audit accepted:

- **STATE-X-R5-Q1** — CLOSED  
- **STATE-X-R5** — CLOSED  
- **FRZ-REC-08** — PASS (checklist scoped)

## Recommended status

```text
STATE-X-R5 = CLOSED @ bcd8157065cc649412b64e9d6ada34be92d4b6a3
STATE-X-R6 = CURRENT / MANDATORY
STATE-X = CURRENT
FRZ-REC-08 = PASS @ accepted SHA (scoped)
FRZ-REC-05 = OPEN (R6 primary)
TRACE-X = NOT ENTERED
```

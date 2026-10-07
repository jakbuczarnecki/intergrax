# STATE-X-R6 — Recovery / Replay / Branching Semantics

| Field | Value |
| --- | --- |
| STATE_X_R6_PRE_AUDIT_HEAD | `bcd8157065cc649412b64e9d6ada34be92d4b6a3` |
| STATE_X_R6_ACCEPTED_CLOSURE_SHA | `3e1c82f224a9f7d5a87555836c5fd80a3fdf22f7` |
| R5 accepted closure SHA | `bcd8157065cc649412b64e9d6ada34be92d4b6a3` |
| Scope | FRZ-REC-05 — explicit resume/retry/replay/partial-recovery semantics; first-class fork disposition |
| Production delta | **0** |
| Status | **STATE-X-R6 CLOSED** (independently accepted @ `3e1c82f224a9f7d5a87555836c5fd80a3fdf22f7`) |

## R5 reconciliation

Independent audit accepted @ `bcd8157065cc649412b64e9d6ada34be92d4b6a3`:

- **STATE-X-R5-Q1** — CLOSED  
- **STATE-X-R5** — CLOSED  
- **FRZ-REC-08** — PASS (scoped; not global FRZ PASS)

Bookkeeping reconciled in roadmap, freeze checklist, and [`STATE_X_R5_BACKUP_RESTORE_RESPONSIBILITY_CERTIFICATION.md`](STATE_X_R5_BACKUP_RESTORE_RESPONSIBILITY_CERTIFICATION.md).

## Operation taxonomy

Mechanical SSOT: `tests/qualification/state_x/_r6_recovery_branching_support.py`

| Kind | Supported | Owner (summary) |
| --- | --- | --- |
| RESUME | yes | LongRunningCoordinator + execution tree resume plan |
| RETRY | yes | ExecutionAttemptRetryService + AttemptLifecycleService |
| NEW_EXECUTION | yes | ExecutionIdentityAuthority + intake |
| INSPECTION_REPLAY | yes | ReplayService (read-only) |
| IDEMPOTENT_RESULT_REPLAY | yes | IdempotencyPreEffectCoordinator |
| PARTIAL_RECOVERY | yes | FanOutPartialRecoveryService |
| FIRST_CLASS_FORK | **no** | NOT SUPPORTED |

## First-class fork disposition

```text
FIRST_CLASS_HISTORICAL_EXECUTION_FORK = NOT SUPPORTED
```

No public `fork()` / `ForkRequest` / `ExecutionForkPort` on HEAD. Anti-fork AST scan: **unclassified fork-like production symbols = 0**.

## Matrices

- Identity / operation semantics: `RECOVERY_OPERATION_MATRIX`  
- Authority: `RECOVERY_AUTHORITY_MATRIX`  
- Side effects: `RECOVERY_SIDE_EFFECT_MATRIX`  
- FRZ-REC-05 structural gate: `assert_frz_rec_05_r6_completeness()`  
- Behavioral registry: `R6_FRZ_REC_05_BEHAVIORAL_EVIDENCE`

## Tenant Isolation Audit

`TENANT_ISOLATION_AUDIT_R6` — **PASS** (fail-closed; adversarial evidence via R6-Q29..Q31 + R3-R2 tenant idempotency replay).

## FRZ mapping

| Criterion | R6 disposition |
| --- | --- |
| **FRZ-REC-05** | **PASS** @ `3e1c82f224a9f7d5a87555836c5fd80a3fdf22f7` |
| FRZ-REC-01..04, 06, 07, 09, 10 | Supporting replay only (not global PASS) |
| FRZ-STA-02, 03, 05, 06, 08 | Supporting |
| FRZ-TEN-08 | Supporting (scoped) |

## Behavioral evidence (current HEAD)

| Area | Command / test |
| --- | --- |
| Resume tree | `test_r6_q05`..`q09`; `test_ue_9c_execution_tree_checkpoint.py` |
| Retry | `test_r6_q10`..`q14`; NPSC-5E/R1 suites |
| New execution mint | `test_r6_q16` |
| Inspection replay | `test_r6_q18` |
| Idempotent replay | `test_r3_r2_q23_retryable_redelivery_canonical_idempotency_replay` |
| Partial recovery bindings | NPSC-5E/R3 final qualification + `test_final_wrong_root_topology_slot_attempt_revision_blocked` |
| Durable retry gate | `test_composition_rejects_production_retry_with_in_memory_store` |
| Full STATE-X | `tests/qualification/state_x` — **559 passed**, 5 skipped (redis) |

## Findings

| Classification | Count |
| --- | --- |
| IN-SCOPE BLOCKER | 0 |
| TRACKED FREEZE DEBT | unchanged |
| ENVIRONMENT/TEST ISSUE | NPSC R3 embedded `uv run pytest` subprocess harness requires `--with cryptography` when collected (pre-existing); behavioral tests pass when run directly |

## Recommended status

```text
STATE-X-R6 = READY FOR AUDIT
STATE-X = CURRENT
FRZ-REC-05 = READY FOR INDEPENDENT CLOSURE REVIEW
next STATE-X stage = NOT ENTERED
TRACE-X = NOT ENTERED
```

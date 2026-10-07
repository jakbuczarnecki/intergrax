# STATE-X-R4 — Task Checkpoint Restore Integrity & Recovery Authority

| Field | Value |
| --- | --- |
| PRE_AUDIT_HEAD / START_HEAD (R4) | `716ed746f8b463681db536804235cedc86adc162` |
| **Independent acceptance SHA (R4 / R4-R1 / R4-R1-Q1)** | **`61faf8f317125b996526ceceaf5754b9c28073d6`** |
| R4-R1 START_HEAD | `6a8252548a8fb81830afac8534acbd5183f9df95` |
| STATE-X-R4-R1-Q1 START_HEAD | `08c9be388e8cfbe1c0271f153497aa748db356b6` |
| Scope | TaskCheckpoint → validated restore → Task materialization → execution re-entry |
| Canonical restore validator | `intergrax/runtime/long_running/checkpoint_resume_validation.py` |
| Canonical persistence owner | `TaskCheckpointPersistence` |
| Status (independent audit) | **CLOSED** @ `61faf8f317125b996526ceceaf5754b9c28073d6` |
| Status (Cursor bookkeeping) | Reconciled; **STATE-X-R5** is current mandatory child |

## Prior accepted reconciliation

Independent acceptance @ `716ed746f8b463681db536804235cedc86adc162`:

- STATE-X-R3-R3 / SX-F12 — CLOSED (bookkeeping)
- STATE-X-R3-R4 / SX-F13 — CLOSED (bookkeeping)
- STATE-X-R3-R5 / SX-F14 — CLOSED (bookkeeping)

## Before / after restore graph

**Before:** checkpoint → `Task.model_validate(task_snapshot)` → optional `is_checkpoint_resumable`.

**After:** checkpoint → `assert_checkpoint_resume_materialization_eligible` (structural) → `Task.model_validate` → execution paths with `assert_checkpoint_resume_eligible` where current authority applies (`restore_if_resuming`, orchestration).

## Closed-world restore inventory

See `tests/qualification/state_x/_r4_task_checkpoint_restore_support.py` (`RESTORE_CONSUMER_INVENTORY`).

## Q2-D1 disposition

**PASS (local):** `terminal_capability_from_task_checkpoint_store()` in `execution_terminal/persistence.py` replaces ad-hoc `isinstance` narrowing in `nexus_loop.py`. No persistence ownership change.

## Backup / restore responsibility (next child only)

| Store | Responsibility |
| --- | --- |
| `TaskCheckpointPersistence` / SQLite default | BACKEND/OPERATOR WITH EXPLICIT PLATFORM CONSISTENCY REQUIREMENTS |
| Execution terminal via checkpoint capability | component of checkpoint store when capability implemented |
| Global backup engine | NOT ENTERED — STATE-X-R5 DR child |

## FRZ mapping (scoped evidence only)

Direct: FRZ-STA-01/02/04/05/06/08; FRZ-REC-01/02/03/04/06/09/10; FRZ-TEN-08.  
Not claimed: FRZ-REC-05, FRZ-REC-08, global FRZ PASS, STATE-X parent CLOSED.

## STATE-X-R4-R1 — Closed-World Restore Consumer Convergence

Canonical structural reader: `validated_task_snapshot_from_checkpoint()` in `checkpoint_resume_validation.py` (composes schema, identity binding, snapshot integrity). Execution/resume paths retain `assert_checkpoint_resume_materialization_eligible` / `assert_checkpoint_resume_eligible`.

Consumer inventory: `tests/qualification/state_x/_r4_r1_restore_consumer_support.py` (`R4_R1_TASK_CHECKPOINT_CONSUMERS`). Raw `Task.model_validate(checkpoint.task_snapshot)` outside allowlist: **0** (`find_raw_task_snapshot_parsers()`).

### Nexus / worker recovery (internal mechanism)

`NexusWorkerRuntime` is an **internal execution-engine mechanism**, not a public platform contract. Its incoming `BackgroundExecutionIdentity` acts as the worker/recovery envelope. After canonical `TaskCheckpoint` restore validation, the internal engine continues the durable logical execution identity stored in the checkpoint (`run_id` / `attempt_id` / root `execution_id`) while preserving the incoming tenant scope. This behavior is an internal realization of platform recovery semantics and does **not** define a public Nexus recovery contract. No `WorkerRecoveryContract`, `NexusRecoveryContract`, or public resume-identity API was introduced for qualification.

Behavioral evidence (no patch of `_reconcile_resume_identity`):

- `test_r4_r1_q1_worker_recovery_positive_runtime_reconciles_identity`
- `test_r4_r1_worker_recovery_cross_tenant_checkpoint_denied`
- `test_r4_r1_worker_recovery_wrong_task_checkpoint_denied`
- `test_r4_r1_worker_recovery_invalid_checkpoint_fail_closed`

Prior report **Q47**: documentation numbering error (DG-001 lineage matrix), **not** STATE-X-R4 mechanical Q01–Q40. R4-R1-Q1 evidence matrix below maps **R4-R1-Q1** pytest rows; R4 qualification file uses overlapping `test_r4_q*` names — do not conflate DG-001 with R4-Q47.

## STATE-X-R4-R1-Q1 — behavioral evidence matrix (R4-Q01..Q40)

Executed on current HEAD (`08c9be388e8cfbe1c0271f153497aa748db356b6` baseline). Mandatory rows map to **executed** pytest names (not `hasattr` / directory checks alone).

| Q | Evidence test(s) | Batch |
| --- | --- | --- |
| Q01 | `test_r4_q01_closed_world_restore_consumer_inventory` | `pytest tests/qualification/state_x/test_state_x_r4_task_checkpoint_restore.py tests/qualification/state_x/_r4_task_checkpoint_restore_qualification_tests.py` |
| Q02 | `test_r4_q02_task_checkpoint_persistence_single_owner` | same |
| Q03 | `test_r4_q03_single_canonical_restore_validator_owner` | same |
| Q04 | `test_r4_q04_no_task_materialization_before_restore_validation` | same |
| Q05 | `test_r4_q05_empty_task_snapshot_rejected` | same |
| Q06 | `test_r4_q06_missing_task_id_in_snapshot_rejected` | same |
| Q07 | `test_r4_q07_missing_tenant_id_in_snapshot_rejected` | same |
| Q08 | `test_r4_q08_snapshot_task_mismatch_rejected` | same |
| Q09 | `test_r4_q09_snapshot_tenant_mismatch_rejected` | same |
| Q10 | `test_r4_q10_unsupported_task_checkpoint_schema_rejected` | same |
| Q11 | `test_r4_q11_unsupported_runtime_checkpoint_schema_rejected` | same |
| Q12 | `test_r4_r1_q12_suspended_reentry_uses_canonical_validation` | `pytest tests/qualification/state_x/_r4_r1_restore_consumer_convergence_tests.py` |
| Q13 | `test_r4_q13_stale_checkpoint_rejected` | R4 qualification file |
| Q14 | `test_r4_q14_tenant_a_checkpoint_tenant_b_resume_denied` | R4 qualification file |
| Q15 | `test_r4_q15_task_a_checkpoint_task_b_resume_denied` | R4 qualification file |
| Q16 | `test_r4_q16_historical_authority_cannot_mint_without_current` | R4 qualification file |
| Q17 | `test_r4_q17_resume_authority_cannot_exceed_current` | R4 qualification file |
| Q18 | `test_r4_q18_narrowed_resume_within_historical_bound` | R4 qualification file |
| Q19 | `test_r4_q19_terminal_checkpoint_state_rejected` | R4 qualification file |
| Q20 | `test_r4_r1_q20_execution_tree_identity_mismatch_rejected` | R4-R1 convergence |
| Q21 | `test_r4_r1_q21_grant_lifecycle_validates_before_projection` | R4-R1 convergence |
| Q22 | `test_r4_r1_q22_missing_degraded_lineage_fail_closed` | R4-R1 convergence |
| Q23 | `test_r4_r1_q23_sealed_wrong_root_lineage_fail_closed` | R4-R1 convergence |
| Q24 | `test_r4_q24_scheduler_validates_before_task_materialization` | R4 qualification file |
| Q25 | `test_r4_r1_q25_scheduled_metadata_authority_negative_replay` → replays `test_r3_r4_q20_q21_*`, `q22`, `q23` | R4-R1 + `pytest tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py -k r3_r4` |
| Q26 | `test_r4_r1_q26_post_authorization_stale_checkpoint_denied` → replays `test_taskcpm_r14`, `r15`, `r16` | `pytest tests/unit/applications/test_task_control_governed_resume.py` |
| Q27 | `test_r4_r1_q27_forged_pause_id_denied` → `test_taskcpm_r17_hitl_pause_id_anti_forgery_still_enforced` | task-control file |
| Q28 | `test_r4_r1_q28_forged_human_request_id_denied` → `test_taskcpm_r17b_hitl_human_request_id_anti_forgery_still_enforced` | task-control file |
| Q29 | `test_r4_r1_q29_missing_approver_evidence_denied` → `test_taskcpm_r21_operator_resume_missing_approver_zero_runner` | task-control file |
| Q30 | `test_r4_r1_q1_worker_recovery_positive_runtime_reconciles_identity` | R4-R1 convergence (`-k worker_recovery`) |
| Q31 | `test_r4_r1_q31_worker_cross_tenant_checkpoint_impossible` + `test_r4_r1_worker_recovery_cross_tenant_checkpoint_denied` | R4-R1 convergence |
| Q32 | `test_r4_q32_worker_preserves_incoming_tenant` (structural) + worker positive tenant assertion in Q30 | R4 qualification + Q30 |
| Q33 | `test_r4_r1_q33_q2_d1_regression_green` | R4-R1 convergence |
| Q34 | `test_r4_r1_q34_r3_r4_scheduler_regression_green` | full `pytest tests/qualification/state_x -k r3_r4` |
| Q35 | `test_r4_q35_q2_d1_terminal_capability_helper` | R4 qualification file |
| Q36 | `test_r4_q36_restore_semantics_not_sqlite_specific` | R4 qualification file |
| Q37 | `test_r4_q37_r3_r4_scheduler_regression_import` + full STATE-X green | `pytest tests/qualification/state_x` |
| Q38 | `test_r4_q38_r3_r5_checkpoint_regression_import` | R4 qualification file |
| Q39 | `test_r4_r1_q39_raw_parser_regression_gate_green` | R4-R1 convergence |
| Q40 | `test_r4_q40_tenant_isolation_adversarial_no_cross_tenant_materialization` | R4 qualification file |

## Tests

`uv run --with cryptography pytest tests/qualification/state_x -k "r4_r1 or worker_recovery" -p no:xdist -q`
`uv run --with cryptography pytest tests/unit/applications/test_task_control_governed_resume.py -p no:xdist -q`
`uv run --with cryptography pytest tests/qualification/state_x/test_state_x_r4_task_checkpoint_restore.py tests/qualification/state_x/_r4_r1_restore_consumer_convergence_tests.py -p no:xdist -q`
`uv run --with cryptography pytest tests/qualification/state_x -p no:xdist -q`

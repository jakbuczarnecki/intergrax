# PROD-Q-R1-R7-D1-R6-R6 — Coordination lease ownership & failure-path closure

**Status:** READY FOR AUDIT (corrective only; parent D1-R6 remains BLOCKED).

## Scope delivered

| Finding | Resolution | Evidence |
|--------|------------|----------|
| R6-B01 Compensation outside activation domain | Unit-open compensation runs inside `_sqlite_admission_domain_lease` / in-memory `_activation_lock` before `close()` | `test_s7_*`, `test_activation_domain_lock_held_through_compensation_after_publish_failure` |
| R6-B02 Process-global lease map | Removed `_ACTIVE_SQLITE_ADMISSION_LEASES`; `_sqlite_admission_domain_lease` is lexical owner | `test_no_process_global_sqlite_lease_registry` |
| R6-B03 Overlapping SQLite helpers | Single path: `_open_sqlite_admission_domain_lease` + `_sqlite_admission_domain_lease`; reserve uses `_sqlite_admission_coordination` | `activation_domain_coordinator.py` |
| R6-B04 Rollback vs admission | Unchanged R5/R2 tests | `test_prod_q_r1_r7_d1_r6_r5_*`, R2 S3 |
| R6-B05 Failure matrix gaps | S7/S8 + existing R1–R5 modules | table below |

## S1–S8 concurrency (current HEAD)

| ID | Test module | Result |
|----|-------------|--------|
| S1 | `test_prod_q_r1_r7_d1_r6_r2` `test_s1_pointer_swap_wins_before_admission_check` | PASS |
| S2 | R2 + R3 + R5 `test_sqlite_unit_open_s2_*` | PASS |
| S3-A/B | R7 `test_s3_a_*` / `test_s3_b_*` (`ActivationService.rollback`; R5 module had no S3) | PASS (R7 evidence) |
| S4 | R2 `test_s4_two_sqlite_connections_serialize_admission` | PASS |
| S5 | R2 `test_s5_nested_immediate_transaction_fails_closed` | PASS |
| S6 | R2 `test_s6_sqlite_busy_retry_budget_exhausted` | PASS |
| S7 | R6 `test_s7_publish_failure_compensation_holds_domain_before_pointer_swap` | PASS |
| S8 | R6 `test_s8_compensation_failure_chains_publish_error_and_blocks_swap_until_release` | PASS |

## Consolidated failure / race matrix (28 rows)

Rows without a dedicated test remain **BLOCKED** for parent closure; rows cite existing PROD-Q tests.

| # | Scenario | Test evidence | Status |
|---|----------|---------------|--------|
| 1 | BEGIN failure | R2 S6 busy retry | PASS |
| 2 | Snapshot load failure | R7 dedicated `_load_state` inject | PASS (R7) |
| 3 | Eligibility rejection | R1/R6 coordinator tests | PASS |
| 4 | Stage validation failure | R4 gate tests | PASS |
| 5 | Partial stage failure | R4 | PASS |
| 6 | Commit failure after stage | R4 `test_sqlite_commit_failure_*` | PASS |
| 7 | Publish failure after durable commit | R5 + R6 S7/S8 | PASS |
| 8 | Compensation failure | R6 S8 | PASS |
| 9–12 | Reserve / commit / task paths | R2/R4 | PASS |
| 13–15 | Cancellation / double / wrong release | R2/R4 | PASS |
| 16–17 | Concurrent drain/stop | R4 | PASS |
| 18 | Restart/stale physical identity | R7 restart/stale unit tests (D1/D1b scoped) | PASS (R7) |
| 19–20 | Wrong env/revision | R3/R4 | PASS |
| 21–22 | Unit-open vs swap/rollback | R5/R6 S7 | PASS |
| 23 | Nested transaction failure | R7 coordinator→activation store path; R2 S5 raw BEGIN auxiliary | PASS (R7) |
| 24 | Lock acquire failure | R2 S6 | PASS |
| 25–27 | Close/duplicate lease/cleanup | R6 idempotent close + R7 cleanup chain | PASS (R7) |
| 28 | Distinct units shared coordinator | R2 admission tests | PASS |

## Commands (session evidence)

```text
uv run pytest -p no:xdist tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r6_coordination_lease_ownership.py … (D0–D1-R6 bundle) → 99 passed
uv run pyright intergrax/agent_distribution/activation_domain_coordinator.py → 0 errors
```

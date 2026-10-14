# PROD-Q-R1-R7-D1-R6-R7 — Lease cleanup, rollback & admission evidence

**Status:** READY FOR AUDIT (corrective child; parent D1-R6 remains BLOCKED).

## Corrective findings addressed

| ID | Resolution | Code / test |
|----|------------|-------------|
| R7-B01 | Acquisition cleanup uses `_cleanup_failed_sqlite_admission_acquisition` + lock release in `finally` path; lock not gated on rollback/close success | `activation_domain_coordinator.py`, R7 fault tests |
| R7-B02 | `commit_durable` fail-closed; uncertain outcome when rollback after failed commit also fails | `commit_durable`, `test_commit_durable_uncertain_when_rollback_also_fails` |
| R7-B03 | S3-A/B use `ActivationService.rollback` + SQLite coordinator | `test_s3_a_*`, `test_s3_b_*` |
| R7-B04 | Dedicated `_load_state` inject tests | `test_load_state_operational_error_*`, `test_load_state_generic_error_*` |
| R7-B05 | Restart/stale unit fail-closed (D1/D1b scope, no D3) | `test_restart_stale_*`, `test_durable_serving_without_gate_*` |
| R7-B06 | Rows #25–27 close/duplicate/cleanup | `test_lease_close_*`, `test_sqlite_lease_close_is_idempotent` (R6), R7 cleanup chain tests |
| R7-B07 | Real nested path: lease + `activation_store.atomic_commit_rollback` → fail-closed (no deadlock) | `SqliteAgentDistributionDatabase._acquire_activation_domain_lock`, `test_nested_activation_store_mutate_*` |

## Consolidated 28-case matrix

| # | Scenario | Test | Status |
|---|----------|------|--------|
| 1 | BEGIN failure / busy retry | R2 `test_s6_sqlite_busy_retry_budget_exhausted` | PASS |
| 2 | Snapshot load failure | R7 `test_load_state_operational_error_releases_lock`, `test_load_state_generic_error_releases_lock` | PASS |
| 3 | Eligibility rejection | R1/R6 coordinator tests | PASS |
| 4 | Stage validation failure | R4 | PASS |
| 5 | Partial stage failure | R4 | PASS |
| 6 | Commit failure after stage | R4 `test_sqlite_commit_failure_*` | PASS |
| 7 | Publish after durable commit | R5/R6 S7/S8 | PASS |
| 8 | Compensation failure | R6 S8 | PASS |
| 9–12 | Reserve / commit / task paths | R2/R4 | PASS |
| 13–15 | Cancellation / double / wrong release | R2/R4 | PASS |
| 16–17 | Concurrent drain/stop | R4 | PASS |
| 18 | Restart/stale physical identity | R7 `test_restart_stale_physical_unit_fail_closed_admission`, `test_durable_serving_without_gate_authorization_cannot_reserve` | PASS (D1/D1b scoped) |
| 19–20 | Wrong env/revision | R3/R4 | PASS |
| 21–22 | Unit-open vs swap/rollback | R5/R6 S7 | PASS |
| 23 | Nested coordinator/store | R7 `test_nested_activation_store_mutate_while_lease_holds_fails_closed` (R2 S5 raw BEGIN superseded for ownership path) | PASS |
| 24 | Lock acquire failure | R2 S6 | PASS |
| 25 | Connection close failure on finalize | R7 `test_lease_close_connection_failure_still_releases_lock` | PASS |
| 26 | Duplicate `close()` | R6 `test_sqlite_lease_close_is_idempotent` | PASS |
| 27 | Cleanup preserves primary + secondary | R7 `test_acquire_cleanup_rollback_failure_*`, `test_acquire_primary_and_cleanup_both_visible` | PASS |
| 28 | Distinct units shared coordinator | R2 admission tests | PASS |

### S3 rollback / admission (ActivationService)

| ID | Test | Status |
|----|------|--------|
| S3-A | `test_s3_a_activation_service_rollback_before_b_admission` | PASS |
| S3-B | `test_s3_b_rollback_waits_for_inflight_sqlite_admission` | PASS |

## Commands (session evidence)

```text
uv run pytest -p no:xdist tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r7_lease_cleanup_rollback_admission_evidence.py tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r{1..6}_*.py tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_activation_domain_coordinator.py tests/unit/agent_distribution/test_agent_distribution_rollback.py → 75 passed
uv run pyright intergrax/agent_distribution/activation_domain_coordinator.py → 0 errors
```

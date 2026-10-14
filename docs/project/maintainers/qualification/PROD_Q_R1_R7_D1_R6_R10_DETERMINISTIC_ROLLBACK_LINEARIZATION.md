# PROD-Q-R1-R7-D1-R6-R10 — Deterministic rollback linearization & D1-R6 final qualification

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (corrective evidence; does **not** close D1-R6, R10, or parents) |
| **START_HEAD** | `39663902d989bb1dc02cac8cbaa34e3f90108c24` |
| **FINAL_COMMIT** | `5f9e737bce3d8a93c2ac298cfce8ab96c364be3b` |
| **Parent** | **D1-R6-R9** findings addressed; **D1-R6** remains **REJECTED / pending independent closure** |
| **ADR** | **ADR-01-R6** — **DEC-09** / **DEC-09-EXT** (ratified; unchanged) |

## R10 corrective findings

| ID | Finding | Resolution | Evidence |
|----|---------|------------|----------|
| **R10-B01** | S3-B did not prove rollback **lock-acquisition attempt** before physical reserve | Test-scoped wrapper on `SqliteAgentDistributionDatabase._acquire_activation_domain_lock` (canonical boundary shared with admission lease + `mutation_transaction`); ordered event trace | `test_s3_b_rollback_waits_for_inflight_sqlite_admission` |
| **R10-B02** | D1-R6 exit criteria dispersed | Consolidated matrix § below @ **FINAL_COMMIT** | This document |
| **R10-B03** | Pyright not independently replayed | Independent replay on coordinator/SQLite/deployment/VENV core scope | § Pyright |
| **R10-B04** | D1/D1b reconstruction overclaim risk | Scoped **fail-closed gate** qualification only; **no** R7-D3 full recovery claim | R7 `test_restart_stale_*`, `test_durable_serving_without_gate_*` |

### S3-B event partial order (deterministic)

Recorded trace steps (indices strictly increasing):

1. `admission_eligibility_verified`
2. `rollback_acquire_attempted` (rollback thread @ `_acquire_activation_domain_lock`, before blocking while admission holds lock)
3. `physical_reservation_completed` (real `VenvBundleColocatedRuntimeDeploymentAdapter.reserve_unit_execution`)
4. `admission_coordination_completed`
5. `rollback_completed` (`ActivationService.rollback` after admission releases domain lock)

**Non-claims:** instrumentation is test-only; production synchronization unchanged.

## S1–S8 replay @ HEAD

| Scenario | Test | Operation | Interleaving | Lock owner during race | Physical | Canonical | Result |
|----------|------|-----------|--------------|------------------------|----------|-----------|--------|
| **S1** | R2 `test_s1_pointer_swap_wins_before_admission_check` | pointer CAS before admit | controlled | coordinator domain | no reserve | B→A swap | **PASS** |
| **S2 in-mem** | R3/R4 `test_s2_admission_wins_concurrent_in_memory` | admit vs swap | barrier | admission thread | in-flight survives swap | pointer moved post-reserve | **PASS** |
| **S2 SQLite** | R2 `test_s2_admitted_in_flight_survives_post_coordination_pointer_swap` | SQLite admit + swap | barrier | admission lease | VENV in-flight | post-swap reject new | **PASS** |
| **S3-A** | R7 `test_s3_a_activation_service_rollback_before_b_admission` | `ActivationService.rollback` then admit | sequential | n/a | 0 in-flight | rev-a | **PASS** |
| **S3-B** | R7 `test_s3_b_rollback_waits_for_inflight_sqlite_admission` | concurrent rollback + admit | events @ lock boundary | T1 admission lease | 1 in-flight through rollback | rev-a; B reject | **PASS** |
| **S4** | R2 `test_s4_two_sqlite_connections_serialize_admission` | two connections | serialized | one lease at a time | single reserve | consistent | **PASS** |
| **S5** | R7 `test_nested_activation_store_mutate_while_lease_holds_fails_closed` | nested store mutate | lease held | admission | fail-closed | no mutation | **PASS** |
| **S6** | R2 `test_s6_sqlite_busy_retry_budget_exhausted` | busy retry | injected busy | n/a | n/a | fail-closed | **PASS** |
| **S7** | R6 `test_s7_publish_failure_compensation_holds_domain_before_pointer_swap` | compensation before swap | failure inject | coordinator | compensated | no swap | **PASS** |
| **S8** | R6 `test_s8_compensation_failure_chains_publish_error_and_blocks_swap_until_release` | compensation failure | failure inject | coordinator | held until release | blocked swap | **PASS** |

## Closed-world synchronization (A1 SQLite bundle)

| Operation | Owner | Caller | Contract | Lock / txn | Physical | Exit invariant |
|-----------|-------|--------|----------|------------|----------|----------------|
| Domain lock | `SqliteAgentDistributionDatabase._activation_domain_sync` | `_acquire_activation_domain_lock` | single non-reentrant mutex | process-wide per DB | none | nested acquire → `ServingExecutionAdmissionError` |
| Admission lease | `_open_sqlite_admission_domain_lease` | `SqliteActivationDomainCoordinator` | `BEGIN IMMEDIATE` snapshot | domain lock + conn txn | after verify | commit or rollback+release lock |
| Rollback durable | `mutation_transaction` | `atomic_commit_rollback` via `ActivationService.rollback` | same `_acquire_activation_domain_lock` | domain lock + IMMEDIATE | drain side-effects post-commit | pointer → prior revision |
| Physical reserve | VENV adapter | `coordinate_physical_execution_reservation` callback | `PhysicalExecutionReservation` | under admission lease | `reserve_unit_execution` | release on failure after reserve |

Admission and rollback **share** `bundle.database._activation_domain_sync` under supported A1 SQLite store bundle topology (`build_activation_domain_coordinator_for_sqlite_bundle`).

## D1-R6 exit criteria ledger (current-HEAD symbols)

| Criterion | Canonical rule | Production symbol | Test | Assertion | Status |
|-----------|----------------|-------------------|------|-----------|--------|
| Pointer eligibility | admit only ACTIVE traffic revision | `verify_serving_execution_eligibility_from_state` | R1/R6 coordinator tests | `ServingExecutionAdmissionError` on mismatch | **PASS** |
| One coordinator authority | no shadow grant registry | `ActivationDomainCoordinator` | R1 corrective, R9 equivalence | no grant nonce paths | **PASS** |
| Transaction snapshot | SQLite IMMEDIATE per operation | `_sqlite_admission_coordination` | R2 S4, R8 | serialized writers | **PASS** |
| Reserve ordering | verify before `physical_reserve` | `SqliteActivationDomainCoordinator.coordinate_physical_execution_reservation` | R2/R4 | reserve after verify | **PASS** |
| Stage/commit/publish | unit-open durable then publish | `coordinate_unit_serving_admission_open` | R5/R6 S7 | pointer CAS guarded | **PASS** |
| Compensation | failed open compensated | `_compensate_unit_open_after_coordination_failure` | R6 S7/S8 | physical reverted | **PASS** |
| SQLite busy bounded | retry budget | `_SQLITE_BUSY_MAX_ATTEMPTS` | R2 S6 | exhausted → error | **PASS** |
| Rollback ordering | rollback vs in-flight admit | `ActivationService.rollback` + domain lock | R7 S3-A/S3-B | S3-B trace | **PASS** |
| Drain/stop | adapter lifecycle | `ActivationService` + VENV adapter | R4, rollback tests | drain/stop errors surfaced | **PASS** |
| Stale unit fail-closed | no admit on stopped unit | VENV gate + coordinator | R7 restart tests | match `stopped` | **PASS** (scoped) |
| A1 topology | shared DB across stores | `build_activation_domain_coordinator_for_sqlite_bundle` | R9, topology tests | mismatch → error | **PASS** |
| Unsupported wiring fail-closed | topology validation | Applications `validate_colocated_serving_admission_topology` | D1b handoff tests | reject unsupported | **PASS** |
| Per-execution release | token release accounting | VENV `PhysicalExecutionReservation` | S3-B, R2 | in_flight 1→0 | **PASS** |
| Typed contracts | Protocol stores/coordinator | `stores.py`, `ActivationDomainCoordinator` | R9 equivalence | provider-neutral | **PASS** |
| No GCA bypass | strict admission path | GCA lifecycle tests | `test_prod_q_r1_r6_strict_gca_lifecycle` | typed admission deps | **PASS** |
| Tenant/env identity | request identity match | `ServingExecutionAdmissionRequest` | R3/R4 negatives | wrong env rejected | **PASS** (local) |

**28-case fault matrix:** unchanged mapping from [R7 qualification](PROD_Q_R1_R7_D1_R6_R7_LEASE_CLEANUP_ROLLBACK_ADMISSION.md) § Consolidated 28-case matrix; rows re-run @ HEAD in R10 regression command.

## Pyright (independent replay)

```text
uv run pyright intergrax/agent_distribution/activation_domain_coordinator.py intergrax/agent_distribution/sqlite_stores.py intergrax/agent_distribution/deployment.py intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py
→ 0 errors, 0 warnings
```

Broader `activation.py` / `stores.py` Protocol stubs report pre-existing `reportReturnType` on `@abstractmethod`-style Protocol members (25 errors) — **TRACKED FREEZE DEBT** for FRZ-TYP; not introduced by R10.

## Regression command (session)

```text
uv run pytest -p no:xdist <18 PROD-Q D1/D1b/D0/D1-R6 modules listed in R7 doc + R10 suite>
→ 155 passed, 1 warning (GCA importlib DeprecationWarning), exit 0
```

## Explicit non-claims

- Does **not** close **PROD-Q**, **PROD-Q-R1-R7**, **D1**, **D1b**, **D1-R6**, **FRZ-PRD** rows, or **R7-D3** recovery.
- Cross-process SQLite writer resistance remains outside OPTION A trust boundary.
- **Recommend:** **D1-R6 READY FOR INDEPENDENT CLOSURE REVIEW** after auditor confirms S3-B trace + matrices on **FINAL_COMMIT** SHA on GitHub.

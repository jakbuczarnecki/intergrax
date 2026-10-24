# TOOL-LIFE-X L3-R9 — Authenticated revocation & production composition evidence

| Field | Value |
| --- | --- |
| START_HEAD | `588655ce656f4b1cdb5ab084847f24af01e21ea5` |
| Scope | MAC-bound supervision, execution/recovery admission, `DURABLE_SINGLE_HOST` composition |
| Multi-host | **BLOCKED** — `SHARED_MULTI_HOST` fail-closed at composition |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Canonical owner

- **Execution/recovery revocation surface:** `intergrax/runtime/external_operations/execution_recovery_external_operation_worker_revocation.py` (`ExecutionRecoveryExternalOperationWorkerRevocation`)
- **Supervision adapter:** `intergrax/runtime/external_operations/execution_recovery_worker_execution_revocation_supervision.py` (`ExecutionRecoveryWorkerExecutionRevocationSupervision`)
- **Durable apply:** `WorkerExecutionOwnershipRevocationRecorder` → SQLite `BEGIN IMMEDIATE` (R8, unchanged linearization)

## Trust / authority

- Composition-bound MAC via `INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY` (`worker_execution_ownership_revocation_trust.py`)
- Public digest recomputation **insufficient** — store verifies MAC + freshness + supervision authority id
- **Admission:** `admit_worker_execution_commit_authority_loss` + `ExecutionRecoveryAdmittedCommitAuthorityLossPolicy` (fail-closed without scope)
- Qualification trust/mac isolated in test module — not production composition

## R9-E01–E15 (scoped)

| Exit | Disposition |
| --- | --- |
| R9-E01 | **ALREADY PROVEN** — execution/recovery modules above |
| R9-E02 | **IMPLEMENTATION REQUIRED → PASS** — MAC + admitted policy |
| R9-E03 | **VALIDATION REQUIRED → PASS** — lifecycle/lease not used in supervision |
| R9-E04 | **IMPLEMENTATION REQUIRED → PASS** — MAC at store boundary |
| R9-E05 | **ALREADY PROVEN** — digest binds tenant/op/worker/epoch/revision |
| R9-E06 | **IMPLEMENTATION REQUIRED → PASS** — `valid_until` + MAC |
| R9-E07 | **ALREADY PROVEN** — R8 SQLite TX |
| R9-E08 | **ALREADY PROVEN** — no parallel supervisor runtime |
| R9-E09 | **IMPLEMENTATION REQUIRED → PASS** — `compose_tool_life_durable_single_host_certified_owner_loss` |
| R9-E10 | **IMPLEMENTATION REQUIRED → PASS** — topology gate |
| R9-E11 | **ALREADY PROVEN** — R8 fencing regressions |
| R9-E12 | **ALREADY PROVEN** — UNKNOWN/reconcile unchanged |
| R9-E13 | **VALIDATION REQUIRED → PASS** — R9 adversarial tests + R7/R8 L2 |
| R9-E14 | **VALIDATION REQUIRED → PASS** — tenant-scoped CAS (R6) |
| R9-E15 | **VALIDATION REQUIRED → PASS** — this package |

## L1 / L2 / L3

| Gate | Command | Result |
| --- | --- | --- |
| L1 | R7 + R8 + R9 files | **40 passed** |
| L2 | R6 + R7 + R8 + R9 + Batch B/C subset | **93 passed** |
| L3 | R8 cross-process / concurrent subset | see `.tmp/session/TOOL-LIFE-X-L3-R9/l3-pytest.log` |

## FRZ (scoped, not global PASS)

| ID | Result |
| --- | --- |
| FRZ-BND-03 | PASS — single owner for physical transitions |
| FRZ-TEN | PASS — tenant-bound store + authorization |
| FRZ-REC | PASS — fail-closed without admission/MAC |
| FRZ-REL-03/04/05 | PASS — L2 regressions |
| FRZ-SEC / FRZ-GOV | PASS — forged digest/MAC rejected |

FINAL_COMMIT: set at publication.

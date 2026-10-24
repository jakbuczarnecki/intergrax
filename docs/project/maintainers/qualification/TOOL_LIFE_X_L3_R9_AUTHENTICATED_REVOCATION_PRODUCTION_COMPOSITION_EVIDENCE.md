# TOOL-LIFE-X L3-R9 — Authenticated revocation & production composition evidence

| Field | Value |
| --- | --- |
| START_HEAD | `fb40ddc78147f3bf099f97ad3c1c08801196df02` |
| Scope | MAC-bound supervision, fail-closed production loss policy, `DURABLE_SINGLE_HOST` composition factory |
| Multi-host | **BLOCKED** — `SHARED_MULTI_HOST` fail-closed at composition |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Independent audit remediation (R9-F01–F04)

| Finding | Disposition |
| --- | --- |
| **R9-F01** CRITICAL — public caller-supplied admission scope | **CORRECTED (Gate B)** — removed `admit_worker_execution_commit_authority_loss` / context-var admission from product surface; production composition defaults to `FailClosedWorkerExecutionCommitAuthorityLossPolicy`. Positive path requires explicit operator `loss_policy` (qualification-only in tests). **Missing for Gate A:** durable commit-authority-loss decision from canonical execution/recovery owner (R8 evidence § supervisor / ADR §4.6). |
| **R9-F02** replay / idempotency | **ALREADY PROVEN (R8 CAS)** + R9 idempotent repeat test; epoch-keyed ledger + `bound_store_revision` freshness on first apply |
| **R9-F03** trust / MAC | **PASS scoped** — composition-bound env key, `hmac.compare_digest`, qualification key isolated in test module, fail-closed without trust |
| **R9-F04** real host composition | **BLOCKED** — `compose_tool_life_durable_single_host_certified_owner_loss` not invoked from canonical application execution host bootstrap (factory-only qualification evidence) |

**Gate result:** **Gate B** — fail-closed correction ready for independent audit.

## Canonical owner

- **Execution/recovery revocation surface:** `execution_recovery_external_operation_worker_revocation.py` (`ExecutionRecoveryExternalOperationWorkerRevocation`)
- **Supervision adapter:** `execution_recovery_worker_execution_revocation_supervision.py` (`ExecutionRecoveryWorkerExecutionRevocationSupervision`)
- **Durable apply:** `WorkerExecutionOwnershipRevocationRecorder` → SQLite `BEGIN IMMEDIATE` (R8, unchanged linearization)
- **Certified loss staging (downstream):** `WorkerExecutionCertifiedOwnerLossCertifier` requires authoritative revocation proof (R7)

## Trust / authority

- Composition-bound MAC via `INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY` (`worker_execution_ownership_revocation_trust.py`)
- Public digest recomputation **insufficient** — store verifies MAC + freshness + supervision authority id
- **Production admission:** fail-closed loss policy; MAC key presence does **not** enable positive revocation
- Qualification trust/mac + `QualificationWorkerExecutionCommitAuthorityLossPolicy` — test modules only

## R9-E01–E15 (post-remediation)

| Exit | Disposition |
| --- | --- |
| R9-E01 | **ALREADY PROVEN** — execution/recovery modules |
| R9-E02 | **PASS (scoped)** — MAC + fail-closed production policy; positive auth requires operator loss policy |
| R9-E03 | **PASS (scoped)** — lifecycle/lease not used in supervision |
| R9-E04 | **PASS (scoped)** — MAC at store boundary |
| R9-E05 | **ALREADY PROVEN** — digest binds tenant/op/worker/epoch/revision |
| R9-E06 | **PASS (scoped)** — `valid_until` + MAC |
| R9-E07 | **ALREADY PROVEN** — R8 SQLite TX |
| R9-E08 | **ALREADY PROVEN** — no parallel supervisor runtime |
| R9-E09 | **PASS (factory)** — `compose_tool_life_durable_single_host_certified_owner_loss` (not host-wired) |
| R9-E10 | **PASS (scoped)** — topology gate |
| R9-E11 | **ALREADY PROVEN** — R8 fencing regressions |
| R9-E12 | **ALREADY PROVEN** — UNKNOWN/reconcile unchanged |
| R9-E13 | **VALIDATION REQUIRED** — L2 after remediation commit |
| R9-E14 | **ALREADY PROVEN** — tenant-scoped CAS (R6) |
| R9-E15 | **VALIDATION REQUIRED** — this package after remediation commit |

## L1 / L2 / L3

| Gate | Command | Result |
| --- | --- | --- |
| L1 | R7 + R8 + R9 unit files | see session log after remediation |
| L2 | R6 + R7 + R8 + R9 consolidated | see session log after remediation |
| L3 | R8 cross-process subset (unchanged semantics) | reuse prior log unless authority batch changes L3 scope |

Logs: `.tmp/session/TOOL-LIFE-X-L3-R9/`

## FRZ (scoped, not global PASS)

| ID | Result |
| --- | --- |
| FRZ-BND-03 | PASS — single owner for physical transitions |
| FRZ-TEN | PASS — tenant-bound store + authorization |
| FRZ-REC | PASS — fail-closed without enforceable loss admission |
| FRZ-REL-03/04/05 | VALIDATION REQUIRED — post-remediation L2 |
| FRZ-SEC / FRZ-GOV | PASS — forged digest/MAC rejected; unsafe admission removed |

FINAL_COMMIT: `d8b2f3b60` (publication SHA — verify on GitHub `development`)

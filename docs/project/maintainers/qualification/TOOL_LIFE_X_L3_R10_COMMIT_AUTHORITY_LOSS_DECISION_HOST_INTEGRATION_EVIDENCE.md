# TOOL-LIFE-X L3-R10 — Commit-authority-loss decision & host integration evidence

| Field | Value |
| --- | --- |
| START_HEAD | `ae71074bad74a7989216c8c4f0a36446d52495cb` |
| Gate | **Gate A (corrective)** — decision evidence only after authoritative revocation TX |
| Scope | `DURABLE_SINGLE_HOST` only; `SHARED_MULTI_HOST` fail-closed |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Independent audit corrective batch (R10-F01–F03)

| Finding | Disposition |
| --- | --- |
| R10-F01 | **CORRECTED** — positive decision minted only in SQLite `BEGIN IMMEDIATE` with revocation insert; `issue_for_bound_operation` read-only / fail-closed without prior revoke |
| R10-F02 | **CORRECTED** — `record_revocation_after_commit_authority_loss` no longer calls decision issuer |
| R10-F03 | **CORRECTED** — decision + revocation single TX; terminal-first / revoke-first races unchanged (R8) |
| R10-F04 | **OPEN (scoped)** — `wire_tool_life_durable_single_host_execution_recovery` present; full LKW bootstrap hook not proven |
| R10-F05 | **OPEN** — independent audit of publication SHA |

## Canonical owner

- **Authority transition:** `WorkerExecutionOwnershipRevocationPort.revoke_worker_execution_ownership` (SQLite colocated store)
- **Admission:** `RepositoryAdmittedWorkerExecutionCommitAuthorityLossPolicy` when `WorkerInstanceRepository` wired; else `FailClosedWorkerExecutionCommitAuthorityLossPolicy`
- **Decision evidence:** minted via `mint_worker_execution_commit_authority_loss_decision` inside revocation TX only
- **Execution/recovery surface:** `ExecutionRecoveryExternalOperationWorkerRevocation.record_revocation_after_commit_authority_loss` → recorder only
- **Host wiring:** `wire_tool_life_durable_single_host_execution_recovery`

## L1 / L2 / L3

| Gate | Command | Result |
| --- | --- | --- |
| L1 | `test_tool_life_x_l3_r10_commit_authority_loss_decision.py` | PASS |
| L2 | R8 + R9 + R10 unit files (single pytest) | **40 passed** — `.tmp/session/TOOL-LIFE-X-L3-R10/l2-pytest.log` |
| L3 | R8 cross-process helpers (recorder-only pre-step) | PASS via R8 suite |

FINAL_COMMIT: set at publication on `development`.

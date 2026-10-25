# TOOL-LIFE-X L3-R10 — Commit-authority-loss decision & host integration evidence

| Field | Value |
| --- | --- |
| START_HEAD | `41097dd3fc74f836cf2acc94c203fab920003dbe` |
| Gate | **Gate B (fail-closed corrective)** — lifecycle/worker-repo injection cannot admit revocation |
| Scope | `DURABLE_SINGLE_HOST` only; `SHARED_MULTI_HOST` fail-closed |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Independent audit corrective batch (R10-F01–F06)

| Finding | Disposition |
| --- | --- |
| R10-F01 | **CORRECTED** — positive decision minted only in SQLite `BEGIN IMMEDIATE` with revocation insert; `issue_for_bound_operation` read-only / fail-closed without prior revoke |
| R10-F02 | **CORRECTED** — `record_revocation_after_commit_authority_loss` no longer calls decision issuer |
| R10-F03 | **CORRECTED** — decision + revocation single TX; terminal-first / revoke-first races unchanged (R8) |
| R10-F04 | **OPEN (scoped)** — `wire_tool_life_durable_single_host_execution_recovery` factory-only; positive recovery disabled (fail-closed default) |
| R10-F05 | **PENDING** — independent audit of publication SHA |
| R10-F06 | **CORRECTED (Gate B)** — removed `RepositoryAdmittedWorkerExecutionCommitAuthorityLossPolicy` and worker-repository auto-admission; lifecycle ≠ commit authority |

## Canonical owner & authority blocker

- **Authority transition (durable fence):** `WorkerExecutionOwnershipRevocationPort.revoke_worker_execution_ownership` (SQLite colocated store, R8)
- **Positive revocation admission (production):** `FailClosedWorkerExecutionCommitAuthorityLossPolicy` only — no lifecycle, missing worker, or repository injection
- **Qualification-only positive path:** explicit `loss_policy=QualificationWorkerExecutionCommitAuthorityLossPolicy()` in tests (not production composition)
- **Missing for Gate A:** canonical execution/recovery operation that proves enforceable ownership invalidation *before* supervision MAC (distinct from worker lifecycle / liveness)
- **Decision evidence:** minted via `mint_worker_execution_commit_authority_loss_decision` inside revocation TX only
- **Execution/recovery surface:** `ExecutionRecoveryExternalOperationWorkerRevocation.record_revocation_after_commit_authority_loss` → recorder only
- **Host wiring:** `wire_tool_life_durable_single_host_execution_recovery` (unactivated positive recovery)

## L1 / L2 / L3

| Gate | Command | Result |
| --- | --- | --- |
| L1 | `test_tool_life_x_l3_r10_commit_authority_loss_decision.py` | **16 passed** — `.tmp/session/TOOL-LIFE-X-L3-R10/l1-pytest.log` |
| L2 | R6–R10 external_operations unit files (single pytest) | **86 passed** — `.tmp/session/TOOL-LIFE-X-L3-R10/l2-pytest.log` |
| L3 | R8 cross-process / restart cases (in R8 suite) | PASS via L2 R8 inclusion |

FINAL_COMMIT: `ec2d74158` on `development` (independent audit pending).

# TOOL-LIFE-X L3-R10 — Commit-authority-loss decision & host integration evidence

| Field | Value |
| --- | --- |
| START_HEAD | `e13b58c5707d42f47af27173c66251f0376676b2` |
| Gate | **Gate A** — canonical decision authority within execution/recovery ownership |
| Scope | `DURABLE_SINGLE_HOST` only; `SHARED_MULTI_HOST` fail-closed |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Canonical owner

- **Decision authority:** `WorkerExecutionCommitAuthorityLossDecisionAuthority` (`worker_execution_commit_authority_loss.py`)
- **Contract:** `WorkerExecutionCommitAuthorityLossDecisionV1` (`contracts/worker_execution_commit_authority_loss_decision.py`)
- **Execution/recovery surface:** `ExecutionRecoveryExternalOperationWorkerRevocation` — `decide_commit_authority_loss_for_bound_operation` then `record_revocation_after_commit_authority_loss`
- **Supervision:** `DurableWorkerExecutionCommitAuthorityLossPolicy` (production default in `compose_tool_life_durable_single_host_certified_owner_loss`)
- **Durable linearization:** SQLite `worker_execution_commit_authority_loss_decisions` + existing `worker_execution_ownership_revocations` (`BEGIN IMMEDIATE`, R8 fencing unchanged)
- **Host wiring:** `wire_tool_life_durable_single_host_execution_recovery` (`applications/_shared/tool_life_execution_recovery_wiring.py`)

## D01–D15 (summary)

| ID | Classification |
| --- | --- |
| D01–D07 | **IMPLEMENTATION REQUIRED → PASS (scoped)** — canonical owner + typed decision + binding |
| D08–D09 | **ALREADY PROVEN (R8)** + R10 idempotent decision/revocation tests |
| D10–D11 | **PASS (scoped)** — R9 supervision/MAC + R10 durable policy |
| D12 | **VALIDATION REQUIRED** — factory + `wire_tool_life_*`; full LKW bootstrap hook **OPEN** |
| D13 | **PASS (scoped)** — topology gate; trust via env MAC key |
| D14 | **PASS (scoped)** — R10 unit matrix + R8/R9 regressions |
| D15 | **VALIDATION REQUIRED** — independent audit of publication SHA |

## L1 / L2 / L3

| Gate | Command | Result |
| --- | --- | --- |
| L1 | `test_tool_life_x_l3_r10_commit_authority_loss_decision.py` | PASS |
| L2 | R8 + R9 + R10 unit files | 38 passed (session log `.tmp/session/TOOL-LIFE-X-L3-R10/l2-pytest.log`) |
| L3 | R8 cross-process helpers (decision pre-step) | PASS via R8 suite |

## FRZ (scoped, not global PASS)

| ID | Result |
| --- | --- |
| FRZ-BND-03 | PASS — single execution/recovery owner |
| FRZ-TEN | PASS — tenant-bound decision + authorization |
| FRZ-REC | PASS — fail-closed without durable decision / explicit `FailClosedWorkerExecutionCommitAuthorityLossPolicy` |
| FRZ-SEC | PASS — direct decision store write rejected; MAC + decision at SQLite apply |

FINAL_COMMIT: _(set at publication)_

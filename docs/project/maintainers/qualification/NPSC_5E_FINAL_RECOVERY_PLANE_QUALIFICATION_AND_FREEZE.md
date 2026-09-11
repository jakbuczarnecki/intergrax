# NPSC-5E Final — Recovery Plane Qualification and Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (enterprise recovery plane final certification)

**Date:** 2026-09-10

**Branch:** `development`

**Task:** NPSC-5E Final — Recovery Plane Final Qualification and Freeze

---

## Purpose

Formal qualification and freeze of the **entire NPSC-5E recovery plane** as one canonical enterprise subsystem composed of three orthogonal mechanisms:

| Mechanism | Responsibility |
| --- | --- |
| **R1** | Bounded retry / attempt transitions (`ExecutionAttemptRetryService` + `AttemptLifecycleService`) |
| **R2** | Durable checkpoint / cross-process resume / logical revision CAS |
| **R3** | Exact failed child / topology-slot partial recovery |

**Production code changed in this task:** `NO`

---

## Provenance

| Label | SHA |
| ----- | --- |
| NPSC-5D Final | `a4a1faca01cd5004e372f235132184a84aa5a6bd` |
| R1 Final | `76603ed266f9f54106bf4718fe8886180a826351` |
| R2 Final | `c030d1d4752513bc11ec2597de4e760ae551a978` |
| R3 Final | `1ed12e21888840e800b632ec66f6b683e5960591` |
| NPSC-5E Final `START_HEAD` | `1ed12e21888840e800b632ec66f6b683e5960591` |
| NPSC-5E Final `START_ORIGIN` | `1ed12e21888840e800b632ec66f6b683e5960591` |

**Drift vs R3 Final (`1ed12e2..origin/development`):** none at task start (A–G recovery contract surfaces unchanged).

---

## Qualification gate

`tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py`

Composes R1/R2/R3 Final qualifications, P0A, DG_001, NPSC-5A/B/C, NPSC-5D Final, HITL R3, attempt/child/terminal/cancellation/checkpoint/long-running/fan-out suites, cross-mechanism composite E2E proofs, static ownership audit, and pre-existing failure reconfirmation.

---

## Recovery plane boundaries

```text
retry ≠ resume ≠ partial topology recovery
```

All three share canonical platform truths only:

```text
Execution identity · lineage · authority · governance
terminal/cancellation · budget/deadline · Nexus scheduling · ChildExecutionPort
```

No `RecoveryEngine`, `UniversalRecoveryManager`, `RecoveryRuntime`, or `EnterpriseRecoveryCoordinator` owns R1+R2+R3.

---

## Ownership matrix

| Concern | Canonical owner |
| --- | --- |
| Execution lifecycle | `ExecutionRuntime` |
| Attempt lifecycle | `AttemptLifecycleService` |
| Retry policy | `ExecutionAttemptRetryService` |
| Durable checkpoint | `LongRunningCoordinator` / `RuntimeCheckpoint` |
| Checkpoint logical revision | `TaskCheckpointPersistence` |
| Partial recovery | `FanOutPartialRecoveryService` (R3) |
| Topology scheduling | `Nexus` |
| Child execution | `ChildExecutionPort` |
| Lineage | `ExecutionLineagePersistence` |
| Terminal | `ExecutionTerminalService` |
| Authority | canonical authority plane |
| Governance | Governance |
| HITL | canonical HITL continuation (`continue_slot`) |
| Fan-in | `BoundedMultiAgentFanOutService` |

---

## Retry semantics (R1 frozen)

- Retry = next attempt of same logical Execution when eligible only.
- Sole attempt mint: `AttemptLifecycleService`.
- Non-retryable: cancellation, terminal deny, governance/authority/trust deny, contract error, budget/deadline, unknown unsafe side effect.
- Retry budget monotonic; no R2/R3 reset; deadline not extended.

See: `NPSC_5E_R1_FINAL_CANONICAL_RETRY_ATTEMPT_QUALIFICATION_AND_FREEZE.md`

---

## Checkpoint / resume semantics (R2 frozen)

- Durable resume restores exact valid checkpoint after interruption.
- Checkpoint is not authority, policy permission, or identity mint.
- `TaskCheckpoint.revision` + `expected_revision` CAS (`N` → `N+1`); stale writers blocked.
- Terminal, cancellation, current governance, and narrowed current authority dominate historical checkpoint.

See: `NPSC_5E_R2_FINAL_CHECKPOINT_DURABLE_RESUME_QUALIFICATION_AND_FREEZE.md`

---

## Partial recovery semantics (R3 frozen)

- Partial recovery = exact failed/interrupted recoverable slot only.
- Successful siblings never replayed; fan-in order and cardinality preserved.
- HITL: `continue_slot`, not `recover_failed_slot`.
- R3 selects slot; R1 controls child attempt retry; R2 owns revision CAS on durable updates.

See: `NPSC_5E_R3_FINAL_CHILD_FANOUT_PARTIAL_RECOVERY_QUALIFICATION_AND_FREEZE.md`

---

## Cross-mechanism state machine

Certified paths (composite E2E in final gate + delegated R1/R2/R3 Final scenarios):

```text
failure → retry → checkpoint → restart → resume → partial fan-out recovery → terminal
```

---

## Identity invariants

- `RunId` preserved across resume; resume does not mint retry attempts.
- `revision ≠ attempt ≠ child execution ≠ topology slot id`.
- `claim ≠ revision`.

---

## Lineage invariants

Checkpoint identity cross-validates durable lineage; resume does not bypass lineage admission.

---

## Authority invariants

Effective resume authority = narrow(current authoritative, historical checkpoint provenance). Revoked or missing authority blocks resume/recovery/child execution.

---

## Governance invariants

Current `PolicyDecision` dominates historical checkpoint policy (`REJECT_GOVERNANCE` on deny).

---

## Terminal / cancellation

`ExecutionTerminalService` and cancellation dominate checkpoint; no terminal resurrection; sealed attempts do not reopen.

---

## Budget / deadline continuity

No reset across retry, resume, or partial recovery; no deadline extension.

---

## Nexus scheduling

Single scheduler; recovery scheduling remains under Nexus / orchestration ports.

---

## Child execution

All recovered children via `ChildExecutionPort`; no direct runner bypass in partial recovery.

---

## HITL

`WAITING_FOR_HUMAN` uses governed continuation; never classified as R1 retry or R3 automatic recovery.

---

## Side-effect safety

`UNKNOWN_UNSAFE` blocks blind retry, blind resume replay, and blind partial recovery.

---

## Concurrency

- Same-slot recovery: one effective execution (idempotent / claim semantics).
- Stale recovery writer blocked by R2 CAS.

---

## Cross-process behavior

SQLite checkpoint store + fresh process adapters restore canonical identity without process-local authority for persisted facts.

---

## Regression matrix

Mandatory frozen suites: R1 Final, R2 Final, R3 Final, P0A, DG_001, NPSC-5A, NPSC-5B Final, NPSC-5C, NPSC-5D Final, HITL R3, Attempt lifecycle, Child execution, Terminal, Cancellation (excluding known restart fixture), Checkpoint store, Long-running, Fan-out.

---

## Static ownership audit

- No forbidden recovery god-object names in `intergrax/`.
- `mint_retry_attempt_id` only in allowlisted attempt lifecycle paths.
- No reflection on authoritative recovery contract surfaces (scoped static scan).

---

## Known pre-existing failures

| Test | Classification |
| --- | --- |
| `test_terminal_cancellation_survives_process_restart` | Fixture expects non-resumable gate (`CheckpointNotResumableError` / `not resumable`); does not contradict recovery invariants |
| `test_build_task_progress_view_aggregates_checkpoints` | Unrelated `partial_results` aggregation; no `intergrax/` drift at R3 Final pin |

---

## Deferred NPSC-5F scope

NPSC-5F may add durable execution evidence, replay evidence, reconstruction, audit history, and diagnostic replay — **without redefining** R1/R2/R3 execution semantics.

---

## Formal verdict

**PASS** — R1, R2, and R3 frozen semantics preserved; orthogonal ownership; no production changes; mandatory regression matrix green; cross-mechanism composites certified.

---

## Freeze statement

> The Intergrax Recovery Plane is composed of three orthogonal frozen mechanisms: R1 owns bounded retry and attempt transitions, R2 owns durable checkpoint/resume and logical revision concurrency, and R3 owns exact failed child/topology-slot partial recovery. None of these mechanisms may mint authority or policy permission, bypass ExecutionRuntime, Nexus, ChildExecutionPort, lineage, terminal truth, or current governance. Retry, resume and partial recovery preserve identity, budget and deadline continuity, fail closed on stale or unsafe state, and never replay successful siblings or unknown external side effects blindly.

---

**NPSC-5E:** `FROZEN / PASS`

**Next:** NPSC-5F — Execution Evidence, Replay & Observability

**Not implied:** Execution Engine COMPLETE (NPSC-5F, scale/resilience, documentation freeze, platform-wide final certification remain).

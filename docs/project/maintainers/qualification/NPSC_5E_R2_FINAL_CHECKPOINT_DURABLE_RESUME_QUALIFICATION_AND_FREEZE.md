# NPSC-5E/R2 Final — Checkpoint & Durable Resume Qualification and Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (final freeze certification)

**Date:** 2026-09-10

**Branch:** `development`

**Task:** NPSC-5E/R2 Final — Checkpoint & Durable Resume Qualification and Freeze

---

## Purpose

Formal freeze of the canonical enterprise **durable checkpoint / durable resume** contract spanning:

```text
R2 base + R2-H1 + R2-H2 + R2-H2-Q1
```

Canonical durable resume:

```text
durable checkpoint
→ exact schema (runtime_checkpoint.v2 / task_checkpoint.v1)
→ exact tenant/task/run/attempt/root identity
→ exact logical checkpoint revision
→ current lineage validation
→ current terminal/cancellation truth
→ current authoritative authority (narrowed)
→ current governance
→ exclusive scheduler claim
→ canonical Execution/Nexus path
```

Checkpoint is **never** authority, policy permission, identity mint, direct execution, or worker bypass.

**Production code changed in this task:** `NO`

---

## Provenance

| Label | SHA |
| ----- | --- |
| R1 Final | `76603ed266f9f54106bf4718fe8886180a826351` |
| R2 implementation | `37276a7e2755847b088fc91da76dee0f96824172` |
| R2-H1 | `7bc0c651ca536ffe1b06ac88e7180b4b62380010` |
| R2-H2 | `4c87483a44341e34667ea5c7868be52b7cc71300` |
| R2-H2-Q1 | `0cc93bed8e432bec66e2fd773185dc68b270bbd4` |
| Final task `START_HEAD` | `bf944912d97bdc47cc6601d20a8fe95883224843` |
| Final task `START_ORIGIN` | `bf944912d97bdc47cc6601d20a8fe95883224843` |

**Drift vs R2-H2-Q1 (`0cc93bed..origin/development`):** class `E` only — VPI offer-level fusion (`platform_proofs/.../verified_product_identification/*`). No A–D checkpoint/recovery drift.

---

## Qualification gate

`tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py`

Composes R1 Final, R2, R2-H1, R2-H2, R2-H2-Q1, P0A, DG_001, NPSC-5D Final, HITL R3, NPSC-5A/B/C, attempt/child/terminal/cancellation/checkpoint/long-running suites plus cross-layer E2E freeze scenarios.

---

## Ownership

| Concern | Owner |
| --- | --- |
| Checkpoint contract | `RuntimeCheckpoint` (`runtime_checkpoint.v2`) |
| Task checkpoint schema | `TaskCheckpoint` (`task_checkpoint.v1`) |
| Resume eligibility | `checkpoint_resume_validation.evaluate_checkpoint_resume_eligibility` |
| Resume coordination | `LongRunningCoordinator` |
| Logical revision CAS | `TaskCheckpointPersistence` |
| Attempt transitions | `AttemptLifecycleService` (resume does not mint) |
| Lineage provenance | `ExecutionLineagePersistence` |
| Terminal truth | `ExecutionTerminalService` |
| Current authority | `Task.execution_authority` (narrowed by checkpoint provenance) |
| Governance | current `PolicyDecision` |
| Scheduler exclusivity | `claim_action` ledger |
| Orchestration | Nexus |
| Child spawn | `ChildExecutionPort` |

---

## Checkpoint schemas

| Schema | Version | Unknown |
| --- | --- | --- |
| Runtime checkpoint | `runtime_checkpoint.v2` | `BLOCKED` |
| Task checkpoint | `task_checkpoint.v1` | `BLOCKED` |

---

## Identity binding

Exact binding on resume: `tenant_id`, `task_id`, `run_id`, `attempt_id`, root `ExecutionId`.

Wrong tenant/task/run/attempt/root → `BLOCKED`. Checkpoint cannot mint `RunId`, `AttemptId`, or `ExecutionId`.

---

## Lineage cross-validation

Checkpoint identity must match authoritative `ExecutionLineagePersistence`. Missing required durable lineage, degraded lineage, sealed attempt, root segment mismatch → `BLOCKED`. Non-durable parent guard (DG_001) preserved.

---

## Terminal / cancellation

`ExecutionTerminalService` and cancellation dominate historical checkpoint state. Terminal success/cancel/deny/fail → resume `BLOCKED`. Cancel-after-checkpoint → resume `BLOCKED`.

---

## Governance freshness

Checkpoint policy state is historical only. Current `PolicyDecision = DENY` → resume `BLOCKED`. Checkpoint is not a policy source.

---

## Authority H1

```text
effective resume authority
= narrow(current authoritative authority, historical checkpoint authority)
```

Current authority required when historical exists. Expansion impossible. Malformed historical authority fails closed. Checkpoint does not rehydrate `restored.execution_authority` into effective authority.

---

## Revision / CAS H2

| Invariant | Result |
| --- | --- |
| Canonical logical order | `TaskCheckpoint.revision` per `(tenant_id, task_id)` |
| First revision | `1` |
| CAS `expected=N` → `N+1` | PASS |
| Stale writer | `BLOCKED` (`StaleCheckpointWriteError`) |
| Revision fork / skip | IMPOSSIBLE |
| `store_sequence` / rowid | physical only |
| `created_at_utc` | metadata only |
| Retry / resume / HITL | do not reset revision stream |
| Duplicate same checkpoint | idempotent |
| Same id different payload | `BLOCKED` |
| Unknown commit retry | no extra revision |
| `get_latest` / `get_by_token` | highest logical revision |
| Stale token | `REJECT_STALE` |

---

## Scheduler claims

Exclusive resume ownership via scheduler ledger. Claim ≠ revision ≠ authority. Concurrent resume → exactly one winner.

---

## Cross-process resume

Fresh store adapter / process boundary preserves canonical revision and identity without process-local-only state.

---

## Idempotency

Duplicate same checkpoint id + binding → idempotent reconciliation. Conflicting duplicate → blocked.

---

## Side-effect safety

Completed nodes restore `prior_output`; no blind rerun. Unknown side-effect without idempotency → no blind replay. No exactly-once external-effects claim.

---

## Retry interoperability

Durable resume ≠ execution-attempt retry. Resume preserves `AttemptId`. Post-resume retry → R1 `ExecutionAttemptRetryService` → `AttemptLifecycleService`; checkpoint layer never transitions attempts.

---

## HITL interoperability

Checkpoint resume ≠ human approval. Governed HITL continuation (R3) unchanged; no checkpoint self-approval.

---

## Child execution interoperability

Resumed parent child spawn remains via `ChildExecutionPort`; child authority ≤ parent. No checkpoint-layer child bypass.

---

## Regression matrix

| Suite | Result |
| --- | --- |
| R1 Final | PASS |
| R2 Original | PASS |
| R2-H1 | PASS |
| R2-H2 | PASS |
| R2-H2-Q1 | PASS |
| P0A | PASS |
| DG_001 | PASS |
| NPSC-5A | PASS |
| NPSC-5B | PASS |
| NPSC-5C | PASS |
| NPSC-5D Final | PASS |
| HITL R3 | PASS |
| Attempt lifecycle | PASS |
| Child execution | PASS |
| Terminal | PASS |
| Cancellation | PASS (minus documented pre-existing) |
| Checkpoint store | PASS |
| Long-running | PASS |

**Unexpected skips:** `0`

---

## Known pre-existing failures

| Test | Baseline | Classification |
| ---- | -------- | -------------- |
| `test_p0c5_cancellation_continuity.py::test_terminal_cancellation_survives_process_restart` | R2 `37276a7` | Fixture calls `persist_checkpoint` with default `TaskState.CREATED`; R2 `assert_checkpoint_persistable` rejects before terminal path. Terminal cancellation durability proven by other p0c5 tests (excluded from matrix). **Does not contradict R2 freeze** — invalid fixture, not lost terminal state. |
| `test_partial_results.py::test_build_task_progress_view_aggregates_checkpoints` | pre-R2 | `human_request_expires_at` aggregation in progress view; `intergrax/runtime/long_running/partial_results.py` not in R2 production diff. Unrelated to resume gate. |

**New failures:** `0`

---

## Static quality

| Tool | Scope | Result |
| ---- | ----- | ------ |
| ruff | final qualification test | PASS |
| ruff | R2 production surface | 1 pre-existing `F401` in `persistence_contract.py` (`ScheduledResume` unused import) |
| pyright | final qualification test | PASS |
| pyright | R2 production surface | pre-existing only (`coordinator.py` RunId/AttemptId — certified at H2-Q1) |

**New static errors:** `0`

---

## Deferred R3 scope

```text
partial sibling recovery
failed fan-out slot continuation
exact failed-child re-execution
preserved sibling successes
partial topology recovery
cross-process exact fan-out slot recovery
```

R3 must build on frozen R2 (revision, resume validator, scheduler claim, lineage, R1 retry) without redefining them.

---

## Formal verdict

```text
NPSC-5E/R2: FROZEN / PASS
NPSC-5E: ACTIVE (R3 remains)
```

---

## Freeze statement

> Canonical durable resume in Intergrax restores only a checkpoint whose schema, tenant/task/run/attempt/root identity, logical revision and durable lineage are valid against current platform truth. A checkpoint is never an authority, policy or identity source. Effective resume authority is derived only from the current canonical authority plane and monotonically constrained by historical checkpoint authority. Logical checkpoint state is protected by provider-neutral revision CAS, so stale writers cannot resurrect superseded state. Terminal, cancellation, governance and lineage truth dominate historical checkpoint data, and resume returns exclusively through the canonical Execution/Nexus path.

**Next:** NPSC-5E/R3 — Child & Fan-Out Partial Recovery

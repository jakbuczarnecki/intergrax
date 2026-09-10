# NPSC-5E — Recovery, Checkpoint & Retry Architecture

> **Stage:** P0 + P0A qualified; **R1 FROZEN / PASS**; **R2 checkpoint durable resume ACTIVE / PASS**; R3 ACTIVE

## P0 inventory

NPSC-5E introduces a recovery plane on top of the frozen NPSC-5D governance baseline. P0 establishes boundaries before retry (R1), checkpoint lineage hardening (R2), and partial recovery (R3).

| Plane | Current owner | Lineage role |
|---|---|---|
| Execution lifecycle | `ExecutionRuntime` | records admission facts only |
| Attempt lifecycle | `AttemptLifecycleService` | per-attempt scope; no attempt mint |
| Checkpoint / resume | `LongRunningCoordinator` | segment close/open continuity |
| Orchestration | Nexus | propagate identity; optional seal on retry |
| Terminal | `ExecutionTerminalService` | lineage seal maps outcome; does not terminate |
| Governance / policy | governance plane | lineage may store provenance snapshots |

## P0A lineage reconciliation

Qualified baseline: `a72c9b568c61e28180756059ae48a99fb56eaa19` (post `a3e719b1c` lineage admission + `a72c9b568` durability hardening).

Hard invariants certified in P0A:

- Lineage ≠ lifecycle
- Lineage ≠ authority (no mint/widen)
- Lineage ≠ policy
- Lineage ≠ checkpoint (`ExecutionLineageRecord` ≠ `RuntimeCheckpoint`)
- Lineage ≠ retry decision
- Lineage ≠ scheduler
- No duplicate execution registry / attempt store / checkpoint store

See: `docs/project/maintainers/qualification/NPSC_5E_P0A_EXECUTION_LINEAGE_BASELINE_QUALIFICATION.md`

## Ownership (canonical)

```text
ExecutionRuntime           = lifecycle root
AttemptLifecycleService    = attempt transition authority
ExecutionLineagePersistence = provenance persistence
Nexus                      = orchestration / scheduling
LongRunningCoordinator     = checkpoint/resume orchestration
Governance                 = WHETHER
Execution authority        = canonical authority owner
```

## Baseline

```text
NPSC-5D FROZEN: a4a1faca01cd5004e372f235132184a84aa5a6bd
NPSC-5E LINEAGE BASELINE: a72c9b568c61e28180756059ae48a99fb56eaa19
```

## R1 — Canonical Execution Retry & Attempt Semantics

**Status:** `FROZEN / PASS` (2026-09-09)

Qualified module: `intergrax/runtime/execution/retry/` + `intergrax/contracts/execution_retry.py`.

Final freeze: `docs/project/maintainers/qualification/NPSC_5E_R1_FINAL_CANONICAL_RETRY_ATTEMPT_QUALIFICATION_AND_FREEZE.md`

### Retry layer taxonomy

| Retry kind | Owner | New AttemptId? | Scope |
|---|---|---:|---|
| Transport retry | Worker transport (`queueing/worker/retry_policy`) | NO | message delivery |
| Step/node retry | Nexus step policy (`StepRetryBudget`, graph node `RetryPolicy`) | NO | single node/step |
| Agent invocation retry | `RetryEngine` (alternate agent within attempt) | NO | agent call |
| **Execution attempt retry** | **`ExecutionAttemptRetryService` + `AttemptLifecycleService`** | **YES** | run attempt |
| Topology recovery | Nexus / `LongRunningCoordinator` | depends on execution request | orchestration |

### Execution attempt retry ownership

```text
Failure classification  → ExecutionFailureKind (contracts/execution_retry)
Retry eligibility       → evaluate_execution_retry_eligibility (runtime/execution/retry/policy)
Backoff                 → compute_backoff_delay (runtime/execution/retry/backoff)
Attempt transition      → AttemptLifecycleService.transition_to_next_attempt (sole AttemptId mint)
Lineage seal            → seal_lineage_attempt RETRY_SUPERSEDED (after successful transition)
Orchestration wiring    → NexusGraphRunner._transition_attempt_for_retry
```

### Failure classification

Typed adapter projects existing `FailureClass` / `FailureResponse` / `ResiliencePolicy` into `ExecutionFailureKind`. No universal god enum; fail-closed on `UNKNOWN`.

Minimum outcomes: retryable transient, retryable timeout, permanent, governance deny, authority deny, trust deny, cancelled, budget exhausted, deadline exceeded, terminal success/deny, contract error, unknown unsafe side-effect, unknown.

### Retry eligibility

Mandatory inputs: classification, `attempt_number` (lifecycle generation), `max_attempts` (`ResiliencePolicy.max_attempts`), cancellation, terminal outcome, global deadline, proposed backoff, side-effect idempotency flag.

Decision contract: `FAIL` | `RETRY` | `CANCEL` only.

### Retry budget

Canonical owner for run/attempt count: **`ResiliencePolicy.max_attempts`** interpreted as **maximum total attempts** (e.g. `3` → attempts 1, 2, 3). Enforced in `evaluate_execution_retry_eligibility` when `attempt_number >= max_attempts`.

Separate budgets (not collapsed): `StepRetryBudget.max_retries`, `RetryPolicy.max_retries` (agent), `RetryCoordinator.max_run_retries`, worker `RetryPolicy.max_retries`.

### Attempt identity

```text
RunId      = preserved
AttemptId  = new (sole authority: AttemptLifecycleService via mint_retry_attempt_id)
Lineage    = predecessor sealed RETRY_SUPERSEDED, successor opened by orchestration
```

### Cancellation / deadline precedence

Cancellation dominates retry. Terminal success/cancel/deny blocks retry. Global deadline is not extended; if `now + backoff >= deadline`, retry is denied.

### Governance / AC-3 interaction

Governance deny, authority deny, and trust deny are non-retryable. `REQUEST_HUMAN` / HITL continuation is not classified as transient execution retry.

### Backoff policy

Replaceable via `BackoffPolicyConfig`: fixed, exponential, jittered, none; capped by `max_delay_seconds`. Resilience policy `backoff` field maps to config.

### Idempotency

Blind retry forbidden when `has_unknown_side_effect` and no idempotency guarantee.

### Non-retryable matrix

| Condition | Retry |
|---|---|
| governance deny | blocked |
| authority deny | blocked |
| trust deny | blocked |
| contract violation | blocked |
| cancellation | blocked |
| terminal success | blocked |
| terminal deny | blocked |
| unknown unsafe side-effect | blocked |
| unknown (unclassified) | fail closed |

See: `docs/project/maintainers/qualification/NPSC_5E_R1_FINAL_CANONICAL_RETRY_ATTEMPT_QUALIFICATION_AND_FREEZE.md`

## R2 — Checkpoint & Durable Resume Hardening

**Status:** `ACTIVE / PASS` (2026-09-09)

Qualified module: `intergrax/runtime/long_running/checkpoint_resume_validation.py` + hardened `LongRunningCoordinator` / `RuntimeCheckpoint`.

Qualification: `docs/project/maintainers/qualification/NPSC_5E_R2_CHECKPOINT_DURABLE_RESUME_HARDENING.md`

### Checkpoint ownership (frozen)

```text
RuntimeCheckpoint              = checkpoint contract (runtime_checkpoint.v2)
checkpoint_builder             = snapshot creation/application
TaskCheckpointPersistence      = persistence port (provider-neutral)
LongRunningCoordinator         = checkpoint/resume coordination
Nexus                          = resume scheduling / topology
ExecutionRuntime               = execution lifecycle
AttemptLifecycleService        = attempt authority (no resume mint)
ExecutionLineagePersistence    = durable provenance (cross-check only)
ExecutionTerminalService       = terminal truth (dominates checkpoint)
Governance / authority plane   = current effective WHETHER / authority
scheduler_claim / ledger       = exclusive scheduler action ownership
```

Checkpoint ≠ authority ≠ policy ≠ permission ≠ scheduler ≠ runtime.

### Schema version gate

- Supported exact version: `runtime_checkpoint.v2`
- Unknown version: fail closed in `RuntimeCheckpoint.validate_canonical()` and `validate_runtime_checkpoint_schema`
- No silent legacy fallback

### Identity binding

On resume, `evaluate_checkpoint_resume_eligibility` validates:

```text
checkpoint.task_id == target TaskId
checkpoint.tenant_id == target tenant
checkpoint.run_id == execution_tree.run_id
checkpoint.attempt_id == execution_tree.attempt_id
execution tree root == optional target root ExecutionId
```

Wrong task/run/attempt/root/tenant: `REJECT_IDENTITY` / `REJECT_TENANT`.

### Lineage cross-validation

When lineage persistence is wired:

- sealed attempt → `REJECT_TERMINAL`
- degraded attempt → `REJECT_LINEAGE`
- active segment root ≠ checkpoint root → `REJECT_LINEAGE`
- required durable lineage missing → `REJECT_LINEAGE`

Non-durable parent guard (DG_001 / `a18e65c`) preserved by lineage admission; resume does not bypass it.

### Terminal / cancellation precedence

`ExecutionTerminalService` and `assert_checkpoint_resumable` dominate checkpoint task state. Terminal success/failed/cancel blocks resume. Cancel-after-checkpoint blocks restore.

### Governance / authority freshness

- Stored checkpoint policy is historical only; current `PolicyDecision` may deny resume (`REJECT_GOVERNANCE`).
- Checkpoint authority is **historical constraint / provenance only** — never an effective authority source.
- Authoritative current authority is `Task.execution_authority` on the resuming task (no checkpoint rehydration path).
- Effective resume authority formula (R2-H1):

```text
effective resume authority
= narrow(authoritative current authority, historical checkpoint authority)
```

- Checkpoint cannot expand authority: `validate_checkpoint_authority_expansion` blocks wider historical authority application.
- `resolve_resume_execution_authority` narrows authoritative current authority; malformed historical provenance fails closed (`REJECT_MALFORMED` / `REJECT_AUTHORITY`).
- Current authority missing while checkpoint historical authority exists: `REJECT_AUTHORITY` (fail closed).

### Budget / attempt continuity

Checkpoint resume does not mint `AttemptId` and does not reset attempt lifecycle generation. R1 `max_attempts` semantics remain frozen for post-resume retry.

### Stale checkpoint semantics

- Canonical checkpoint ordering key: durable logical `checkpoint_revision` per `(tenant_id, task_id)` stream; `get_latest` / `get_by_token` order `checkpoint_revision DESC`.
- `store_sequence` / SQLite `rowid` is physical persistence sequence only — not canonical logical version.
- `created_at_utc` is auxiliary metadata only; timestamps cannot determine canonical checkpoint state.
- `TaskCheckpointPersistence.save(..., expected_revision=N)` performs atomic compare-and-set: successor revision `N+1` commits only when canonical revision is `N`.
- A checkpoint writer based on superseded logical revision cannot commit a new canonical checkpoint revision (`StaleCheckpointWriteError`).
- Superseded checkpoint vs store `get_latest`: `REJECT_STALE`
- Store remains append-only history; canonical latest follows validated revision chain; stale late-writer physical insert order cannot resurrect old state

### Duplicate / concurrent resume

- Scheduler ledger `claim_action` provides one winner for exclusive actions.
- Resume coordination remains in `LongRunningCoordinator`; no second runtime/scheduler/checkpoint engine.

### Cross-process resume

`SQLiteTaskCheckpointStore` + fresh process adapter restore canonical identity without process-local ContextVar dependency for persisted facts.

### Side-effect safety

Completed execution-tree nodes restore `prior_output` and are not blindly replayed. Interrupted nodes without idempotency remain pending (no blind replay).

### Provider neutrality

Generic coordinator depends on `TaskCheckpointPersistence` / `TaskCheckpointReader` ports only.

## Future boundaries

### R3 — Partial Recovery / Fan-out Continuation

- Preserve NPSC-5B two-level fan-out lineage.
- Exact slot resume must not create false sibling lineage.
- Governed HITL continuation ≠ failure retry.

## Deferred / out of scope

- `RecoveryLineageManager`, `RetryLineageEngine`, `ExecutionLineageRuntime` — forbidden
- KV lineage adapter
- R3 partial failed-slot fan-out recovery
- NPSC-5F evidence/replay store

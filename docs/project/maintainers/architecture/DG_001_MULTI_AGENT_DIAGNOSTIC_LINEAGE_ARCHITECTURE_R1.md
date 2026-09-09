# DG-001 — Multi-agent diagnostic execution lineage architecture (R1)

> **Task:** `DG-001-CROSS-SYSTEM-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ARCHITECTURE-R1-TERMINAL-CONTINUITY-CORRECTION`  
> **Resolves:** GAP-R1-01 from `DG-001-CROSS-SYSTEM-DIAGNOSTIC-COMPATIBILITY-AUDIT-R1`  
> **Mode:** architecture decision + contract ownership audit — **no implementation**  
> **Branch:** `development`  
> **START HEAD:** `36c2be6a03411e117250c56c4519d759682b8032`  
> **Ancestry verified:** `c09fa302` (terminal continuity correction base) · prior R1 correction chain

---

## 1. Problem statement

Execution runtime deterministically knows the direct parent→child execution edge at the child admission boundary:

```text
ChildExecutionRunner
  → parent ExecutionId from active context
  → mint child ExecutionId
  → ExecutionIdentityBinding(
        execution_id=child,
        parent_execution_id=parent
    )
```

DIAG-2 `ExecutionReconstructor` consumes only:

```text
RuntimeEventPersistence
  + CausalEvidencePersistence
```

Those canonical diagnostic read sources provide **execution membership within a run** (`RuntimeEvent.execution_id`) but **not** `parent_execution_id` and **no** parent→child `CausalRelationKind`. Diagnostics therefore cannot deterministically reconstruct direct parent→child edges or nested execution trees.

This is a **read-side architecture gap**, not an Execution Engine correctness defect. The platform already owns canonical Execution Tree machinery that must be evaluated for reuse before any parallel lineage store.

---

## 2. Current authority model

### Identity authority

| Role | Component | Contract |
| ---- | --------- | -------- |
| Root `RunId` / `AttemptId` / `ExecutionId` minting | `intergrax/runtime/execution/identity_authority.py` | `mint_root_execution_identity()` |
| Child `ExecutionId` minting | `identity_authority.mint_child_execution_id()` invoked by `ChildExecutionRunner` | `ExecutionId` |
| Retry `AttemptId` minting | `identity_authority.mint_retry_attempt_id()` via `AttemptLifecycleService` | `AttemptId` |
| Active identity binding | `bind_active_execution_identity()` in `intergrax/contracts/execution_identity.py` | contextvars: `run_id`, `attempt_id`, `execution_id`, `parent_execution_id` |

**Identity authority ≠ lineage fact authority.** Minting a child `ExecutionId` establishes the runtime admission fact; durable lineage persistence and structural projection are separate concerns.

### Execution lineage scope (R1 terminal continuity — attempt vs admission split)

**Repo audit:** no existing public type carries the full attempt persistence key; execution admission identity is already owned by `ExecutionIdentityBinding`.

| Existing contract | Fields present | Gap |
| ----------------- | -------------- | --- |
| `ExecutionIdentityBinding` | `run_id`, `attempt_id`, `execution_id`, `parent_execution_id` | no `tenant_id`, no `task_id` |
| `RootExecutionContext` | `run_id`, `attempt_id`, `execution_id`, `tenant_id` | no `task_id` (optional extension at task composition only) |
| `ActiveExecutionIdentityState` | same as binding | no `tenant_id`, no `task_id` |
| `ActiveExecutionTaskScopePort` | resolves `TaskId` from active execution | no `tenant_id`; lookup port, not admission scope carrier |
| `ExecutionRequest` | neutral work-intent | **must not** receive identity/lifecycle scope for lineage hook consumption |

**Selected minimal new abstractions (REQUIRED):**

```text
ExecutionLineageAttemptScope  (immutable @dataclass, frozen=True)
  tenant_id: str
  task_id: TaskId
  run_id: RunId
  attempt_id: AttemptId
```

Attempt-level APIs (`open_attempt`, `seal_attempt`, `read_seal`, `list_admissions_for_attempt`, segment continuity) accept **attempt scope only** — no accidental `execution_id`.

```text
ExecutionLineageAdmissionRecord  (immutable append-only row)
  scope: ExecutionLineageAttemptScope
  execution_id: ExecutionId
  parent_execution_id: ExecutionId | None
  admission_position: int  (monotonic within attempt)
  optional: graph_node_id (structural ref only)
```

Admission-specific identity (`execution_id`, `parent_execution_id`) belongs to the admission record, not attempt scope.

**Alias rule:** if the name `ExecutionLineageScope` is retained in implementation, it **must** be attempt-only (equivalent to `ExecutionLineageAttemptScope`). It must **not** embed `execution_id` or `parent_execution_id`.

**ROOT TASK_ID SOURCE:**

```text
Task.task_id: TaskId
  (intergrax.runtime.task.task.Task)
```

Composition boundary: `execute_root_task(task, ...)` and `HostTaskExecution.execute(task, ...)`. On task-based paths, `ExecutionLineageAttemptScope` is composed from `task.tenant_id` + `task.task_id` + active `RunId`/`AttemptId`. **Generic runtime compatibility:** `resolve_root_execution_context(...)` and `RootExecutionContext` remain valid for non-task execution paths; lineage persistence is wired **only** when lineage capability is active. Optional minimal extension: `RootExecutionContext.task_id: TaskId | None`, populated at task composition; fail-closed validation when lineage persistence is enabled and `task_id` is absent. No `ExecutionRequest` introspection; no `getattr` / reflection / arbitrary metadata.

**CHILD ATTEMPT SCOPE SOURCE:**

```text
parent ExecutionLineageAttemptScope  (tenant_id, task_id, run_id, attempt_id)
```

Child admissions inherit attempt scope unchanged. Child `execution_id` + `parent_execution_id` come from canonical `ExecutionIdentityBinding` at child admission — nested children do **not** depend on root `execution_id` stored in attempt scope.

**ADMISSION SCOPE DELIVERY (single pattern — no alternatives):**

**Pattern A — hook constructed with immutable typed attempt scope + admission record fields.**

The lineage concrete `ExecutionAdmissionHook` implementation is constructed per admission with an immutable `ExecutionLineageAttemptScope` plus admission identity from `ExecutionIdentityBinding`. `admit(request)` uses only constructor-bound scope/record + `ExecutionLineagePersistence` port; it **never** inspects `request` for identity or scope.

| Admission | Scope construction |
| --------- | ------------------ |
| Root | `ExecutionRuntime.execute()` builds attempt scope from task composition (when lineage wired) + root `ExecutionIdentityBinding` for admission record |
| Child | `ChildExecutionRunner.execute()` inherits parent attempt scope + child `ExecutionIdentityBinding` for admission record |

Parent attempt scope is bound attempt-scoped at root admission (alongside active execution identity) so children inherit without request introspection.

### Lineage authority (corrected)

Four distinct roles — **no single component owns all four**:

```text
LINEAGE FACT AUTHORITY:
  ChildExecutionRunner / ExecutionRuntime root admission
    → ExecutionIdentityBinding.parent_execution_id
    → durable write via ExecutionLineagePersistence at admission boundary

CANONICAL STRUCTURAL MODEL:
  ExecutionTreeSnapshot / ExecutionCheckpointEntry
    → validated tree invariants (single root per attempt, acyclic, known parents)

IN-MEMORY RECORDER:
  ExecutionTreeRecorder
    → process-local structural projection; NOT universal lineage truth origin

DURABLE SOURCE:
  ExecutionLineagePersistence
    → immutable structural admission records scoped per attempt

DIAGNOSTICS:
  consumer only — reads durable lineage + joins runtime status evidence
```

**Canonical direct parent→child fact** is born at the **child admission boundary** when `ChildExecutionRunner` mints child identity and binds `ExecutionIdentityBinding`. `ExecutionTreeRecorder.record_child_started()` is an **in-memory structural projection** on graph paths — it must converge to the same admission truth, but it is **not** the universal lineage fact authority.

`RunBudgetLedger` records `parent_execution_id` for budget allocation — budget semantics, not execution lineage (see §8 Option C).

---

## 3. Current write paths

| Stage | Writer | What is written | Durability |
| ----- | ------ | --------------- | ---------- |
| Root admission | `ExecutionRuntime` | Root `ExecutionIdentityBinding` (`parent_execution_id = None`) | Process memory only |
| Child admission | `ChildExecutionRunner` | `ExecutionIdentityBinding` in active context | Process memory only |
| Child budget grant | `ExecutionBudgetLedger.grant_child_budget()` | `PersistedBudgetRecord.parent_execution_id` when durable ledger wired | `RunBudgetPersistence` CAS on grant (budget domain) |
| Child tree projection | `ExecutionTreeRecorder.record_child_started()` via `GraphExecutor` | `ExecutionCheckpointEntry` with `parent_execution_id` | Process memory (`ExecutionTreeRecorder._snapshot`) |
| Node completion | `record_graph_node_completion()` + `sync_execution_tree_to_task()` | Updated entry status / `prior_output`; tree copied into `task.runtime.orchestration.runtime_checkpoint` | Process memory (task object) |
| Long-running pause | `LongRunningCoordinator.persist_checkpoint()` → `TaskCheckpointPersistence.save()` | Full `RuntimeCheckpoint.execution_tree` inside `TaskCheckpoint` | Durable — **only when** `long_running.enabled` and `checkpoint_on_pause` and pause checkpoint fires |
| DIAG-2 sources | `RuntimeEventPersistence.append()` | `RuntimeEvent` with `execution_id`, no `parent_execution_id` | Durable observability stream |
| Causal evidence | `CausalEvidencePersistence.append()` | `TRANSPORT_TASK_TRIGGERED_EXECUTION` only | Durable cross-boundary transport fact |

**Coverage gap:** durable lineage admission is absent today. `record_child_started` is invoked only from `GraphExecutor._execute_node_in_child()` and writes in-memory only. Other `ChildExecutionRunner` paths mint child identity but do **not** durably persist lineage admission.

---

## 4. Current durable sources

| Source | Contains `parent_execution_id`? | Scope | Diagnostic read today? |
| ------ | ------------------------------- | ----- | ---------------------- |
| `ExecutionTreeSnapshot` in `TaskCheckpoint.runtime` | Yes | Long-running pause checkpoints (attempt-scoped snapshot) | `TaskCheckpointReader` — not wired into `ExecutionReconstructor` |
| In-memory `ExecutionTreeRecorder` / `task.runtime_checkpoint` | Yes | Active process, per attempt | No |
| `RunBudgetLedgerSnapshot.records` | Yes (budget record) | Per-run budget | No — wrong semantic domain |
| `RuntimeEvent` stream | No | Per-run events | Yes (DIAG-2) |
| `PlatformCausalEvidence` | No parent→child relation | Transport→execution | Yes (DIAG-2) |

---

## 5. Crash / failure durability

### Critical crash scenario

```text
parent running
  → child ExecutionId minted (ChildExecutionRunner)
  → child context bound (ExecutionIdentityBinding)
  → in-memory record_child_started (GraphExecutor path only)
  → child work begins
  → PROCESS CRASH before durable lineage admission write
```

**Verdict:** the parent→child edge is **not** retained in any canonical **lineage** durable source today.

Supporting evidence:

- `ExecutionTreeRecorder` is in-memory until `sync_execution_tree_to_task` (still in-memory) or `TaskCheckpointPersistence.save` (long-running pause only).
- `RunBudgetPersistence` may retain `parent_execution_id` on `grant_child_budget` when a durable budget ledger is wired, but that is **budget allocation truth**, not execution lineage authority.

### Success / failure path matrix

| Moment | Edge in runtime | Edge durable (lineage) | Diagnostics can read | Exact durable source when YES |
| ------ | --------------- | ---------------------- | -------------------- | ------------------------------ |
| Child minted | Contextvars + optional budget record | NO (lineage) | NO | — |
| `record_child_started` (in-memory) | `ExecutionTreeRecorder` | NO | NO | — |
| Child started (work running) | Same | NO | NO | — |
| Child failed (no pause checkpoint) | In-memory tree entry `FAILED` | NO | NO | — |
| Child completed (no pause checkpoint) | In-memory tree entry `COMPLETED` | NO | NO | — |
| Process crashed mid-child | Lost (lineage) | NO | NO | — |
| Long-running checkpoint persisted | `ExecutionTreeSnapshot` | YES (resume projection) | Not via DIAG-2 | `TaskCheckpointPersistence` → `TaskCheckpoint.runtime.execution_tree` |
| Resume from checkpoint | Tree restored into recorder | YES (last saved snapshot) | Not via DIAG-2 | Same checkpoint store |

---

## 6. ExecutionTreeSnapshot semantics (Option A truth audit)

`ExecutionTreeSnapshot` is **attempt-scoped**. Each attempt owns its own execution tree. Multiple attempts within one `RunId` produce isolated trees — never merged into a single snapshot.

| # | Question | Answer |
| - | -------- | ------ |
| 1 | Created for every canonical run? | **No.** Recorder starts in `GraphExecutor.execute()` for graph runs; not universal for all `ChildExecutionRunner` paths. |
| 2 | When does root entry appear? | `ExecutionTreeRecorder.start_root()` at graph execute start (or `from_snapshot` on resume). Root durable admission must precede root delegate work when lineage persistence is active. |
| 3 | When does child entry appear? | `record_child_started()` before child node work in `GraphExecutor._execute_node_in_child()` — in-memory projection only today. |
| 4 | Before child work? | **Yes** on covered graph paths for in-memory recorder; durable admission write required before delegate work (remediation). |
| 5 | Nested children represented? | **Yes** — each child entry carries its direct `parent_execution_id`; `validate_tree()` enforces single root per attempt, known parents, acyclic graph. |
| 6 | Failure before durable admission loses edge? | **Yes** — in-memory only until `ExecutionLineagePersistence` write succeeds. |
| 7 | Normal successful execution always persists tree? | **No** — only if long-running checkpoint path fires (resume projection, not lineage authority). |
| 8 | Failed execution always persists tree? | **No** — same condition. |
| 9 | Snapshot exists after process end? | **Only** if checkpoint was saved or lineage admission store written; otherwise lost. |
| 10 | Diagnostics public read of snapshot? | **No** — `ExecutionReconstructor` does not consume lineage persistence or `TaskCheckpointReader`. |

**Conclusion:** `ExecutionTreeSnapshot` is the correct **canonical structural model** but is **not** presently a sufficient **durable forensic source** for Diagnostics without admission-time durable writes, attempt-scoped persistence, and a public read contract.

---

## 7. Option A — Reuse existing Execution Tree (SELECTED WITH CORRECTED CONTRACTS)

```text
Execution admission (root + child)
  → ExecutionIdentityBinding (lineage fact authority)
  → ExecutionLineagePersistence (durable immutable admission records)
  → ExecutionTreeRecorder / ExecutionTreeSnapshot (structural projection)
  → public read contract (attempt-scoped)
  → ExecutionReconstructor lineage projection + runtime status join
  → DiagnosticReadService operator view
```

### Advantages

- Reuses existing `ExecutionTreeSnapshot` validation (`validate_tree()`, `build_execution_tree_resume_plan`).
- No duplicate tree semantics; same structural model serves checkpoint/resume projection and diagnostics read assembly.
- Nested depth, fan-out, and direct edges (`E4.parent = E2`) are already modeled.
- Conflict policy already exists in recorder: duplicate `execution_id` raises `ValueError`; tree validation rejects unknown parents and cycles.

### Risks (current state)

- No admission-time durable write today.
- Edge may exist only in-memory until crash.
- Not all child paths record into the tree today.
- Checkpoint persistence is pause-scoped resume projection, not lineage authority.
- Retention tied to checkpoint store unless admission store is attempt-scoped.

### Verdict

**`SELECTED WITH CORRECTED CONTRACTS`**

Conditions for implementation:

1. **Admission-time durable write** of immutable structural admission records at root and child boundaries — not only on long-running pause.
2. **Attempt-scoped persistence** — `tenant_id + task_id + run_id + attempt_id`; one root per attempt tree.
3. **Universal recording** at admission for all canonical root and child paths via existing `ExecutionAdmissionHook`.
4. **Public attempt-scoped read contract** for lineage assembly (see §14).
5. **`ExecutionReconstructor` integration** with typed attempt-level completeness semantics (§14).
6. **Terminal lineage seal** correlated with canonical attempt/retry closure authorities — **not** `ExecutionRuntime` return (§13, §14).

---

## 8. Option B — Canonical causal evidence

Proposed semantics:

```text
parent ExecutionId
  → CausalRelationKind.EXECUTION_SPAWNED_CHILD (hypothetical)
  → child ExecutionId
```

### Assessment

| Criterion | Finding |
| --------- | ------- |
| Semantic fit | Parent→child is **execution structural metadata**, not a cross-boundary causal fact like transport→execution. |
| Schema today | `RuntimeExecutionRef` lacks `execution_id`. Extension required. |
| Durability | `CausalEvidencePersistence` is durable and already consumed by DIAG-2. |
| Duplicate authority risk | High unless Execution Tree admission is demoted — contradicts REUSE FIRST. |
| Admission timing | Writable at child boundary, but duplicates tree truth. |

### Verdict

**`VIABLE_WITH_CONDITIONS`** as a **durable forensic projection only** — but **rejected as SELECTED** because it creates dual truth unless Execution Tree admission is demoted, and structural lineage already belongs to Execution admission machinery.

---

## 9. Option C — Other existing public contract

| Candidate | `execution_id` + `parent_execution_id`? | Verdict |
| --------- | --------------------------------------- | ------- |
| `TaskCheckpointPersistence` / `TaskCheckpoint.runtime.execution_tree` | Yes | **Partial** — resume/checkpoint projection only; not lineage authority; pause-scoped |
| `RunBudgetPersistence` / `PersistedBudgetRecord` | Yes | **Rejected** — budget allocation domain |
| `RuntimeEventPersistence` | No `parent_execution_id` | Not applicable for structural edges |
| `CausalEvidencePersistence` | No parent→child relation | Not applicable |
| Active contextvars | Yes (in-process) | Not durable |

### Verdict

**`NO_EXISTING_OPTION_C`** as a complete solution.

---

## 10. Comparison matrix

Scores: **Strong** / **Partial** / **Weak** / **N/A** — qualitative.

| Criterion | A: Execution Tree | B: Causal Evidence | C: Existing contract |
| --------- | ----------------: | -----------------: | -------------------: |
| Canonical ownership | Strong | Weak | Partial |
| Durability | Partial → Strong after admission write | Strong | Partial |
| Crash safety | Weak today → Strong after admission write | Strong if added | Weak |
| Attempt isolation | Strong (after correction) | Weak | Partial |
| Nested execution | Strong | Strong if typed refs added | Partial |
| Retry / resume | Strong (new attempt = new tree) | Weak | Partial |
| Tenant integrity | Strong | Strong | Strong |
| Idempotency | Strong | Strong | Partial |
| No duplicate truth | Strong | Weak | Partial |
| Backward compatibility | Strong (`UNAVAILABLE` for old runs) | Strong | Strong |

---

## 11. Selected architecture

```text
SELECTED_OPTION: OPTION_A
```

**Execution admission at root and child boundaries is lineage fact authority; `ExecutionLineagePersistence` is durable source; `ExecutionTreeSnapshot` is canonical structural model; Diagnostics consumes via public read path.**

Rationale:

1. REUSE FIRST — existing `ExecutionAdmissionHook`, `ExecutionTreeSnapshot`, validation, and resume planning.
2. Option A gaps are **integration, durability timing, and contract precision** — not missing domain model.
3. Option B misclassifies structural lineage as cross-boundary causal evidence.
4. Causal evidence remains **not** primary lineage authority.

---

## 12. Authority and ownership

| Concern | Owner | Must not own |
| ------- | ----- | ------------ |
| Lineage fact authority | `ExecutionRuntime` / `ChildExecutionRunner` admission + `ExecutionIdentityBinding` | Diagnostic Engine |
| Canonical structural model | `ExecutionTreeSnapshot` / `ExecutionCheckpointEntry` (validated projection) | Diagnostics |
| In-memory recorder | `ExecutionTreeRecorder` | Diagnostics; not durable truth |
| Identity minting | `identity_authority` / `ChildExecutionRunner` | Diagnostics |
| Durable lineage storage | `ExecutionLineagePersistence` (attempt-scoped) | Diagnostics |
| Runtime status / failure | `RuntimeEventPersistence` / DIAG-2 runtime evidence | Lineage persistence |
| Operator diagnostic projection | `ExecutionReconstructor` → `DiagnosticReadService` | — |
| Resume/checkpoint projection | `TaskCheckpoint.runtime.execution_tree` | Lineage authority |

**Hierarchy:**

```text
Execution admission fact
        ↓
durable lineage persistence (ExecutionLineagePersistence)
        ↓
canonical structural lineage read (attempt-scoped)
        ↓
ExecutionReconstructor (joins runtime status from RuntimeEventPersistence)
```

`TaskCheckpoint.runtime.execution_tree` is **resume/checkpoint projection** — not a competing durable lineage authority.

**Checkpoint vs durable lineage conflict:**

```text
lineage persistence says: E2.parent = E1
checkpoint says:         E2.parent = E3
```

→ **HARD INTEGRITY FAILURE** — never last-write-wins. Checkpoint cannot overwrite canonical immutable parent mapping.

**Diagnostic Engine consumes execution truth; it does not create execution truth.**

Decision System remains **out of scope** for execution parent→child edges.

---

## 13. Write semantics

### Lineage fact birth

Canonical direct parent→child fact is born at child admission:

```text
ChildExecutionRunner
  → parent ExecutionId from active context
  → mint child ExecutionId
  → ExecutionIdentityBinding(
        execution_id=child,
        parent_execution_id=parent
    )
  → durable lineage admission write (required remediation)
```

Root lineage fact is born at root admission with `parent_execution_id = None`.

### Root admission flow

```text
ExecutionRuntime root identity
  → ExecutionBoundary
  → existing ExecutionAdmissionHook
  → ExecutionLineagePersistence root admission (durable, if lineage persistence active)
  → root delegate
```

**Root entry must be durably admitted before root delegate work** when lineage persistence is active.

### Child admission flow

```text
ChildExecutionRunner child identity
  → ExecutionBoundary
  → existing ExecutionAdmissionHook
  → ExecutionLineagePersistence child admission (durable, before delegate)
  → child delegate
```

**Child entry must be durably admitted before child delegate work** when lineage persistence is active.

### Implementation boundary (logical — not implemented in this task)

```text
ROOT:
  ExecutionRuntime
    → ExecutionBoundary
    → ExecutionAdmissionHook implementation (lineage recorder)
    → ExecutionLineagePersistence root admission
    → delegate

CHILD:
  ChildExecutionRunner
    → ExecutionBoundary
    → same ExecutionAdmissionHook abstraction
    → ExecutionLineagePersistence child admission
    → delegate
```

`ExecutionTreeRecorder.record_child_started()` and `GraphExecutor`-specific recording must later converge to this shared admission path — no duplicate recording authority. **Not implemented in correction task.**

### Reuse existing `ExecutionAdmissionHook`

The repo already provides `ExecutionAdmissionHook[RequestT]`. Lineage recording is a **concrete reusable implementation** of this existing protocol (implementation name TBD, e.g. lineage admission recorder/hook).

**`ExecutionTreeAdmissionHook` as a new public protocol: NOT REQUIRED.**

**Scope delivery contract:** lineage hook receives `ExecutionLineageAttemptScope` + admission identity via **constructor binding (Pattern A)** at each `ExecutionBoundary` construction. `admit(request)` must not use `getattr`, reflection, application-specific request typing, arbitrary metadata, or request payload inspection to obtain `task_id`, `tenant_id`, or execution identity.

**Neutral `ExecutionRequest` invariant:** identity/lifecycle scope stays execution-infrastructure concern; do not extend `ExecutionRequest` with `task_id`, tenant, run, or execution identity solely for lineage hook consumption.

### Durable data model (resolved — no ambiguity)

**Selected model: append-only immutable structural admission records.**

`ExecutionLineagePersistence` stores **immutable** `ExecutionLineageAdmissionRecord` rows. It does **not** store mutable `ExecutionCheckpointEntry` status updates and does **not** perform full snapshot CAS.

**Why not `ExecutionCheckpointEntry`?**

`ExecutionCheckpointEntry` is checkpoint/resume-oriented and mutable — it carries `ExecutionCheckpointStatus`, `prior_output`, and completion mutations via `record_graph_node_completion()`. Using it as the durable lineage store would make lineage persistence a second owner of mutable execution status, violating single-authority. Immutable admission records separate **structural truth** (who is parent of whom) from **runtime status** (succeeded/failed), which remains on `RuntimeEventPersistence`.

**`ExecutionLineageAdmissionRecord` minimum fields:**

```text
scope: ExecutionLineageAttemptScope
execution_id: ExecutionId
parent_execution_id: ExecutionId | None
admission_position (monotonic within attempt)
optional: graph_node_id (structural ref only)
```

**Excluded:** prompts, raw payloads, PII, raw exceptions, runtime status, failure details.

**Durable attempt continuity records** (same store — no second diagnostics pipeline):

```text
open_attempt(scope)                         → attempt OPEN
open_segment(scope, segment_identity)       → SEGMENT_OPEN (durable marker before segment work)
close_segment_for_resume(scope, segment_id) → clean resumable suspension
mark_degraded(scope, reason_code)           → durable monotonic degradation truth
seal_attempt(scope, closure_kind)           → attempt closed (see below)
read_attempt_lineage_state(scope)           → OPEN | DEGRADED | SEALED + segment facts
```

**Terminal seal record** (immutable row correlated with closure authority):

```text
scope: ExecutionLineageAttemptScope
closure_kind = RETRY_SUPERSEDED
              | TERMINAL_COMPLETED
              | TERMINAL_FAILED
              | TERMINAL_CANCELLED
              | ATTEMPT_LINEAGE_DEGRADED
sealed_at_position
```

Required for typed completeness — see §14.

### Attempt closure authorities (R1 terminal continuity — corrected)

**`ExecutionRuntime.execute()` normal return ≠ attempt closure.** Long-running pause/resume may return from `ExecutionRuntime` while the attempt remains OPEN:

```text
ExecutionRuntime return
  → WAITING / PAUSED / resumable task state
  → checkpoint (TaskCheckpoint projection)
  → resume
  → SAME AttemptId
  → further executions / new process segment
```

Return from `ExecutionRuntime` or `ExecutionBoundary` does **not** automatically terminalize the attempt and must **not** trigger lineage seal.

Reuse existing lifecycle authorities:

| Closure kind | Canonical authority | Lineage action |
| ------------ | ------------------- | -------------- |
| Retry supersession (A1 → A2) | `AttemptLifecycleService.transition_to_next_attempt(...)` succeeds | Seal A1 (`RETRY_SUPERSEDED` or `ATTEMPT_LINEAGE_DEGRADED`); open A2; **do not seal A2 at transition** |
| Final COMPLETED | `ExecutionTerminalService.commit_terminal_outcome(..., COMPLETED)` | Seal final active attempt (`TERMINAL_COMPLETED` or degraded variant) |
| Final FAILED | `ExecutionTerminalService.commit_terminal_outcome(..., FAILED)` | Seal final active attempt (`TERMINAL_FAILED` or degraded variant) |
| Final CANCELLED | `ExecutionTerminalService.record_cancellation(...)` / `commit_terminal_outcome(..., CANCELLED)` | Seal final active attempt (`TERMINAL_CANCELLED` or degraded variant) |
| Resumable pause | `LongRunningCoordinator.persist_checkpoint()` / task lifecycle WAITING states | Attempt **OPEN**; `close_segment_for_resume` only — **no seal** |

**Typed composition boundary for final terminal seal** (both facts available):

```text
NexusLoop._commit_durable_terminal_authority(task)
  → terminal_outcome_from_task_state(task.state)
  → require_active_execution_identity() → active AttemptId
  → ExecutionTerminalService.commit_terminal_outcome(...)
  → lineage seal_attempt(scope, closure_kind=TERMINAL_*)
```

Diagnostics **reads** seal only; Diagnostics does **not** own or write seal.

```text
RETRY_CLOSURE_AUTHORITY:
  AttemptLifecycleService successful transition_to_next_attempt(...)

FINAL_TERMINAL_AUTHORITY:
  ExecutionTerminalService canonical terminal outcome commit
  (COMPLETED | FAILED | CANCELLED)

PAUSE_BEHAVIOR:
  Resumable states (WAITING_FOR_HUMAN, checkpoint pause, scheduled resume,
  other canonical resumable states) → attempt remains OPEN;
  clean segment close/continuation marker only

RESUME_BEHAVIOR:
  Validate previous segment continuity for same AttemptId;
  open new segment; attempt remains OPEN; no new attempt lineage tree

RAW_EXCEPTION_BEHAVIOR:
  Raw exception from ExecutionBoundary / ExecutionRuntime does NOT seal attempt.
  Seal allowed only when exception path leads to canonical terminal outcome commit
  (typically FAILED) via ExecutionTerminalService.

PROCESS_CRASH_BEHAVIOR:
  Missing clean segment closure → on resume/re-entry attempt lineage permanently
  PARTIAL/DEGRADED; COMPLETE forbidden; do not infer missing admissions absent
```

**Forbidden (removed contract):**

```text
ExecutionRuntime.execute()
  → boundary returns
  → attempt terminal          ← WRONG
  → seal                      ← WRONG
```

### Process segment continuity (enterprise-safe)

Each execution process segment of the same `AttemptId` must be durably recognizable:

```text
SEGMENT_OPEN (durable marker written — fail-closed if unavailable)
     ↓
execution work (root/child admissions within segment)
     ↓
either:
  clean resumable suspension  → close_segment_for_resume
  retry closure               → AttemptLifecycle transition seals attempt
  final closure               → ExecutionTerminalService + seal_attempt
```

If the process disappears **without** clean segment closure, the next resume/re-entry must detect the incomplete prior segment and force attempt lineage to **PARTIAL/DEGRADED** for forensic completeness. Missing admissions must not be guessed absent.

**Store unavailable at segment open:** **FAIL-CLOSED FOR SEGMENT OPEN** — do not begin a new root/resume segment unless a minimal durable continuity marker (`open_segment`) can be written. No ad-hoc second durability pipeline; reuse `ExecutionLineagePersistence` continuity records.

### Seal legality and degradation (R1 terminal continuity)

```text
child/root admission persistence availability failure (living segment)
        ↓
mark_degraded(scope, reason_code)  (durable — REQUIRED before any later COMPLETE seal)
        ↓
AttemptLineageDegradationState rebind degraded=True  (runtime monotonic indicator)
        ↓
ATTEMPT_LINEAGE_SEALED / TERMINAL_* COMPLETE lineage prohibited
        ↓
seal_attempt writes ATTEMPT_LINEAGE_DEGRADED or TERMINAL_* with degraded completeness
        ↓
Diagnostics completeness → PARTIAL (never COMPLETE for degraded seal)
```

**Mandatory crash scenario (must be architecturally impossible to false-COMPLETE):**

```text
A1 OPEN
  → child admission persistence fails
  → in-memory degraded=true
  → execution continues
PROCESS CRASH
  → in-memory degradation lost
resume same A1
  → store recovered
  → execution continues
  → terminal
```

**Required guarantee:** `A1 != COMPLETE` unless durable continuity proves no unknown lineage gap. If unclean segment or durable `mark_degraded` absent → PARTIAL/DEGRADED on resume; terminal seal may occur but lineage completeness remains non-COMPLETE.

**`AttemptLineageDegradationState` (RUNTIME_ONLY optimization):**

```text
@dataclass(frozen=True)
AttemptLineageDegradationState:
  degraded: bool = False
```

- **Owner:** execution infrastructure; bound attempt-scoped at root admission via `bind_active_attempt_lineage_degradation()` (parallel to `bind_active_execution_identity()`).
- **Set:** monotonic rebind/mark operation (`degraded=False → degraded=True`) — **not** mutation of frozen object; triggered when durable `mark_degraded` succeeds or when reading durable degraded state on segment resume.
- **Read:** fast-path legality check within living process; **not** sole source of truth after process restart.
- **Canonical completeness truth:** durable `mark_degraded` + segment continuity records in `ExecutionLineagePersistence`.
- **Not stored in:** `Task` metadata, `Problem`, Decision System, or raw dict.
- **Inherited:** children read parent attempt degradation; cannot clear degradation within the same attempt.

Structural conflicts remain hard integrity failures (admission rejected), distinct from availability degradation.

### Idempotency

**Root re-admission (same attempt):**

| Condition | Result |
| --------- | ------ |
| Same `attempt_id` + same root `execution_id` | Idempotent success |
| Same `attempt_id` + different root `execution_id` | Hard integrity conflict (unless canonical retry created new attempt) |

**Child re-admission:**

| Condition | Result |
| --------- | ------ |
| Same child + same parent | Idempotent success |
| Same child + different parent | Hard integrity conflict |

### Conflict policy

**Hard integrity failure** — never last-write-wins:

- Structural parent mismatch (`E3.parent = E1` then `E3.parent = E2`)
- Cross-tenant parent/child admission
- Cycle or invalid parent reference
- Checkpoint parent mapping conflicts with durable lineage

### Tenant isolation

All reads and writes scoped by `tenant_id + task_id + run_id + attempt_id`. Cross-tenant parent/child admission **MUST FAIL**.

### Cycle prevention

Reject admissions that would introduce cycles. Validate on read assembly using same invariants as `ExecutionTreeSnapshot.validate_tree()`.

### Persistence scope

```text
tenant_id
task_id
run_id
attempt_id
```

Each attempt owns its own execution tree. **Multiple root entries in one attempt tree are forbidden.**

Run-level diagnostic API (if exposed) returns an explicit collection of attempt trees:

```text
attempt A1 → tree (single root)
attempt A2 → tree (single root)
attempt A3 → tree (single root)
```

Never aggregate multiple attempts into one merged tree.

---

## 14. Read semantics

### Scope

`ExecutionReconstructor` reads lineage in scope:

```text
tenant_id
task_id
run_id
attempt_id
```

Do **not** aggregate multiple attempts into one tree.

### Minimal read capabilities

Assemble structural tree from immutable `ExecutionLineageAdmissionRecord` rows for the attempt scope. Reuse validation invariants from `ExecutionTreeSnapshot` — **do not** introduce `DiagnosticExecutionTree` or parallel hierarchy types.

| Capability | Source |
| ---------- | ------ |
| Root execution | Admission record with `parent_execution_id is None` (exactly one per attempt) |
| Direct parent | `record.parent_execution_id` |
| Direct children | Records where `parent_execution_id == execution_id` |
| Bounded full tree | All admission records for attempt scope with limit |
| Status / failure | **Read-side join** from `RuntimeEventPersistence` — not lineage store |
| Completeness | Typed attempt-level enum (see below) |

### Status / failure linkage (no dual authority)

```text
Parent→child structural tree:  ExecutionLineagePersistence
Runtime failure/status:        RuntimeEventPersistence / DIAG-2 runtime evidence
ExecutionReconstructor:        read-side join
```

Lineage persistence is **not** a second runtime-status store.

### Integration point

```text
ExecutionLineagePersistence (read, attempt-scoped)
  + RuntimeEventPersistence (status join)
  → ExecutionReconstructor.reconstruct_execution() [extended]
  → lineage projection with completeness
  → DiagnosticReadService occurrence view
```

### Completeness protocol (attempt-level — mandatory)

Fail-open execution policy requires typed completeness — **never infer `COMPLETE` from a structurally valid-looking entry list alone.**

```text
attempt lineage OPEN (+ durable segment continuity)
       ↓
root + child durable admissions
       ↓
canonical closure authority fires (retry transition OR terminal outcome commit)
       ↓
lineage seal correlated with that authority
       ↓
COMPLETE (lineage completeness — distinct from execution outcome)
```

| Status | Meaning |
| ------ | ------- |
| `COMPLETE` | Attempt canonically closed; no further admissions; all required continuity segments clean; no durable degradation / unclean segment gap; structural tree validates; seal present with allowed closure_kind |
| `PARTIAL` | Some admissions missing, admission write failed but execution continued, unclean segment after crash, truncated page, degraded seal, or seal absent |
| `UNAVAILABLE` | No durable lineage evidence for attempt; store unreachable on read; process crashed before any admission |

**`COMPLETE` semantics (lineage completeness):**

- Attempt was actually closed via retry supersession or terminal authority — **not** because `ExecutionRuntime` returned.
- Attempt cannot accept further canonical admissions.
- All required continuity segments are clean (or closure is degraded-only by policy).
- No persistent/detected admission gap; structural tree validates.
- Terminal/retry closure authority confirmed and correlated with seal.

**Execution outcome vs lineage completeness (orthogonal):**

```text
Attempt outcome = FAILED   + Lineage completeness = COMPLETE   → LEGAL
Attempt outcome = CANCELLED + Lineage completeness = COMPLETE  → LEGAL
```

when all structural facts are durable and attempt was canonically closed without degradation gaps.

**Never** return `COMPLETE` when:

- Admission write failed but execution continued without durable `mark_degraded`
- Process crashed before clean segment closure (unclean segment → permanent PARTIAL/DEGRADED)
- Store was unavailable during admission and durable degradation not recorded
- Terminal/retry seal does not exist
- Seal is degraded (`ATTEMPT_LINEAGE_DEGRADED` or terminal seal with degraded completeness)
- Only in-memory `AttemptLineageDegradationState` indicates degradation after restart without durable record

**Never** infer `parent = root` for missing edges.

### False COMPLETE prevention (formal invariant)

```text
No sequence of:
  - store availability failure,
  - process crash,
  - checkpoint resume,
  - retry,
  - terminal completion

may cause COMPLETE unless durable continuity proves that no unknown lineage gap exists.
```

Durable proof requires: segment continuity records, admission rows, durable degradation markers, and seal legality derived from those records — not from ContextVar alone.

### Final lineage closure model

```text
INITIAL ATTEMPT:
  open_attempt → attempt OPEN → open_segment → root/child admissions

RESUMABLE PAUSE:
  close_segment_for_resume (clean) → attempt remains OPEN

RESUME SAME ATTEMPT:
  validate segment continuity → open_segment → attempt remains OPEN

RETRY:
  AttemptLifecycleService transition A1→A2 succeeds
  → A1 seal (RETRY_SUPERSEDED or DEGRADED)
  → A2 open_attempt (OPEN; independent tree)

FINAL SUCCESS / FAILURE / CANCEL:
  ExecutionTerminalService canonical terminal commit
  + known active AttemptId
  → seal_attempt for that final attempt

PROCESS CRASH:
  missing clean segment closure
  → attempt PARTIAL/DEGRADED permanently
  → later COMPLETE forbidden
```

### Crash semantics (R1 terminal continuity)

| Scenario | Completeness |
| -------- | ------------ |
| Crash before any admission or before seal | `UNAVAILABLE` or `PARTIAL` |
| Admission persistence failure + crash without durable mark_degraded | `PARTIAL` — no COMPLETE possible after resume |
| Unclean segment restart (no close_segment_for_resume) | `PARTIAL` / permanently degraded |
| Successful admissions + pause (clean segment) + resume | Eligible for COMPLETE only after canonical final/retry seal |
| Valid non-degraded seal after canonical closure authority | `COMPLETE` eligible |

### Retry isolation (R1 terminal continuity)

Each new `AttemptId` within the same `RunId`:

```text
AttemptLifecycleService.transition_to_next_attempt succeeds
  → previous attempt sealed (RETRY_SUPERSEDED | DEGRADED)
  → new ExecutionLineageAttemptScope (new attempt_id)
  → open_attempt → OPEN lineage state
  → open_segment → new root admission path
  → fresh runtime degradation indicator (from durable state = false unless prior policy)
  → eventual seal only via terminal or next retry transition
```

Seal for attempt A1 does not affect attempt A2. A1 closure is triggered **only** by successful retry transition — not by `ExecutionRuntime` return. Diagnostics reads attempt trees independently.

---

## 15. Integrity invariants

1. Exactly **one root** per attempt tree (`parent_execution_id is None`).
2. Every non-root entry references an existing parent `execution_id` in the same attempt scope.
3. Graph is acyclic.
4. `execution_id` is unique within the attempt lineage store.
5. `(child_execution_id → parent_execution_id)` is immutable once durably admitted.
6. Retry creates a **new** `AttemptId` within the same `RunId`; each attempt owns its own execution-tree scope.
7. `ExecutionIds` for a new attempt must follow canonical `ExecutionRuntime` / child identity authority and **must never be inferred from the previous attempt**.

### Retry semantics (safe contract)

```text
Retry creates a new AttemptId within the same RunId.

Each attempt owns its own execution-tree scope.

ExecutionIds for the new attempt must follow canonical ExecutionRuntime /
child identity authority and must never be inferred from the previous attempt.
```

Current retry behavior:

- Preserves `RunId`
- Mints new `AttemptId`
- Rebinding active attempt clears active `execution_id` and `parent_execution_id`

**No claim that `ExecutionId` is stable across retry attempts** — each attempt mints identities through canonical admission paths.

---

## 16. Failure isolation

**Classification:** execution parent→child lineage is a **diagnostics / observability evidence requirement**, not an execution integrity gate (unlike budget reservation or authority policy).

### Storage availability failure

**Admission persistence availability failure:**

- **Execution may continue** (fail-open for execution) **only if** durable degradation can be recorded before any later COMPLETE seal; otherwise treat as unclean segment risk
- Durable `mark_degraded(scope, ...)` → attempt lineage **DEGRADED**
- Runtime `AttemptLineageDegradationState` rebind for in-process monotonic checks
- COMPLETE lineage completeness → **FORBIDDEN** while degraded or after unclean segment
- Diagnostics: `PARTIAL` or `UNAVAILABLE` — never `COMPLETE`

**Seal persistence availability failure:**

- Canonical closure authority already committed (retry transition or terminal outcome)
- Seal write fails → Diagnostics `PARTIAL` (structural admissions may exist; seal absent)
- Does not retroactively erase durable degradation if admissions succeeded

### Structural conflict

Examples: parent mismatch, cross-tenant violation, cycle, invalid parent, checkpoint vs durable lineage parent conflict:

- **Hard integrity failure** at admission or read reconciliation
- Child admission rejected where applicable

| Failure | Execution continues? | Diagnostics behavior |
| ------- | -------------------- | -------------------- |
| Lineage durable write fails (availability) | **Yes** | `PARTIAL` / `UNAVAILABLE` — never `COMPLETE` |
| Lineage structural conflict | **No** at admission (or hard failure on read reconcile) | Integrity error |
| Lineage read missing | N/A | `UNAVAILABLE` — no inference |
| Checkpoint vs lineage parent conflict | N/A (reconcile on read/resume) | **HARD INTEGRITY FAILURE** |

Budget grant failure remains independent and may still fail-closed child admission — budget integrity, not lineage.

---

## 17. Backward compatibility

Historical runs without admission-time lineage persistence:

```text
lineage_completeness = UNAVAILABLE or PARTIAL
```

Never default missing parent to root. Never reconstruct tree from heuristics.

---

## 18. Security

Lineage durable records contain **structural facts only** — see §13 `ExecutionLineageAdmissionRecord` fields.

**Exclude:** prompts, agent payloads, PII, secrets, raw exceptions.

---

## 19. Scale / performance requirements (next task)

- Support 1 parent + thousands of direct children via paginated `list_admissions_for_attempt(scope, ...)`.
- Support nesting depth ≥ 3 with bounded traversal.
- Index by `(tenant_id, task_id, run_id, attempt_id)` and `execution_id`.
- Run-level views return explicit attempt-tree collections — no merged multi-attempt trees.
- Idempotent append suitable for concurrent child fan-out (sibling admissions).

### Persistence public API (attempt-scoped — R1 terminal continuity)

All attempt-level methods accept `ExecutionLineageAttemptScope` only:

```text
open_attempt(scope)
open_segment(scope, segment_identity)
admit_root(scope, record_fields...)      # parent_execution_id = None in record
admit_child(scope, record_fields...)     # parent_execution_id required in record
close_segment_for_resume(scope, segment_identity)
mark_degraded(scope, reason_code)
seal_attempt(scope, closure_kind)        # RETRY_SUPERSEDED | TERMINAL_* | ATTEMPT_LINEAGE_DEGRADED
list_admissions_for_attempt(scope, ...)  # paginated
read_seal(scope, ...)
read_attempt_lineage_state(scope)
```

`seal_attempt` rejects COMPLETE-eligible closure when durable degradation or unclean segment continuity forbids it. Invoked from:

- retry transition handler (A1 superseded), correlated with `AttemptLifecycleService` success
- terminal commit handler (`NexusLoop._commit_durable_terminal_authority`), correlated with `ExecutionTerminalService` success

**Not** from `ExecutionRuntime.execute()` return path.

---

## 20. Future qualification scenarios

| ID | Scenario |
| -- | -------- |
| Q1 | Root + single child |
| Q2 | Root + 3 siblings |
| Q3 | Nested tree depth ≥ 3 |
| Q4 | One child failure |
| Q5 | Partial sibling failure |
| Q6 | Retry (new `AttemptId`, isolated attempt tree) |
| Q7 | Resume from checkpoint |
| Q8 | Process crash after child admission (lineage durability proof) |
| Q9 | Duplicate edge admission (idempotent) |
| Q10 | Conflicting parent edge (integrity failure) |
| Q11 | Cross-tenant rejection |
| Q12 | Historical run without lineage evidence (`UNAVAILABLE`) |
| Q13 | Root admission persistence (durable before root delegate) |
| Q14 | Two attempts in one run remain isolated |
| Q15 | Attempt A1 and A2 each have exactly one root |
| Q16 | Missing terminal seal → never `COMPLETE` |
| Q17 | Failed admission write + execution continues → `PARTIAL`/`UNAVAILABLE` |
| Q18 | Checkpoint vs durable lineage parent conflict → hard integrity failure |
| Q19 | Lineage hook receives `task_id` without request introspection |
| Q20 | Child inherits exact parent task/tenant scope |
| Q21 | Admission storage failure prohibits COMPLETE seal |
| Q22 | Seal storage failure → `PARTIAL` |
| Q23 | Retry A2 has independent OPEN/seal lifecycle |
| Q24 | Stale/degraded attempt cannot later be falsely sealed COMPLETE |
| Q25 | `ExecutionRuntime` returns resumable state → attempt NOT sealed |
| Q26 | Pause + resume same `AttemptId` → same OPEN attempt lineage |
| Q27 | Root delegate raises without canonical terminal commit → no COMPLETE seal |
| Q28 | Terminal FAILED with complete structural evidence → lineage COMPLETE allowed |
| Q29 | Terminal CANCELLED with complete structural evidence → lineage COMPLETE allowed |
| Q30 | Retry A1→A2 → A1 closes, A2 opens independently |
| Q31 | Admission write failure + crash + same-attempt resume → COMPLETE impossible |
| Q32 | Unclean process segment restart → lineage permanently PARTIAL/DEGRADED |
| Q33 | Generic non-task `ExecutionRuntime` path remains compatible |
| Q34 | Segment-open persistence unavailable → fail-closed segment open proven |
| Q35 | Lineage scope contains attempt identity only; execution identity in admission record |

---

## 21. Explicitly rejected alternatives

| Alternative | Reason |
| ----------- | ------ |
| `ExecutionTreeRecorder.record_child_started()` as universal lineage fact authority | Wrong boundary — admission fact born at `ExecutionIdentityBinding`; recorder is projection |
| `ExecutionTreeAdmissionHook` new public protocol | `ExecutionAdmissionHook` already exists |
| Mutable `ExecutionCheckpointEntry` as durable lineage store | Second owner of runtime status; checkpoint-specific |
| Full snapshot CAS as durable model | Ambiguous with mutable checkpoint entries; prefer append-only immutable admissions |
| Run-scoped persistence without `attempt_id` | Violates attempt isolation |
| `DiagnosticExecutionTree` / parallel lineage registry | REUSE FIRST — duplicate authority |
| Inference from event ordering / timestamps | Non-canonical; forbidden |
| `RunBudgetPersistence` as lineage authority | Budget domain |
| Option B as SELECTED | Structural metadata misclassified; dual truth |
| `TaskCheckpoint` as lineage authority | Resume projection only |
| `COMPLETE` without canonical closure authority seal | Incomplete completeness contract; ExecutionRuntime return is not closure |

---

## 22. Implementation boundary

```text
Writer (runtime):
  ROOT:
    ExecutionRuntime
      → ExecutionBoundary
      → ExecutionAdmissionHook (lineage recorder implementation)
      → ExecutionLineagePersistence.admit_root()
      → delegate

  CHILD:
    ChildExecutionRunner
      → ExecutionBoundary
      → ExecutionAdmissionHook (same abstraction)
      → ExecutionLineagePersistence.admit_child()
      → delegate

  PROJECTION (converge later — not this task):
    ExecutionTreeRecorder.record_child_started()
      → subordinate to shared admission path

Canonical structural model (existing):
  ExecutionTreeSnapshot
  ExecutionCheckpointEntry (in-memory / checkpoint projection only)

Durable contract (new):
  ExecutionLineagePersistence
  ExecutionLineageAdmissionRecord (immutable append-only)

Read integration:
  ExecutionReconstructor
    + ExecutionLineagePersistence (attempt-scoped)
    + RuntimeEventPersistence (status join)
    → lineage projection (completeness-aware)
  DiagnosticReadService
    → operator occurrence view

Checkpoint relationship:
  TaskCheckpointPersistence embeds ExecutionTreeSnapshot for resume only
  Conflict with durable lineage → HARD INTEGRITY FAILURE
```

### Decision block

```text
SELECTED_OPTION:
OPTION_A

ATTEMPT_SCOPE_TYPE:
  ExecutionLineageAttemptScope (tenant_id, task_id, run_id, attempt_id)
  ExecutionLineageAdmissionRecord carries execution_id + parent_execution_id

ROOT_TASK_SCOPE_DELIVERY:
  Optional RootExecutionContext.task_id: TaskId | None at task composition paths only;
  lineage persistence wired only when capability active; fail-closed when enabled without task_id;
  resolve_root_execution_context(...) unchanged for generic non-task ExecutionRuntime paths

CHILD_SCOPE:
  Inherit attempt scope only; execution identity from ExecutionIdentityBinding

ADMISSION_SCOPE_DELIVERY:
  Pattern A — lineage ExecutionAdmissionHook constructed with immutable ExecutionLineageAttemptScope
  + admission record fields per ExecutionBoundary; admit(request) never inspects request

RETRY_CLOSURE_AUTHORITY:
  AttemptLifecycleService successful transition_to_next_attempt(...)

FINAL_TERMINAL_AUTHORITY:
  ExecutionTerminalService.commit_terminal_outcome / record_cancellation
  composed at NexusLoop._commit_durable_terminal_authority with active AttemptId

PAUSE_BEHAVIOR:
  Resumable states → attempt OPEN; close_segment_for_resume only; no seal

RESUME_BEHAVIOR:
  Validate segment continuity; open_segment; same AttemptId; attempt OPEN

RAW_EXCEPTION_BEHAVIOR:
  Raw boundary/runtime exception does not seal; seal only after canonical terminal commit

PROCESS_CRASH_BEHAVIOR:
  Unclean segment → permanent PARTIAL/DEGRADED; COMPLETE forbidden after resume

PROCESS_SEGMENT_CONTINUITY:
  open_segment / close_segment_for_resume / unclean detection via durable continuity records

EXECUTION_RUNTIME_RETURN_IS_TERMINAL:
  NO

SEAL_LEGALITY:
  COMPLETE lineage only when durable continuity proves no unknown gap;
  seal correlated with retry transition or terminal authority — not ExecutionRuntime return

ADMISSION_FAILURE_POLICY:
  Fail-open execution MAY continue if durable mark_degraded succeeds;
  crash before durable degradation → unclean segment → PARTIAL on resume

DEGRADATION_DURABILITY:
  Canonical truth: durable mark_degraded + continuity records in ExecutionLineagePersistence;
  AttemptLineageDegradationState ContextVar = RUNTIME_ONLY monotonic indicator

FALSE_COMPLETE_PREVENTION:
  Formal invariant — no crash/resume/retry/terminal sequence yields COMPLETE without durable proof

GENERIC_EXECUTION_RUNTIME_COMPATIBILITY:
  YES — non-task paths remain valid without mandatory task_id

FAILED_ATTEMPT_COMPLETE_LINEAGE_ALLOWED:
  YES (orthogonal execution outcome vs lineage completeness)

CANCELLED_ATTEMPT_COMPLETE_LINEAGE_ALLOWED:
  YES

LINEAGE_FACT_AUTHORITY:
  ExecutionRuntime / ChildExecutionRunner admission boundary
  → ExecutionIdentityBinding (parent_execution_id)
  → durable write via ExecutionAdmissionHook → ExecutionLineagePersistence

CANONICAL_STRUCTURAL_MODEL:
  ExecutionTreeSnapshot / ExecutionCheckpointEntry
  (validated tree invariants; attempt-scoped; in-memory + checkpoint projection)

CANONICAL_DURABLE_SOURCE:
  ExecutionLineagePersistence
  (immutable admissions + segment continuity + degradation + seal per attempt)

PERSISTENCE_SCOPE:
  tenant_id + task_id + run_id + attempt_id (attempt APIs — no execution_id)

ROOT_WRITE_BOUNDARY:
  ExecutionRuntime → ExecutionBoundary → ExecutionAdmissionHook (scope-bound)
  → open_segment → ExecutionLineagePersistence.admit_root → root delegate

CHILD_WRITE_BOUNDARY:
  ChildExecutionRunner → ExecutionBoundary → ExecutionAdmissionHook (scope-bound)
  → ExecutionLineagePersistence.admit_child → child delegate

EXISTING_ADMISSION_PROTOCOL:
  ExecutionAdmissionHook[RequestT] (reuse — no new protocol)

COMPLETENESS_PROTOCOL:
  OPEN + segment continuity + admissions + canonical closure authority + seal → COMPLETE
  degraded / unclean segment / missing seal / crash → PARTIAL or UNAVAILABLE

FAILURE_POLICY:
  Segment open: fail-closed if continuity marker unavailable
  Admission availability: fail-open only with durable mark_degraded
  Structural conflict → hard integrity failure

CHECKPOINT_RELATIONSHIP:
  TaskCheckpoint.runtime.execution_tree = resume/checkpoint projection only
  Durable lineage parent mapping wins; conflict → HARD INTEGRITY FAILURE

RETRY_SEMANTICS:
  A1 sealed on successful AttemptLifecycle transition; A2 open_attempt independently

STATUS_SOURCE:
  RuntimeEventPersistence / DIAG-2 runtime evidence (read-side join in ExecutionReconstructor)

NEW_ABSTRACTIONS_REQUIRED:
  ExecutionLineagePersistence: REQUIRED
  ExecutionLineageAdmissionRecord: REQUIRED
  ExecutionLineageAttemptScope: REQUIRED (ExecutionLineageScope alias allowed if attempt-only)
  AttemptLineageDegradationState: RUNTIME_ONLY
  Durable lineage attempt-state / segment-state contract: REQUIRED (same store — REUSED)
  New admission protocol: NO
  New terminal authority: NO
  New attempt authority: NO
  ExecutionTreeAdmissionHook protocol: NOT REQUIRED — reuse ExecutionAdmissionHook

DIAGNOSTICS_CORE_CHANGE_REQUIRED:
  YES (read projection + completeness + status join — ExecutionReconstructor / read models)

EXECUTION_ENGINE_CHANGE_REQUIRED:
  YES (admission durable write; segment continuity; seal at retry/terminal composition boundaries)

CAUSAL_EVIDENCE_CHANGE_REQUIRED:
  NO
```

---

## 23. Final verdict

```text
GAP_R1_01: RESOLVED_ARCHITECTURALLY (CORRECTED CONTRACTS)

WHO owns parent→child lineage fact?
  → ChildExecutionRunner / ExecutionRuntime admission + ExecutionIdentityBinding

WHO owns durable lineage?
  → ExecutionLineagePersistence (attempt-scoped immutable admissions)

WHO owns structural model?
  → ExecutionTreeSnapshot (validated; attempt-scoped)

WHEN does the edge become canonical?
  → At admission boundary (ExecutionIdentityBinding); durably at ExecutionLineagePersistence write

WHERE is it durable?
  → ExecutionLineagePersistence (not TaskCheckpoint alone)

WHAT happens on crash?
  → Today: lineage lost (BLOCKING for forensic completeness)
  → After remediation: admission durable write preserves edge before delegate work

HOW does Diagnostics read it?
  → ExecutionReconstructor + ExecutionLineagePersistence (attempt-scoped) + completeness seal
  → seal correlated with AttemptLifecycleService retry transition or ExecutionTerminalService commit

HOW do we avoid duplicate authority?
  → Single admission fact path; immutable structural store; status from RuntimeEventPersistence;
    checkpoint is projection only; no causal duplicate; budget ledger excluded;
    ExecutionRuntime return is not closure authority
```

---

## 24. Recommended next task

**`DG-001-MULTI-AGENT-EXECUTION-LINEAGE-ADMISSION-PERSISTENCE-R1`**

Minimal implementation surface:

1. Define `ExecutionLineageAttemptScope`, `ExecutionLineageAdmissionRecord`, and `AttemptLineageDegradationState` (runtime-only) typed contracts.
2. Optional `RootExecutionContext.task_id: TaskId | None` at task composition boundary only.
3. Define `ExecutionLineagePersistence` public ABC (attempt-scoped: `open_attempt`, `open_segment`, `admit_root`, `admit_child`, `close_segment_for_resume`, `mark_degraded`, `seal_attempt`, `list_admissions_for_attempt`, `read_seal`, `read_attempt_lineage_state`).
4. Implement scope-bound `ExecutionAdmissionHook` lineage recorder at root and child admission.
5. Wire seal at retry transition handler (`AttemptLifecycleService`) and terminal commit handler (`NexusLoop._commit_durable_terminal_authority` + `ExecutionTerminalService`) — **not** at `ExecutionRuntime` return.
6. Extend `ExecutionReconstructor` with attempt-scoped lineage projection, status join, and `LineageCompleteness`.
7. Harness host wiring for persistence adapter.
8. Execute qualification scenarios Q1–Q35.

---

## 25. Tests executed (supporting evidence)

Focused regression at correction START HEAD — prior evidence from R1 draft:

- `tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py`
- `tests/unit/runtime/long_running/test_runtime_checkpoint.py`
- `tests/unit/runtime/execution/test_child_execution.py`
- `tests/unit/runtime/execution/test_graph_executor_child_execution.py`
- `tests/unit/runtime/nexus/execution/test_ue_8ar1_execution_tree_authority.py`
- `tests/unit/runtime/execution/test_ue_11e_resume_recovery.py`
- `tests/unit/runtime/diagnostics/test_execution_reconstruction.py`
- `tests/unit/runtime/observability/test_causal_evidence_contract.py`
- `tests/unit/runtime/observability/test_durable_causal_evidence_persistence.py`

**No new tests in correction task.**

---

## 26. Confirmations

- No production changes in this task
- No parallel Execution Tree model
- No duplicate lineage authority
- Existing `ExecutionAdmissionHook` reused — no `ExecutionTreeAdmissionHook` protocol
- Attempt trees isolated (`tenant_id + task_id + run_id + attempt_id`)
- Root persistence defined (durable before root delegate)
- No heuristic lineage inference
- No Diagnostics bypass of canonical contracts
- No Decision System changes
- No causal evidence change
- No private API requirement
- No `getattr` / `setattr` / dynamic dict contracts proposed
- No identity copied into neutral `ExecutionRequest`
- No request introspection for lineage scope
- `ExecutionLineageAttemptScope` + `ExecutionLineageAdmissionRecord` + durable segment continuity contracts defined
- `AttemptLineageDegradationState` runtime-only; durable degradation via `mark_degraded`
- Terminal seal at retry/terminal composition boundaries — **not** `ExecutionRuntime` return
- False COMPLETE prevented by durable continuity + seal legality (not ContextVar alone)
- Same AttemptId resume handled via segment continuity
- Generic non-task `ExecutionRuntime` paths preserved
- No branch / worktree / history rewrite

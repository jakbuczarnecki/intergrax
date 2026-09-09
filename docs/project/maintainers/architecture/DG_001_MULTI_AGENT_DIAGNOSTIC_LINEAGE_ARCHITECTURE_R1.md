# DG-001 — Multi-agent diagnostic execution lineage architecture (R1)

> **Task:** `DG-001-CROSS-SYSTEM-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ARCHITECTURE-R1`  
> **Resolves:** GAP-R1-01 from `DG-001-CROSS-SYSTEM-DIAGNOSTIC-COMPATIBILITY-AUDIT-R1`  
> **Mode:** architecture decision + contract ownership audit — **no implementation**  
> **Branch:** `development`  
> **START HEAD:** `74aff023983c126d0bfd65a0e6c79a194f931353`  
> **Ancestry verified:** `d5c5b1cf` (cross-system correction) · `91568550` (B4 revalidation)

---

## 1. Problem statement

Execution runtime deterministically knows the direct parent→child execution edge at the child admission boundary:

```text
ChildExecutionRunner / ExecutionBoundary
  → ExecutionIdentityBinding.parent_execution_id
  → child ExecutionId minted under active parent
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

**Identity authority ≠ lineage authority.** Minting a child `ExecutionId` does not by itself establish durable or diagnostic-readable parent linkage.

### Lineage authority

```text
LINEAGE_AUTHORITY:
ExecutionTreeRecorder.record_child_started()
  → ExecutionCheckpointEntry.parent_execution_id
  → ExecutionTreeSnapshot (validated tree)
```

Owned by the **runtime execution / checkpoint layer** (`intergrax/runtime/long_running/execution_tree_checkpoint.py`), integrated at graph execution via `GraphExecutor._execute_node_in_child()`.

`ExecutionIdentityBinding.parent_execution_id` in `ChildExecutionRunner` is the **runtime admission fact** consumed to call `record_child_started`; the recorder snapshot is the **authoritative structural tree state** for resume, checkpoint rebuild, and (after remediation) diagnostics.

**No competing canonical lineage authority exists today.** `RunBudgetLedger` records `parent_execution_id` for budget allocation — budget semantics, not execution lineage (see §8 Option C).

---

## 3. Current write paths

| Stage | Writer | What is written | Durability |
| ----- | ------ | --------------- | ---------- |
| Child admission | `ChildExecutionRunner` | `ExecutionIdentityBinding` in active context | Process memory only |
| Child budget grant | `ExecutionBudgetLedger.grant_child_budget()` | `PersistedBudgetRecord.parent_execution_id` when durable ledger wired | `RunBudgetPersistence` CAS on grant (budget domain) |
| Child tree admission | `ExecutionTreeRecorder.record_child_started()` via `GraphExecutor` | `ExecutionCheckpointEntry` with `parent_execution_id` | Process memory (`ExecutionTreeRecorder._snapshot`) |
| Node completion | `record_graph_node_completion()` + `sync_execution_tree_to_task()` | Updated entry status / `prior_output`; tree copied into `task.runtime.orchestration.runtime_checkpoint` | Process memory (task object) |
| Long-running pause | `LongRunningCoordinator.persist_checkpoint()` → `TaskCheckpointPersistence.save()` | Full `RuntimeCheckpoint.execution_tree` inside `TaskCheckpoint` | Durable — **only when** `long_running.enabled` and `checkpoint_on_pause` and pause checkpoint fires |
| DIAG-2 sources | `RuntimeEventPersistence.append()` | `RuntimeEvent` with `execution_id`, no `parent_execution_id` | Durable observability stream |
| Causal evidence | `CausalEvidencePersistence.append()` | `TRANSPORT_TASK_TRIGGERED_EXECUTION` only | Durable cross-boundary transport fact |

**Coverage gap:** `record_child_started` is invoked only from `GraphExecutor._execute_node_in_child()`. Other `ChildExecutionRunner` paths (orchestration slot work, application child ports) mint child identity but do **not** currently call `ExecutionTreeRecorder`.

---

## 4. Current durable sources

| Source | Contains `parent_execution_id`? | Scope | Diagnostic read today? |
| ------ | ------------------------------- | ----- | ---------------------- |
| `ExecutionTreeSnapshot` in `TaskCheckpoint.runtime` | Yes | Long-running pause checkpoints | `TaskCheckpointReader` — not wired into `ExecutionReconstructor` |
| In-memory `ExecutionTreeRecorder` / `task.runtime_checkpoint` | Yes | Active process | No |
| `RunBudgetLedgerSnapshot.records` | Yes (budget record) | Per-run budget | No — wrong semantic domain |
| `RuntimeEvent` stream | No | Per-run events | Yes (DIAG-2) |
| `PlatformCausalEvidence` | No parent→child relation | Transport→execution | Yes (DIAG-2) |

---

## 5. Crash / failure durability

### Critical crash scenario

```text
parent running
  → child ExecutionId minted (ChildExecutionRunner)
  → child context bound
  → record_child_started (in-memory tree)
  → child work begins
  → PROCESS CRASH before next TaskCheckpoint save
```

**Verdict:** the parent→child edge is **not** retained in any canonical **lineage** durable source today.

Supporting evidence:

- `ExecutionTreeRecorder` is in-memory until `sync_execution_tree_to_task` (still in-memory) or `TaskCheckpointPersistence.save` (long-running pause only).
- `RunBudgetPersistence` may retain `parent_execution_id` on `grant_child_budget` when a durable budget ledger is wired, but that is **budget allocation truth**, not execution lineage authority. Using it for diagnostics would violate single-authority and mis-label structural edges as budget facts.

### Success / failure path matrix

| Moment | Edge in runtime | Edge durable (lineage) | Diagnostics can read | Exact durable source when YES |
| ------ | --------------- | ---------------------- | -------------------- | ------------------------------ |
| Child minted | Contextvars + optional budget record | NO (lineage) | NO | — |
| `record_child_started` | `ExecutionTreeRecorder` | NO | NO | — |
| Child started (work running) | Same | NO | NO | — |
| Child failed (no pause checkpoint) | In-memory tree entry `FAILED` | NO | NO | — |
| Child completed (no pause checkpoint) | In-memory tree entry `COMPLETED` | NO | NO | — |
| Process crashed mid-child | Lost (lineage) | NO | NO | — |
| Long-running checkpoint persisted | `ExecutionTreeSnapshot` | YES | Not via DIAG-2 | `TaskCheckpointPersistence` → `TaskCheckpoint.runtime.execution_tree` |
| Resume from checkpoint | Tree restored into recorder | YES (last saved snapshot) | Not via DIAG-2 | Same checkpoint store |

---

## 6. ExecutionTreeSnapshot semantics (Option A truth audit)

| # | Question | Answer |
| - | -------- | ------ |
| 1 | Created for every canonical run? | **No.** Recorder starts in `GraphExecutor.execute()` for graph runs; not universal for all `ChildExecutionRunner` paths. |
| 2 | When does root entry appear? | `ExecutionTreeRecorder.start_root()` at graph execute start (or `from_snapshot` on resume). |
| 3 | When does child entry appear? | `record_child_started()` before child node work in `GraphExecutor._execute_node_in_child()`. |
| 4 | Before child work? | **Yes** — `record_child_started` precedes agent execution in that path. |
| 5 | Nested children represented? | **Yes** — each child entry carries its direct `parent_execution_id`; `validate_tree()` enforces single root, known parents, acyclic graph. |
| 6 | Failure before checkpoint persistence loses edge? | **Yes** — in-memory only until `TaskCheckpointPersistence.save`. |
| 7 | Normal successful execution always persists tree? | **No** — only if long-running checkpoint path fires. |
| 8 | Failed execution always persists tree? | **No** — same condition. |
| 9 | Snapshot exists after process end? | **Only** if checkpoint was saved; otherwise lost. |
| 10 | Diagnostics public read of snapshot? | **No** — `ExecutionReconstructor` does not consume `TaskCheckpointReader` or tree persistence. |

**Conclusion:** `ExecutionTreeSnapshot` is the correct **semantic and validation authority** but is **not** presently a sufficient **durable forensic source** for Diagnostics without an admission-time durable write and a public read contract.

---

## 7. Option A — Reuse existing Execution Tree

```text
ExecutionTreeRecorder (admission)
  → durable ExecutionTreeSnapshot / ExecutionCheckpointEntry store (new admission write)
  → public read contract
  → ExecutionReconstructor lineage projection
  → DiagnosticReadService operator view
```

### Advantages

- Reuses existing `ExecutionCheckpointEntry.parent_execution_id`, `validate_tree()`, resume planning (`build_execution_tree_resume_plan`).
- No duplicate tree semantics; same model serves checkpoint/resume and diagnostics.
- Nested depth, fan-out, and direct edges (`E4.parent = E2`) are already modeled.
- Conflict policy already exists: duplicate `execution_id` raises `ValueError`; tree validation rejects unknown parents and cycles.

### Risks (current state)

- Checkpoint persistence is **pause-scoped**, not admission-scoped.
- Edge may be recorded later than failure if only in-memory.
- Crash loses last edges.
- Diagnostics lacks public read integration.
- Not all child paths record into the tree today.
- Retention tied to checkpoint store unless admission store is run-scoped.

### Verdict

**`VIABLE_WITH_CONDITIONS`**

Conditions for SELECTED:

1. **Admission-time durable write** of each new `ExecutionCheckpointEntry` (or full snapshot CAS per run) — not only on long-running pause.
2. **Universal recording** at child admission for all canonical `ChildExecutionRunner` paths.
3. **Public run-scoped read contract** for `ExecutionTreeSnapshot` entries (see §13).
4. **`ExecutionReconstructor` integration** with completeness semantics (§14).

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
| Semantic fit | Parent→child is **execution structural metadata**, not a cross-boundary causal fact like transport→execution. Overloading `PlatformCausalEvidence` blurs DIAG-1 transport boundary purpose. |
| Schema today | `RuntimeExecutionRef` lacks `execution_id`; only `task_id`, `run_id`, `attempt_id`, `tenant_id`. Extension required. |
| Durability | `CausalEvidencePersistence` is durable and already consumed by DIAG-2. |
| Duplicate authority risk | High unless Execution Tree is demoted to projection-only — contradicts REUSE FIRST and existing resume/checkpoint ownership. |
| Admission timing | Writable at child boundary, but duplicates tree truth. |

### Verdict

**`VIABLE_WITH_CONDITIONS`** as a **durable forensic projection only** — but **rejected as SELECTED** because it creates dual truth unless Execution Tree is demoted, and structural lineage already belongs to Execution Tree machinery.

---

## 9. Option C — Other existing public contract

| Candidate | `execution_id` + `parent_execution_id`? | Verdict |
| --------- | --------------------------------------- | ------- |
| `TaskCheckpointPersistence` / `TaskCheckpoint.runtime.execution_tree` | Yes | **Partial** — durable only on long-running pause; wrong lifecycle for all-run forensic lineage |
| `RunBudgetPersistence` / `PersistedBudgetRecord` | Yes | **Rejected** — budget allocation domain; not lineage authority; would mislead diagnostics |
| `RuntimeEventPersistence` | No `parent_execution_id` | Not applicable |
| `CausalEvidencePersistence` | No parent→child relation | Not applicable |
| Active contextvars | Yes (in-process) | Not durable — not applicable |

### Verdict

**`NO_EXISTING_OPTION_C`** as a complete solution. `TaskCheckpointPersistence` is a **partial** carrier of `ExecutionTreeSnapshot` but does not satisfy admission-time durability or universal coverage.

---

## 10. Comparison matrix

Scores: **Strong** / **Partial** / **Weak** / **N/A** — qualitative, not point totals.

| Criterion | A: Execution Tree | B: Causal Evidence | C: Existing contract |
| --------- | ----------------: | -----------------: | -------------------: |
| Canonical ownership | Strong | Weak (structural fact in causal store) | Partial (checkpoint only) |
| Durability | Partial → Strong after admission write | Strong | Partial |
| Crash safety | Weak today → Strong after admission write | Strong if added | Weak |
| Availability on success | Partial | Strong if added | Partial |
| Availability on failure | Partial | Strong if added | Partial |
| Nested execution | Strong | Strong if typed refs added | Partial |
| Retry / resume | Strong (existing resume plan) | Weak (attempt vs execution conflation risk) | Partial |
| Tenant integrity | Strong (via run/task scope + validation) | Strong | Strong |
| Idempotency | Strong (`duplicate execution_id` fails) | Strong (evidence_id idempotency) | Partial |
| Read efficiency | Strong (bounded tree by run) | Strong (indexed append-only) | Partial |
| Retention alignment | Configurable per lineage store | Aligns with causal retention | Checkpoint retention only |
| No duplicate truth | Strong | Weak | Partial |
| Backward compatibility | Strong (`UNAVAILABLE` for old runs) | Strong | Strong |
| Implementation complexity | Medium (extend existing tree path) | Medium–High (new relation + dual-write risk) | Low but insufficient |

---

## 11. Selected architecture

```text
SELECTED_OPTION: OPTION_A
```

**Existing Execution Tree is canonical lineage truth; Diagnostics must reuse it via a public durable read path with admission-time persistence.**

Rationale:

1. REUSE FIRST — `ExecutionTreeSnapshot` / `ExecutionCheckpointEntry` already own parent→child semantics, validation, resume, and checkpoint integration.
2. Option A gaps are **integration and durability timing**, not missing domain model.
3. Option B misclassifies structural lineage as cross-boundary causal evidence and invites dual authority.
4. Option C has no complete existing durable contract.

---

## 12. Authority and ownership

| Concern | Owner | Must not own |
| ------- | ----- | ------------ |
| Lineage structural truth | `ExecutionTreeRecorder` + `ExecutionTreeSnapshot` (runtime execution layer) | Diagnostic Engine |
| Identity minting | `identity_authority` / `ChildExecutionRunner` | Diagnostics |
| Durable lineage storage | Platform persistence adapter (run-scoped; see §22) | Diagnostics |
| Operator diagnostic projection | `ExecutionReconstructor` → `DiagnosticReadService` | — |

**Diagnostic Engine consumes execution truth; it does not create execution truth.**

Decision System (`DecisionIdentity`, `DecisionExecutionLineage`) remains **out of scope** for execution parent→child edges. The selected design does not block a future Decision→Execution join (GAP-R1-02).

---

## 13. Write semantics

### When the edge becomes canonical

1. **Runtime canonical:** `ExecutionTreeRecorder.record_child_started(execution_id, parent_execution_id, …)` — same process, before child work on covered paths.
2. **Durable canonical (required remediation):** immediate append/CAS of the `ExecutionCheckpointEntry` to the run-scoped lineage store at child admission — **after** in-memory recorder acceptance, **before** child delegate work proceeds (or atomically with recorder acceptance in one bounded write).

### Write boundary

```text
ChildExecutionRunner.execute()
  → (existing) mint + budget + identity bind
  → NEW: ExecutionTreeAdmissionHook (public protocol)
       → ExecutionTreeRecorder.record_child_started()
       → ExecutionLineagePersistence.append_entry()  [new public contract — see §22]
```

Centralizing admission recording in a hook invoked from `ChildExecutionRunner` (or a single orchestration adapter) fixes the GraphExecutor-only coverage gap.

### Idempotency

- Re-admission with same `execution_id` + same `parent_execution_id` → no-op success (align with `ExecutionTreeRecorder` duplicate guard or explicit idempotent append).
- Re-admission with same `execution_id` + **different** `parent_execution_id` → **hard integrity failure** (`ExecutionLineagePersistenceConflictError` or equivalent).

### Conflict policy

**Hard integrity failure** — never last-write-wins. Matches `ExecutionTreeRecorder` duplicate `execution_id` behavior and `RuntimeEventPersistence` fingerprint conflict pattern.

### Tenant isolation

All reads and writes scoped by `tenant_id` + `task_id` + `run_id`. Cross-tenant parent/child admission **MUST FAIL** at persistence boundary.

### Cycle prevention

Reuse `ExecutionTreeSnapshot.validate_tree()` / `_assert_acyclic()` on bounded read assembly; reject entries that would introduce cycles.

---

## 14. Read semantics

### Minimal read capabilities

Reuse `ExecutionTreeSnapshot` / `ExecutionCheckpointEntry` — **do not** introduce `DiagnosticExecutionTree` or parallel hierarchy types.

Logical projection (name illustrative; implement reuse-first):

| Capability | Source |
| ---------- | ------ |
| Root execution | Entry with `parent_execution_id is None` |
| Direct parent | `entry.parent_execution_id` |
| Direct children | Entries where `parent_execution_id == execution_id` |
| Bounded full tree | All entries for `(tenant_id, task_id, run_id)` with limit |
| Status / failure linkage | `ExecutionCheckpointStatus` + optional `prior_output` summary fields (no payloads) |
| Incomplete evidence | Lineage completeness enum (reuse pattern from `RuntimeHistoryCompleteness`) |

### Integration point

```text
ExecutionLineagePersistence (read)
  → ExecutionReconstructor.reconstruct_execution() [extended]
  → lineage projection with completeness
  → DiagnosticReadService occurrence view
```

### Completeness semantics (reuse DIAG-2 pattern)

Extend reconstruction output with lineage completeness aligned to `RuntimeHistoryCompleteness`:

| Status | Meaning |
| ------ | ------- |
| `COMPLETE` | All entries for run loaded within bounds; tree validates |
| `PARTIAL` | Truncated page, missing admission records on some paths, or checkpoint-only historical data |
| `UNAVAILABLE` | No durable lineage evidence for run |

**Never** return a partial tree as complete. **Never** infer `parent = root` for missing edges.

---

## 15. Integrity invariants

1. Exactly one root per run attempt tree snapshot scope.
2. Every non-root entry references an existing parent `execution_id` in the same run scope.
3. Graph is acyclic.
4. `execution_id` is unique within the run lineage store.
5. `(child_execution_id → parent_execution_id)` is immutable once admitted.
6. Retry creates a **new** `AttemptId` via `AttemptLifecycleService`; lineage entries remain tied to the executions minted under their attempts — diagnostics must not infer tree shape from attempt ordering alone.
7. `ExecutionId` is stable across retry attempts within the same execution invocation scope; retry transitions mint new attempt, not new execution identity for the same logical retry boundary (per `AttemptLifecycleService`).

---

## 16. Failure isolation

**Classification:** execution parent→child lineage is a **diagnostics / observability evidence requirement**, not an execution integrity gate (unlike budget reservation or authority policy).

**Policy: observability degradation (fail-open for execution, fail-closed for lineage truth)**

| Failure | Execution continues? | Diagnostics behavior |
| ------- | -------------------- | -------------------- |
| Lineage durable write fails after in-memory record | **Yes** | Mark lineage `PARTIAL` / `UNAVAILABLE`; surface degradation flag |
| Lineage write conflict (duplicate parent mismatch) | **No** — integrity violation | Hard error at admission |
| Lineage read missing | N/A | `UNAVAILABLE` — no inference |

Budget grant failure remains independent and may still fail-closed child admission — that is budget integrity, not lineage.

---

## 17. Backward compatibility

Historical runs without admission-time lineage persistence:

```text
lineage_completeness = UNAVAILABLE or PARTIAL
```

Never default missing parent to root. Never reconstruct tree from:

- `node_id`, `agent_id`, event ordering, `correlation_id`, timestamps, or `DELEGATION_GRANTED` payloads.

---

## 18. Security

Lineage durable records contain **structural facts only**:

```text
tenant_id, task_id, run_id, attempt_id,
parent_execution_id, child_execution_id,
relation marker (implicit in entry shape),
admission timestamp or monotonic position,
optional graph_node_id (structural ref, not heuristic lineage)
```

**Exclude:** prompts, agent payloads, PII, secrets, raw exceptions. `prior_output` in checkpoints is resume-oriented; lineage read for diagnostics should expose status/summary fields only through existing bounded contracts.

---

## 19. Scale / performance requirements (next task)

- Support 1 parent + thousands of direct children via paginated `list_entries_for_run`.
- Support nesting depth ≥ 3 with bounded traversal (no full event-store scan).
- Index by `(tenant_id, task_id, run_id)` and `execution_id`.
- Avoid N+1 cross-run queries; lineage read is run-scoped.
- Idempotent append suitable for concurrent child fan-out (sibling admissions).

---

## 20. Future qualification scenarios

| ID | Scenario |
| -- | -------- |
| Q1 | Root + single child |
| Q2 | Root + 3 siblings |
| Q3 | Nested tree depth ≥ 3 |
| Q4 | One child failure |
| Q5 | Partial sibling failure |
| Q6 | Retry (new `AttemptId`, stable execution lineage per attempt) |
| Q7 | Resume from checkpoint |
| Q8 | Process crash after child admission (lineage durability proof) |
| Q9 | Duplicate edge admission (idempotent) |
| Q10 | Conflicting parent edge (integrity failure) |
| Q11 | Cross-tenant rejection |
| Q12 | Historical run without lineage evidence (`UNAVAILABLE`) |

---

## 21. Explicitly rejected alternatives

| Alternative | Reason |
| ----------- | ------ |
| Diagnostic-owned Execution Tree | Violates ownership — Diagnostics consumes, does not author |
| `DiagnosticExecutionTree` / parallel lineage registry | REUSE FIRST — duplicate authority |
| Inference from event ordering / timestamps | Non-canonical; forbidden |
| Inference from `node_id` / `agent_id` | Non-canonical; forbidden |
| `parent_execution_id` only in arbitrary event payload | Not typed canonical contract |
| Duplicate lineage store separate from Execution Tree model | Duplicate semantics |
| Diagnostics writing lineage from `ChildExecutionRunner` | Wrong ownership boundary |
| Problem metadata as lineage storage | Wrong domain |
| Decision System as execution lineage authority | Out of scope; separate identity domain |
| `RunBudgetPersistence` as lineage authority | Budget domain; dual truth |
| Option B as SELECTED (causal evidence as primary lineage authority) | Structural metadata misclassified; dual truth with Execution Tree |

---

## 22. Implementation boundary

```text
Writer (runtime):
  ChildExecutionRunner
    → ExecutionTreeAdmissionHook [new public protocol at admission boundary]
    → ExecutionTreeRecorder.record_child_started() [existing]
    → ExecutionLineagePersistence.append_entry() [new public ABC]

Canonical contract (existing models):
  ExecutionTreeSnapshot
  ExecutionCheckpointEntry
  ExecutionCheckpointStatus

Persistence (new adapter, existing schema):
  ExecutionLineagePersistence
    implementations: SQLite / harness host wiring (plugin boundary)

Read integration:
  ExecutionReconstructor
    + ExecutionLineagePersistence
    → lineage projection (completeness-aware)
  DiagnosticReadService
    → operator occurrence view (unchanged orchestration pattern)
```

`TaskCheckpointPersistence` continues to embed `ExecutionTreeSnapshot` for **resume/checkpoint** semantics. Admission lineage store may share storage backend but must not couple diagnostic reads to pause-only checkpoint lifecycle.

### Decision block

```text
SELECTED_OPTION: OPTION_A

CANONICAL_LINEAGE_AUTHORITY:
  ExecutionTreeRecorder / ExecutionCheckpointEntry / ExecutionTreeSnapshot
  (intergrax/runtime/long_running/execution_tree_checkpoint.py)

CANONICAL_DURABLE_SOURCE:
  ExecutionLineagePersistence storing ExecutionCheckpointEntry rows per run
  (new public contract; schema reuse from ExecutionTreeSnapshot)

WRITE_BOUNDARY:
  ChildExecutionRunner admission hook → recorder → ExecutionLineagePersistence.append_entry

READ_BOUNDARY:
  ExecutionReconstructor (+ ExecutionLineagePersistence) → DiagnosticReadService

FAILURE_POLICY:
  Durable lineage write failure → execution continues; lineage marked PARTIAL/UNAVAILABLE
  Lineage conflict → hard integrity failure at admission

BACKWARD_COMPATIBILITY:
  Pre-remediation runs → lineage UNAVAILABLE/PARTIAL; no root inference

PRODUCTION_COMPONENTS_REUSED:
  ExecutionTreeSnapshot, ExecutionCheckpointEntry, ExecutionTreeRecorder,
  ChildExecutionRunner, ExecutionReconstructor, CausalEvidencePersistence (unchanged),
  RuntimeEventPersistence (unchanged), TaskCheckpointPersistence (resume path unchanged)

PRODUCTION_COMPONENTS_REQUIRING_CHANGE:
  ChildExecutionRunner (admission hook wiring),
  GraphExecutor (delegate recording to shared admission hook),
  ExecutionReconstructor (lineage projection),
  harness diagnostic wiring (inject ExecutionLineagePersistence)

NEW_ABSTRACTIONS_REQUIRED:
  ExecutionLineagePersistence (public ABC — append + bounded list_for_run)
  ExecutionTreeAdmissionHook (public protocol — optional if hook folded into single admission coordinator)

DIAGNOSTICS_CORE_CHANGE: YES (read projection only — ExecutionReconstructor / read models)

EXECUTION_ENGINE_CHANGE: YES (admission durable write + universal recording — no semantic fork)

CAUSAL_EVIDENCE_CHANGE: NO
```

---

## 23. Final verdict

```text
GAP_R1_01: RESOLVED_ARCHITECTURALLY

WHO owns parent→child truth?
  → ExecutionTreeRecorder / ExecutionCheckpointEntry (runtime execution layer)

WHEN does the edge become canonical?
  → At record_child_started (runtime); durably at admission append (required remediation)

WHERE is it durable?
  → ExecutionLineagePersistence (new run-scoped store reusing ExecutionTreeSnapshot entries)
  → TaskCheckpointPersistence remains resume projection, not sole forensic source

WHAT happens on crash?
  → Today: lineage lost (BLOCKING for forensic completeness)
  → After remediation: admission durable write preserves edge before child work

HOW does Diagnostics read it?
  → ExecutionReconstructor + ExecutionLineagePersistence with completeness semantics

HOW do we avoid duplicate authority?
  → Single lineage model (Execution Tree); no causal duplicate; budget ledger excluded
```

---

## 24. Recommended next task

**`DG-001-MULTI-AGENT-EXECUTION-LINEAGE-ADMISSION-PERSISTENCE-R1`**

Minimal implementation surface:

1. Define `ExecutionLineagePersistence` public ABC (`append_entry`, `list_entries_for_run` with pagination).
2. Wire `ExecutionTreeAdmissionHook` at `ChildExecutionRunner` admission for all child paths.
3. Extend `ExecutionReconstructor` with lineage projection + `LineageCompleteness`.
4. Harness host wiring for persistence adapter.
5. Execute qualification scenarios Q1–Q12.

---

## 25. Tests executed (supporting evidence)

Focused regression at START HEAD — **100 passed** (13.2s):

- `tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py`
- `tests/unit/runtime/long_running/test_runtime_checkpoint.py`
- `tests/unit/runtime/execution/test_child_execution.py`
- `tests/unit/runtime/execution/test_graph_executor_child_execution.py`
- `tests/unit/runtime/nexus/execution/test_ue_8ar1_execution_tree_authority.py`
- `tests/unit/runtime/execution/test_ue_11e_resume_recovery.py`
- `tests/unit/runtime/diagnostics/test_execution_reconstruction.py`
- `tests/unit/runtime/observability/test_causal_evidence_contract.py`
- `tests/unit/runtime/observability/test_durable_causal_evidence_persistence.py`

Log: `.tmp/session/dg-001-lineage-r1/pytest.log`

---

## 26. Confirmations

- No production changes in this task
- No parallel Execution Tree model
- No duplicate lineage authority
- No Diagnostics bypass of canonical contracts
- No Decision System changes
- No private API requirement
- No `getattr` / `setattr` / dynamic dict contracts proposed
- No branch / worktree / history rewrite

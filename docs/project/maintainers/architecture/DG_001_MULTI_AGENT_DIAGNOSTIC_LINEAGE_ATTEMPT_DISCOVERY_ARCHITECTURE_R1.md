# DG-001 — Multi-agent diagnostic lineage attempt discovery architecture (R1)

> **Task:** `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ATTEMPT-DISCOVERY-ARCHITECTURE-R1`  
> **Mode:** docs-only architecture — **no production implementation**  
> **Branch:** `development`  
> **START_HEAD:** `600e9ca8979b787287154f8b2303a2b1eaf61a86`  
> **Ancestry verified:** `156f1defb42235fb5c3d720a41f1fcb873ad542e` · `c8225da1aa40dd82195070f9ab193d66631bba1a`  
> **READ_INTEGRATION_BASE:** `c8225da1aa40dd82195070f9ab193d66631bba1a`  
> **Supersedes attempt-discovery claim in:** `docs/project/maintainers/qualification/DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_READ_INTEGRATION_R1.md`

---

## 1. Problem statement

Independent audit identified an architecture gap:

```text
ExecutionLineagePersistence may hold a durable AttemptId
that is not yet present in RuntimeEventPersistence or CausalEvidencePersistence.
```

DIAG-2 `ExecutionReconstructor` currently discovers attempts only as:

```text
RuntimeEvent attempts UNION CausalEvidence attempts
```

Lineage is per-attempt enrichment keyed by known `attempt_id`. There is no run-scoped, bounded, durable discovery authority for lineage-only attempts.

This document defines the canonical discovery contract and selects the architecture correction path. It does **not** implement production code.

---

## 2. Write-path trace — counterexample proof

### 2.1 Production call chain (root execution)

| Step | Component | Call site | Durable side effect |
| ---- | --------- | --------- | ------------------- |
| 1 | `HostTaskExecution.execute` | `intergrax/runtime/execution/host_task.py` | none |
| 2 | `Execution.execute` → `ExecutionRuntime.execute` | `intergrax/runtime/execution/runtime.py` | none |
| 3 | `activate_root_execution_lineage` | `intergrax/runtime/execution/lineage/root_activation.py` | **`open_attempt` + `open_segment`** |
| 4 | `ExecutionBoundary.execute` | `intergrax/runtime/execution/boundary.py` | admission hooks before delegate |
| 5 | `ExecutionLineageRootAdmissionHook.admit` | `intergrax/runtime/execution/lineage/admission.py` | **`admit_root`** |
| 6 | delegate (`HostTaskExecution` inner runtime) | after admission | RuntimeEvent publication begins |
| 7 | transport / boundary paths | during execution | CausalEvidence publication (when applicable) |

**Ordering contract (verified in production wiring):**

```text
open_attempt
  → open_segment
  → (ExecutionBoundary created)
  → admit_root          # admission hook, before delegate
  → delegate.execute    # RuntimeEvent + CausalEvidence publication
```

`activate_root_execution_lineage` is invoked in `ExecutionRuntime.execute` **before** `ExecutionBoundary` construction. `admit_root` runs inside `ExecutionBoundary._run_admission_and_delegate` **before** `delegate.execute`.

Test proof: `tests/unit/runtime/execution/lineage/test_execution_lineage_admission_order.py::test_root_admission_before_delegate` — delegate is never reached when `admit_root` raises.

### 2.2 Durable lineage before diagnostic facts

`open_attempt` creates durable attempt meta via `put_if_absent` on the attempt partition:

```189:211:intergrax/runtime/execution/lineage/persistence.py
    def open_attempt(
        self, scope: ExecutionLineageAttemptScope
    ) -> ExecutionLineageAttemptState:
        partition = execution_lineage_partition_key(scope)
        ...
        initial = _initial_attempt_state(scope)
        created = self._store.put_if_absent(
            _PartitionRow(
                partition, _META_ROW, encode_execution_lineage_attempt_state(initial)
            ),
        )
```

`open_segment` durably updates attempt state and creates segment row in the same partition.

Neither step requires or publishes RuntimeEvent or CausalEvidence.

### 2.3 Legal crash window A — before root admission

```text
open_attempt        → durable attempt meta exists
open_segment        → durable segment + attempt state exists
PROCESS CRASH       → before ExecutionBoundary admit_root completes
```

**Post-crash durable facts:**

| Store | AttemptId present? |
| ----- | ------------------ |
| ExecutionLineagePersistence | **YES** — attempt meta + OPEN segment |
| RuntimeEventPersistence | **NO** — delegate never ran |
| CausalEvidencePersistence | **NO** — no transport evidence yet |

**Rejection of “repo already guarantees discovery fact before lineage admission”:**

No public contract, production call site, or test requires RuntimeEvent or CausalEvidence before `open_attempt`. The opposite ordering is explicit: lineage activation precedes boundary admission, which precedes delegate execution and observability publication.

### 2.4 Legal crash window B — retry attempt before DIAG-2 facts

```text
A1 discovered normally (RuntimeEvent and/or CausalEvidence)
AttemptLifecycle transition A1 → A2
activate_root_execution_lineage for A2 → open_attempt + open_segment durable
PROCESS CRASH before first A2 RuntimeEvent or CausalEvidence
```

A2 is durable in lineage but invisible to current DIAG-2 attempt union.

### 2.5 Legal crash window C — segment without root admission

```text
open_attempt → open_segment → crash before admit_root
```

May leave:

```text
OPEN segment without root admission record
UNCLEAN predecessor segment (if resume predecessor chain)
```

This is a **transitional / degraded crash history** state, not automatic hard corruption. Reader semantics must distinguish lifecycle state from integrity defects (§12, §14).

---

## 3. Current discovery verdict

### 3.1 Implementation audit

`_build_attempts` in `intergrax/runtime/diagnostics/execution_reconstruction.py`:

```263:270:intergrax/runtime/diagnostics/execution_reconstruction.py
    attempt_ids = sorted(
        set(causal_by_attempt) | set(events_by_attempt),
        key=lambda attempt_id: _attempt_projection_order_key(
            attempt_id,
            causal_by_attempt=causal_by_attempt,
            events_by_attempt=events_by_attempt,
        ),
    )
```

Lineage reader is consulted **only for attempt_ids already in the union**. No lineage enumeration API exists on `ExecutionLineageReader`.

### 3.2 Verdict

```text
EXISTING_DIAG2_DISCOVERY_COMPLETE: NO
COUNTEREXAMPLE_CONFIRMED: YES
```

The historical PASS in READ_INTEGRATION_R1 attempt-discovery section is **invalid** and is superseded by this document.

---

## 4. AttemptLifecycle audit

### 4.1 Authority model

`AttemptLifecycleService` (`intergrax/runtime/execution/attempt_lifecycle/service.py`) owns **durable attempt transitions** (initial + retry). `AttemptLifecycleState` stores:

```text
run_id, active_attempt_id, previous_attempt_id, generation, transition_reason
```

One record per `(tenant_id, run_id)` — CAS-updated on each retry.

### 4.2 Audit questions

| Question | Answer | Evidence |
| -------- | ------ | -------- |
| Full run attempt history? | **NO** | Only `active_attempt_id` + one `previous_attempt_id` link; no append-only attempt log |
| Bounded list of every AttemptId? | **NO** | No `list_attempts_for_run` API on `AttemptLifecycleStore` |
| Mandatory when lineage enabled? | **NO** | `ExecutionRuntime` lineage gate is `execution_lineage_persistence is not None`; lifecycle is separately wired in Nexus/queue/background paths |
| Same-attempt resume correct? | **YES** | Resume reuses `attempt_id`; lifecycle initial record is idempotent for same attempt |
| Historical order without inference? | **NO** | Reconstructing full chain requires walking `previous_attempt_id` from current head only; missing intermediate transitions are not recoverable; no bounded enumeration |

### 4.3 Decision

```text
ATTEMPT_LIFECYCLE_REUSE: NO
```

AttemptLifecycle remains **retry transition authority only**. It must not become DIAG-2 discovery authority without a separate architecture decision expanding its contract.

---

## 5. Solution option matrix

| Option | Correctness | Crash safety | Ordering | Atomicity | Scalability | Tenant isolation | Coupling | Back-compat | Authority duplication risk |
| ------ | ----------- | ------------ | -------- | --------- | ----------- | ---------------- | -------- | ----------- | -------------------------- |
| **A** RuntimeEvent + CausalEvidence only | **FAIL** — misses lineage-only attempts | N/A | Event/causal only | N/A | Bounded per store | YES | Low | YES | None |
| **B** AttemptLifecycle as discovery | **FAIL** — no full history, not mandatory | Partial | generation ≠ presentation order | Single CAS row | O(1) read | YES | High — conflates retry with discovery | Breaks separation | **HIGH** |
| **C** Lineage-owned run discovery index | **PASS** — index-before-attempt invariant | Stale index superset legal | `discovery_position` monotonic | Partition-local atomic rows; cross-partition eventual | Bounded paginated list | YES | Low — same persistence capability | Historical runs unaffected | **LOW** — projection only |
| **D** Mandatory pre-admission RuntimeEvent | Theoretically complete | Depends on observability store | Event position | Couples admission to observability | Bounded | YES | **HIGH** — observability becomes admission gate | Breaking | Medium |
| **E** Mandatory causal evidence before admission | **FAIL** — not universal for all root paths | N/A | N/A | N/A | N/A | YES | **HIGH** | Breaking | Medium |
| **F** Checkpoint-based discovery | **FAIL** — checkpoint may not exist | N/A | N/A | N/A | N/A | YES | Couples DIAG-2 to resume | No | **HIGH** |
| **G** DocumentStore cross-partition scan | **FAIL** — unbounded, heuristic partition discovery | N/A | Undefined | N/A | **UNBOUNDED** | Risky | Store-specific | No | Medium |

### 5.1 Rejected heuristics (absolute)

```text
lexical AttemptId ordering
ExecutionId ordering as attempt order
timestamp guessing
checkpoint order
document partition scan
"latest looking" record
event-time inference
```

---

## 6. Selected architecture — OPTION_C

### 6.1 Rationale (reuse-first)

No existing canonical authority provides:

```text
all AttemptIds · bounded · durable · ordered · tenant-scoped · crash-safe
```

RuntimeEvent and CausalEvidence cover attempts that reached observability publication. AttemptLifecycle covers transition state, not discovery enumeration. Checkpoints are resume projections, not forensic discovery.

**OPTION_C** extends the existing `ExecutionLineagePersistence` capability with a **discovery projection** — not a second store, not lifecycle authority, not identity authority.

```text
SELECTED_OPTION: OPTION_C
DISCOVERY_AUTHORITY: ExecutionLineagePersistence run-scoped discovery projection
DISCOVERY_IS_LIFECYCLE_AUTHORITY: NO
NEW_ATTEMPT_AUTHORITY: NO
NEW_LINEAGE_STORE: NO
```

---

## 7. Run scope contract

```text
ExecutionLineageRunScope
    tenant_id: str
    task_id: TaskId
    run_id: RunId
```

**Excluded:** `attempt_id`, `execution_id`, `segment_id`.

```text
RUN_SCOPE: ExecutionLineageRunScope
```

---

## 8. Discovery record contract

```text
ExecutionLineageAttemptDiscoveryRecord
    run_scope: ExecutionLineageRunScope
    attempt_id: AttemptId
    discovery_position: int   # monotonic within run discovery projection, >= 1
```

Properties:

- Immutable fact row, append-only within the run discovery projection.
- `discovery_position` is **presentation/discovery order** — not AttemptLifecycle `generation`, not retry number, not identity, not timestamp.
- No reusable canonical attempt-order fact exists outside this projection for lineage-only attempts.

```text
DISCOVERY_RECORD: ExecutionLineageAttemptDiscoveryRecord
DISCOVERY_ORDERING_FACT: discovery_position
```

---

## 9. Index-first invariant

Registration in the run discovery index must occur **before** any per-attempt lineage state can become durable:

```text
register_discovery_entry(run_scope, attempt_id)
  BEFORE
open_attempt(attempt_scope)
```

Required invariant:

```text
durable attempt lineage  ⇒  prior durable run discovery entry exists
```

Never the reverse.

Production write-path insertion point: `activate_root_execution_lineage` — call discovery registration immediately before `persistence.open_attempt(scope)`.

Retry path: register new `attempt_id` at `ExecutionAttemptRetryService` transition boundary before first lineage activation for the new attempt.

```text
INDEX_FIRST_INVARIANT: register_discovery_entry BEFORE open_attempt; durable lineage ⇒ prior discovery entry
```

---

## 10. Cross-partition crash semantics

Run discovery index partition ≠ per-attempt lineage partition. No cross-partition transaction.

### 10.1 Index succeeds, attempt open crashes

```text
discovery entry exists · attempt state absent
```

**Legal.** Discovery index is the **candidate attempt set** (superset). Stale discovery entry does not assert forensic lineage attempt existence.

Reader semantics:

```text
discovery entry + ABSENT attempt state  →  candidate-only (no lineage enrichment)
discovery entry + AVAILABLE attempt state  →  full lineage reconstruction
```

### 10.2 Attempt state without discovery entry

```text
attempt state exists · discovery entry absent
```

**Impossible** on correct write path. If detected on read → **INTEGRITY ERROR**.

```text
CROSS_PARTITION_CRASH_SEMANTICS:
  stale discovery superset legal
  attempt-without-discovery-entry = integrity defect
STALE_DISCOVERY_ENTRY_SEMANTICS: candidate-only; not forensic existence truth
ATTEMPT_WITHOUT_DISCOVERY_ENTRY: integrity defect if detected
```

---

## 11. Idempotency and concurrency

### 11.1 Repeated registration

Same `(tenant_id, task_id, run_id, attempt_id)` → idempotent; same `discovery_position`; no duplicate positions for one `AttemptId`.

### 11.2 Same-attempt resume

```text
same AttemptId · new root ExecutionId
```

Does **not** create a new discovery entry.

```text
SAME_ATTEMPT_RESUME_NEW_DISCOVERY_RECORD: NO
```

### 11.3 Concurrent first registration of same attempt

Partition-local `put_if_absent` (or equivalent) → one canonical record, one position. No last-write-wins.

### 11.4 Concurrent registration of different legal AttemptIds

Allowed. Positions assigned atomically within the run discovery partition (monotonic counter or position row CAS). AttemptLifecycle retry transitions serialize new attempt minting at the service layer; concurrent distinct attempts without lifecycle remain possible in theory but position assignment must be partition-atomic.

---

## 12. Bounded read API

New reader method on `ExecutionLineageReader` (architecture contract only):

```text
list_attempts_for_run(
    run_scope: ExecutionLineageRunScope,
    limit: int,
    cursor: str | None = None,
) -> ExecutionLineageAttemptDiscoveryPage
```

Requirements:

```text
bounded · paginated · stable · tenant-scoped · deterministic
```

Forbidden: `list_all_attempts()`.

```text
BOUNDED_DISCOVERY: YES
UNBOUNDED_SCAN: NO
```

---

## 13. Ordering contract for ExecutionReconstruction.attempts

### 13.1 Candidate set (union)

```text
ATTEMPT_DISCOVERY_UNION:
  RuntimeEvent attempts
  ∪ CausalEvidence attempts
  ∪ LineageDiscovery attempts (from list_attempts_for_run)
```

Duplicate `attempt_id` across sources → single `ReconstructedAttempt`.

### 13.2 Presentation order (deterministic)

For each `attempt_id` in the union, compute sort key:

| Attempt class | Primary order key | Tie-break |
| ------------- | ----------------- | --------- |
| Indexed lineage discovery | `(0, discovery_position)` | `str(attempt_id)` for stable display only |
| RuntimeEvent present, no discovery index | `(1, first ExecutionEventPosition)` | `str(attempt_id)` |
| CausalEvidence only, no discovery index | `(2, recorded_at, evidence_id)` | per `causal_evidence_query_order_key` |
| Lineage-only without discovery index | **must not occur** post-correction | N/A |

**Critical separation:**

```text
presentation/discovery order  ← discovery_position (and event/causal for non-indexed historical)
AttemptLifecycle retry authority  ← generation / transition (NOT used for DIAG-2 ordering)
```

Lexical `AttemptId` ordering is **never** semantic truth for lineage-only attempts.

### 13.3 Historical backward compatibility

Executions before lineage R1 discovery index:

```text
no discovery index
```

Reconstructed from RuntimeEvent ∪ CausalEvidence only — unchanged.

```text
HISTORICAL_BACKWARD_COMPATIBILITY: pre-index runs use RuntimeEvent ∪ CausalEvidence only; no full-history migration required
```

---

## 14. Checkpoint non-authority

```text
CHECKPOINT_AS_ATTEMPT_DISCOVERY_AUTHORITY: NO
CHECKPOINT_AS_DISCOVERY_AUTHORITY: NO
```

Checkpoints may not exist; they remain resume projections only.

---

## 15. RuntimeEvent and CausalEvidence admission options

**OPTION_D (mandatory pre-admission RuntimeEvent):** Would make observability persistence availability a structural execution admission dependency. Violates separation of concerns; observability is not universal admission authority.

**OPTION_E (mandatory causal evidence):** Transport evidence is not emitted on every legal root execution path. Extending `CausalRelationKind` as a workaround requires separate justification — rejected here.

---

## 16. READ_INTEGRATION_R1 correction items

The following implement defects in current read integration must be corrected in the next implementation task.

### 16.1 Truncation semantics

```text
TRUNCATION_SEMANTICS:
  truncated segment/admission pages MUST NOT be validated as complete forensic snapshot
  legal missing unseen records ≠ corruption
```

When `segments_truncated` or `admissions_truncated`, structural validation applies only to **loaded** records; completeness → `TRUNCATED`, not integrity failure for unseen tail.

### 16.2 Stable snapshot semantics

```text
STABLE_SNAPSHOT_SEMANTICS:
  reader captures attempt_state.generation
  loads all pages + seal under that generation observation
  if generation changed between read start and seal read → bounded retry from scratch
  never compose state from two durable generations
```

Prevents torn reconstruction across concurrent writers.

### 16.3 State / seal consistency

After stable snapshot:

```text
STATE_SEAL_CONSISTENCY:
  state.sealed == (seal is not None)

  if sealed:
    state.closure_kind is not None
    state.closure_kind == seal.closure_kind
    state.degraded == seal.degraded

  if not sealed:
    state.closure_kind is None
    seal is None
```

Mismatch after stable snapshot → **INTEGRITY ERROR** (not PARTIAL, not OPEN).

Current `_derive_completeness` does not enforce seal/state field equality — correction required.

### 16.4 Provider error translation

```text
PROVIDER_ERROR_TRANSLATION:
  DocumentStoreExecutionLineagePersistence operational read failures
    → ExecutionLineageUnavailableError
  structural decode / cursor / scope corruption
    → ExecutionLineageIntegrityError (unchanged)
```

### 16.5 Partial vs corruption

```text
missing fact because history is known degraded/truncated  →  PARTIAL or TRUNCATED
contradictory durable fact  →  INTEGRITY ERROR

examples:
  degraded / SEGMENT_UNCLEAN known gap  →  PARTIAL
  bounded data omitted  →  TRUNCATED
  same execution two conflicting real parents  →  INTEGRITY ERROR
  stable state unsealed but seal exists  →  INTEGRITY ERROR
```

### 16.6 Root-admission crash window (read semantics)

```text
OPEN segment without root admission + unsealed attempt  →  OPEN transitional / PARTIAL degraded crash history
OPEN segment without root admission + sealed attempt with contradictory seal  →  INTEGRITY ERROR
```

Do not treat missing root admission alone as corruption without lifecycle context.

---

## 17. No second authority

```text
NEW ATTEMPT IDENTITY AUTHORITY: NO
NEW ATTEMPT LIFECYCLE AUTHORITY: NO
NEW EXECUTION TREE: NO
NEW LINEAGE STORE: NO
```

Discovery index is part of existing `ExecutionLineagePersistence` / `ExecutionLineageReader` — discovery projection only.

---

## 18. Next implementation task

```text
NEXT_TASK: DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION
```

That task implements together:

```text
attempt discovery (OPTION_C)
truncation-safe reconstruction
stable snapshot retry
state/seal consistency
production provider error translation
```

---

## 19. Final decision block

```text
STATUS: PASS

EXISTING_DIAG2_DISCOVERY_COMPLETE: NO

COUNTEREXAMPLE_CONFIRMED: YES

SELECTED_OPTION: OPTION_C

DISCOVERY_AUTHORITY: ExecutionLineagePersistence run-scoped discovery projection (ExecutionLineageReader.list_attempts_for_run)

DISCOVERY_IS_LIFECYCLE_AUTHORITY: NO

RUN_SCOPE: ExecutionLineageRunScope { tenant_id, task_id, run_id }

DISCOVERY_RECORD: ExecutionLineageAttemptDiscoveryRecord { run_scope, attempt_id, discovery_position }

DISCOVERY_ORDERING_FACT: discovery_position (monotonic per run; not lifecycle generation)

INDEX_FIRST_INVARIANT: register_discovery_entry BEFORE open_attempt; durable lineage ⇒ prior discovery entry

CROSS_PARTITION_CRASH_SEMANTICS: stale discovery superset legal; attempt-without-discovery-entry = integrity defect

SAME_ATTEMPT_RESUME_NEW_DISCOVERY_RECORD: NO

HISTORICAL_BACKWARD_COMPATIBILITY: pre-index runs use RuntimeEvent ∪ CausalEvidence only

ATTEMPT_DISCOVERY_UNION: RuntimeEvent ∪ CausalEvidence ∪ LineageDiscovery

CHECKPOINT_AS_DISCOVERY_AUTHORITY: NO

UNBOUNDED_SCAN: NO

TRUNCATION_SEMANTICS: truncated pages ≠ complete snapshot; unseen tail ≠ corruption

STABLE_SNAPSHOT_SEMANTICS: generation-guarded bounded retry; no cross-generation compose

STATE_SEAL_CONSISTENCY: sealed/state/seal field equality enforced after stable snapshot

PROVIDER_ERROR_TRANSLATION: operational backend failures → ExecutionLineageUnavailableError

NEW_ATTEMPT_AUTHORITY: NO

NEW_LINEAGE_STORE: NO

NEXT_TASK: DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION
```

# DG-001 — Multi-agent diagnostic lineage attempt discovery architecture (R1)

> **Task:** `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ATTEMPT-DISCOVERY-ARCHITECTURE-R1-ROLLOUT-CORRECTION`  
> **Mode:** docs-only architecture — **no production implementation**  
> **Branch:** `development`  
> **START_HEAD:** `dcc7b60ad5b9635759dd2c2ac5c9d91dfe5012db`  
> **Ancestry verified:** `8ac2b40d86a46b0568c3814bc3f8323d8ee7f375` · `156f1defb42235fb5c3d720a41f1fcb873ad542e` · `c8225da1aa40dd82195070f9ab193d66631bba1a`  
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
DISCOVERY_IS_ATTEMPT_LIFECYCLE_AUTHORITY: NO
NEW_ATTEMPT_AUTHORITY: NO
NEW_LINEAGE_STORE: NO
```

---

## 7. Run scope and discovery partition

### 7.1 Run scope

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

### 7.2 Run discovery partition

Canonical logical partition within the same `ExecutionLineagePersistence` capability — **not** a second store:

```text
execution-lineage discovery run partition
  tenant_id
  task_id
  run_id
```

All run-scoped discovery rows (`ExecutionLineageDiscoveryRunState`, `ExecutionLineageAttemptDiscoveryRecord`) live in this partition. Per-attempt lineage rows remain in per-attempt partitions.

---

## 8. Run discovery coordination state

Minimal durable projection metadata:

```text
ExecutionLineageDiscoveryRunState
    run_scope: ExecutionLineageRunScope
    generation: int              # monotonic; advances on every successful position allocation
    next_discovery_position: int   # next position to assign; >= 1
```

This is **discovery projection coordination state** only. It is **not** attempt lifecycle, execution status, or retry authority.

Logical meta row key (architecture contract): deterministic run-scoped key equivalent to `meta:discovery_run_state` within the run discovery partition.

```text
DISCOVERY_RUN_STATE: ExecutionLineageDiscoveryRunState
```

---

## 9. Discovery record contract

```text
ExecutionLineageAttemptDiscoveryRecord
    run_scope: ExecutionLineageRunScope
    attempt_id: AttemptId
    discovery_position: int   # monotonic within run discovery projection, >= 1
```

Properties:

- Immutable fact row, append-only within the run discovery projection.
- Logical row key: `attempt:<AttemptId>` (or equivalent deterministic key). **Not** timestamp-based.
- `discovery_position` is **presentation/discovery order** — not AttemptLifecycle `generation`, not retry number, not execution identity, not event position, not timestamp.
- No reusable canonical attempt-order fact exists outside this projection for lineage-only attempts.

```text
DISCOVERY_RECORD: ExecutionLineageAttemptDiscoveryRecord
DISCOVERY_ORDERING_FACT: discovery_position
```

---

## 10. Legacy rollout problem

`ExecutionLineageAttemptState` existed before the run discovery projection. Therefore durable legacy lineage attempts may legally exist **without** `ExecutionLineageAttemptDiscoveryRecord`.

**Forbidden global invariant:**

```text
attempt state exists
+
discovery entry absent
=
always integrity error
```

That would be false corruption for data created before discovery-v1 deployment.

**Correct rule:** absence of a discovery row alone does **not** distinguish legacy from corruption. The distinction must come from an explicit durable attempt contract marker (§11).

---

## 11. Discovery contract marker and codec migration

### 11.1 Typed durable distinction

Minimal explicit contract on `ExecutionLineageAttemptState`:

```text
ExecutionLineageAttemptState
    ...
    discovery_contract_version: int | None
```

Semantics:

| Value | Meaning |
| ----- | ------- |
| `None` | **LEGACY_ATTEMPT** — created before discovery-v1 index-first contract |
| `1` | **DISCOVERY_V1_REQUIRED_ATTEMPT** — created under discovery-v1 index-first contract |

Properties required of this marker:

- typed
- durable
- explicit
- non-temporal
- non-heuristic

**Forbidden** for legacy vs post-v1 distinction:

```text
deployment timestamp
commit SHA
AttemptId lexical range
created_at guess
environment version inference
```

```text
LEGACY_DISCOVERY_DISTINCTION: discovery_contract_version on ExecutionLineageAttemptState
DISCOVERY_CONTRACT_VERSION: None = legacy; 1 = discovery-v1 index-first required
```

### 11.2 Codec migration contract

Current lineage durable rows use attempt-state schema version **1** (`intergrax/runtime/execution/lineage/codecs.py`, `_SCHEMA_VERSION = 1`).

Backward-compatible migration — version **only** the attempt-state codec contract:

```text
attempt-state schema v1 (existing durable rows)
  → on read: discovery_contract_version = None (implicit legacy)

attempt-state schema v2 (new writes)
  → explicit discovery_contract_version field in payload
  → reader accepts v1 and v2; v1 decode yields discovery_contract_version = None
```

Requirements:

- Existing v1 durable rows remain readable after v2 rollout.
- Do **not** bump schema version on segment, admission, or seal codecs unless those records need the new field (they do not).
- Writer sets `discovery_contract_version=1` only via `open_attempt(..., discovery_contract_version=1)` on the post-v1 path (§12).

```text
CODEC_MIGRATION: attempt-state v1 → implicit None; v2 → explicit discovery_contract_version; reader backward-compatible
```

### 11.3 Post-v1 hard invariant

For:

```text
attempt_state.discovery_contract_version == 1
```

there must exist **exactly one** `ExecutionLineageAttemptDiscoveryRecord` for the same `(tenant_id, task_id, run_id, attempt_id)`.

Absence → **INTEGRITY ERROR** (true index-first violation).

For:

```text
discovery_contract_version is None
```

absence of discovery entry is **legal legacy state**.

```text
POST_V1_INDEX_REQUIRED: YES
LEGACY_ATTEMPT_WITHOUT_INDEX: LEGAL
POST_V1_ATTEMPT_WITHOUT_INDEX: INTEGRITY_ERROR
```

---

## 12. Index-first invariant

Registration in the run discovery index must occur **before** any post-v1 per-attempt lineage state can become durable:

```text
register_discovery_entry(run_scope, attempt_id)
    ↓ durable success
open_attempt(
    attempt_scope,
    discovery_contract_version=1,
)
```

Required invariant:

```text
post-v1 durable attempt state (discovery_contract_version == 1)
  ⇒
discovery entry already durable
```

Never the reverse. If discovery registration fails → **`open_attempt` MUST NOT execute** (fail closed).

**Single canonical write seam** — no secondary discovery registration in `ExecutionAttemptRetryService`:

```text
activate_root_execution_lineage
  → register_discovery_entry(run_scope, attempt_id)
  → open_attempt(attempt_scope, discovery_contract_version=1)
  → open_segment(...)
```

This path covers **both** first root attempt and retry minted attempts (A2, A3, …): each first lineage activation for a new `AttemptId` passes through `activate_root_execution_lineage`. Retry lifecycle authority remains in `AttemptLifecycleService`; discovery registration is **not** coupled to `AttemptLifecycle` / `RetryService` for convenience.

```text
INDEX_FIRST: register discovery before post-v1 open_attempt; discovery failure blocks open_attempt
RETRY_DISCOVERY_REGISTRATION_IN_RETRY_SERVICE: NO
```

---

## 13. Atomic first registration

For a **new** `AttemptId` within a run:

```text
read ExecutionLineageDiscoveryRunState

allocate position = next_discovery_position

ONE PARTITION-ATOMIC BATCH:
    put-if-absent attempt discovery row (key: attempt:<AttemptId>)
    +
    CAS/replace run discovery meta:
        generation += 1
        next_discovery_position += 1
```

Required invariant:

```text
attempt discovery row created
  IFF
counter advancement committed
```

**Forbidden:** `put row` then `CAS counter` as two independent writes.

### 13.1 Reuse existing atomic capability

Architecture **must** reuse `PartitionAtomicDocumentStore.execute_partition_atomic_batch` for DocumentStore provider (`DocumentStoreExecutionLineagePersistence` already requires `PartitionAtomicDocumentStore`).

`InMemoryExecutionLineagePersistence`: same logical atomic semantics under one lock / transaction boundary.

Do **not** introduce a new transaction framework.

```text
DISCOVERY_ATOMICITY: discovery row + position counter = one partition-atomic operation
DISCOVERY_ATOMIC_BATCH: PartitionAtomicDocumentStore.execute_partition_atomic_batch
```

---

## 14. Cross-partition crash semantics

Run discovery partition ≠ per-attempt lineage partition. No cross-partition transaction.

### 14.1 Index succeeds, attempt open crashes

```text
register_discovery_entry → durable success
PROCESS CRASH
open_attempt never executed
```

Result:

```text
discovery row exists · attempt state absent
```

**Legal.** Discovery index is the **candidate attempt set** (superset). Stale discovery entry does not assert forensic attempt existence.

**Projection behavior:** if **only** a discovery row exists — no attempt state, no RuntimeEvent, no CausalEvidence — **do not create `ReconstructedAttempt`**. The entry is a candidate, not existence truth. It may be omitted from final projection or surfaced only in internal discovery diagnostics — never as a real execution attempt.

If discovery entry exists **and** RuntimeEvent and/or CausalEvidence exists but attempt state is absent → reconstruct attempt from canonical runtime/causal sources; lineage enrichment → **ABSENT**. Not a hard fail.

```text
STALE_DISCOVERY_ENTRY: legal candidate-only
STALE_DISCOVERY_ONLY_VISIBLE_AS_REAL_ATTEMPT: NO
```

### 14.2 Attempt state without discovery entry — legacy vs post-v1

```text
attempt state exists · discovery entry absent
```

| `discovery_contract_version` | Verdict |
| ---------------------------- | ------- |
| `None` (legacy) | **LEGAL LEGACY** — reconstruct lineage from existing per-attempt durable facts |
| `1` (post-v1) | **INTEGRITY ERROR** — true index-first violation |

**Forbidden:** treating all attempt-without-discovery cases as corruption (§10).

```text
CROSS_PARTITION_CRASH_SEMANTICS:
  stale discovery superset legal
  legacy attempt without index legal when discovery_contract_version is None
  post-v1 attempt without index = integrity defect
```

---

## 15. Idempotency and concurrency

### 15.1 Repeated registration

Same `(tenant_id, task_id, run_id, attempt_id)` → idempotent; returns existing record and existing `discovery_position`; **does not** increment counter.

Conflicting persisted record for same key (different `discovery_position` or payload) → **INTEGRITY ERROR**.

### 15.2 Same-attempt resume

```text
same AttemptId · new root ExecutionId
```

Does **not** create a new discovery entry.

```text
SAME_ATTEMPT_RESUME_NEW_DISCOVERY_ROW: NO
```

### 15.3 Concurrent first registration of same attempt

Partition-local `put_if_absent` inside atomic batch → one canonical record, one position. No last-write-wins.

### 15.4 Concurrent registration of different legal AttemptIds

Parallel registration of A2 and A3 must yield **unique** `discovery_position` values without lost update, duplicate position, or last-write-wins. CAS / atomic conflicts → **bounded retry**.

```text
CONCURRENT_POSITION_ALLOCATION: partition-atomic batch + bounded CAS retry
```

---

## 16. Bounded read API

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

## 17. Run discovery stable snapshot

`list_attempts_for_run()` is paginated. Under concurrent registration, the reader must not compose a candidate set from multiple discovery generations.

**Algorithm:**

```text
run_state_before = read_discovery_run_state()

load bounded discovery pages

run_state_after = read_discovery_run_state()

if generation unchanged:
    snapshot stable

if generation changed:
    retry whole discovery read (bounded)
```

Requirements:

- Retries bounded — no infinite loop.
- Do **not** merge discovery rows from different `ExecutionLineageDiscoveryRunState.generation` values and declare the set COMPLETE.
- If generation changes through all bounded retries → explicit read result (not integrity error — concurrent append is not corruption):

```text
DISCOVERY_READ_TRUNCATED   # preferred
# or typed UNSTABLE / UNAVAILABLE per existing read model
```

**Separation from per-attempt snapshot:**

| Coordination state | Protects |
| ------------------ | -------- |
| `ExecutionLineageDiscoveryRunState.generation` | candidate attempt set for run discovery pagination |
| `ExecutionLineageAttemptState.generation` | per-attempt forensic reconstruction (segments, admissions, seal) |

Do **not** use run discovery generation as a substitute for attempt lineage generation, or vice versa.

```text
RUN_DISCOVERY_STABLE_SNAPSHOT: generation-guarded bounded retry on run discovery meta; DISCOVERY_READ_TRUNCATED on persistent churn
RUN_SNAPSHOT ≠ ATTEMPT_SNAPSHOT: separate generation contracts
```

---

## 18. Ordering contract for ExecutionReconstruction.attempts

### 18.1 Candidate set (union)

```text
ATTEMPT_DISCOVERY_UNION:
  RuntimeEvent attempts
  ∪ CausalEvidence attempts
  ∪ indexed discovery attempts (from stable list_attempts_for_run snapshot)
  ∪ legacy lineage attempts (discovery_contract_version is None, with or without runtime/causal facts)
```

**Exclude** from union: discovery-only stale candidates (discovery row without attempt state, RuntimeEvent, or CausalEvidence) — §14.1.

Duplicate `attempt_id` across sources → single `ReconstructedAttempt`.

### 18.2 Presentation order — explicitly non-chronological (OPTION 1)

No canonical comparable order exists between legacy attempts and discovery-v1 indexed attempts. **Do not** present `ExecutionReconstruction.attempts` tuple order as execution chronology.

Sort keys are **stable display order** only:

| Attempt class | Primary order key | Tie-break |
| ------------- | ----------------- | --------- |
| Indexed discovery-v1 (`discovery_contract_version == 1`) | `(0, discovery_position)` | `str(attempt_id)` |
| Legacy / runtime (`discovery_contract_version is None`, RuntimeEvent present) | `(1, first ExecutionEventPosition)` | `str(attempt_id)` |
| Legacy / causal only | `(2, recorded_at, evidence_id)` | per `causal_evidence_query_order_key` |
| Legacy lineage-only (state present, no runtime/causal) | `(3, str(attempt_id))` | — |

**Mixed rollout consequence:** indexed A2 may sort **before** legacy A1 in display order. This is legal and expected — not lifecycle truth.

DIAG-3 / DIAG-4 must **not** use tuple order as lifecycle or retry authority.

Optional typed provenance (implementation may add if consumer value exists): `DISCOVERY_POSITION` · `RUNTIME_EVENT_POSITION` · `CAUSAL_EVIDENCE_ORDER` · `LEGACY_LINEAGE_ONLY`. Not required for decoration alone.

**Critical separation:**

```text
presentation/discovery order  ← discovery_position (indexed) or event/causal keys (legacy)
AttemptLifecycle retry authority  ← generation / transition (NOT used for DIAG-2 ordering)
discovery_position  ≠  ExecutionEventPosition  (not comparable across classes)
```

Lexical `AttemptId` ordering is **never** semantic truth.

```text
MIXED_ATTEMPT_ORDER_SEMANTICS: stable non-chronological display classes; not execution chronology
```

### 18.3 Mixed rollout canonical example

```text
Run R1

A1: legacy lineage (discovery_contract_version=None), no discovery row, RuntimeEvents exist
    → discovery deployment of discovery-v1
A2: discovery_position=1, discovery_contract_version=1
A3: discovery_position=2, discovery_contract_version=1
```

| Concern | Behavior |
| ------- | -------- |
| Candidate union | A1 from RuntimeEvent (+ legacy lineage enrichment); A2/A3 from discovery index + lineage; no stale-only rows |
| Lineage validation | A1: legacy without index → legal; A2/A3: post-v1 requires matching discovery row |
| Presentation order | A2 (class 0, pos 1), A3 (class 0, pos 2), A1 (class 1, event position) — **A2 may appear before A1** |

```text
MIXED_RUN_SUPPORTED: YES
```

### 18.4 Historical backward compatibility — final

```text
legacy attempt without discovery row = legal
  iff
explicit durable attempt contract identifies it as legacy (discovery_contract_version is None)
```

Not:

```text
absence of discovery row alone → legacy vs corruption inference
```

Pre-index runs with no lineage at all: reconstructed from RuntimeEvent ∪ CausalEvidence only — unchanged.

```text
HISTORICAL_BACKWARD_COMPATIBILITY: explicit discovery_contract_version marker; no migration heuristic
```

---

## 19. Checkpoint non-authority

```text
CHECKPOINT_AS_ATTEMPT_DISCOVERY_AUTHORITY: NO
CHECKPOINT_AS_DISCOVERY_AUTHORITY: NO
```

Checkpoints may not exist; they remain resume projections only.

---

## 20. RuntimeEvent and CausalEvidence admission options

**OPTION_D (mandatory pre-admission RuntimeEvent):** Would make observability persistence availability a structural execution admission dependency. Violates separation of concerns; observability is not universal admission authority.

**OPTION_E (mandatory causal evidence):** Transport evidence is not emitted on every legal root execution path. Extending `CausalRelationKind` as a workaround requires separate justification — rejected here.

---

## 21. READ_INTEGRATION_R1 correction items

The following implement defects in current read integration must be corrected in the next implementation task.

### 21.1 Truncation semantics

```text
TRUNCATION_SEMANTICS:
  truncated segment/admission pages MUST NOT be validated as complete forensic snapshot
  legal missing unseen records ≠ corruption
```

When `segments_truncated` or `admissions_truncated`, structural validation applies only to **loaded** records; completeness → `TRUNCATED`, not integrity failure for unseen tail.

### 21.2 Per-attempt stable snapshot semantics

```text
STABLE_SNAPSHOT_SEMANTICS:
  reader captures attempt_state.generation
  loads all pages + seal under that generation observation
  if generation changed between read start and seal read → bounded retry from scratch
  never compose state from two durable generations
```

```text
PER_ATTEMPT_STABLE_SNAPSHOT:
  attempt_state.generation before read
  → load segment/admission pages + seal
  → attempt_state.generation after read
  generation changed → bounded retry from scratch
  never compose state from two attempt generations
```

Run discovery stable snapshot (§17) uses **separate** `ExecutionLineageDiscoveryRunState.generation` — not interchangeable.

### 21.3 State / seal consistency

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

### 21.4 Provider error translation

```text
PROVIDER_ERROR_TRANSLATION:
  DocumentStoreExecutionLineagePersistence operational read failures
    → ExecutionLineageUnavailableError
  structural decode / cursor / scope corruption
    → ExecutionLineageIntegrityError (unchanged)
```

### 21.5 Partial vs corruption

```text
missing fact because history is known degraded/truncated  →  PARTIAL or TRUNCATED
contradictory durable fact  →  INTEGRITY ERROR

examples:
  degraded / SEGMENT_UNCLEAN known gap  →  PARTIAL
  bounded data omitted  →  TRUNCATED
  same execution two conflicting real parents  →  INTEGRITY ERROR
  stable state unsealed but seal exists  →  INTEGRITY ERROR
```

### 21.6 Root-admission crash window (read semantics)

```text
OPEN segment without root admission + unsealed attempt  →  OPEN transitional / PARTIAL degraded crash history
OPEN segment without root admission + sealed attempt with contradictory seal  →  INTEGRITY ERROR
```

Do not treat missing root admission alone as corruption without lifecycle context.

---

## 22. No second authority

```text
NEW ATTEMPT IDENTITY AUTHORITY: NO
NEW ATTEMPT LIFECYCLE AUTHORITY: NO
NEW EXECUTION TREE: NO
NEW LINEAGE STORE: NO
```

Discovery index is part of existing `ExecutionLineagePersistence` / `ExecutionLineageReader` — discovery projection only.

```text
DISCOVERY_IS_ATTEMPT_LIFECYCLE_AUTHORITY: NO
ATTEMPT_LIFECYCLE_AUTHORITY_CHANGED: NO
```

---

## 23. Implementation test scenarios (frozen)

| ID | Scenario |
| -- | -------- |
| D1 | New run first attempt discovery registration + index-first open |
| D2 | Repeated registration idempotent (same position, no counter bump) |
| D3 | Concurrent same AttemptId — single row/position |
| D4 | Concurrent different AttemptIds — unique positions, no lost update |
| D5 | Discovery persisted + crash before open_attempt — stale candidate legal |
| D6 | Post-v1 attempt state without discovery row — integrity error |
| D7 | Legacy v1 attempt state without discovery row — legal |
| D8 | Mixed legacy A1 + indexed A2/A3 — union, validation, display order |
| D9 | Same-attempt resume — no new discovery row |
| D10 | Paginated run discovery read |
| D11 | Run discovery generation changes during pagination — bounded retry |
| D12 | Persistent discovery writer churn → DISCOVERY_READ_TRUNCATED |
| D13 | Stale discovery-only candidate not surfaced as forensic attempt |

---

## 24. Next implementation task

```text
NEXT_TASK: DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION
```

That task implements together:

```text
attempt discovery (OPTION_C)
discovery_contract_version marker + attempt-state codec v2
run discovery atomic registration + stable snapshot
truncation-safe reconstruction
per-attempt stable snapshot retry
state/seal consistency
production provider error translation
legacy vs post-v1 integrity rules
```

---

## 25. Final decision block

```text
STATUS: PASS

SELECTED_OPTION: OPTION_C

DISCOVERY_AUTHORITY: ExecutionLineagePersistence run-scoped discovery projection

DISCOVERY_IS_ATTEMPT_LIFECYCLE_AUTHORITY: NO

LEGACY_DISCOVERY_DISTINCTION: ExecutionLineageAttemptState.discovery_contract_version (None = legacy, 1 = discovery-v1)

POST_V1_INDEX_REQUIRED: YES

DISCOVERY_CONTRACT_VERSION: None = LEGACY_ATTEMPT; 1 = DISCOVERY_V1_REQUIRED_ATTEMPT

INDEX_FIRST: register_discovery_entry BEFORE open_attempt(discovery_contract_version=1); discovery failure blocks open_attempt

DISCOVERY_RUN_STATE: ExecutionLineageDiscoveryRunState { run_scope, generation, next_discovery_position }

DISCOVERY_ATOMICITY: discovery row + position counter = one partition-atomic operation (PartitionAtomicDocumentStore)

DISCOVERY_POSITION: monotonic per run projection; not retry_number / lifecycle generation / event position / timestamp

RUN_DISCOVERY_STABLE_SNAPSHOT: generation-guarded bounded retry; DISCOVERY_READ_TRUNCATED on persistent churn

STALE_DISCOVERY_ENTRY: legal candidate-only

STALE_DISCOVERY_ONLY_VISIBLE_AS_REAL_ATTEMPT: NO

LEGACY_ATTEMPT_WITHOUT_INDEX: LEGAL

POST_V1_ATTEMPT_WITHOUT_INDEX: INTEGRITY_ERROR

MIXED_RUN_SUPPORTED: YES

MIXED_ATTEMPT_ORDER_SEMANTICS: stable non-chronological display classes (indexed class 0, legacy runtime class 1, causal class 2, legacy lineage-only class 3); not execution chronology

SAME_ATTEMPT_RESUME_NEW_DISCOVERY_ROW: NO

ATTEMPT_LIFECYCLE_AUTHORITY_CHANGED: NO

CHECKPOINT_AUTHORITY: NO

UNBOUNDED_SCAN: NO

NEW_LINEAGE_STORE: NO

NEW_EXECUTION_TREE: NO

TRUNCATION_SEMANTICS: unseen records because bounded max reached ≠ corruption

PER_ATTEMPT_STABLE_SNAPSHOT: attempt_state.generation-guarded bounded retry

STATE_SEAL_CONSISTENCY: sealed/state/seal field equality enforced after stable snapshot

PROVIDER_ERROR_TRANSLATION: operational DocumentStore read error → ExecutionLineageUnavailableError

NEXT_TASK: DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION
```

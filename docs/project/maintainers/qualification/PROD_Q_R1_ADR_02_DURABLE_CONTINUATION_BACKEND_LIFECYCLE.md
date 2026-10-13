# PROD-Q-R1-ADR-02 — Durable Continuation Backend & Lifecycle

| Field | Value |
|-------|-------|
| **Status** | **PENDING / NOT APPROVED** |
| **Date** | 2026-10-13 |
| **START_HEAD (ADR-02-R1)** | `27e713748f2bc6f2acb8f751b728bf4c80c1d37f` |
| **START_HEAD (ADR-02-R2)** | `3a8a8301b993a18a0a6419a21b417b7ee564ecd9` |
| **START_HEAD (ADR-02-R3)** | `cd295d7c97ef8a123b4ec4f02540a9a4bce3436d` |
| **FINAL_COMMIT (docs only)** | *(set at commit — ADR-02-R3)* |
| **Revision** | **ADR-02-R3** — serialized mutation / persistence transaction boundary (corrects R2) |
| **Prior revision** | **ADR-02-R2** — **REJECTED** (independent audit: invalid lock release before durable commit; §2.2.0) |
| **Branch** | `development` |
| **Parent** | **PROD-Q-R1** — **BLOCKED** |
| **Related** | **ADR-01** [`PROD_Q_R1_ADR_01_DURABLE_CONTINUATION_OWNERSHIP.md`](PROD_Q_R1_ADR_01_DURABLE_CONTINUATION_OWNERSHIP.md) (**B1** composition ownership, PENDING) · GR-5-R2 · GR-10-R12 · STATE-X **SX-F17** · COMPAT-X `execution_continuation_durable_state.v1` |
| **Does not close** | **PROD-Q-R1** · **FRZ-PRD-04** · global FRZ PASS |

---

## 1. Question

Can an **existing production-qualified** `ExecutionContinuationStateStore` implementation satisfy **B1** (composition-owned durable injection) without a separate persistence architecture decision, or is a **new production persistence responsibility** required?

**Scope:** analysis only — no store, wrapper, fallback, migration, or alternate recovery authority.

---

## 2. Decision (independent recommendation)

**Classification: Case B** — existing file + export/restore mechanism is **reusable**; **production-qualified durability lifecycle is incomplete** without an accepted seal policy and composition owner.

```text
STOP — ARCHITECTURE DECISION REQUIRED (backend / lifecycle)
```

| Layer | Verdict |
|-------|---------|
| **B1 composition binding** (ADR-01) | **Compatible** — inject any `ExecutionContinuationStateStore` with `is_durable is True` through existing harness/Nexus surfaces |
| **`ExecutionContinuationDurableStateFilePersistence`** | **IMPLEMENTED / NOT QUALIFIED** — snapshot I/O only; **not** an `ExecutionContinuationStateStore`; **no** automatic seal on CAS success |
| **Production-qualified physical persistence** | **MISSING** — no sanctioned composition root constructs, retains, seals, and restores a restart-correct durable backend for GCA strict startup |
| **Existing in-tree store implementations** | **IMPLEMENTED / NOT QUALIFIED** for production restart (qualification and lab patterns only) |

**Maximum status:** **READY FOR ARCHITECTURE DECISION** (not CLOSED).

### 2.1 Architecture proposal (approval only — no implementation)

**Physical layer (reuse):** `execution_continuation_durable_state.v1` envelope (`export_durable_continuation_state` / `restore_durable_continuation_backing`) + `ExecutionContinuationDurableStateFilePersistence._atomic_write` (temp file + `Path.replace`).

**Missing piece (minimal extension):** a **single process-lifetime lifecycle owner** (intended home: `ProductionProcessComposition` field + factory after ADR-01 B1 acceptance) that:

1. **Startup:** `load_state_store()` → `ReconstructedDurableExecutionContinuationStateStore` over restored backing (or empty backing + optional missing file policy).
2. **Runtime:** expose one shared store view over mutable `ExecutionContinuationDurableBacking` (same CAS as `persistence.py`; **no second CAS implementation**).
3. **Durability contract (must be chosen explicitly):** synchronous `persist_from_backing` after **each successful** store mutation **or** a documented weaker cadence — default qualification expectation for HITL pause/resume is **per-mutation seal** so acknowledged `ExecutionContinuationService` CAS cannot be lost to crash.
4. **Shutdown:** optional final seal (redundant if per-mutation); **shutdown-only sealing is insufficient** for crash safety.

**Rejected without ADR:** inferring durability from `ReconstructedDurableExecutionContinuationStateStore.is_durable=True` while mutations never reach disk; fabricating SQLite/platform KV adapter in this record; synchronous seal wrapper shipped without explicit seal-policy acceptance.

---

## 2.2 ADR-02-R3 — Serialized durable mutation transaction (independent recommendation)

**Scope:** persistence semantics for production `ExecutionContinuationStateStore` — single transaction boundary covering validation, CAS, snapshot, persist, publication, ACK. **No production code in this revision.**

**Evidence @ `cd295d7c97ef8a123b4ec4f02540a9a4bce3436d`:** `ExecutionContinuationDurableBacking._lock` is a non-reentrant `threading.Lock` (`persistence.py`); public store methods and `export_durable_continuation_state` each acquire it independently; `persist_from_backing` exports under lock then writes **outside** lock (`durable_state_file.py`); service ACK follows in-memory CAS only (`service.py`).

```text
STOP — ARCHITECTURE DECISION REQUIRED (mutation commit protocol + seal policy acceptance)
```

**Maximum status:** **READY FOR ARCHITECTURE DECISION** (not CLOSED).

### 2.2.0 ADR-02-R2 audit disposition (why R3 exists)

Independent audit **rejected ADR-02-R2 §2.2.2 P0–P5** because **P3 released `backing._lock` before durable commit completed**. That ordering:

- allowed **in-process readers** to observe a mutation that might still be rolled back on seal failure;
- allowed a **second mutator** to advance backing and seal while the first seal was still in flight → **reverse-order durable snapshots**;
- made **rollback after a concurrent successful seal** unsafe (could regress durable authority);
- left **in-memory authority ahead of disk** visible to other threads during the vulnerability window.

R3 replaces P0–P5 with one **serialized transaction** that holds the serialization lock across validation, in-memory apply, snapshot, and physical persist, using **lock-held internal mutation/export** (not nested public store calls).

### 2.2.1 Non-atomic fact (do not assume)

`compare_and_swap` / `begin_current_episode_if_predecessor_allows` and `persist_from_backing` are **independent steps** on the current code path:

1. Service methods return the updated `PendingExecutionContinuation` **only after in-memory CAS returns `True`** (`service.py`).
2. File seal is **never** invoked from the store or service on that path.
3. `export_durable_continuation_state` snapshots under `backing._lock`; `_atomic_write` uses `*.tmp` + `Path.replace` with **no** `fsync` / directory sync (`durable_state_file.py`).

Therefore **semantic commit** (in-memory CAS) and **durable commit** (media write) are not coupled unless an explicit production protocol is accepted and implemented.

### 2.2.2 Mutation transaction — required production protocol (proposed)

**Name:** **serialized seal-before-ACK** (single-writer, one lock scope per successful mutating store operation).

**Authority:** one `ExecutionContinuationDurableBacking` per process composition; one `ExecutionContinuationDurableStateFilePersistence` (or accepted delegate) owned by the **composition lifecycle** (ADR-01 **B1** binding). **No second CAS engine** — reuse the **same** CAS / episode-begin semantics as `InMemoryExecutionContinuationStateStore` on shared backing; production façade orchestrates the transaction only.

**Mutating operations in scope:** `compare_and_swap`, `begin_current_episode_if_predecessor_allows`, and any future production path that changes backing without going through those methods (**forbidden**).

**Single serialized transaction (logical phases — one `backing._lock` acquisition on the mutating path):**

| Phase | Action | Reader / mutator visibility |
|-------|--------|------------------------------|
| **T1** | Acquire `backing._lock` once at production façade entry (mutating methods only) | All `load` / resolve paths block until **T7** |
| **T2** | Validate expected state (CAS equality / episode-predecessor rules) via **lock-held internal primitives** — **not** public `compare_and_swap` / `begin_current_episode_if_predecessor_allows` (those re-acquire `_lock` → **deadlock**) | Unchanged backing if validation fails |
| **T3** | Apply mutation to backing (same semantics as today's `persistence.py` bodies under lock) | Still uncommitted from caller's perspective until **T7** |
| **T4** | Build complete durable snapshot payload from backing **under the same lock** (equivalent to `export_durable_continuation_state` without nested acquire) | Snapshot matches post-**T3** backing |
| **T5** | Physical persist: `_atomic_write` (temp + `Path.replace`) **while still holding `backing._lock`** | Disk unchanged until replace succeeds; in-process observers still blocked |
| **T6** | **Publication:** on **T5** success, backing and on-disk envelope are aligned for this mutation; no separate in-memory publish step | Committed state |
| **T7** | Release lock; return `True` to caller (**ACK**) | Readers observe committed state |

**Validation failure (T2):** release lock; return `False` / raise per existing contract; **no** **T4**–**T5**.

**Persist failure (T5) with known failure:** roll back **T3** under the **same** lock **before** **T7** when rollback is safe (no ACK was delivered; no concurrent mutator ran — guaranteed by lock). Release lock; return `False` or raise `STORE_QUERY_FAILED`; disk remains at last successful **T5**.

**ACK rule (qualification default):** `ExecutionContinuationService` must not observe a successful mutating store result unless **T5** completed successfully. Production façade **must not** return `True` until **T5** succeeds.

**Idempotent client / lost response:** If **T5** succeeded and **T7** returned `True` but the caller did not observe the response, retry with the same `expected` receives **`False` / `STALE_REVISION`** at service layer; durable snapshot already holds `updated` — client must **read** current episode. No durable regression from retry.

**Locking evaluation:** existing `backing._lock` **is sufficient** to serialize the full transaction **if** the production façade (a) holds it across **T1**–**T6**, and (b) invokes **lock-assumed** mutation + export helpers instead of public store methods or `export_durable_continuation_state`. Holding the lock across I/O blocks readers and other mutators for seal latency — **accepted** for minimal Option A (single-writer process).

**Rejected patterns:**

| Pattern | Why rejected |
|---------|----------------|
| R2 **P3** (release lock, then `_atomic_write`) | Uncommitted in-memory visibility; concurrent seal reordering; unsafe rollback |
| External lock + call public `compare_and_swap` | Non-reentrant deadlock |
| Async / batched seal without epoch gate | Out-of-order `replace` → stale snapshot can overwrite newer seal |
| Second in-memory authority / shadow CAS | Violates single semantic owner |

**Implementation boundary (architecture only — not authorized in this commit):**

- Extract or duplicate **lock-held** CAS / episode-begin / export snapshot builders from `persistence.py` (e.g. `_*_locked` helpers) callable only when `backing._lock` is already held.
- Production façade calls `_atomic_write` (or a thin delegate) from **T5** without calling `persist_from_backing` as-is (that function releases lock between export and write).
- Read methods continue to use short `with backing._lock` sections; they never overlap committed visibility with uncommitted **T3** because mutators hold the lock through **T5**.

**Canonical persistence primitives:** `_atomic_write` + envelope schema are **necessary**; `export_durable_continuation_state` / `persist_from_backing` **as wired today** are **not sufficient** for **T1**–**T7** without the lock-held split above.

### 2.2.3 Concurrency — serialization and durability ordering

| Actor | During transaction **T1**–**T6** (lock held) | After **T7** |
|-------|-----------------------------------------------|--------------|
| **Reader** (`load`, resolve) | Blocks; **cannot** observe uncommitted **T3** | Sees last committed backing |
| **Mutator A** | Holds lock | — |
| **Mutator B** | Blocks on **T1**; cannot start until A reaches **T7** | Sees A's commit or CAS conflict on retry |
| **Cross-process reader** | N/A (single-writer file) | Sees disk only after A's completed **T5** (`replace`) |

| Constraint | Rule |
|------------|------|
| **Writers** | **Single production process** per continuation file. Multi-writer file access **not** in scope. |
| **Mutation serialization** | At most one mutating transaction per backing; lock spans **T1**–**T6** inclusive. |
| **Durability ordering** | Seal *N* completes (**T5**) before mutator *N+1* acquires lock (**T1**). **No async seal** on qualification path without new ADR + monotonic `snapshot_epoch` reject-old apply. |
| **Identity invariants** | Unchanged GR-5: four-ID current episode; CAS on `continuation_id` + full snapshot equality; no alternate recovery authority. |

### 2.2.4 Persistence failure and crash semantics

Backend for minimal scope: **Option A** — `execution_continuation_durable_state.v1` + temp + `Path.replace` @ R3 HEAD.

#### 2.2.4.1 Persistence failure matrix (runtime)

| Scenario | In-memory after handling | On-disk | Caller ACK | Required store behavior |
|----------|--------------------------|---------|------------|-------------------------|
| Failure **before** `replace` (e.g. `.tmp` `write_text` error) | Rolled back **T3** under lock if safe | Prior envelope intact | **No** | `False` / `STORE_QUERY_FAILED` |
| Exception **during** `replace` | Rolled back **T3** if durable outcome **known** prior; if **unknown**, **no** optimistic rollback | Prior or new complete file per FS; may be **ambiguous** | **No** unless **T5** known success | If outcome **uncertain** → **fail-closed** / `RECONCILIATION_REQUIRED` (§2.2.4.3) |
| **T5** success, ACK lost on wire | Committed | Updated envelope | Service may retry STALE | Durable truth = committed; read converges |
| Process crash **before** **T5** completes | Lost (new process) | Prior envelope (`.tmp` may be orphan) | **No** (ACK only after **T5**) | Restart loads prior seal |
| Process crash **after** **T5**, before/during **T7** | Lost | Updated if `replace` completed | Uncertain to client; durable holds commit | Restart loads new envelope; client reads converge |
| **T5** success, crash before **T7** | Lost | Updated | Client uncertain | Same as row above — **no second seal regression** when lock serialization enforced |

**Rollback safety:** rollback of **T3** is **mandated** only when (1) ACK not yet released, (2) lock still held, (3) durable outcome of **T5** is **known failed**. **Do not** roll back memory to a stale snapshot if durable media may already reflect a **newer** successful **T5** (R2 failure mode). Uncertain I/O → reconciliation, not silent rollback.

#### 2.2.4.2 Crash / restart matrix (process-crash consistency)

| Event | On-disk after recovery | Restart-qualified semantics |
|-------|------------------------|----------------------------|
| Normal exit | Last successful **T5** | Last sealed snapshot |
| `SIGKILL` after ACK (**T7**) | Last successful **T5** | Matches last ACK |
| Crash during `.tmp` write | Prior main file | Prior seal |
| Crash during `replace` | FS-dependent; typical: old **or** new complete file, not torn main | **Atomic rename** level only |
| OS crash / power loss | FS + hardware dependent | **Not qualified** — no `fsync` / dir sync @ HEAD |

#### 2.2.4.3 Commit outcome uncertain after I/O failure

If **T5** cannot classify success vs failure (partial write, ambiguous `replace`, driver error without stable post-condition):

- Production store enters **explicit fail-closed** or **`RECONCILIATION_REQUIRED`** state: **no further mutating** operations on that backing until reload from disk or operator-approved reconciliation proves whether main file matches memory.
- **Must not** continue with divergent in-memory vs durable authority.
- **Must not** assume rollback is safe.

### 2.2.5 Durability levels and PROD-Q qualification stance

| Level | Claimed @ Option A + §2.2.2? | Notes |
|-------|------------------------------|-------|
| **Process-crash consistency** (single-writer, seal-before-ACK) | **Yes — target of this protocol** | ACK only after **T5**; lock prevents uncommitted visibility |
| **Atomic file replacement** (`Path.replace`) | **Yes — bounded** | Completed replace → prior or next full envelope |
| **OS-crash / power-loss durability** | **No** | `Path.replace` without `fsync` / directory sync; **do not claim** |

**PROD-Q reference qualification:** serialized seal-before-ACK + rename-level atomicity is **sufficient as the documented default** for HITL pause/resume restart **subject to independent audit** of the production façade implementing **T1**–**T7**. **FRZ-PRD-04** remains **OPEN** until full PROD-Q chain PASS. **Power-loss / stable-media** durability remains an **OPEN** additional requirement (separate sync policy ADR + evidence) — not inferable from `replace` alone.

### 2.2.6 Ownership (unchanged intent, R3 precision)

| Role | Owner @ proposed production shape |
|------|-------------------------------------|
| **Semantic / CAS authority** | `ExecutionContinuationStateStore` contract; logic in `persistence.py` on shared backing — **one** engine |
| **Composition / injection** | ADR-01 **B1** — `ProductionProcessComposition` retains one store instance, shared harness → Nexus |
| **Physical persistence** | Composition-scoped `ExecutionContinuationDurableStateFilePersistence` (path policy at deploy); **only** component that calls `_atomic_write` for continuation |
| **Mutation→durability coupling** | **New** production store façade (see §2.2.7) — lifecycle owner invokes seal policy; **not** `ExecutionContinuationService` |
| **Recovery** | Startup: `load_state_store()` → `ReconstructedDurableExecutionContinuationStateStore`; **no** parallel export/import recovery authority |

### 2.2.7 Backend decision — Options A / B / C (@ R3 HEAD)

| Option | Verdict | Contract consequences | Migration / smallest valid implementation | Crash guarantees | Limitations |
|--------|---------|----------------------|-------------------------------------------|------------------|-------------|
| **A. File persistence + §2.2.2 serialized transaction** | **RECOMMENDED (pending approval)** | Production store implements **T1**–**T7** on shared backing + existing envelope; `is_durable=True` only when façade wired; service unchanged | ADR-01 B1 field + factory: restore on activate; **one new class** (file-backed façade) + **lock-held** helpers; reuse `_atomic_write` | Post-**T5** restart per §2.2.4; rename-level atomicity | Lock held across I/O; single-writer; no power-loss claim without sync ADR |
| **B. Existing transactional adapter via continuation contract** | **REJECTED for PROD-Q-R1 minimal scope** | Would require mapping full store CAS + current-episode invariants into row/KV transactions | No continuation-shaped adapter @ HEAD (`ProductionPlatformPersistence` / SQLite patterns serve other domains) | SQLite `synchronous=FULL` proven **only** for those contracts | New schema + adapter = **Option C** work |
| **C. New production adapter unavoidable** | **PARTIAL — façade required, not new semantics** | If **A** accepted: new code is **coupling façade + lifecycle + lock-held helpers**, not a second CAS or alternate envelope | Same as A; optional later: sync-hardened writer replaces **T5** implementation only | Depends on chosen physical layer | Cannot avoid new production type until an existing store implements §2.2.2 — **none** @ R3 HEAD |

If §2.2.2 **serialized seal-before-ACK** is rejected in favor of weaker cadence (e.g. shutdown-only or async seal), retain:

```text
STOP — ARCHITECTURE DECISION REQUIRED
```

and do **not** ship a provisional wrapper without accepted policy.

### 2.2.8 Minimum implementation stages (after ADR acceptance only)

1. **Accept** §2.2.2 **T1**–**T7** + §2.2.4–§2.2.5 durability claims (process-crash + rename-level only).
2. **ADR-01 B1:** composition owns backing + file persistence path + store façade reference.
3. **Implement** production store façade: mutating methods execute §2.2.2 **T1**–**T7** via lock-held helpers (no nested public CAS); read methods use short lock sections only (no seal).
4. **Startup:** `load_state_store()` or empty backing + missing-file policy; **no** `wire_execution_continuation_state_store(None)` on strict path.
5. **Qualification:** GR-10-R12 true process restart on composition path; negative suite §2.2.9; STATE-X SX-F17 reconciliation; FRZ-PRD-04 remains **OPEN** until independent PASS.

### 2.2.9 Required negative qualification tests (architecture-mandated)

| ID | Intent |
|----|--------|
| **NEG-SEAL-01** | Inject **T5** failure after successful **T3** → in-memory rollback under lock; disk unchanged; service does not return success |
| **NEG-SEAL-02** | Post-**T5** restart in new OS process → loaded state matches last ACK mutation (records + `current_by_identity`) |
| **NEG-SEAL-03** | Corrupt / truncated main file → `CORRUPT_CONTINUATION_STATE` on load (existing unit pattern) |
| **NEG-SEAL-04** | Concurrent mutators: one completes **T7**, other `STALE_REVISION` / CAS `False`; durable file equals successful mutation only |
| **NEG-SEAL-05** | `is_durable=True` store without seal wiring → static/composition gate **fail closed** (no silent in-memory production authority) |
| **NEG-SEAL-06** | Lost-response simulation: after ACK, retry same CAS with old `expected` → STALE; `load` returns committed `updated` |
| **NEG-SEAL-07** | Documented **non-goal:** power-loss without sync — test must not assert survival unless sync ADR adds post-**T5** fsync/dir-sync |
| **NEG-SEAL-08** | Concurrent mutations A then B: durable seal order matches commit order; **no** reverse-order durable seals (regression guard for R2 P3) |
| **NEG-SEAL-09** | Reader during in-flight transaction: cannot observe **T3** state before **T7** (blocked on `backing._lock`) |
| **NEG-SEAL-10** | **T5** failure on mutator A does not erase or regress durable state from a prior successful **T5** on another identity / earlier commit |
| **NEG-SEAL-11** | Stale in-flight seal (hypothetical async or miswired path) cannot `replace` a newer on-disk envelope — qualification uses serialized **T1**–**T7** only; miswire must fail closed |
| **NEG-SEAL-12** | Simulated ambiguous **T5** outcome → store enters fail-closed / reconciliation-required; no further mutating ACK until reconciled |
| **NEG-SEAL-13** | Façade does not call public `compare_and_swap` / `export_durable_continuation_state` while holding `backing._lock` (deadlock / static guard) |

---

## 3. Mandatory implementation inventory (@ START_HEAD)

### 3.1 Contract & composition (not persistence backends)

| Symbol | Role | Classification |
|--------|------|----------------|
| `ExecutionContinuationStateStore` (`intergrax/contracts/execution_continuation_state_store.py`) | Semantic owner contract (GR-5-R2): CAS, current episode per four-ID | Contract — **not** a store |
| `wire_execution_continuation_state_store` | `None` → `InMemoryExecutionContinuationStateStore` | Factory — **TEST-ONLY default** in strict production |
| `wire_execution_engine_continuation_dependencies` | GR-5 bundle: service + lifecycle driver | **Composition owner** for EE/Nexus wiring; inherits injected store |
| `reconnect_execution_engine_continuation_dependencies` | Process B over **same** durable store instance | Reconnect-qualified only when store already restart-durable |
| `export_durable_continuation_state` / `execution_continuation_state_store_from_durable_export` | `execution_continuation_durable_state.v1` envelope | Serialization — **not** survival across crash without external I/O + lifecycle |
| `validate_execution_continuation_for_composition` | Strict `is_durable` gate | Policy — GCA fails closed today (`execution_continuation_state_store=None`) |

### 3.2 Store implementations (`intergrax/runtime/execution/continuation/persistence.py`)

| Implementation | Physical mechanism | Survives process exit? | CAS / identity | Restore / corruption | Init / shutdown | Production callers | Qualification evidence |
|----------------|-------------------|------------------------|----------------|----------------------|-----------------|--------------------|-------------------------|
| `InMemoryExecutionContinuationStateStore` | In-process `dict` + `threading.Lock` | **No** | Full CAS + current-episode invariants under lock | N/A (ephemeral) | Per `wire_*` or explicit construct | Lab default via `wire_execution_continuation_state_store(None)`; tests | GR-5-R2 unit suites |
| `ExecutionContinuationDurableBacking` | In-process shared dict (docstring: *reference qualification only*) | **No** unless exported/sealed externally | Same as in-memory via shared backing | Via export/restore path only | Test harness / qualification | **None** in GCA production path | GR-10-R12, SESSION-01, UCA-6C restart harness |
| `BackingExecutionContinuationStateStore` | Alias to live backing | **No** (`is_durable=False`; reconnect client) | Inherited | N/A for restart | `backing_execution_continuation_state_store(backing)` | Qualification compositions (live process) | `test_session_01_*`; GR-10-R12 reconnect gates |
| `ReconstructedDurableExecutionContinuationStateStore` | In-process backing after `restore_durable_continuation_backing` | **Only after** explicit deserialize of durable export | Inherited; `is_durable=True` | Fail-closed `CORRUPT_CONTINUATION_STATE` on envelope violations | `execution_continuation_state_store_from_durable_export` | Tests, scenario baselines, opt-in strict GCA **unit** probes (constructed in test, not launcher) | `test_gr10_r12_restart_from_durable_export`, `test_gr10_r12_r1_restart_pause_resolve_resume`; COMPAT-X schema |
| `restore_durable_continuation_backing` / `reconstructed_durable_execution_continuation_state_store` | Factory over export | Same as reconstructed store | — | — | — | Qualification only | GR-5-R5 restart identity tests (custom durable plugins) |

**Do not infer production durability from `is_durable=True` alone** — flag is set on reconstructed in-memory views after JSON restore, not on continuous disk-backed mutation.

### 3.3 File persistence adapter (`durable_state_file.py`)

| Implementation | Physical mechanism | Survives process exit? | Atomicity | Production callers | Evidence |
|----------------|-------------------|------------------------|-----------|-------------------|----------|
| `ExecutionContinuationDurableStateFilePersistence` | JSON file; write via temp + `replace` | **Yes** if `persist_from_backing` invoked | Atomic file replace on seal | **None** on sanctioned GCA / `ProductionProcessComposition` | `test_uca6c_durable_continuation_file_persistence.py`; `Uca6cTrueRestartDurableBackends.seal_continuation_for_process_death` (qualification harness); STATE-X inventory |

**Gap:** no production wiring from live CAS mutations → seal → restart `load_state_store()` at composition root.

### 3.4 Non-contract / incompatible surfaces

| Surface | Classification | Reason |
|---------|----------------|--------|
| `InMemoryContinuationStateStore` (`applications/governed_contractor_application/host/stores.py`) | **INCOMPATIBLE** | GCA host `GovernedContinuationRequest` by `task_id` — not `ExecutionContinuationStateStore` / four-ID CAS |
| `ExecutionContinuationStateStoreReadAdapter` | Adapter | Read port only |
| Test `_PluginDurableContinuationStore`, `_CustomDurableContinuationStore`, etc. | **TEST-ONLY** | Gate / plugin proofs |

### 3.5 Decision matrix summary

| Candidate | Status |
|-----------|--------|
| `InMemoryExecutionContinuationStateStore` | **TEST-ONLY** (strict production forbidden) |
| `BackingExecutionContinuationStateStore` | **IMPLEMENTED / NOT QUALIFIED** (reconnect, not restart) |
| `ReconstructedDurableExecutionContinuationStateStore` + export/restore | **IMPLEMENTED / NOT QUALIFIED** for production (semantic restart proof only) |
| `ExecutionContinuationDurableStateFilePersistence` | **IMPLEMENTED / NOT QUALIFIED** (no composition lifecycle owner) |
| SQLite / `ProductionPlatformPersistence` continuation adapter | **MISSING** |
| Sanctioned GCA composition-provided store | **MISSING** (ADR-01 B1 not implemented) |

---

## 3.6 File persistence call graph (@ `27e713748f2bc6f2acb8f751b728bf4c80c1d37f`)

```text
ExecutionContinuationService (service.py)
  → store.compare_and_swap / begin_current_episode_if_predecessor_allows
      → InMemoryExecutionContinuationStateStore | BackingExecutionContinuationStateStore
          → ExecutionContinuationDurableBacking._by_id / _current_by_identity (in-process)

Qualification seal (explicit only):
  Uca6cTrueRestartDurableBackends.seal_continuation_for_process_death (uca6c_r6_r5_9_r1_true_restart_durable_backends.py)
    → ExecutionContinuationDurableStateFilePersistence.persist_from_backing(backing)
        → export_durable_continuation_state(backing)  [lock; records + current_by_identity deep copy]
        → _atomic_write(JSON)  [*.tmp → replace]

Restart load (qualification / tests):
  ExecutionContinuationDurableStateFilePersistence.load_state_store()
    → load_export_payload() → execution_continuation_state_store_from_durable_export
        → restore_durable_continuation_backing → ReconstructedDurableExecutionContinuationStateStore

Production GCA @ START_HEAD: no edge from factory/composition to file persistence or backing.
```

**Callers of `ExecutionContinuationDurableStateFilePersistence`:** `testing_support/uca6c_r6_r5_9_r1_true_restart_durable_backends.py`, `tests/unit/runtime/execution/continuation/test_uca6c_durable_continuation_file_persistence.py`, STATE-X inventory references only.

**Callers of `persist_from_backing`:** same harness `seal_continuation_for_process_death` and unit test `test_host_a_file_persist_host_b_load_same_identity` — **never** from `ExecutionContinuationService` or composition root.

---

## 3.7 Critical verification (evidence @ START_HEAD)

| # | Question | Answer | Evidence |
|---|----------|--------|----------|
| 1 | Persist each committed mutation vs snapshots only? | **Snapshots only** on explicit `persist_from_backing` | No hook from `compare_and_swap` (`persistence.py` L143–160) to file layer; seal after pause in UCA test (`test_uca6c_…` L71–72) |
| 2 | State atomically committed before write success? | **For file seal:** yes — one envelope written via temp+replace; **for service acknowledgement:** in-memory CAS only | `durable_state_file.py` L72–78; `service.py` L77–86 returns after in-memory CAS |
| 3 | Crash between mutation and seal lose acknowledged update? | **Yes** | CAS success mutates backing only; disk unchanged until `persist_from_backing`; harness models death via explicit seal + fresh backing (`uca6c_…` L100–105) |
| 4 | Current-episode pointers and records committed consistently? | **At seal time:** yes — single locked export includes `records` + `current_by_identity`; restore validates cross-references | `export_durable_continuation_state` L255–275; `_validate_restored_store_invariants` L296–317 |
| 5 | Partial writes, corruption, restore failures? | Temp file + `replace`; missing file → `NOT_FOUND`; bad JSON/envelope → `CORRUPT_CONTINUATION_STATE` | `durable_state_file.py` L41–60, L72–83; unit tests L87–102 |
| 6 | Concurrent writers? | **Explicitly single-writer / single-process** — no file locking or multi-writer protocol | Class has no lock; qualification uses one Host A seal then Host B load |
| 7 | Authorized owner via `ProductionProcessComposition`? | **No** @ START_HEAD — dataclass has no continuation field | `production_process_composition.py` L33–39 |
| 8 | Reusable without new CAS implementation? | **Yes** — file adapter is I/O; CAS remains `InMemoryExecutionContinuationStateStore` logic on shared backing | `BackingExecutionContinuationStateStore` shares backing dicts L231–234 |

### Mutation → durability → acknowledgement flow

**Current @ HEAD (not production-qualified):**

```text
Client → ExecutionContinuationService → store.compare_and_swap (short lock)
  → on True: ACK to caller
  → (no automatic seal) persist_from_backing [export under lock, write outside lock]
```

**Required after ADR-02-R3 acceptance (architecture target — not implemented @ HEAD):**

```text
Client → ExecutionContinuationService → production façade mutator
  → T1..T7: single lock scope — validate, CAS, snapshot, _atomic_write, ACK
```

### Crash / restart failure matrix

| Event | In-memory authority | On-disk state | After restart (no seal) | After seal before crash |
|-------|---------------------|---------------|-------------------------|-------------------------|
| CAS success, immediate crash | Lost | Stale / missing | Prior sealed snapshot or empty | Last sealed snapshot |
| CAS success, seal completes | Lost (new process) | Updated envelope | N/A | `load_state_store` restores CAS state |
| Crash during `write_text` on `.tmp` | Lost | Prior file intact (replace not done) | Prior snapshot | Prior snapshot |
| Corrupt main file | — | Unreadable | `CORRUPT_CONTINUATION_STATE` on load | Same |

---

## 4. Ownership model (@ START_HEAD)

| Dimension | Owner |
|-----------|-------|
| **Semantic owner** | `ExecutionContinuationStateStore` contract + `ExecutionContinuationService` (GR-5) |
| **Physical storage owner** | **Unassigned** for production — qualification assigns operator/backend responsibility for file path in UCA harness only |
| **Composition owner (runtime)** | `wire_execution_engine_continuation_dependencies` / harness `execution_continuation_state_store` kwarg; **not** `ProductionProcessComposition` |
| **Durability policy** | `validate_execution_continuation_for_composition` |

---

## 5. Reconciliation with ADR-01 (B1)

- **B1** remains the independently recommended **composition ownership** pattern (mirror `provider_invocation_store`).
- **B1 alone is insufficient** without ADR-02 acceptance of: (1) which physical backend, (2) when state is sealed to durable media, (3) crash consistency between backing and file/DB, (4) single shared instance through Nexus (GR-10-R12).
- **Smallest future implementation scope (after both ADRs accepted):** binding-only — add composition field + forward to `create_governed_contractor_process_app`; construct **approved** durable store at deploy/activate using an **accepted** backend factory (file seal pattern, platform KV, or other ADR-chosen adapter). **No** new semantic store logic required if backend honors existing CAS contract.

---

## 6. Minimum qualification evidence before production wiring

Independent of B1 merge, backend must show:

1. **True process restart** (OS-level new process): pause/WAITING state survives; resume authority unchanged (GR-10-R12 R1-style).
2. **Durability not reconstruction-only**: continuous or explicitly defined seal points from live store to durable media; no silent in-memory authority after restart.
3. **Composition root** constructs once, shares instance harness → Nexus, tears down on process exit.
4. **Corrupt / partial file** → fail-closed (`CORRUPT_CONTINUATION_STATE` / policy equivalent).
5. **STATE-X SX-F17** and **COMPAT-X** `execution_continuation_durable_state.v1` parity if envelope used; **production delta = 0** on acceptance SHA.
6. **FRZ-PRD-04** remains **OPEN** until full PROD-Q chain PASS + GitHub audit.

---

## 7. STATE-X / GR-10 compatibility

- **STATE-X:** SX-F17 classifies `BackingExecutionContinuationStateStore` and `ReconstructedDurableExecutionContinuationStateStore` as family components with `canonical_truth=True` at mechanism level — this certifies **contract behavior in qualification inventory**, not GCA production composition binding. Moving ownership to `ProductionProcessComposition` requires STATE-X row reconciliation (ADR-01 §8); no contradiction if backend ADR records physical owner.
- **GR-10-R12:** Orchestration/HITL gates require durable injected store and prove export/restore **simulation** of restart; they do **not** certify sanctioned launcher persistence.
- **GR-5:** Episode/CAS semantics implemented in `persistence.py`; durable progress gates consume store via wired service.

---

## 8. FRZ / blockers

| ID | State |
|----|-------|
| **PROD-Q-R1** | **BLOCKED** |
| **FRZ-PRD-04** | **OPEN** |
| **ADR-01 B1** | **PENDING / NOT APPROVED** |
| **ADR-02 backend** | **PENDING / NOT APPROVED** (this record) |
| **Unresolved for closure** | **Explicit approval** of ADR-02-R3 §2.2.2 serialized seal-before-ACK (**T1**–**T7**); B1 composition field; production façade + lock-held helpers; qualification true-restart on sanctioned launcher path |

**Independent-audit blockers (ADR-02-R3):**

| Blocker | Verification target |
|---------|---------------------|
| **Lock scope** | Production mutating path holds `backing._lock` from **T1** through **T5**; no public store/export call that re-acquires `_lock` |
| **Visibility** | No reader observes pre-**T7** mutation; NEG-SEAL-09 |
| **Seal ordering** | NEG-SEAL-08 / NEG-SEAL-11 |
| **Failure / ambiguity** | NEG-SEAL-01, NEG-SEAL-12; no silent memory–disk divergence |
| **Durability claims** | Process-crash + `replace` only; power-loss **OPEN** (§2.2.5) |
| **R2 regression** | Any seal-after-unlock design remains **invalid** |

---

## 9. Minimal future implementation scope (after ADR acceptance)

1. **ADR-01 B1:** `ProductionProcessComposition.execution_continuation_state_store` (or approved lifecycle handle) + GCA forward — binding only.
2. **ADR-02:** composition-scoped factory: restore on activate, shared `ExecutionContinuationDurableBacking`, inject `ReconstructedDurableExecutionContinuationStateStore` (or approved delegate) with **`is_durable=True` only when seal policy is wired**.
3. **Seal policy implementation (ADR-02-R3):** production store façade — §2.2.2 **T1**–**T7** (lock-held serialized seal-before-ACK); reuse `_atomic_write` + lock-held export; **no** new episode/CAS semantics.
4. Qualification: GR-10-R12 true process restart + corrupt file fail-closed on composition path; STATE-X SX-F17 row update; production delta = 0.

**Not in scope:** alternate envelope, second continuation authority, production wiring in this analysis commit.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

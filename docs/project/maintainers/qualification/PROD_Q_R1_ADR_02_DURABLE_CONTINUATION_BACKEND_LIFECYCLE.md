# PROD-Q-R1-ADR-02 — Durable Continuation Backend & Lifecycle

| Field | Value |
|-------|-------|
| **Status** | **PENDING / NOT APPROVED** |
| **Date** | 2026-10-13 |
| **START_HEAD (analysis)** | `27e713748f2bc6f2acb8f751b728bf4c80c1d37f` |
| **FINAL_COMMIT (docs only)** | `402851511b42329d695166e853b40929ebf580ef` |
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

```text
Client → ExecutionContinuationService.apply_resolution / resume / _commit_lifecycle_transition
  → store.compare_and_swap (in-process, lock)
  → on True: return updated PendingExecutionContinuation to caller  [ACK]
  → (no automatic step) persist_from_backing
  → optional later: export snapshot → atomic file replace  [DURABLE only if seal runs]
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
| **Unresolved for closure** | Accept seal cadence (per-mutation vs weaker); B1 composition field; qualification true-restart on sanctioned launcher path |

---

## 9. Minimal future implementation scope (after ADR acceptance)

1. **ADR-01 B1:** `ProductionProcessComposition.execution_continuation_state_store` (or approved lifecycle handle) + GCA forward — binding only.
2. **ADR-02:** composition-scoped factory: restore on activate, shared `ExecutionContinuationDurableBacking`, inject `ReconstructedDurableExecutionContinuationStateStore` (or approved delegate) with **`is_durable=True` only when seal policy is wired**.
3. **Seal policy implementation:** delegate store wrapping successful mutations with `persist_from_backing` **or** equivalent — reuse existing export/file atomic write; **no** new episode/CAS semantics.
4. Qualification: GR-10-R12 true process restart + corrupt file fail-closed on composition path; STATE-X SX-F17 row update; production delta = 0.

**Not in scope:** alternate envelope, second continuation authority, production wiring in this analysis commit.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

# Enterprise Scale & Resilience — W3-A Checkpoint & Recovery Scaling Inventory

**Task:** Enterprise Scale & Resilience/W3-A — Checkpoint & Recovery Scaling Inventory  
**Character:** Architecture inventory + qualification only (no scaling implementation, no new managers).  
**Production code changed:** NO  
**Architecture companion:** [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md), [`NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md`](../architecture/NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md)

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `bf532e8148b35110ab82bdba047bff69e2a0dda1` |
| Branch | `development` |

**Read scope (this task):** `intergrax/runtime/execution/` (checkpoint, recovery, resume, persistence, execution state); qualification + architecture docs for execution/recovery/checkpoint/scheduler.  
**Out of scope for code reads (referenced via frozen arch only):** `intergrax/runtime/long_running/*` (TaskCheckpoint CAS, scheduler `claim_due`, `LongRunningCoordinator`).

---

## ETAP 1 — Inventory obecnego modelu

Platforma ma **dwa równoległe durable plane** w execution hosting:

| Obszar | Owner | Kontrakt | Implementacja (execution slice) | Test |
|--------|-------|----------|----------------------------------|------|
| **Checkpoint creation (task / runtime)** | `LongRunningCoordinator` + `checkpoint_builder` (long_running); Nexus/orchestration applies snapshot | `RuntimeCheckpoint` v2, `TaskCheckpoint` | `orchestration.py`, `host_task.py` — resume bind / `apply_runtime_checkpoint_to_task`; nie tworzy blobów bez long_running | R2 frozen suites under `tests/unit/runtime/architecture/test_npsc5e_r2_*`, long_running store tests |
| **Checkpoint creation (decision)** | Decision lifecycle host + contracts (semantic shape); execution hosts port | `DecisionCheckpointState`, `DecisionFinalizationKey` | `decision_checkpoint_persistence.py`; brak tworzenia snapshotów w samym porcie | Decision durability conformance (sparse vs task plane) |
| **Checkpoint storage (task)** | `TaskCheckpointPersistence` | `save(..., expected_revision=)`, append history | Wywołania z `fan_out_partial_recovery.py` → store w long_running | `test_npsc5e_r2_h2_checkpoint_revision_stale_writer_protection.py` |
| **Checkpoint storage (decision)** | `DecisionCheckpointPersistence` | load/save per finalization key | `sqlite_decision_checkpoint_persistence.py`, `in_memory_decision_checkpoint_persistence.py` | In-memory / integration via decision recovery helpers |
| **Revision / CAS (task stream)** | `TaskCheckpointPersistence` + R2-H2 | Logical `checkpoint_revision`; `StaleCheckpointWriteError` | `fan_out_partial_recovery.py` (`expected_revision=checkpoint.revision`) | R2-H2 mandatory regression |
| **Revision / CAS (attempt)** | `AttemptLifecycleService` | `AttemptLifecycleStore.compare_and_swap` | `attempt_lifecycle/service.py` (retry loop max 8) | `test_attempt_lifecycle.py`, R1 final qual |
| **Revision / CAS (decision SQLite)** | Brak dedykowanego ownera revision | Last-write-wins UPSERT | `sqlite_decision_checkpoint_persistence.py` — `ON CONFLICT DO UPDATE` bez `expected_revision` | **Gap** — brak frozen CAS na decision stream |
| **Recovery start (topology partial)** | `FanOutPartialRecoveryService` | `PartialRecoveryRequest`, `TopologyRecoverySnapshot` | `fan_out_partial_recovery.py` | R3 final qual |
| **Recovery start (decision durable)** | `decision_recovery.py` helpers | `resume_decision_from_durable_state`, reconciliation z finalization | `decision_recovery.py` | Via decision/finalization conformance |
| **Resume execution** | `LongRunningCoordinator` + Nexus orchestration | R2 eligibility gates | `orchestration.py` — `build_task_checkpoint_resume_plan`, `bind_active_execution_resume_plan`; `active_execution_resume.py` (ContextVar plan) | R2 final E2E qual |
| **Scheduler claim** | `LongRunningScheduler` + SQLite store (arch) | `claim_due` / `complete_claim`, scheduler ledger `claim_action` | Nie w execution/ — arch P0 inventory | long_running scheduler tests |
| **Lease handling** | Checkpoint store lease on scheduled rows; idempotency `claim` (arch) | Fence + expiry | execution: `StaleClaimError` on attempt CAS only | ENTERPRISE P0 inventory row |
| **Durable evidence** | RuntimeEvent bus + optional persistence | `ExecutionFailureEvidenceRequest` | `failure_evidence/runtime_event_recorder.py` | DIAG R2 qual |
| **Terminal truth (dominates resume)** | `ExecutionTerminalService` | Terminal record CAS / put_if_absent | `execution_terminal/service.py`, persistence adapters | P0C-5/6 tests |

---

## ETAP 2 — Model wykonania (opis, bez zmian)

Dwa ścieżki współistnieją: **(A) long-running task checkpoint** steruje continuity grafu/Nexus; **(B) decision checkpoint** steruje lifecycle decyzji (osobny klucz finalization).

```text
Execution start (ExecutionRuntime + root context)
      |
      v
Execute step (Nexus GraphExecutor / delegate)
      |
      v
Checkpoint write (LongRunningCoordinator → TaskCheckpointPersistence.save;
                 opcjonalnie decision save via DecisionCheckpointPersistence)
      |
      v
Continue (graph batches; partial topology snapshot on fan-out)
      |
      v
Failure (classification → R1 retry OR partial recovery OR terminal)
      |
      v
Recovery (FanOutPartialRecoveryService / resume eligibility / decision_recovery)
      |
      v
Resume from checkpoint (orchestration resume plan + stale writer / terminal / lineage gates)
```

**Decision path (równoległy):** lifecycle transitions → checkpoint snapshot on host events → terminal via `persist_terminal_decision_state` (finalization commit **before** checkpoint).

---

## ETAP 3 — Checkpoint ownership

### Kto tworzy checkpoint?

| Plane | Twórca snapshotu | Gdzie w execution |
|-------|-------------------|-------------------|
| Task / runtime | `checkpoint_builder` (long_running) wywoływany z coordinatora | Execution **aplikuje** istniejący checkpoint: `host_task.py`, `orchestration.py` |
| Decision | Decision lifecycle host (Tier-2/agents) + contracts | Execution **persistuje** przez port: `save_decision_checkpoint`, `persist_terminal_decision_state` |
| Topology partial | `capture_topology_recovery_snapshot` (long_running) po fan-out | `fan_out_partial_recovery.py` aktualizuje `runtime.topology_recovery` |

### Kiedy zapis?

- **Task:** po ważnych punktach orchestracji long-running (scheduler-driven resume, coordinator policy) — **nie** po każdym node step domyślnie (arch R2).
- **Decision:** na zdarzeniach lifecycle/finalization (host), terminal: commit finalization then checkpoint.
- **Partial recovery:** po udanym `recover_failed_slot`, CAS na revision.
- **Po failure:** R1 mintuje nowy attempt (osobny store); R3 mutuje checkpoint ze snapshotem slotów.

### Czy checkpoint jest źródłem prawdy?

| Aspekt | Task checkpoint | Decision checkpoint |
|--------|-----------------|---------------------|
| Execution continuity | **Tak** (przy braku terminal) | Częściowo — semantic decision state |
| Terminal outcome | **Nie** — `ExecutionTerminalService` dominuje | **Nie** — `DecisionFinalizationPersistence` authoritative outcome |
| Authority / policy | **Nie** — historical only (R2-H1 narrow) | Governance plane |
| Lineage / audit | **Nie** — cross-check only | **Nie** |
| Attempt identity | Bound in checkpoint; **mint** tylko `AttemptLifecycleService` | N/A |

**Rozdzielenie (frozen P0A):** `ExecutionLineageRecord` ≠ `RuntimeCheckpoint`; failure evidence (RuntimeEvent) ≠ checkpoint state.

---

## ETAP 4 — CAS / Revision analysis

| Element | Task checkpoint stream | Attempt lifecycle | Decision SQLite checkpoint |
|---------|------------------------|-------------------|----------------------------|
| Revision number | Tak — `checkpoint_revision` per `(tenant_id, task_id)` | `generation` field | **Brak** logical revision w store |
| Conflict detection | `StaleCheckpointWriteError` | CAS failure / `StaleClaimError` | Ostatni writer wygrywa |
| Concurrent update protection | Atomic CAS w persistence port | `compare_and_swap` on raw bytes | `BEGIN IMMEDIATE` + UPSERT |
| Retry on conflict | Caller-dependent (R3 maps to `PartialRecoveryError`) | Do 8 iteracji w service | Brak |
| Ownership | LongRunning + R2-H2 | `AttemptLifecycleService` | `SQLiteDecisionCheckpointPersistence` |

---

## ETAP 5 — SQLite / storage contention

| Ryzyko | Ocena | Mechanizm dziś |
|--------|-------|----------------|
| **Hot tenant** (wiele task_id / decision keys) | **P1** — jeden plik DB, connection per op, WAL; wiele streamów → lock wait | Brak shardingu; tenant_id tylko partycja logiczna |
| **Hot execution stream** (same task, many checkpoint writes) | **P1** — serializacja zapisów na streamie task; CAS retry pressure | R2 revision CAS poprawia poprawność, nie zwiększa throughput |
| **Decision key hot spot** | **P1** — brak revision CAS; contention = lost updates semantyczne | UPSERT bez expected revision |
| **Recovery storm** (10k resume po outage) | **P1** — równoległe resume + Nexus graph bez globalnego throttle | Scheduler `claim_due` **limit** batch; brak recovery admission na execution plane |

**Evidence (arch):** ENTERPRISE P0 — „Checkpoint DB: SQLite file lock / connection per operation”.

---

## ETAP 6 — Scheduler / lease analysis

Implementacja claim/lease w **long_running store** (poza execution/). Semantyka z arch:

| Element | Status |
|---------|--------|
| Lease owner | Scheduler instance + row lease w checkpoint DB |
| Lease duration | Store-defined expiry (poll loop renews via claim semantics) |
| Renewal | Re-claim / poll cycle |
| Duplicate execution protection | `claim_action` ledger + resume eligibility + terminal guard |
| Recovery after worker death | Lease expiry → row reclaimable; resume re-admission; attempt CAS unchanged |

W **execution/**: `AttemptLifecycleService` używa **CAS**, nie time-lease; `local_execution_capacity_admission` to **capacity permit**, nie scheduler lease.

---

## ETAP 7 — Orphan work

| Scenariusz | Zachowanie |
|------------|------------|
| Worker dies mid-execution | In-process state (ContextVar resume plan, active bindings) **utracone**; durable checkpoint może nadal wskazywać RUNNING / pending nodes |
| Worker dies mid checkpoint write | R2: partial write + CAS — stale/invalid revision chain; late writer → `StaleCheckpointWriteError` |
| Worker dies mid recovery | Slot recovery może być partial; checkpoint CAS failure → `PartialRecoveryError` STALE |
| **checkpoint RUNNING, brak workera** | **Tak, możliwe** — durable task state ≠ live worker; scheduler może ponowić claim resume po lease; brak automatycznego „kill RUNNING” bez terminal/cancel path |

Cooperative cancel + terminal persistence reduce but do not eliminate orphan RUNNING until resume or terminal commit.

---

## ETAP 8 — Multi-worker assumptions

| Klasa | Przykłady |
|-------|-----------|
| **Process-local** | `active_execution_resume.py`, `active_decision_checkpoint_persistence.py` ContextVars; `InMemoryAttemptLifecycleStore`; GraphExecutor semaphores; `LocalExecutionCapacityAdmission` |
| **Durable** | SQLite task checkpoint store; decision checkpoint/finalization SQLite; attempt/terminal stores when wired to KV/document/SQLite |
| **Distributed-ready (primitives)** | Task checkpoint CAS + scheduler lease claims; attempt lifecycle CAS; terminal put_if_absent; idempotency claim (arch) |

**Gaps:** brak distributed execution admission; decision checkpoint bez revision CAS; fair tenant partitioning absent (W1 ignores tenant on capacity).

---

## ETAP 9 — Recovery storm inventory (10k failed → DB returns → simultaneous recover)

| Mechanizm | Status |
|-----------|--------|
| Recovery admission | **Partial** — resume eligibility gates (terminal, lineage, governance); brak globalnego „recovery slot” portu |
| Recovery concurrency limit | **Scheduler claim `limit` only** — nie cap na równoległe graph resume |
| Retry protection | R1 `max_attempts` + backoff/jitter — dotyczy attempt retry, nie mass resume |
| Jitter | R1 backoff — optional per policy |
| Fairness | **Brak** per-tenant recovery fairness |
| Backpressure | Graph `GRAPH_BACKPRESSURE` optional; W1 root admission optional — **nie** recovery-specific |

---

## ETAP 10 — Klasyfikacja ryzyk

### P0

- **Podwójne wykonanie slotu** przy równoległym resume bez claim — mitigowane scheduler ledger + R2 stale rejection; wymaga poprawnej konfiguracji durable terminal + CAS (fail-closed qual).
- **Authority expansion on resume** — mitigowane R2-H1; regression required on changes.
- **Decision checkpoint concurrent writers** — brak CAS → teoretyczna utrata spójności semantic state przy multi-writer (P0 jeśli multi-worker zapisuje ten sam finalization key).

### P1

- SQLite hotspot (tenant/task stream).
- Recovery storm bez execution-plane throttle.
- Orphan RUNNING until lease/resume/terminal.
- Decision store last-write-wins under load.

### P2

- Brak metryk contention per store.
- Optymalizacja connection pooling / shard-by-tenant (defer do ADR).

---

## Architectural decisions required (before scaling)

1. **Single vs dual checkpoint plane** — czy skalowanie dotyczy tylko `TaskCheckpointPersistence` czy też decision store revision model.
2. **Decision checkpoint CAS** — ADR czy wyrównać do R2-H2 (`expected_revision`) czy scalić ownership do task stream.
3. **Recovery admission** — nowy port vs rozszerzenie W1 + scheduler limit (bez `RecoveryManager`).
4. **Store topology** — shard per tenant / read replicas vs replace SQLite (ADR only).
5. **Orphan RUNNING policy** — automatic terminal vs lease-only resume (semantyka execution unchanged).

---

## Final report (W3-A)

```text
STATUS: QUALIFIED INVENTORY COMPLETE (no production changes)

START_HEAD: bf532e8148b35110ab82bdba047bff69e2a0dda1
FINAL_HEAD: c8a4a868769bb567588fa81479dc127c8fe9defd
COMMIT_SHA: c8a4a8687

FILES_CHANGED: docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W3_A_CHECKPOINT_RECOVERY_SCALING_INVENTORY.md

PRODUCTION_CODE_CHANGED: NO

CHECKPOINT_OWNER: LongRunningCoordinator + checkpoint_builder (task); Decision lifecycle host + DecisionCheckpointPersistence port (decision semantic)

RECOVERY_OWNER: LongRunningCoordinator (resume); FanOutPartialRecoveryService (R3 partial); decision_recovery helpers (decision durable)

STORAGE_OWNER: TaskCheckpointPersistence / SQLiteTaskCheckpointStore (task); SQLiteDecisionCheckpointPersistence + DecisionFinalizationPersistence (decision); AttemptLifecycleStore (attempt)

CAS_MODEL: Task checkpoint logical revision CAS (R2-H2); AttemptLifecycle compare_and_swap; Decision SQLite UPSERT without revision CAS (gap)

LEASE_MODEL: Scheduler row lease + claim_action ledger (long_running); execution capacity permit (W1) orthogonal; attempt uses CAS not TTL lease

PROCESS_LOCAL_COMPONENTS: ContextVar resume/checkpoint bindings; in-memory stores; GraphExecutor semaphores; LocalExecutionCapacityAdmission

DURABLE_COMPONENTS: SQLite task checkpoint; decision checkpoint/finalization; attempt/terminal stores when wired; RuntimeEvent evidence when bus persistence set

DISTRIBUTED_READY_COMPONENTS: Task checkpoint CAS; scheduler claim; attempt CAS; terminal put_if_absent

SQLITE_CONTENTION_RISK: HIGH under hot tenant/task streams and recovery storm — single-file WAL, per-op connection

RECOVERY_STORM_RISK: MEDIUM-HIGH — batch claim limit only; no global recovery concurrency or tenant fairness

ORPHAN_WORK_RISK: MEDIUM — durable RUNNING vs dead worker until lease/resume/terminal; ContextVar state not cross-process

MULTI_WORKER_GAPS: Decision checkpoint no revision CAS; no distributed execution admission; tenant-fair recovery absent

P0_RISKS: Resume authority expansion (mitigated frozen); decision multi-writer LWW on same key; duplicate exclusive action if ledger bypassed

P1_RISKS: SQLite lock contention; recovery storm; orphan RUNNING; aligned R1 retry load on shared providers (W2 partial)

P2_RISKS: Observability of store contention; connection lifecycle optimization

NEW_CONTRACTS: NONE

NEW_MANAGERS: NONE (forbidden by W3 guardrails)

ANY_ADDED: qualification doc only

TYPE_IGNORE_ADDED: NO

GETATTR_SETATTR: NO

TESTS: not run (docs-only inventory)

RUFF: N/A

PYRIGHT: N/A

NEW_FAILURES: N/A

CROSS_SESSION_CONTAMINATION: NO

ARCHITECTURAL_DECISIONS_REQUIRED: decision CAS alignment; recovery admission; store sharding; orphan RUNNING policy (see ETAP 10)

PUSH: NO (operator instruction)

NEXT_TASK: W3-B ADR draft for checkpoint store scaling + recovery admission (no implementation until accepted)
```

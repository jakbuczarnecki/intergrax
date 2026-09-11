# ADR-ENTERPRISE-CHECKPOINT-CONSISTENCY-RECOVERY-ADMISSION: Checkpoint Consistency & Recovery Admission

| Field | Value |
|-------|-------|
| **Status** | **Accepted** — Option C event append + snapshot CAS frozen (§2.1–§2.2); recovery **start** admission implemented for `TASK_RESUME`, `PARTIAL_TOPOLOGY`, and `DECISION_DURABLE` (§3). Post-start execution-width gap for `DECISION_DURABLE` remains documented (§3.4); start-only admission is wired at `decision_durable_recovery_handoff`. |
| **Date** | 2026-09-11 |
| **Baseline** | W3-A inventory on `development`; task checkpoint CAS frozen in NPSC-5E R2-H2 |
| **Related** | [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W3_A_CHECKPOINT_RECOVERY_SCALING_INVENTORY.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W3_A_CHECKPOINT_RECOVERY_SCALING_INVENTORY.md) · [`NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md`](NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md) · W1-A `ExecutionCapacityAdmissionPort` · W2 `DependencyConcurrencyAdmissionPort` |
| **Planned contract module (W3-C)** | `intergrax/contracts/recovery_admission.py` (name frozen here; no code in W3-B) |

---

## 1. Context

Przed skalowaniem wielu workerów i odpornością na awarie platforma utrzymuje **kilka niezależnych rodzajów stanu**. Nie wolno ich scalać w jeden „globalny stan wykonania” ani w centralny manager.

| Plane | Rola | Przykładowy owner (dziś) | Źródło prawdy? |
|-------|------|---------------------------|----------------|
| **Checkpoint** | Kontynuacja wykonania (task/runtime topology; decision semantic snapshot) | `TaskCheckpointPersistence`; `DecisionCheckpointPersistence` | Task: tak dla resume grafu (przy braku terminal). Decision: częściowo — semantic load |
| **Lineage** | Audyt / cross-check tożsamości wykonania | `ExecutionLineageRecord` (osobny od checkpoint) | Nie — historyczny cross-check |
| **Evidence** | Zdarzenia failure / diagnostyka | RuntimeEvent bus + opcjonalna persistence | Nie — append evidence, nie resume |
| **Terminal state** | Nieodwracalny wynik wykonania / decyzji | `ExecutionTerminalService`; `DecisionFinalizationPersistence` | **Tak** — dominuje resume |
| **Decision state (governance)** | Lifecycle decyzji, finalization | Host decyzji + finalization commit | Outcome: finalization; snapshot: materializacja |

**Niezmienne rozdzielenie (frozen P0A / W3-A):**

```text
checkpoint ≠ lineage ≠ evidence ≠ terminal ≠ decision finalization outcome
```

- **Checkpoint** to materializowany punkt kontynuacji, nie pełna historia decyzji ani audyt policy.
- **Evidence** to zdarzenia operacyjne; replay evidence ≠ restore checkpoint.
- **Terminal** zawsze wygrywa z resume z checkpointu (execution i decision paths).

Problem skalowania: task checkpoint ma **revision CAS** (R2-H2); decision checkpoint w SQLite jest dziś **last-write-wins UPSERT** bez `expected_revision` — luka P0 przy multi-writer na tym samym `DecisionFinalizationKey`. Równolegle brak **recovery admission** na execution plane — po outage możliwa **recovery storm** (W3-A ETAP 9).

W3-B nie implementuje storage ani coordinatorów. Ustalamy modele przed W3-C.

---

## 2. Decision checkpoint consistency

Dotyczy **decision plane** (`DecisionCheckpointPersistence`). Task plane pozostaje przy logical `checkpoint_revision` + `StaleCheckpointWriteError` (bez zmiany semantyki R2-H2).

### Opcja A — Revision CAS (snapshot)

Model:

```text
read snapshot @ revision 10
write snapshot @ revision 11 with expected_revision=10
if revision changed → reject (StaleDecisionCheckpointWriteError)
```

| Zalety | Wady |
|--------|------|
| Prosty, zgodny z task checkpoint R2-H2 | Konflikt writerów → retry / conflict handling |
| Chroni przed cichym nadpisaniem semantic state | **Nie** utrwala pełnej historii decyzji |
| Niski koszt odczytu (jeden snapshot) | Hot key → presja retry na tym samym finalization key |

### Opcja B — Append-only decision history

Model:

```text
DecisionCreated → DecisionApproved → DecisionRejected → DecisionFinalized → …
                      ↓
              DecisionSnapshot (projection)
```

| Zalety | Wady |
|--------|------|
| Pełna historia, audyt, zgodność z AI governance | Wymaga materializacji aktualnego stanu |
| Naturalny model „kto co zatwierdził i kiedy” | Większa złożość odczytu i storage |
| Bezpieczny multi-writer na poziomie event append | Konieczna polityka kompaktacji / snapshot lag |

### Opcja C — Event history + CAS-guarded snapshot (hybrid)

**Uzasadnienie:** Governance wymaga historii (B); resume i load wymagają szybkiego snapshotu z ochroną przed lost update (A). To nie jest sztuczna trzecia ścieżka — W3-A wskazuje jednocześnie gap CAS **oraz** potrzebę audytu decyzji.

Model:

1. **Append-only** `DecisionEvent` (immutable, monotonic `event_sequence` per finalization key) — **authoritative audit trail**; append semantics frozen in §2.1.
2. **Materialized** `DecisionCheckpointState` ze **`snapshot_revision`** (lub `last_applied_sequence`) — zapis tylko przez `save(..., expected_revision=)` / equivalent — **osobny konflikt** od event append (§2.2).
3. Po udanym event append: materializacja / replay → snapshot save; konflikt snapshot **nie** usuwa już utrwalonego eventu (§2.2).

Task checkpoint **nie** scala się z decision stream (dual plane unchanged).

### 2.1 Authoritative `DecisionEvent` append (W3-C contract)

Event stream jest **authoritative** — dwa równoległe writery **nie mogą** skutecznie appendować dwóch różnych eventów jako tego samego następnego `event_sequence`.

**Compare-and-append** (nazwy ilustracyjne do W3-C; semantyka obowiązkowa):

```text
append(
    event,
    expected_last_sequence=N,
)

atomic condition (storage-enforced, not app-only):
    current_last_sequence == N

success:
    event.event_sequence = N + 1
    persist event (immutable)

conflict:
    StaleDecisionEventAppendError   # typed; authoritative history conflict
```

| Invariant | Wymaganie |
|-----------|-----------|
| `expected_last_sequence` | **Tak** — każdy append musi deklarować oczekiwany ostatni sequence na `DecisionFinalizationKey` |
| Atomowość | Warunek `current_last_sequence == N` i insert następnego eventu **jedna operacja atomowa** (transaction + constraint lub store-native CAS) |
| Multi-writer safety | **Zabronione:** `SELECT MAX(event_sequence)` → `next = max + 1` → `INSERT` **bez** atomowej ochrony — to race multi-writer |
| Storage | Co najmniej **`UNIQUE (finalization_key, event_sequence)`** (lub semantycznie równoważny constraint); nie polegać wyłącznie na application-side check |
| Idempotencja | Każdy event ma stabilny **`event_id`** (lub istniejący immutable identifier w payload). Retry tego samego logical append: **`same event_id` → idempotent replay OR deterministic duplicate rejection** — bez globalnego idempotency managera |
| vs snapshot CAS | **Event append conflict** = konflikt authoritative history (`StaleDecisionEventAppendError`). **Snapshot CAS conflict** = konflikt materialized projection (`StaleDecisionCheckpointWriteError` / equivalent). Snapshot CAS **nie zastępuje** event append CAS |

**Po poprawnym append (sequence N+1):**

```text
append event @ N+1
    → materialize / replay events through N+1
    → snapshot save(expected_revision=R)

snapshot CAS loses:
    reload newer snapshot
    → replay authoritative events from stream
    → retry materialization
    (do not delete or roll back appended event because projection CAS lost)
```

W3-C: osobny port / persistence dla decision events (małe kontrakty, DI — **nie** `CheckpointManager`).

### 2.2 Snapshot materialization (unchanged intent, frozen separation)

Semantyka revision CAS na snapshot (Opcja A) pozostaje na **projekcji** only. Writers konkurujący o materializację tej samej projekcji: reload + replay z authoritative stream po `StaleDecisionCheckpointWriteError`.

### Rekomendacja (decision checkpoint)

**Przyjąć Opcję C (event history + CAS-guarded snapshot)** dla W3-C i dalszego skalowania.

Uzasadnienie:

| Kryterium | Ocena |
|-----------|--------|
| Skalowanie / multi-worker | Compare-and-append serializuje authoritative sequence; snapshot CAS wykrywa równoległą materializację |
| Audyt | Event stream jest authoritative audit trail |
| AI governance | Historia transitions jest wymagana; snapshot alone (A) niewystarczający |
| Distributed execution | Event log + revision na projekcji mapuje się na przyszły shared store bez centralnego managera |

**Task checkpoint:** bez zmian — revision CAS (R2-H2). **Decision:** nie utrzymywać LWW UPSERT bez revision w produkcji multi-worker.

W3-C scope (design only here): rozszerzyć port persistence o `expected_revision` na snapshot **oraz** port compare-and-append dla decision events (§2.1).

---

## 3. Recovery admission design

### Ownership

| Concern | Owner |
|---------|--------|
| **Czy recovery może wystartować** (permit) | **Runtime recovery plane** — wywołujący (`LongRunningCoordinator` resume path, `FanOutPartialRecoveryService`, `decision_recovery` helpers) przez wstrzyknięty port |
| Wykonanie recovery (graph, slot mutate, checkpoint save) | Istniejące serwisy (bez nowego managera) |
| Checkpoint zapis | `TaskCheckpointPersistence` / decision ports (bez zmiany ownership) |
| Retry / backoff | R1 attempt lifecycle (ortogonalne) |
| Scheduler batch | `claim_due` limit (ortogonalne — nie zastępuje recovery admission) |

Port **nie** wykonuje recovery, **nie** zapisuje checkpointów, **nie** zarządza retry.

### Kontrakt (W3-C) — `RecoveryAdmissionPort`

Analogia do W1-A / W2:

```text
request = RecoveryAdmissionRequest(...)
permit = await recovery_admission.acquire(request)
# ... durable eligibility, claim/lease, execution-width handoff (§3.3) ...
await permit.release()   # once, after safe handoff — not immediately after acquire
```

Propozowane elementy (names illustrative until W3-C):

| Symbol | Rola |
|--------|------|
| `RecoveryAdmissionPort` | Async pluginable admission |
| `RecoveryAdmissionRequest` | `tenant_id`, `task_id`, `run_id`, `attempt_id`, **recovery_kind** (enum: `TASK_RESUME`, `PARTIAL_TOPOLOGY`, `DECISION_DURABLE`, …), opcjonalne metadata |
| `RecoveryAdmissionPolicy` | **`max_concurrent_recovery_starts`** (nie `max_concurrent_recoveries` — start-only permit **nie** limituje liczby trwających recovery), overload mode, optional wait timeout |
| `RecoveryAdmissionPermit` | `release()` exactly once |

**Injection:** composition root; domyślnie `None` = legacy (brak throttlingu), spójnie z W1 optional port.

### Overload behavior

| Mode | Dozwolony |
|------|-----------|
| `REJECT` | Tak — `RecoveryAdmissionExceededError` |
| `WAIT_WITH_TIMEOUT` | Tak — `RecoveryAdmissionTimeoutError` |
| `WAIT_FOREVER` | **Zabroniony** (jak W1/W2) |

Walidacja policy: mirror `ExecutionCapacityPolicy` — timeout wymagany iff `WAIT_WITH_TIMEOUT`.

### Permit lifetime

| Wariant | Opis |
|---------|------|
| **A — tylko start recovery** | Permit od acquire do **bezpiecznego handoff** post-start execution width (§3.3) — **nie** „acquire → wejście w funkcję → natychmiastowy release” |
| **B — całe recovery** | Permit przez cały czas trwania resume/graph recovery |

**Rekomendacja: A (start-only permit)** — bez zmiany głównej decyzji.

Długotrwałe recovery (graf, partial fan-out) **nie** powinno blokować puli **start** admission godzinami. Policy limituje **`max_concurrent_recovery_starts`**, nie równoległe pełne recovery.

W3-C implementuje wyłącznie **recovery start admission** — bez `RecoveryExecutionManager` / `RecoveryManager`.

**Handoff invariant (obowiązkowy minimum przed `release()`):**

```text
RecoveryAdmission acquire
    → durable recovery eligibility confirmed (terminal / lineage / governance gates)
    → durable claim / lease ownership acquired where applicable
    → recovered work handed off to canonical post-start execution-capacity owner (§3.3)
       OR recovery kind reaches another explicitly bounded execution owner
    → release RecoveryAdmissionPermit
```

Gdy ścieżka resume po admission wchodzi pod **`ExecutionCapacityAdmissionPort`** (W1): **nigdy** nie zwalniać recovery-start permit **przed** ustaleniem execution-capacity ownership (permit W1 acquired). Unikać luki:

```text
release recovery permit → brak execution permit → unbounded recovered work starts
```

**Interakcja z W1:** `ExecutionCapacityAdmissionPort` = root execution slots; `RecoveryAdmissionPort` = recovery **start** slots. Ortogonalne; nie rozszerzać W1 o recovery.

### 3.3 Post-start execution-width handoff (W1 × recovery kinds)

Start-only admission **nie** zastępuje limitu szerokości wykonania po starcie. Dla każdego `recovery_kind` W3-C musi wire’ować release permit zgodnie z tabelą.

| RECOVERY KIND | START ADMISSION OWNER | DURABLE CLAIM OWNER | POST-START EXECUTION CAPACITY OWNER | SAFE HANDOFF POINT (release permit) |
|---------------|----------------------|---------------------|-------------------------------------|-------------------------------------|
| **`TASK_RESUME`** | Caller at `LongRunningCoordinator` resume entry (`RecoveryAdmissionPort`) | `LongRunningScheduler` / task store **atomic claim** + ledger (`claim_due`, lease) where applicable | Optional **`ExecutionCapacityAdmissionPort`** on `ExecutionRuntime` when resume re-enters root execute; **`GraphExecutor`** optional caps; platform **fan-out bounds** on resumed orchestration; scheduler **`claim_due` batch limit** (coarse cross-worker) | After eligibility + **durable claim** (if path uses claim); and when W1 wired on resume execute path, **after** `ExecutionCapacityAdmissionPort.acquire` succeeds; else after claim + binding resume plan to governed execution entry (no gap before bounded graph/root path) |
| **`PARTIAL_TOPOLOGY`** | `FanOutPartialRecoveryService` entry | Task checkpoint stream + **`expected_revision` CAS** on partial recovery save; topology snapshot on checkpoint | Parent **`ActiveGovernedExecutionTask`** context; **`bounded_multi_agent_fanout`** (`MAX_FAN_OUT_CONCURRENCY`, item caps); **`GraphExecutor`** optional caps via orchestration adapter — **not** a second W1 root on typical partial path | After policy evaluation + delegation to **`FanOutCoordinationSlotExecutor` / orchestration submission** within existing governed task and fan-out bounds |
| **`DECISION_DURABLE`** | `resume_decision_from_durable_state_with_recovery_admission` (`decision_durable_recovery_handoff`) | **Terminal supremacy** + load **`DecisionFinalizationPersistence`** / checkpoint per key (no scheduler claim) | **Brak** dedykowanego process-wide execution-width owner na tej ścieżce dziś — per-key lifecycle host only | After durable materialization (`resume_decision_from_durable_state`) completes — **start-only** permit; see §3.4 for post-start width gap |

### 3.4 `DECISION_DURABLE` — post-start width gap

| Flag | Value |
|------|--------|
| **`POST_START_RECOVERY_WIDTH_GAP`** | **Tak** — mass concurrent `DECISION_DURABLE` starts can overload checkpoint/finalization I/O without a real execution-width owner (distinct from W1 root slots) |
| **Start admission (W3-C3)** | **`RecoveryKind.DECISION_DURABLE`** + `max_concurrent_recovery_starts` via `LocalRecoveryAdmission`; wired at `decision_durable_recovery_handoff` — limits concurrent **starts** only |
| **Action for post-start width** | **`ARCHITECTURAL DECISION REQUIRED`** for cross-key concurrency / backpressure beyond start admission ( **nie** auto-tworzyć `RecoveryManager`) |
| **ADR slice status** | Start admission **implemented**; post-start width policy still open |

`TASK_RESUME` / `PARTIAL_TOPOLOGY`: handoff invariants **Accepted** for W3-C wiring subject to optional W1 on resume path (§3.3).

**Tenant_id:** metadata na request (obserwowalność); **nie** partycjonuje slotów w pierwszej implementacji process-local (fairness — out of scope W3-B).

---

## 4. Orphan RUNNING policy

Scenariusz:

```text
worker started execution
      ↓
durable state / scheduler row: RUNNING (or claim held)
      ↓
worker crash
      ↓
brak live workera; checkpoint może nadal wskazywać pending work
```

### Model A — Lease

`RUNNING` + **wygasły lease** na wierszu schedulera / claim → wiersz **recoverable** (re-claim, resume po eligibility).

### Model B — Explicit substates

`RUNNING` → `RECOVERY_PENDING` → `RECOVERING` → `COMPLETED`.

### Model C — Terminal authority

Terminal record (execution lub decision finalization) **zawsze** pierwszeństwo — resume z checkpointu **zabronione** gdy terminal istnieje.

### Wybór

**Przyjąć Model A (lease-gated orphan recovery) z obowiązkowym Model C (terminal supremacy).**

| Aspekt | Decyzja |
|--------|---------|
| Wykrywanie orphan | Scheduler poll + **lease expiry** na durable claim row (long_running store — arch) |
| Kiedy | Po `lease_expires_at` (store-defined); nie natychmiast po crash |
| Kto przejmuje | Inny worker przez **atomic claim** / resume eligibility — bez centralnego coordinatora |
| Stany RECOVERY_PENDING | **Odrzucone** w W3 — unikamy rozszerzania enum execution state; recovery flow istnieje w serwisach |
| Terminal | `ExecutionTerminalService` / decision finalization **blokuje** resume niezależnie od lease |

Process-local ContextVar (resume plan, active bindings) **nie** przetrwa crash — recovery musi opierać się na **durable** checkpoint + claim + terminal gates.

---

## 5. Multi-worker readiness

| Element | Process local | Durable | Distributed ready |
|---------|---------------|---------|-------------------|
| **Checkpoint (task)** | Coordinator/builder w procesie | SQLite / store + **revision CAS** | **Tak** (CAS + stream key) |
| **Checkpoint (decision snapshot)** | ContextVar binding | SQLite UPSERT today → **CAS + events (W3-C)** | **Po W3-C** (event + snapshot revision) |
| **Decision state (events)** | Host transitions | Brak dedykowanego durable event log today | **Po W3-C** (append port) |
| **Recovery (logic)** | GraphExecutor semaphores | Resume via checkpoint + terminal | Częściowo — wymaga admission + claim |
| **Lease** | — | Scheduler row lease (long_running) | **Tak** (durable expiry) |
| **Admission (root)** | `LocalExecutionCapacityAdmission` | — | Port pluginable; local first |
| **Admission (recovery)** | Brak | — | **W3-C** — port; local then distributed |
| **Scheduler claim** | Poll loop | `claim_due` + ledger | **Tak** (atomic claim) |

---

## 6. Failure scenarios

### Scenario 1 — Concurrent checkpoint write (same execution / task stream)

| Writer | Zachowanie (task, R2-H2) |
|--------|---------------------------|
| Worker A | `save(expected_revision=n)` → success → revision n+1 |
| Worker B | `save(expected_revision=n)` → **StaleCheckpointWriteError** |

**Dane:** nie tracimy poprawnego stanu — jeden writer wygrywa, drugi musi reload + retry policy (R3 → `PartialRecoveryError` STALE where applicable).

**Decision (dziś):** LWW — **ryzyko cichej utraty**; po W3-C: **compare-and-append** on event stream (§2.1) + konflikt snapshot revision (`StaleDecisionCheckpointWriteError`) — **osobne** błędy.

### Scenario 2 — Database outage

Storage niedostępne → masowe failure resume/save/claim.

| Obszar | Ocena |
|--------|--------|
| Retry | Store-level errors propagują do caller; **nie** globalny retry manager |
| Recovery storm | Po powrocie DB tysiące workerów może resume równolegle — **brak admission dziś** (P1) |
| Backpressure | W1 root optional; scheduler `limit` batch only — **niewystarczające** |
| W3-C mitigacja | `RecoveryAdmissionPort` REJECT/WAIT_WITH_TIMEOUT + jitter na R1 (osobny ADR) |

Fail-closed: brak silent skip checkpoint/terminal.

### Scenario 3 — Worker crash (RUNNING orphan)

| Pytanie | Odpowiedź (Model A + C) |
|---------|-------------------------|
| Kto wykrywa? | Scheduler / store lease expiry + opcjonalne health (out of scope) |
| Kiedy? | Po wygaśnięciu lease, nie przy samym crash |
| Kto przejmuje? | Worker który wygra **claim** i przejdzie eligibility (terminal, lineage, governance, **recovery admission**) |

Podwójne wykonanie slotu: mitigacja **claim ledger + attempt CAS + terminal** (W3-A P0).

---

## 7. Recovery storm analysis

| Mechanizm | Status (pre W3-C) | Po ADR / W3-C |
|-----------|-------------------|---------------|
| Recovery admission | **Brak** dedykowanego portu | **Accepted design** — `RecoveryAdmissionPort` |
| Recovery concurrency limit | Scheduler `claim_due` limit only | + process-local max concurrent recovery **starts** |
| Tenant fairness | **Brak** | Out of scope implementacji (ADR only) |
| Retry protection | R1 `max_attempts` + backoff | Bez zmian (retry redesign out of scope) |
| Backpressure | Graph optional; nie recovery-specific | Recovery admission = wejściowy backpressure |
| Jitter | R1 backoff optional | Zachęca się przy mass resume (policy caller), nie w porcie |

---

## 8. Out of scope (W3-B / ten ADR)

Poza zaakceptowanym designem W3-C:

- ❌ Distributed recovery coordinator / global execution coordinator
- ❌ `RecoveryManager`, `CheckpointManager`, `PersistenceManager`, `DistributedRecoveryCoordinator`
- ❌ Database migration, sharding, read replicas
- ❌ Tenant fairness implementation
- ❌ Observability / metrics changes
- ❌ Retry redesign (R1)
- ❌ Implementacja produkcyjna portów (W3-C)
- ❌ Scalanie checkpoint z lineage lub evidence

---

## 9. W3-C implementation checklist (informative)

1. `intergrax/contracts/recovery_admission.py` — port + policy + errors (mirror W1 shape).
2. `LocalRecoveryAdmission` (preferred: `intergrax/runtime/resilience/` or `intergrax/runtime/execution/`) — process-local slots.
3. Wire at resume / partial recovery / decision durable entrypoints — **optional** `None` default.
4. Decision: compare-and-append event port (§2.1) + `expected_revision` on snapshot save; `UNIQUE(finalization_key, event_sequence)`; deprecate LWW for multi-worker configs.
5. Recovery admission: `max_concurrent_recovery_starts`; wire handoff table §3.3; **`DECISION_DURABLE` start admission** at `decision_durable_recovery_handoff` (§3.4 post-start width still open).
6. Qualification tests for admission + stale decision event append + stale snapshot write (no new managers).

---

## 10. Summary decisions

| Topic | Decision |
|-------|----------|
| Planes | checkpoint ≠ lineage ≠ evidence ≠ terminal — **no merge** |
| Task checkpoint consistency | Keep **revision CAS** (R2-H2) |
| Decision checkpoint consistency | **Event history + CAS snapshot (Option C)**; **compare-and-append** + storage unique sequence (§2.1–§2.2) |
| Recovery admission | **`RecoveryAdmissionPort`**, start-only permit, **`max_concurrent_recovery_starts`**, handoff §3.3; **`DECISION_DURABLE` start admission** owned by `decision_durable_recovery_handoff` (§3.4 post-start width gap remains) |
| Orphan RUNNING | **Lease expiry + reclaim**; **terminal always wins** |
| Central managers | **Forbidden** |

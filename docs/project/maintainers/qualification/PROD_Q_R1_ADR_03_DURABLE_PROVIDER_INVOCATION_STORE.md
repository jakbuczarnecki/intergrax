# PROD-Q-R1-ADR-03 — Durable Provider Invocation Store (PIS-ADR-01)

| Field | Value |
|-------|-------|
| **Status** | **READY FOR ARCHITECTURE DECISION** (independent recommendation — **not ratified**) |
| **Date** | 2026-10-13 |
| **START_HEAD** | `fddb01cf56ec94d8b612912a2e4819e011ca56a5` |
| **Branch** | `development` |
| **Parent** | **PROD-Q-R1** — **BLOCKED** |
| **Slice** | **PROD-Q-R1-PIS-ADR-01** · **R7 Case B** — missing production `ProviderInvocationStore` |
| **Related** | **ADR-01** [`PROD_Q_R1_ADR_01_DURABLE_CONTINUATION_OWNERSHIP.md`](PROD_Q_R1_ADR_01_DURABLE_CONTINUATION_OWNERSHIP.md) · **ADR-02** [`PROD_Q_R1_ADR_02_DURABLE_CONTINUATION_BACKEND_LIFECYCLE.md`](PROD_Q_R1_ADR_02_DURABLE_CONTINUATION_BACKEND_LIFECYCLE.md) · GR-7-A3 · GR-7-A7 · GR-10-R13 · **PROD-Q** SSOT [`PROD_Q_PLATFORM_PRODUCTION_QUALIFICATION.md`](PROD_Q_PLATFORM_PRODUCTION_QUALIFICATION.md) |
| **Does not close** | **PROD-Q-R1** · **FRZ-PRD-04** · global FRZ PASS |

**Scope:** architecture decision only — **no** store implementation, schema migration, factory wiring, launcher changes, or pytest in this slice.

---

## 1. Question

What **production-qualified physical backend** satisfies the existing `ProviderInvocationStore` contract for strict GCA / reference production composition, without inventing exactly-once delivery or enlarging the port?

**Evidence @ `fddb01cf56ec94d8b612912a2e4819e011ca56a5`:**

| Artifact | Role |
|----------|------|
| `intergrax/contracts/provider_invocation_store.py` | Authoritative port (GR-7-A3) |
| `intergrax/contracts/provider_invocation.py` | `provider_invocation.v1` / `provider_invocation_outcome.v1` records |
| `intergrax/runtime/enterprise_reliability/provider_invocation_lifecycle.py` | Write-before-effect helpers (`persist_provider_invocation_*`) |
| `intergrax/runtime/governance/orchestration_consequential_effect_reliability_boundary.py` | Topology slot: intent → dispatch → outcome; **no blind retry** on unresolved intent |
| `applications/governed_contractor_application/host/provider_invocation_lifecycle.py` | `GovernedProviderInvocationDispatchGate` — intent durability **before** `execute()` |
| `applications/governed_contractor_application/host/production_external_work_composition.py` | Strict gate: `is_durable is True` required |
| `intergrax/applications/_shared/production_process_composition.py` | Composition field `provider_invocation_store` (binding precedent) |
| `intergrax/applications/_shared/reference_production_process_lifecycle.py` | Opens **continuation** only; **caller** supplies `provider_invocation_store` today |
| `DurableTestProviderInvocationStore` / `testing_support` doubles | Qualification fixtures — `is_durable=True` **without** cross-restart media |
| **No** in-tree `ProviderInvocationStore` SQLite adapter | **Case B** gap |

---

## 2. Non-negotiable semantics (contract authority)

The port and helpers below are **fixed** for this decision. Implementation must preserve:

1. Durable invocation intent before external provider dispatch.
2. One canonical record per `invocation_id`.
3. Identical repeated invocation writes are no-ops (`provider_invocations_equivalent`).
4. Conflicting invocation payloads fail closed (`ProviderInvocationConflictError`).
5. Outcome writes are durable and idempotent (`provider_invocation_outcomes_equivalent`).
6. Conflicting final outcomes fail closed (`ProviderInvocationOutcomeConflictError`).
7. Restart restores intents and known outcomes (`is_durable` + media-backed store).
8. Missing outcome does **not** imply provider execution failure (`ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY` — GR-7-A7).
9. Uncertain provider result must **not** automatically trigger another external side effect (orchestration `OrchestrationConsequentialEffectUncertaintyError`; recovery is **decision-only** via `provider_invocation_recovery.py`).
10. Provider-neutral abstraction and domain ownership unchanged — **no** provider semantics in the store.

**Explicitly out of scope:** exactly-once delivery; provider idempotency keys; operation-status lookup; reconciliation execution — separate obligations.

---

## 3. Backend options

### 3.1 Option A — Dedicated SQLite `ProviderInvocationStore` (recommended)

| Dimension | Specification |
|-----------|----------------|
| **Canonical owner** | **Process composition lifecycle** — same tier as `ProductionProcessComposition.provider_invocation_store` and `ReferenceProductionProcessLifecycle` (extend lifecycle opener to construct/retain/close the store; **not** GCA harness defaults, **not** per-request factories). |
| **Module home (proposed)** | `intergrax/runtime/persistence/` or `intergrax/runtime/enterprise_reliability/` — `SQLiteProviderInvocationStore` + `open_provider_invocation_store_at(path)` mirroring `sqlite_opens.py` patterns. |
| **Schema & versioning** | Dedicated DB file (or dedicated table namespace **not** shared with KV terminal/attempt/checkpoint rows). Tables keyed by `invocation_id` (TEXT PK): `provider_invocation_intents` (canonical JSON `provider_invocation.v1`), `provider_invocation_outcomes` (canonical JSON `provider_invocation_outcome.v1`). Schema meta row `provider_invocation_store_schema` version `1`. **No** conflation with `distributed_kv_entries` or continuation envelopes. |
| **Transactions & concurrency** | Process-local **serialized writer** (`threading.RLock` or single connection discipline, same class as `SqliteDistributedKVStore`). Each `put_invocation` / `put_outcome`: `BEGIN IMMEDIATE` → read existing by PK → equivalence no-op or conflict raise → insert → `COMMIT`. `get_*`: read-only transaction. WAL + `PRAGMA synchronous=FULL` (align with sanctioned SQLite KV reference). |
| **Durability-before-ACK** | `put_*` returns only after successful `COMMIT`. Callers (`persist_provider_invocation_intent`, dispatch gates) already treat persistence failure as **no dispatch**. |
| **Restart & corruption** | Restart: reopen same `db_path`; `get_invocation` / `get_outcome` restore state. Corrupt file / `sqlite3` operational error: surface `ProviderInvocationPersistenceError` at helper boundary; strict composition must not substitute in-memory store. Missing file on first open: create empty schema (policy mirror continuation `missing_file_policy` if lifecycle needs fail-closed deploy). |
| **Migration** | Greenfield `schema_version=1`; future additive columns via meta table — **no** mapping from qualification doubles. |
| **Operational limits** | Single-writer process model (reference production V1); not a distributed invocation log; backup = file copy when process stopped. |
| **Smallest valid implementation** | One store class implementing the five port methods + `is_durable=True`; opener with path; lifecycle wires path next to continuation state file; `close()` flushes connection. **No** changes to `ProviderInvocationStore` ABC. Conflict/idempotency logic may delegate to existing equivalence helpers **or** enforce via PK + JSON compare — must match in-memory reference behavior in `InMemoryProviderInvocationStore`. |

### 3.2 Option B — Existing production storage adapter (`ProductionPlatformPersistence` / platform KV)

| Dimension | Verdict |
|-----------|---------|
| **Owner** | `ProductionPlatformPersistence` is composition-owned for **effective profile / terminal / attempt** primitives — **not** invocation lifecycle. |
| **Semantics** | KV `set`/`get` has **no** invocation/outcome split, no `ProviderInvocationConflictError` / outcome conflict typing, no paired intent+outcome invariants. Mapping invocation records into tenant KV would **conflate** domains and duplicate conflict rules ad hoc. |
| **Conclusion** | **REJECTED** — does not satisfy contract without a **new** domain-specific adapter (= Option A work, with worse coupling). |

### 3.3 Option C — Other backend

**Not required.** Option A meets all contractual requirements using existing sanctioned SQLite infrastructure (`sqlite_composition`, `SqliteDistributedKVStore` PRAGMA discipline, `sqlite_opens` opener pattern). Reserve Option C only if a future multi-process deployment forbids file SQLite — would need a **separate** ADR; not evidenced for PROD-Q-R1.

---

## 4. Decision (independent recommendation)

**Select Option A** — dedicated SQLite implementation of the existing `ProviderInvocationStore` contract, opened and retained at the **reference production process lifecycle** boundary and bound on `ProductionProcessComposition.provider_invocation_store`.

**Classification:** **R7 Case B** — contract and composition binding exist; **production-qualified durable media implementation is missing**.

**Maximum status:** **READY FOR ARCHITECTURE DECISION** (awaiting operator ratification — no implementation in this slice).

---

## 5. Contract & transaction matrix

| Operation | Caller | Precondition | Store transaction | On success | On failure |
|-----------|--------|--------------|-------------------|------------|------------|
| `put_invocation` | `persist_provider_invocation_intent` (may pre-read) | None | Serialize → read PK → equiv no-op / conflict / insert → commit | Intent durable | `ProviderInvocationConflictError` or `ProviderInvocationPersistenceError` — **no** provider dispatch |
| `get_invocation` | Gates, recovery, orchestration | None | Read | `ProviderInvocation` or `None` | `ProviderInvocationPersistenceError` |
| `put_outcome` | `persist_provider_invocation_outcome` | Intent should exist (callers enforce workflow) | Serialize → read PK → equiv no-op / conflict / insert → commit | Outcome durable | Conflict or persistence error — **no** implied re-dispatch |
| `get_outcome` | Gates, recovery, orchestration | None | Read | Outcome or `None` | Persistence error |
| Orchestration slot `execute_admitted_effect` | Topology reliability boundary | If outcome exists → classify; if intent without outcome → **Uncertainty** (no execute) | Uses store reads + above puts | Single external execute after intent commit | Uncertainty / definitive failure — **no** blind retry |
| External work dispatch | `GovernedProviderInvocationDispatchGate` | Intent persisted before `execute()` | Same as `put_invocation` | Provider callable runs | Intent persistence failure → abort before execute |

**ACK rule:** Provider dispatch and orchestration `execute()` run only after `put_invocation` commits successfully (existing code order preserved).

---

## 6. Outcome & recovery (canonical — no new recovery authority)

Recovery **decisions** remain in `intergrax/contracts/enterprise_reliability/provider_invocation_recovery.py` (GR-7-A7). The store only supplies durable facts. **No** second recovery authority in the SQLite adapter.

| Scenario | Durable state after | Runtime behavior (existing) | Store obligation |
|----------|---------------------|----------------------------|------------------|
| Intent committed, provider not yet called | Intent only | Retry may re-enter gate; `persist_provider_invocation_intent` no-ops if equivalent | Intent readable after restart |
| Provider called, outcome not recorded | Intent only | `CRASH_AMBIGUITY`; orchestration: intent without outcome → `OrchestrationConsequentialEffectUncertaintyError` | Intent survived commit before dispatch |
| Provider returned success, outcome persistence failed | Intent only (outcome absent) | Uncertainty; **no** automatic re-execute | Failed `put_outcome` must not ACK outcome |
| Process restarted, unresolved invocation | Intent ± outcome | Reload via `get_*`; recovery policy selects among allowed actions — **not** blind provider retry | `is_durable` + media |
| Identical repeated intent | Unchanged intent | No-op at `persist_provider_invocation_intent` | Equiv check |
| Conflicting repeated intent | Prior intent | `ProviderInvocationConflictError` | PK + payload mismatch |
| Conflicting outcome | Prior outcome | `ProviderInvocationOutcomeConflictError` | PK + payload mismatch |
| DB commit failed | Prior committed state | `ProviderInvocationPersistenceError`; no success ACK to caller | Rollback transaction |

**Missing outcome ≠ failure.** **Unknown outcome ≠ permission to re-dispatch** without explicit recovery/reconciliation/eligibility paths.

---

## 7. Composition & lifecycle

**Preserve ownership chain:**

```text
ReferenceProductionProcessLifecycle (process opener / shutdown)
  → ProductionProcessComposition.provider_invocation_store
    → create_governed_contractor_backend_app / strict topology
      → build_harness_host_runtime
        → GovernedExternalWorkProductionRuntime / orchestration reliability boundary
```

**Required lifecycle behaviors (future implementation):**

1. **Construct once** per process at `open_reference_production_process_lifecycle` (or successor) — **parallel** to continuation file open; inject instance into `create_reference_production_process_composition(..., provider_invocation_store=store)`.
2. **Retain** for full serve lifetime on `ProductionProcessComposition` — same object identity strict gates expect (`test_prod_q_r1_*` composition binding).
3. **Shutdown:** `close_process_resources()` closes SQLite connection after continuation close (order: stop accepting work → close invocation store → close continuation store — exact order to be fixed in implementation slice; **no** ownership transfer to GCA `factory.py` or FastAPI shutdown hooks).
4. **Startup failure:** if composition build fails after store open, close store in `except` path (mirror continuation `store.close()` in `open_reference_production_process_lifecycle`).

**Forbidden:** synthesizing `InMemoryProviderInvocationStore` / `DurableTestProviderInvocationStore` on strict production paths; inferring durability from `is_durable` on test doubles.

---

## 8. Qualification requirements (future negative / restart tests)

Implementation slice must add focused tests (not in this ADR slice):

| ID | Requirement |
|----|-------------|
| **PIS-N1** | Repeated identical `put_invocation` / `persist_provider_invocation_intent` — single row, no error |
| **PIS-N2** | Conflicting invocation same `invocation_id` — `ProviderInvocationConflictError` |
| **PIS-N3** | Outcome persist + idempotent repeat; conflicting outcome — `ProviderInvocationOutcomeConflictError` |
| **PIS-N4** | Concurrent threads same `invocation_id` — exactly one winner; other sees conflict or no-op |
| **PIS-N5** | Simulated commit failure (injecting store double or faulting backend) — caller receives persistence error; **no** dispatch side-effect counter increment |
| **PIS-N6** | OS-process restart (new store instance, same `db_path`) — intent and outcome visible |
| **PIS-N7** | After restart, intent without outcome — orchestration / recovery surfaces uncertainty, **no** second external provider call in default strict path |
| **PIS-N8** | Corrupt or locked DB — `ProviderInvocationPersistenceError`; strict startup/composition fails closed |
| **PIS-N9** | Strict composition rejects `is_durable=False` (existing gates + SQLite store `is_durable=True`) |
| **PIS-N10** | Unresolved outcome after simulated crash — **no** duplicate external provider dispatch without explicit recovery action |

---

## 9. Blockers & FRZ obligations

| Item | State |
|------|-------|
| **PROD-Q-R1** | **BLOCKED** — pending ratified backend + implementation + independent GitHub audit |
| **FRZ-PRD-04** | **OPEN** — production deploy path requires durable invocation store on sanctioned composition, not qualification doubles |
| **R1-SQLITE-ENV-01** | Remains in PROD-Q SSOT; dedicated invocation DB must respect **PROD-Q-R1-R3** isolation (no implicit `build/intergrax.db` repair) |
| **ADR-01 / ADR-02** | Continuation ownership/backend — **orthogonal**; invocation store must **not** share continuation CAS or episode files |
| **Operator ratification** | This document is **not** closure evidence |

---

## 10. Exit summary

| Deliverable | Value |
|-------------|-------|
| **Backend recommendation** | **Option A** — dedicated SQLite `ProviderInvocationStore` |
| **Ownership** | `ReferenceProductionProcessLifecycle` → `ProductionProcessComposition.provider_invocation_store` → GCA strict / orchestration topology |
| **Smallest implementation** | Single SQLite adapter + path opener + lifecycle open/close wiring; schema v1; no port expansion |
| **Tests required** | §8 **PIS-N1**–**PIS-N10** (future slice) |
| **Maximum qualification status** | **READY FOR ARCHITECTURE DECISION** |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

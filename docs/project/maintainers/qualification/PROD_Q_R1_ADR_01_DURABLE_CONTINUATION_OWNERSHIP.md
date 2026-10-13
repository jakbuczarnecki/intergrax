# PROD-Q-R1-ADR-01 — Durable Continuation Ownership & Production Composition

| Field | Value |
|-------|-------|
| **Status** | **ACCEPTED (architecture)** — **B1** ratified @ `392c1ac360baa7f69beab6eb691e45adfc1f4bcb`; composition field **not implemented** in DUR-01 slice |
| **Date** | 2026-10-13 |
| **START_HEAD (analysis)** | `27e713748f2bc6f2acb8f751b728bf4c80c1d37f` |
| **ADR-02 (backend)** | [`PROD_Q_R1_ADR_02_DURABLE_CONTINUATION_BACKEND_LIFECYCLE.md`](PROD_Q_R1_ADR_02_DURABLE_CONTINUATION_BACKEND_LIFECYCLE.md) — **ACCEPTED (Option A / T1–T7)**; façade slice `SerializedSealExecutionContinuationStateStore` — **IMPLEMENTED / NOT PRODUCTION-QUALIFIED** until independent audit |
| **Branch** | `development` |
| **Parent** | **PROD-Q-R1** — **BLOCKED** |
| **Related evidence** | **PROD-Q-R1-R4** [`PROD_Q_R1_R4_PRODUCTION_STARTUP_PATH.md`](PROD_Q_R1_R4_PRODUCTION_STARTUP_PATH.md) · GR-10-R12 qualification · STATE-X `ExecutionContinuationStateStore` inventory |
| **Does not close** | **PROD-Q-R1** · **FRZ-PRD-04** · global FRZ PASS |

---

## 1. Problem statement

The sanctioned GCA production entry `create_governed_contractor_process_app(process_composition=…)` reaches `validate_execution_continuation_for_composition` with `execution_continuation_state_store=None` and fails closed (`NON_DURABLE_CONTINUATION_STORE`). Strict production requires an explicit **durable** `ExecutionContinuationStateStore` (`is_durable is True`). No authorized binding exists from `ProductionProcessComposition` or the process-app factory surface to a process-lifetime durable store.

---

## 2. Semantic state (ExecutionContinuationStateStore)

**Owner contract:** `intergrax/contracts/execution_continuation_state_store.py` (GR-5-R2).

**State:** Canonical `PendingExecutionContinuation` snapshots keyed by `continuation_id`, with compare-and-swap updates and a single **current episode** per exact four-ID `ExecutionContinuationIdentity` (task, run, attempt, execution). Lifecycle transitions and HITL/orchestration resume authority consume this store via `ExecutionContinuationPort` / `ExecutionContinuationService` wired in `wire_execution_engine_continuation_dependencies`.

**Not in scope of this store:** AP-9/AP-10 lifecycle (`ProductionAgentPlatformRuntime.stores`), effective profile KV/document (`ProductionPlatformPersistence`), task checkpoints (`TaskCheckpointPersistence`), execution terminal KV, or provider invocation records (`ProviderInvocationStore`).

---

## 3. Current ownership (current HEAD)

| Responsibility | Canonical owner | GCA strict path today |
|----------------|-----------------|------------------------|
| Durability policy | `validate_execution_continuation_for_composition` (`durability_policy.py`) | Enforced when `production_mode` |
| Store injection into harness | `build_harness_host_runtime(..., execution_continuation_state_store=…)` | Only via `create_governed_contractor_backend_app` kwarg |
| Nexus / host sharing | `nexus_loop` + `nexus_host_execution` (GR-10-R12 gates) | Same injected store instance |
| Process composition root | `ProductionProcessComposition` — AP runtime, optional `provider_invocation_store` | **No** continuation field |
| Sanctioned entry | `create_governed_contractor_process_app` | Does **not** pass continuation kwarg |
| Reference composition factory | `create_reference_production_process_composition` | Does **not** construct continuation store |
| Durable implementations | `ReconstructedDurableExecutionContinuationStateStore` + export/restore (`persistence.py`) | Qualification / test pattern; **no** production SQLite adapter in tree |

**Precedent:** `provider_invocation_store` on `ProductionProcessComposition` is required for strict GCA when `process_composition` is set — same composition-root pattern for another execution-adjacent durable port.

---

## 4. Decision matrix (candidates)

| ID | Approach | Exactly-one owner | Sanctioned entry | Replaceable backend | STATE-X / GR-10 impact | Migration |
|----|----------|-------------------|------------------|---------------------|------------------------|-----------|
| **B1** | Add `execution_continuation_state_store` to `ProductionProcessComposition`; process launcher / deploy-activate constructs once; `create_governed_contractor_process_app` forwards to backend factory | **Yes** — mirrors `provider_invocation_store` | **Yes** | **Yes** — contract ABC unchanged | **Compatible** — strengthens composition-root ownership | Composition constructors + GCA `main.py` threading; static gates |
| **B2** | Extend `ProductionPlatformPersistence` bundle with continuation factory (same `db_path` / KV universe) | **Yes** — if bundle remains sole composition-owned primitive | **Yes** (via composition) | **Yes** if adapter is swappable | **Needs ADR detail** — must not conflate KV terminal/attempt with continuation CAS semantics | Platform persistence API + reference builder |
| **B3** | New kwarg on `create_governed_contractor_process_app` only (caller supplies store) | **Weak** — splits launcher vs composition root | **Partial** | Yes | Compatible but **duplicates** injection surfaces | Small API change; poor lifecycle cohesion |
| **A** | “Backend kwarg is sufficient” | N/A | **No** — sanctioned path cannot call backend with store today | N/A | N/A | **Rejected** — fails PROD-Q-R1-R4 evidence |
| **C** | Parallel store from harness defaults | **No** — forbidden | Violates durability policy | No | **Invalidates** GR-10-R12 | **Rejected** |

**Accepted:** **B1** — independently re-confirmed @ `27e713748f2bc6f2acb8f751b728bf4c80c1d37f` (composition ownership only). **Durable backend selection and process-lifetime persistence lifecycle** are **out of scope here** and remain **unresolved** — see **PROD-Q-R1-ADR-02** (`STOP — ARCHITECTURE DECISION REQUIRED` for production-qualified physical store). B1 does **not** imply startup-qualified or fully implemented continuation durability.

---

## 5. Lifecycle & recovery (expected after approval)

- **Construct / retain:** Once per production **process**, at the same lifecycle tier as `ProductionProcessComposition` (deploy/activate or reference launcher), not per HTTP request or per `build_harness_host_runtime` call.
- **Share:** Single store instance injected into harness → Nexus (existing GR-10-R12 invariant).
- **Crash/restart:** Survives only if `is_durable is True` on a restart-qualified implementation **and** an accepted physical backend has persisted state to durable media per **ADR-02** seal policy. `is_durable` alone does not imply on-disk survival.
- **Close:** Process shutdown / composition teardown may invoke an ADR-02–approved seal hook; **shutdown without a successful seal does not guarantee crash durability** — acknowledged in-memory CAS can be lost if the process dies before media write.
- **No implicit in-memory fallback** in strict mode.

---

## 6. Explicit disposition

**Option A** (existing binding sufficient for sanctioned entry): **not supported** — **B1** binding still required for sanctioned GCA path.

**Physical implementation @ DUR-01 slice:** does **not** represent production-qualified deployment; independent GitHub audit required.

**Option C** (duplicate ownership): **not observed** — gap is missing binding, not overlapping stores.

**Maximum qualification status:** **READY FOR ARCHITECTURE DECISION** (not CLOSED).

---

## 7. Minimal implementation scope (after approval only)

1. Record accepted choice (B1 and/or B2) in an **Accepted** ADR revision.
2. Extend `ProductionProcessComposition` (and reference launcher / `create_reference_production_process_composition` or approved construction helper) to own and retain one durable `ExecutionContinuationStateStore`.
3. Thread store through `create_governed_contractor_process_app` → `create_governed_contractor_backend_app` (no silent `None`).
4. Architecture/static gates: sanctioned path must not call strict factory without composition-provided durable store.
5. Targeted unit tests (`test_prod_q_r1_r4_production_startup_path.py` and composition continuity gates); **no** weakening of `validate_execution_continuation_for_composition`.

**Out of scope:** new persistence backend, FRZ-PRD-04 PASS, dependency admission materialization (PROD-Q-R1-R1), full HTTP reproduction.

---

## 8. FRZ / regression obligations

| ID | Obligation |
|----|------------|
| **FRZ-PRD-04** | Remains **OPEN** until independent qualification PASS after implementation + audit |
| **PROD-Q-R1** | Remains **BLOCKED** until continuation ownership + remaining startup deps qualified |
| **GR-10-R12** | Re-run orchestration continuation qualification gates on changed composition surfaces |
| **STATE-X** | Reconcile durable responsibility row for continuation if composition ownership moves; no rewrite of closed certification SHAs |

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

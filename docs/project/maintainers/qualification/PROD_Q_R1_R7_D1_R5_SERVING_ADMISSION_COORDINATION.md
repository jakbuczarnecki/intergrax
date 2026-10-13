# PROD-Q-R1-R7-D1-R5 — Serving Admission Coordination Architecture Decision

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (architecture / evidence only — **no production code**) |
| **Branch** | `development` |
| **START_HEAD** | `36492c847000fe67ef0081bbf3d54f2a3f56d5d3` |
| **Corrective child (this wave)** | **PROD-Q-R1-R7-D1-R5-R1** — SQLite activation domain & single-writer enforcement (§11) |
| **Parent** | **PROD-Q-R1-R7** — **BLOCKED** |
| **Corrective parent** | **PROD-Q-R1-R7-D1** / **D1b** — **BLOCKED** (independent **D1-R4** audit **REJECTED**) |
| **Prior corrective** | **PROD-Q-R1-R7-D1-R4** — **REJECTED** (independent GitHub audit) |
| **ADR baseline** | **ADR-01-R5** — **DEC-01–DEC-08 RATIFIED** |
| **Proposed ADR extension** | **R6 coordination** — **PROPOSED** ([`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) §22) — **not ratified** |
| **Next implementation (after operator approval)** | **PROD-Q-R1-R7-D1-R6** — bounded corrective code wave |
| **Does not close** | **PROD-Q-R1** · **PROD-Q-R7** · **FRZ-PRD-04** · **FRZ-PRD-05** · **FRZ-PRD-08** · any global FRZ PASS |

## 1. Canonical preflight (@ audited HEAD)

| Check | Result |
|-------|--------|
| Branch | `development` |
| HEAD | `fb05b7e52a21a5f698efea4fc5fc4a27c7292d41` (= **EXPECTED_START_HEAD** for D1-R5-R1 / **DEC-09 ratification gate**) |
| Program | **PROD-Q** — **CURRENT / BLOCKED** (roadmap) |
| Parent ordering | **PROD-Q-R1** **BLOCKED** → **PROD-Q-R1-R7** **BLOCKED** → **D1-R5** corrective (no unrelated stage) |
| **DEC-01–DEC-08** | **RATIFIED** (ADR-01-R5) |
| D1 / D1b code @ HEAD | **IMPLEMENTED** (VENV colocated adapter, handoff, store-backed per-execution admission) — **NOT QUALIFIED** |
| **FRZ-PRD-04/05/08** | **OPEN** (unchanged) |

### Independent D1-R4 rejection — root causes (ARC-01–ARC-04)

| ID | Summary | Disposition in R5 |
|----|---------|-------------------|
| **ARC-01** | Process-global `_PENDING_GRANT_ISSUANCES` + nonce + `consume_serving_execution_admission_grant` = second issuance authority | **Eliminate** in **D1-R6**; admission = single bounded operation under activation domain |
| **ARC-02** | Grant verified in one critical section, consumed later; pointer can advance (stale evidence) | **Eliminate** separable consumable grant; verify + physical effect in **one** domain boundary |
| **ARC-03** | `ActivationService._environment_locks` (per service instance), `AgentDistributionStoreState._activation_lock` (per bundle), SQLite `SqliteAgentDistributionDatabase._activation_lock` (per DB object), adapter `_UnitAdmissionGate` — not one cross-topology domain | **Document** supported vs unsupported topology; **A1 single-process** enforcement in **DEC-09 (proposed)** |
| **ARC-04** | `VenvBundleColocatedServingUnitRuntimeAccessor.admit_execution` → `admit_unit_execution` without store re-verify | **Production graph** must use only `StoreBackedColocatedServingUnitExecutionAdmission`; deprecate direct accessor admit on sanctioned wiring (**D1-R6**) |

---

## 2. Closed-world inventory (@ `36492c84…`)

Legend: **CS** = concurrency / serialization domain.

| Operation | Contract | Semantic owner | Physical owner | Composition owner | Canonical store | CS domain | Direct callers (production-relevant) | Bypass potential |
|-----------|----------|----------------|----------------|-------------------|-----------------|-----------|--------------------------------------|------------------|
| `prepare_candidate` / `commit_activation` | `ActivationService` | Agent Distribution | — | `ReferenceProductionLifecycleLauncher` (ref) | deployment + serving + activation stores | `ActivationService._environment_lock` + store mutations | launcher, tests, future prod opener | Separate `ActivationService` instance → **no shared env lock** |
| `atomic_swap_serving_revision` / CAS | `ApplicationEnvironmentServingStore` | Agent Distribution | — | — | serving record | in-mem: `state._activation_lock`; SQLite: `mutation_transaction` | `ActivationService`, tests | Second SQLite **connection object** in another process → **no shared lock** |
| `verify_current_serving_execution_admission_eligibility` | `serving_execution_admission.py` | Agent Distribution | — | handoff / per-exec admission | serving + deployment + revision | none alone | handoff authorize, `StoreBackedColocatedServingUnitExecutionAdmission` | Returns snapshot; **not** authority if decoupled from CS |
| `run_serving_execution_admission_critical_section` | same | Agent Distribution | — | Applications | in-mem serving store only | `run_under_activation_domain(state)` | handoff, per-exec admit | **Fails closed** on SQLite (`require_shared_activation_domain_state`) |
| `authorize_colocated_serving_execution_after_activation` | `serving_runtime_handoff.py` | Agent Distribution (verify) + adapter (physical open) | adapter `authorize_serving_execution_admission` | `bind_colocated_serving_runtime_after_activation` | same | in-mem activation domain | post-activate bind | Grant path + `serving_admission_authorized` unit flag |
| `authorize_serving_execution_admission` | adapter | Agent Distribution evidence + adapter | `VenvBundleColocatedRuntimeDeploymentAdapter` | bind after handoff | — | adapter `RLock` | composition bind | **Forged grant** blocked by nonce; **stale grant** partially blocked by critical-depth check only |
| `admit_unit_execution` / `release_unit_execution` | adapter | — (physical only) | adapter `_UnitAdmissionGate` | accessor / tests | — | per-unit lock | accessor, D1 tests | **Identifier-only** if reachable without store verify |
| `StoreBackedColocatedServingUnitExecutionAdmission.admit_execution` | Apps shared | Agent Distribution verify + adapter physical | gate delegate | `bind_colocated_serving_runtime_after_activation` | shared stores | in-mem activation domain | `ColocatedServingUnitAdmissionTaskExecutionPort` (GCA) | SQLite store bundle today → **BLOCKED** |
| `establish_serving_runtime_handoff` | Apps shared | Applications | — | bind | — | — | bind, tests | Does not admit execution |
| `ColocatedServingUnitAdmissionTaskExecutionPort.execute` | Apps shared | Applications (accounting) | via handoff admission port | GCA `factory.py` | — | inherits admission port | scheduler / task execution | None if handoff uses store-backed admission |
| `attest_serving_unit` | `RuntimeDeploymentAdapter` | AD orchestrates, adapter observes | adapter | `ActivationService` | deployment instance | adapter lock | activation only | — |

**GCA reachability (strict factory):** `factory.py` → `ColocatedServingUnitAdmissionTaskExecutionPort` → `handoff.colocated_execution_admission` (store-backed when bound via `bind_colocated_serving_runtime_after_activation`). **No** direct `RuntimeDeploymentAdapter` import in GCA host.

---

## 3. Architecture decision — OPTION A vs OPTION B

### OPTION A — Single-authority A1 process (**RECOMMENDED**)

**Claim:** For **DEC-01 A1** first production topology, one sanctioned OS process owns:

1. One `VenvBundleColocatedRuntimeDeploymentAdapter` instance (all physical units).
2. One durable store bundle (`SqliteAgentDistributionStoreBundle` or shared in-mem `AgentDistributionStoreState`) — **one** `SqliteAgentDistributionDatabase` object per environment universe in-process.
3. One production process composition (`open_production_process_lifecycle` target) wiring launcher + adapter + stores.
4. All pointer mutations and execution admission for that environment participate in **one activation domain** (extend `run_under_activation_domain` to SQLite via `SqliteAgentDistributionDatabase._activation_lock` / `mutation_transaction` — **not** a new business registry).

**Enforcement evidence (existing + proposed):**

| Writer | Bypass without A1 contract? | A1 mitigation |
|--------|----------------------------|---------------|
| `ActivationService` in-process | Env lock is per service instance | **Single** activation service per composition |
| Admin / rollback CLI (future) | Separate process + SQLite file | **Forbidden** in A1 without OPTION B; fail-closed if second process opens same DB without coordination |
| Second SQLite connection (same process) | Same `_activation_lock` on shared `SqliteAgentDistributionDatabase` | **Wire one DB object** (DEC-04) |
| `admit_unit_execution` direct | Public on adapter | **D1-R6:** production wiring uses store-backed port only; tests may call adapter directly |
| Grant forgery | Nonce registry | **Remove** registry (**D1-R6**) |

**Mandatory condition status:** Enforceable **iff** operator accepts **DEC-09 (proposed)** + **DEC-09-EXT** (§11) and **D1-R6** implements domain bridge + topology validation + removes grant split. **Not** enforceable for multi-process launcher + worker without OPTION B; **not** enforceable from `_activation_lock` alone (**D1-R5-R1**).

### OPTION B — Store-coordinated admission (**NOT RECOMMENDED for A1 first wave**)

Cross-process ordering via durable semantic contract (leases / fencing / transactional admission). Aligns with multi-admin topology but requires **new ratified coordination contract** beyond ADR-01-R5. **Out of scope for D1-R5**; **do not implement** here.

### Recommendation

**Adopt OPTION A** with **DEC-09 (proposed)** as explicit production topology gate. Defer OPTION B until a separate architecture ratification when multi-process serving is in scope.

---

## 4. Target solution shape (implementation contract for D1-R6)

**Single authoritative operation:** `admit_colocated_serving_execution_now(...)` (name indicative — exact API in implementation child).

Logical steps **inside one activation-domain critical section**:

1. Resolve application / environment / unit from handoff or task context.
2. Read current serving record, revision, deployment instance (**steps 2–6 of verify**).
3. Assert ACTIVE + SERVING + pointer + unit + digest match.
4. Optionally assert activation-commit snapshot not stale (`assert_activation_commit_matches_current_serving`) on handoff-open path only.
5. **Physical:** increment in-flight on `_UnitAdmissionGate` (or open gate once at handoff boundary per revised design).
6. **Exit CS** promptly after reservation.
7. Execute via existing host port.
8. Release in-flight on completion / cancel / exception.

**Handoff boundary (once per activation):** replace `ServingExecutionAdmissionGrant` + `consume_*` + `authorize_serving_execution_admission(grant)` with **`open_serving_unit_execution_gate_after_activation(...)`** running verify + `open_admission()` inside the **same** CS — **no transferable token**.

**Pointer change during execution:** already-admitted work **need not** be killed; new admissions fail closed on superseded pointer / drain / stop. Drain policy unchanged (FRZ-PRD-05).

**Serialization property:** *linearizable* with respect to serving pointer CAS and per-unit in-flight accounting for the **supported** activation domain.

---

## 5. Lifecycle & concurrency matrix (summary)

| Transition | Semantic owner | Trigger | Precondition | Physical | CS | Failure | Negative test (D1-R6) |
|------------|----------------|---------|--------------|----------|-----|---------|------------------------|
| → PREPARING | AD | `prepare_candidate` | validated revision | `adapter.prepare` | env lock | FAILED | missing artifact |
| PREPARING → READY | AD | prepare complete | readiness evidence | unit exists | env lock | FAILED | attest fail on reuse |
| READY → SERVING | AD | `commit_activation` | governance authz | optional authorize path | activation domain + CAS | rollback | non-SERVING admit rejected |
| SERVING → DRAINING | AD | `begin_drain` | prior SERVING | `begin_drain` closes gate | env lock | — | admit after drain rejected |
| DRAINING → STOPPED | AD + adapter | drain complete + stop | in-flight = 0 | `stop` | adapter lock | FAILED | stop with in-flight policy |
| * → FAILED | AD | prepare/attest fail | — | invalidate unit | env lock | — | restart does not resurrect |

**Race scenarios:**

| Scenario | Supported A1? | Disposition |
|----------|---------------|-------------|
| verify vs pointer CAS | in-mem shared state | **PASS** with single CS (D1-R6) |
| verify grant then swap then consume | current code | **FAIL** ARC-02 — remove grant |
| forward activation vs new admission | shared domain | **PASS** when serialized |
| rollback vs admission | shared stores | **PASS** fail-closed |
| duplicate authorization | adapter flag | handoff once; per-exec accounting |
| multiple `ActivationService` | — | **BLOCKED** — forbidden topology |
| separate store bundles | — | **BLOCKED** |
| separate SQLite connections / processes | — | **BLOCKED** (OPTION B) |
| adapter restart | — | **BLOCKED** until D3 recovery qualified |

---

## 6. Future child — PROD-Q-R1-R7-D1-R6

**Scope (files):**

| File | Change |
|------|--------|
| `intergrax/agent_distribution/serving_execution_admission.py` | Remove `_PENDING_GRANT_ISSUANCES`, nonce, `consume_*`, `object.__setattr__`, `_ACTIVATION_CRITICAL_DEPTH`; add SQLite activation-domain bridge; single-shot admit API |
| `intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py` | Replace grant authorize with domain-scoped open; tighten `admit_unit_execution` visibility for production |
| `intergrax/applications/_shared/serving_runtime_handoff.py` | Handoff authorize without grant object |
| `intergrax/applications/_shared/colocated_serving_unit_execution_admission.py` | Use unified admit API |
| `intergrax/applications/_shared/venv_bundle_colocated_serving_unit_runtime_accessor.py` | Remove or guard direct `admit_execution` from production path |
| Tests: `test_prod_q_r1_r7_d1_r4_*`, `d1_r3`, `d1b`, adapter tests, new SQLite CS tests | Update + 30-point matrix |

**Explicit non-goals:** new lock lease DB, Redis registry, schema change, GCA / Governance edits.

---

## 7. Acceptance specification (30 points — for D1-R6; **not PASS** until implemented)

### Authority

1. Manually constructed grant cannot authorize. → **no grant type** / non-constructible admission evidence.
2. Authentic but stale snapshot cannot authorize.
3. READY cannot admit work.
4. Non-ACTIVE revision cannot admit.
5. Non-SERVING deployment cannot admit.
6. Wrong pointer/unit/environment/digest cannot admit.
7. Missing canonical state fails closed.
8. Valid current unit admits exactly once per execution.
9. No independent adapter permission path (sanctioned wiring graph).

### Concurrency

10. Pointer swap before admission causes rejection.
11. Admission before pointer swap — defined ordering preserved.
12. Drain before admission causes rejection.
13. Admission before drain — in-flight counted.
14. Rollback prevents new work on superseded unit.
15. Two concurrent admissions preserve accounting.
16. Multiple service instances — **unsupported topology fails closed**.
17. Independent-process config fails closed.

### Physical lifecycle

18. Stopped/failed unit cannot admit.
19. Restarted process does not resurrect old unit.
20. Exception/cancellation releases in-flight.
21. Cleanup does not corrupt another unit.
22. Runtime/resolver identity continuity intact.

### Enterprise regression

23. D0 attestation/idempotency protected.
24. D1/D1b contracts preserved (handoff, GCA port).
25. Structural replaceability preserved.
26. GCA cannot bypass admission boundary.
27. No shadow registry / second authority.
28. Targeted Pyright zero new diagnostics.
29. Supported concurrency — deterministic tests.
30. Unsupported topologies explicitly rejected.

---

## 8. Enterprise audit matrix (this task)

| Area | RATIFIED | PROPOSED | IMPLEMENTED | NOT QUALIFIED | BLOCKED |
|------|----------|----------|-------------|---------------|---------|
| DEC-01 A1 topology | ✓ | | partial adapter | prod qual | |
| DEC-09 coordination | | ✓ R6 | | | |
| Grant registry authority | | remove | exists @ HEAD | | ARC-01 |
| Store-backed per-exec admit | | | ✓ in-mem | SQLite path | |
| OPTION B multi-process | | deferred | | | |
| FRZ-PRD-04/05/08 | | | | | OPEN |

---

## 9. FRZ contribution (evidence only)

| FRZ | Contribution |
|-----|----------------|
| **FRZ-PRD-04** | Defines startup ordering; D1-R6 closes admission slice only |
| **FRZ-PRD-05** | Drain vs admit races specified |
| **FRZ-PRD-08** | Bypass graph documented; D1-R6 closes accessor bypass |
| **FRZ-PRD-03** | Unsupported topology fail-closed |
| **FRZ-PRD-11** | Concurrency matrix |

All global FRZ remain **OPEN**.

---

## 10. Unresolved findings

| Finding | Class |
|---------|-------|
| ARC-01–04 | **IN-SCOPE BLOCKER** → **D1-R6** |
| SQLite execution admission CS | **IN-SCOPE BLOCKER** → **D1-R6** |
| Multi-process same env | **BLOCKED** — OPTION B or forbid |
| FRZ-PRD-04/05 global | **TRACKED FREEZE DEBT** |
| D1-R4 tests green ≠ audit PASS | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** for qual closure |

**Roadmap disposition:** **PROD-Q-R1** stays **BLOCKED**; **D1/D1b** stay **BLOCKED** pending **D1-R6** + audit; do not promote roadmap row to CLOSED.

---

## 11. PROD-Q-R1-R7-D1-R5-R1 — SQLite activation domain & single-writer (corrective)

| Field | Value |
|-------|-------|
| **Parent** | **PROD-Q-R1-R7-D1-R5** (coordination proposal — not ratified) |
| **Status** | **READY FOR AUDIT** (evidence only) |
| **Does not ratify** | **DEC-09**, **DEC-09-EXT**, **D1**, **D1b** |

### 11.1 Root cause confirmation (ROOT-01 … ROOT-05)

| ID | Finding @ `36492c84…` | Code anchor |
|----|----------------------|-------------|
| **ROOT-01** | `SqliteAgentDistributionDatabase._activation_lock` is `threading.RLock` — shared only within one Python object / process (`sqlite_stores.py` L135–137, L276–277). | Second process opening the same path does not acquire this lock. |
| **ROOT-02** | `mutation_transaction()` = object lock + `BEGIN IMMEDIATE` + load/mutate/save on **one** connection (L276–289). Execution admission has **no** equivalent qualified boundary on SQLite (`serving_execution_admission.py` L101–108 raises for non-in-memory serving store). | Pointer CAS uses `_mutate`; per-exec admit uses `run_serving_execution_admission_critical_section` → **blocked** on SQLite bundle today. |
| **ROOT-03** | SQLite store reads call `_refresh()` → `db.load_state()` on a **new** connection per call (`_SqliteMutationMixin` L496–497, L841–847). `verify_current_serving_execution_admission_eligibility` performs **three** independent refreshes (serving, revision, instance). | Not one transactional snapshot; generations can disagree under concurrent mutation (another connection/process). |
| **ROOT-04** | No API rejects: second `SqliteAgentDistributionDatabase(db_path)`; second process; duplicated launcher; admin tool; separate writer connection. | `runtime_locks` table exists in schema but is **not** wired as environment writer lease @ HEAD. |
| **ROOT-05** | D1-R4 shadow authority: `_PENDING_GRANT_ISSUANCES` + nonce (`serving_execution_admission.py` L31–90). DEC-09 rejects replacement via another registry/flag. | DEC-09-EXT must be **domain serialization**, not a token store. |

**Nested-transaction hazard (implementation-grounded):** `_mutate` is non-reentrant across connections: a second `mutation_transaction` on the same thread would block on `_activation_lock` (RLock re-enters) but open a **second** `BEGIN IMMEDIATE` — risk of `SQLITE_BUSY`, deadlock, or partial ordering. **D1-R6** must not nest `_mutate` inside an admission CS without proof; prefer **one** coordinator entry that holds the active transaction state for reads + ordered physical side effect.

### 11.2 Extended closed-world matrix (mutations + admission)

| Contract / operation | Semantic owner | Composition owner | Caller | Store / backend | Lock | Transaction boundary | Cross-process | Failure |
|---------------------|----------------|-------------------|--------|-----------------|------|----------------------|---------------|---------|
| `prepare_candidate` / `commit_activation` | `ActivationService` | prod / ref launcher | launcher, tests | SQLite `_mutate` / in-mem | `ActivationService._environment_lock` (per service instance) + store | `mutation_transaction` / in-mem `_activation_lock` | **None** (Python lock) | fail-closed AD errors |
| `atomic_swap_serving_revision` | serving store | — | `ActivationService` | SQLite / in-mem | via `_mutate` / `_activation_lock` | single IMMEDIATE txn | second process can write same file | CAS mismatch → error |
| `verify_current_serving_execution_admission_eligibility` | `serving_execution_admission` | handoff / store-backed port | handoff, `StoreBackedColocatedServingUnitExecutionAdmission` | per-store `_refresh` | none | **none** (multi-read) | uncoordinated | `ServingExecutionAdmissionError` |
| `run_serving_execution_admission_critical_section` | same | Applications | store-backed admit | in-mem only @ HEAD | `run_under_activation_domain` | in-mem mutex | N/A | **fail closed** SQLite |
| `StoreBackedColocatedServingUnitExecutionAdmission.admit_execution` | AD verify + adapter physical | bind path | GCA task port | shared stores | inherits CS | verify + `admit_execution` not atomic on SQLite | blocked today | `ServingRuntimeHandoffError` wrap |
| `admit_unit_execution` | — (physical) | — | accessor, tests | adapter gate | adapter `RLock` | none | none | identifier-only bypass if reachable |
| `SqliteRuntimeRegistryProjectionDescriptorStore.put` | AP-10 descriptor | — | coordinator | separate `BEGIN IMMEDIATE` on `_db._connect()` | store `_lock` | **parallel** txn to AD bundle | same file contention | `SQLITE_BUSY` / `ValueError` |

**Serving / ACTIVE record writers (closed world):** `ActivationService` (`prepare_candidate`, `commit_activation`, rollback/drain/stop orchestration) → `DeploymentInstanceStore.update_instance`, `ApplicationEnvironmentServingStore.atomic_swap_serving_revision`, `ApplicationEnvironmentActivationStore.atomic_commit_*`, `RuntimeRevisionService` state transitions; tests and future admin CLIs if wired to same stores. **No** GCA / Governance direct serving mutation @ HEAD.

### 11.3 A1 authority boundary (evidence-backed answers)

**A. Single-writer ownership**

| # | Answer |
|---|--------|
| 1 | **Intended** owner: one sanctioned OS process running production composition (`wire_production_process_lifecycle` / durable opener) holding one `VenvBundleColocatedRuntimeDeploymentAdapter` + one `SqliteAgentDistributionStoreBundle.database`. |
| 2 | **Yes** — any process may construct `SqliteAgentDistributionDatabase(path)`; no rejection @ HEAD. |
| 3 | **Yes** — second object can call `_mutate` / store APIs and mutate serving records if file is writable. |
| 4 | **None today.** **Proposed:** composition **startup topology validation** (DEC-09-EXT) fail-closed before traffic; does **not** by itself prevent a hostile second process — see OPTION A vs A2. |
| 5 | Two concurrent startups: **undefined** @ HEAD; SQLite last-writer wins / `SQLITE_BUSY` — not a deterministic winner contract. |
| 6 | Abrupt termination: durable READY/SERVING remains; physical unit gone — D0 attest on reuse; **no** writer lease cleanup. |
| 7 | Stale vs live ownership: **not distinguished** in storage; only adapter attestation on activation paths in **same** process. |
| 8 | Admin rollback tools: **same gap** — must be same composition graph or **OPTION B**. |

**B. SQLite transactional semantics**

- **Activation mutations:** qualified **per operation** via `mutation_transaction` (IMMEDIATE + full-state snapshot in memory).
- **Admission ordering:** **not qualified** on SQLite; in-memory path serializes with pointer transitions only when `InMemoryApplicationEnvironmentServingStore` shares `AgentDistributionStoreState`.
- **Proposed DEC-09-EXT ordering (one CS entry):** (1) enter domain bridge (process-local mutex + optional hold active IMMEDIATE txn for read generation); (2) load consistent serving + revision + instance view; (3) pointer/generation check; (4) physical `admit_execution`; (5) release domain — on exception rollback txn, **no** admission permission; on `SQLITE_BUSY` bounded retry then fail-closed.
- **Independent connections during held write txn:** additional `load_state()` connections may observe stale or concurrent states; must not be used for eligibility inside the same logical operation without shared txn/coordinator.

**C. Topology classification**

| Topology | Disposition |
|----------|-------------|
| One `ActivationService` + one in-mem `AgentDistributionStoreState` | **SUPPORTED WITH PROOF** (tests use `run_serving_execution_admission_critical_section`) |
| Two `ActivationService` instances, shared in-mem state | **MUST FAIL CLOSED** (separate `_environment_lock` maps) |
| One `SqliteAgentDistributionDatabase`, multiple stores on same `_db` | **SUPPORTED** for mutations (shared `_activation_lock`); admission **REQUIRES NEW ARCHITECTURE** (domain bridge) |
| Two `SqliteAgentDistributionDatabase` same path | **MUST FAIL CLOSED** (duplicate writer objects; lock not shared) |
| Two connections, one process | Reads via `_refresh`; writes serialize per object — cross-connection snapshot **REQUIRES NEW ARCHITECTURE** for admission |
| Two OS processes, one SQLite file | **MUST FAIL CLOSED** for A1 writer claim; **OPTION B** or strict external ops forbiddance |

### 11.4 Options disposition

| Option | Summary | @ HEAD |
|--------|---------|--------|
| **A1 — Strict process-exclusive owner** | One process owns adapter + mutations + admission. | **Recommended** for DEC-01 A1 **iff** operator ratifies DEC-09 + **DEC-09-EXT**: composition validation + SQLite domain bridge + consistent read — **not production-ready** until D1-R6 + audit. Exclusivity vs **hostile** second process: **not enforceable** without authorized lease mechanism → operational + fail-closed config. |
| **A2 — Readers in second process** | Second process may read SQLite. | **REQUIRES NEW ARCHITECTURE** to prove read-only; SQLite file permissions are **environment** concern, not platform contract @ HEAD. Writer attempt: **undefined** — treat as **MUST FAIL CLOSED** for qualification. |
| **B — Store-coordinated multi-process** | Shared coordination contract for transitions. | **Alternative**; SQLite could be first store **only** with new ratified coordinator spanning processes — **not** ratified; defer. |

**STOP — ARCHITECTURE DECISION REQUIRED:** Ratify **DEC-09-EXT** (`ActivationDomainCoordinator` protocol + composition binding) before D1-R6 claims SQLite admission PASS.

### 11.5 Recommended architecture (proposal — not permission)

| Element | Owner |
|---------|-------|
| Semantic activation / serving truth | **Agent Distribution** (`ActivationService`, canonical stores) |
| Composition / topology gate | **Applications shared** production lifecycle opener |
| Physical unit + in-flight gate | **VenvBundleColocatedRuntimeDeploymentAdapter** |
| Execution admission verify | **Agent Distribution** (`serving_execution_admission` — no grant registry) |
| GCA | Consumer via `StoreBackedColocatedServingUnitExecutionAdmission` only |
| Enforcement point | **Composition startup** (reject unsupported topology) + **domain bridge** at admit / pointer mutation linearization |
| Unsupported | Multi-process writers; duplicate DB objects; grant tokens; direct accessor admit on prod graph |

### 11.6 Serialization model (linearization points)

| Operation | Precondition | Owning contract | Pointer / txn domain | Physical effect | On failure |
|-----------|--------------|-----------------|----------------------|-----------------|------------|
| Forward activation | validated revision, governance | `ActivationService.commit_activation` | `mutation_transaction` / env lock | prior drain optional | rollback activation txn |
| Rollback | ACTIVE target | `ActivationService` + `atomic_commit_rollback` | same | adapter stop/drain | fail-closed |
| Open serving admission (handoff) | SERVING + attested unit | handoff bind | **in-mem CS today** | `open_admission` / gate | no durable permission |
| Reserve in-flight execution | current pointer + SERVING instance | store-backed admit | **domain bridge (proposed)** | `admit_execution` | no reservation |
| Begin drain | SERVING | `ActivationService.begin_drain` | env lock + adapter | close gate | — |
| Close admission | drain policy | adapter + AD | adapter lock | gate closed | — |
| Stop unit | drain complete | adapter `stop` | adapter | unit stopped | FAILED state |

**Ordering rules (proposal):** Admission **after** pointer swap with **stale identity** → **reject**. Admission **before** swap with **live** pointer at verify time → **legitimate**; after swap, new admits see new pointer; in-flight drain rules unchanged. Prior activation **result objects** never authorize execution.

### 11.7 D1-R6 scope delta (conditional on ratification)

| File | Current → required | Owner | Test | Forbidden |
|------|-------------------|-------|------|-----------|
| `serving_execution_admission.py` | in-mem-only CS → **domain bridge** + remove grant registry | AD | SQLite concurrent pointer/admit tests | new nonce registry |
| `sqlite_stores.py` | isolated txns → **coordinator hook** (no schema) | AD | busy/rollback tests | OS file lock |
| `in_memory_stores.py` | `run_under_activation_domain` → implement shared protocol | AD | in-mem regression | — |
| `colocated_serving_unit_execution_admission.py` | call bridge | Apps | GCA port tests | bypass verify |
| `serving_runtime_handoff.py` | gate open inside domain | Apps | handoff tests | grant object |
| `venv_bundle_colocated_runtime_deployment_adapter.py` | domain-scoped open; narrow `admit_unit_execution` | AD | adapter tests | second authority |
| `venv_bundle_colocated_serving_unit_runtime_accessor.py` | guard direct admit | Apps | negative wiring | prod reachability |
| `production_process_composition.py` / lifecycle opener | **topology validation** | Apps | fail-closed startup tests | silent multi-writer |
| `reference_production_lifecycle.py` | align durable wiring | Apps | D1/D1b regressions | duplicate bundle |

### 11.8 Acceptance specification (32 points — **NOT PASS** until D1-R6 implemented)

Items 1–30 from §7 remain; add:

31. **D0/D1/D1b** rollback and attestation tests remain green after domain bridge.

32. Material failures emit actionable evidence **without** creating authority (no READY inference, no grant).

(§7 items 1–30 unchanged — design evidence does **not** mark PASS.)

### 11.9 Enterprise audit matrix (D1-R5-R1)

| Requirement | HEAD evidence | Proposed target | Status | Gap |
|-------------|---------------|-----------------|--------|-----|
| Cross-process single-writer | none | composition gate + ops policy | **BLOCKED** | no lease @ HEAD |
| SQLite admission CS | fail-closed | DEC-09-EXT bridge | **BLOCKED** | D1-R6 |
| Consistent multi-store read | independent `_refresh` | coordinator snapshot | **BLOCKED** | D1-R6 |
| No shadow registry | grant dict exists | remove | **BLOCKED** | D1-R6 |
| FRZ-PRD-03 unsupported topology | partial docs | startup validation | **OPEN** | D1-R6 |
| FRZ-PRD-11 concurrency | in-mem tests only | SQLite determinism | **OPEN** | D1-R6 |

### 11.10 Tenant Isolation Audit (D1-R5-R1 scope)

**Scope:** single application environment activation domain; tenant binding via `ApplicationEnvironmentTenantResolver` + governance on control-plane mutation — **unchanged**. **State isolation:** serving records keyed by `(application_id, application_environment_id)`; cross-environment write requires store API with distinct keys — no new tenant leak introduced by proposal. **Cross-process same file:** operational misconfiguration could merge environments only if operators share one DB for multiple envs — **fail-closed topology validation** must forbid unintended sharing (**FRZ-TEN** supporting). **N/A:** per-tenant SQLite file sharding — not in DEC-09 scope.

### 11.11 Post-step discovery & roadmap completeness

| Question | Result |
|----------|--------|
| New parent blocker? | **No** — corrective under **D1-R5** / **D1/D1b**; **PROD-Q-R1-R7** remains **BLOCKED**. |
| Mandatory task misclassified as debt? | **No** — D1-R6 explicitly carries SQLite domain + grant removal. |
| Additional children? | **D1-R6** only after **DEC-09** + **DEC-09-EXT** ratification; no new roadmap row. |
| Canonical ordering intact? | **Yes** — D1-R5-R1 does not reorder roadmap stages. |
| Downstream FRZ change? | **FRZ-PRD-11** gains explicit SQLite concurrency evidence requirement when D1-R6 lands. |

---

## 12. PROD-Q-R1-R7-DEC-09-RATIFICATION-GATE — final decision package (@ `fb05b7e5…`)

| Field | Value |
|-------|-------|
| **Classification** | Corrective decision gate under **PROD-Q-R1-R7** (not a new canonical roadmap stage) |
| **Status** | **READY FOR AUDIT** (architecture / evidence only — **no ratification**, **no D1-R6**) |
| **START_HEAD** | `fb05b7e52a21a5f698efea4fc5fc4a27c7292d41` |
| **Answers** | Can **DEC-01 A1** admit serving on one sanctioned process + existing SQLite **without a second authority?** **Yes, conditionally** — **OPTION A** + ratified **DEC-09** + **DEC-09-EXT**, then **D1-R6** + independent audit. **Not** today @ HEAD. |

### 12.A Decision matrix

| Option | Supported topology | Enforced invariants | Assumptions | Known gaps | New approvals | Recommendation |
|--------|-------------------|---------------------|-------------|------------|---------------|----------------|
| **A — One co-located owner** | One OS process; one adapter; one shared `SqliteAgentDistributionDatabase` ref across store bundle; one `ActivationService`; sanctioned production composition | Pointer CAS + admission linearized in **one activation domain**; no grant registry; store-backed GCA path only; unsupported wiring rejected at **composition startup** | Operators do **not** run a second writer process on the same DB path; file permissions / deployment layout prevent hostile co-writers (**operational** — **NOT GUARANTEED** by platform alone) | External SQLite writer **NOT GUARANTEED** detectable; `_activation_lock` process-local; admission on SQLite **fail-closed** today | Ratify **DEC-09** + **DEC-09-EXT**; authorize **D1-R6** only after ratification | **RECOMMENDED** |
| **B — Store-mediated coordination** | Multiple processes, one durable store | Single cross-process serialization contract for pointer + admission | New lease/fencing or equivalent **ratified** contract; not ADR-01-R5 @ HEAD | Entire coordination model **unimplemented** | Separate architecture decision beyond DEC-09 | **Defer** |
| **C — No safe model** | — | — | — | Would block A1 entirely | — | **Reject** — A is viable under bounded trust boundary |

### 12.B Ownership diagram (current → proposed)

**Current (@ HEAD):**

```text
Governance (control plane) ──authz──► ActivationService ──► stores (_mutate / mutation_transaction)
GCA ──► ColocatedServingUnitAdmissionTaskExecutionPort ──► StoreBacked… ──► verify (3× _refresh) + CS (in-mem only)
                                              └─► adapter.admit_unit_execution (physical; bypass if unwired)
serving_execution_admission: global _PENDING_GRANT_ISSUANCES + nonce (shadow issuance)
```

**Proposed (post DEC-09 + DEC-09-EXT + D1-R6):**

```text
Applications production opener ──topology validate──► fail-closed if unsupported
ActivationService / stores ──semantic truth──► ActivationDomainCoordinator (AD-owned protocol)
  └─ linearize: eligibility read (one snapshot) → physical in-flight reserve → execute → release
RuntimeDeploymentAdapter ──physical only──► _UnitAdmissionGate (no permission semantics)
GCA ──consumer──► store-backed admission port only (no adapter import)
```

### 12.C Fail-closed — unsupported topologies (no hand-waving)

| Topology | How identified | Who rejects | When | Guaranteed? | External writer bypass? | In-flight work | Operator diagnosis |
|----------|----------------|-------------|------|-------------|-------------------------|----------------|-------------------|
| SQLite serving store without domain bridge | `require_shared_activation_domain_state` → `None` | `run_serving_execution_admission_critical_section` | Per admit / handoff CS | **Yes** (today) | N/A — admit blocked | N/A | `ServingExecutionAdmissionError`: requires shared activation domain store state |
| Two `SqliteAgentDistributionDatabase` same path (one process) | **Proposed:** composition wiring audit (single `_db` identity) | Production lifecycle opener | Startup before traffic | **Yes** if validation implemented (**D1-R6**) | Second object can still write if constructed outside composition — **NOT GUARANTEED** | Admits only if domain intact | Startup failure + structured topology reason |
| Two OS processes, one writable SQLite file | **Proposed:** deployment policy + optional startup self-check (PID singleton) — **policy**, not crypto | Composition / launcher contract | Startup | **NOT GUARANTEED** | **Yes** — independent `sqlite3.connect` | Undefined races; may corrupt logical state | Ops: multiple processes on same path; FRZ-PRD-03 evidence gap until OPTION B or ops controls |
| Two `ActivationService` instances | **Proposed:** single service on composition graph | Composition validation | Startup | **Yes** if enforced | In-process duplicate still possible if tests construct — prod graph only | Separate env locks — races | Unsupported topology error at open |
| Direct `VenvBundleColocatedServingUnitRuntimeAccessor.admit_execution` | Static prod wiring graph | **D1-R6:** narrow visibility / no sanctioned path | Bind / execute | **Yes** when wiring closed | Tests/tools may still call | Physical only — no store proof | Code review + negative wiring test |
| Multi-process launcher + worker (writer) | Config / process model | **Must fail closed** — not OPTION A | Startup | **Yes** when classified | **Yes** without OPTION B | — | Explicit OPTION B required message |
| Grant forgery | Missing / bad nonce | `consume_serving_execution_admission_grant` | Authorize | **Partial** — registry is second authority (**ARC-01**) | Forged nonce without register fails | — | Remove registry in **D1-R6** |

### 12.D SQLite execution-admission contract (DEC-09-EXT — **PROPOSED**)

**Operation sequence:** `current canonical eligibility → serialized in-flight reservation → (release coordination domain) → execute → release in-flight`

| Concern | Rule |
|---------|------|
| Authoritative serving state | **Agent Distribution** stores behind one `SqliteAgentDistributionDatabase` (or shared in-mem `AgentDistributionStoreState`) |
| Consistent read | Inside coordinator: **one** `mutation_transaction` **read generation** *or* single `load_state(conn)` on the **same** connection used for the logical operation — **not** three independent `_refresh()` calls |
| Activation CAS vs admission | Pointer mutations and admission **must** share **ActivationDomainCoordinator** mutex; mutations keep existing `_mutate` / `mutation_transaction`; admission must not nest a second IMMEDIATE txn without holding coordinator |
| `mutation_transaction` / `_mutate` / `_refresh` | `_mutate`: holds `_activation_lock`, one IMMEDIATE txn, full snapshot. `_refresh`: new connection `load_state()` — **unsafe** for eligibility alone. Coordinator must not call store getters that `_refresh` mid-reservation without shared txn |
| Adapter side effects | Physical `admit_execution` **after** eligibility proven inside domain; if physical fails, no durable “permission”; compensate in-flight counter on adapter |
| Post-reservation failure | Exception before execute: release in-flight; no grant survives |
| `SQLITE_BUSY` | Bounded retry inside coordinator; then **fail closed** — no admit, no partial grant |
| Two DB objects | Locks **not** shared — **unsupported**, fail closed at composition |
| Guarantee scope | **Process-local** serialization **proven** with shared `_db`; **cross-process** exclusivity **NOT GUARANTEED** |

**`ActivationDomainCoordinator` (name indicative — platform Protocol, AD module):**

1. `run_activation_domain[T](db, fn)` — process-local re-entrant mutex on the **canonical** `SqliteAgentDistributionDatabase` (extends `_activation_lock` semantics to admission path).
2. `read_eligibility_snapshot(...)` — serving + revision + deployment from **one** loaded `AgentDistributionStoreState` generation.
3. `reserve_colocated_execution(...)` — verify + adapter in-flight increment **before** releasing mutex **or** documented hold through reserve only (no long-lived txn across agent run).

**STOP — ARCHITECTURE DECISION REQUIRED:** Operator ratifies **DEC-09-EXT** wording above before **D1-R6** starts.

### 12.E Findings F01–F06 reconciliation

| ID | Status @ `fb05b7e5…` | Disposition |
|----|---------------------|-------------|
| **F01** | Confirmed — `RLock` per DB object | OPTION A + shared `_db`; cross-process **NOT GUARANTEED** |
| **F02** | Confirmed — admission not in `mutation_transaction` boundary | DEC-09-EXT coordinator |
| **F03** | Confirmed — independent `_refresh` in verify | Single snapshot in coordinator |
| **F04** | Confirmed — no writer exclusion API | Ops trust boundary + fail-closed topology; detect **NOT GUARANTEED** |
| **F05** | Confirmed — `_PENDING_GRANT_ISSUANCES` | Remove in **D1-R6** |
| **F06** | Confirmed — accessor / `admit_unit_execution` | Sanctioned graph store-backed only (**D1-R6**) |

### 12.F Authority graph table (closed world @ HEAD)

| Contract | Owner | Direct callers (evidence) | Store | Lock | Transaction | Physical effect | Bypass risk |
|----------|-------|---------------------------|-------|------|-------------|-----------------|-------------|
| `ActivationService.commit_activation` | AD | launcher / tests (`activation.py`) | SQLite `_mutate` / in-mem | `_environment_lock` + store | `mutation_transaction` | adapter orchestration | Second service instance |
| `atomic_swap_serving_revision` | AD serving store | `ActivationService` | SQLite / in-mem | via `_mutate` | IMMEDIATE per op | — | Other process connection |
| `verify_current_serving_execution_admission_eligibility` | AD | handoff, `StoreBackedColocatedServingUnitExecutionAdmission` | per-store `_refresh` | none | none | — | Stale multi-read |
| `run_serving_execution_admission_critical_section` | AD | handoff, store-backed admit | in-mem only | `run_under_activation_domain` | in-mem | — | SQLite → **blocked** |
| `authorize_colocated_serving_execution_after_activation` | AD + adapter | `bind_colocated_serving_runtime_after_activation` | stores | in-mem CS | grant path | `authorize_serving_execution_admission` | ARC-02 stale grant |
| `StoreBackedColocatedServingUnitExecutionAdmission` | AD + Apps | GCA `factory.py` → task port | shared | in-mem CS | verify + physical | `admit_execution` | SQLite blocked; accessor bypass |
| `admit_unit_execution` | adapter physical | accessor, tests | — | adapter lock | none | in-flight++ | Identifier-only |
| `mutation_transaction` | AD sqlite | `_mutate` on all SQLite stores | full snapshot | `_activation_lock` | BEGIN IMMEDIATE | persists JSON rows | Second DB object / process |

**Absent / not found @ HEAD:** separate `deployment.py` admission entry (orchestration lives in `activation.py` + adapter); production `open_production_process_lifecycle` may be reference-only — sanctioned path documented as `create_reference_production_process_composition` / `ReferenceProductionLifecycleLauncher` until durable opener lands (**TRACKED FREEZE DEBT**).

### 12.G Decision request (operator)

Authorize **exactly one** of the following (default recommendation: **1 + 2**):

1. **Ratify DEC-09 (OPTION A)** — A1 single-process topology gate per ADR §22 table (rules 1–7), accepting that **hostile cross-process SQLite writers are NOT GUARANTEED** excluded without operational controls or future OPTION B.
2. **Ratify DEC-09-EXT** — `ActivationDomainCoordinator` contract per §12.D (composition topology validation + domain bridge + consistent snapshot + ordered physical reserve; **forbidden** without separate decision: OS file lock, PID lease file, lease table, shadow permission DB).
3. **Reject OPTION A** — record **OPTION C** disposition; keep **PROD-Q-R1-R7** **BLOCKED** until a new decision (likely OPTION B scope).

**Explicit non-authorization in this gate:** opening **D1-R6** implementation, promoting DEC-09/EXT to RATIFIED in ADR without operator message, closing **PROD-Q-R1-R7** / **D1** / **D1b**, any FRZ PASS.

### 12.H Enterprise audit matrix (ratification gate)

| Requirement | HEAD evidence | Proposed | PASS/PARTIAL/BLOCKED | Residual risk |
|-------------|---------------|----------|----------------------|---------------|
| Contracts over implementations | ADR DEC-01–08 RATIFIED | DEC-09/EXT protocols | **PARTIAL** | Implementation ≠ qualified |
| Layer boundaries | Tier rules in AGENTS.md | unchanged | **PASS** | — |
| No shadow authority | `_PENDING_GRANT_ISSUANCES` | remove D1-R6 | **BLOCKED** | ARC-01 |
| Bypass resistance | accessor admit, SQLite CS gap | topology + wiring | **BLOCKED** | F04,F06 |
| Fail-closed unsupported | SQLite admit raises | startup validation | **PARTIAL** | cross-process |
| Transaction consistency | `_mutate` qualified | coordinator read | **BLOCKED** | F03 |
| Cross-process guarantees | none | ops boundary | **BLOCKED** | NOT GUARANTEED |
| Governance ≠ execution | GCA consumer only | unchanged | **PASS** | — |
| FRZ-PRD-03/04/05/08/11 | OPEN checklist | evidence in §11–12 | **OPEN** | freeze debt |

### 12.I Tenant isolation & roadmap (gate scope)

**Tenant isolation:** unchanged — environment-scoped store keys; governance on activation mutations; DEC-09 does not introduce cross-tenant shared admission state. Risk: **operational** sharing of one SQLite file across environments → **fail-closed topology validation** (FRZ-TEN supporting).

**Roadmap:** **PROD-Q** → **PROD-Q-R1** **BLOCKED** → **PROD-Q-R1-R7** **BLOCKED** → **D1/D1b** **BLOCKED** → **D1-R5** / **D1-R5-R1** / **DEC-09 gate** = **READY FOR AUDIT** → next mandatory dependency: **operator ratification** then **D1-R6**. No new roadmap row; do not close parent.

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

# PROD-Q-R1-R7-D1-R5 — Serving Admission Coordination Architecture Decision

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (architecture / evidence only — **no production code**) |
| **Branch** | `development` |
| **START_HEAD** | `7495bcdd5abaf3f279f363af7cb1f45e869ba32b` |
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
| HEAD | `7495bcdd5abaf3f279f363af7cb1f45e869ba32b` (= **EXPECTED_START_HEAD**) |
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

## 2. Closed-world inventory (@ `7495bcdd…`)

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

**Mandatory condition status:** Enforceable **iff** operator accepts **DEC-09 (proposed)** A1 topology contract and **D1-R6** extends activation domain to SQLite + removes grant split. **Not** enforceable for multi-process launcher + worker without OPTION B.

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

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

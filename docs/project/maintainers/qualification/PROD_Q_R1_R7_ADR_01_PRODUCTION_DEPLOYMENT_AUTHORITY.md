# PROD-Q-R1-R7-ADR-01 — Production Deployment Authority, Topology & Process Ownership

| Field | Value |
|-------|-------|
| **Status** | **READY FOR ARCHITECTURE DECISION** (proposed — **not ratified**) |
| **Date** | 2026-10-13 |
| **Baseline SHA** | `7ac1da1499897a4f61d4f26fd2e6477dd4dfed9c` |
| **Branch** | `development` |
| **Parent** | **PROD-Q-R1** — **BLOCKED** |
| **Slice** | **PROD-Q-R1-R7-ADR-01** (architecture only; **no implementation** in this commit) |
| **Related evidence** | **PROD-Q-R1-R4** [`PROD_Q_R1_R4_PRODUCTION_STARTUP_PATH.md`](PROD_Q_R1_R4_PRODUCTION_STARTUP_PATH.md) · **R5** [`PROD_Q_R1_R5_STRICT_STARTUP_LIFECYCLE.md`](PROD_Q_R1_R5_STRICT_STARTUP_LIFECYCLE.md) · **R6** [`PROD_Q_R1_R6_STRICT_STARTUP_LIFECYCLE.md`](PROD_Q_R1_R6_STRICT_STARTUP_LIFECYCLE.md) · **ADR-01/02/03** (continuation / invocation) · **PIS-02** (`open_reference_production_process_lifecycle`) |
| **Does not close** | **PROD-Q-R1** · **PROD-Q-R7** · **FRZ-PRD-04** · **FRZ-PRD-05** · any global FRZ PASS |

**Roadmap alignment:** Canonical [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) lists **PROD-Q** as **OPEN / BLOCKED ON** `PROD-Q-R1`, `PROD-Q-P1..P3`. **PROD-Q-R1-R7-ADR-01** is a **proposed child of PROD-Q-R1**, not yet a ratified roadmap row — no conflict with canonical ordering; operator ratification required before roadmap promotion.

---

## 1. Baseline SHA and canonical scope

| Item | Value |
|------|-------|
| **Evidence lock** | `7ac1da1499897a4f61d4f26fd2e6477dd4dfed9c` |
| **Mandatory stage** | **PROD-Q** (CURRENT per freeze checklist) |
| **Parent program** | **PROD-Q-R1** — Production Startup / Dependency Materialization Corrective — **BLOCKED** |
| **This slice** | Architecture decision for deployment authority only |
| **Next mandatory task (after ratification)** | **PROD-Q-R1-R7-D1** (production `RuntimeDeploymentAdapter`) unless operator reorders |

Applicable freeze rows (unchanged **OPEN**): **FRZ-PRD-01**, **FRZ-PRD-03**, **FRZ-PRD-04**, **FRZ-PRD-05**, **FRZ-PRD-07**, **FRZ-PRD-08**; supporting **FRZ-TYP-***, **FRZ-PLG-***, **FRZ-RPL-***, **FRZ-CTR-***, **FRZ-TEN-*** where implicated below.

---

## 2. Root cause (code-supported)

Production strict serving for GCA is intentionally **fail-closed** until an **activated** `ProductionProcessComposition` exists (`applications/governed_contractor_application/host/main.py`: `create_app` / `run()` reject unactivated entry). **Deployment and activation semantics** are owned by **Agent Distribution** (`ActivationService` + durable stores). Today every sanctioned lifecycle wiring path injects **`FakeInMemoryRuntimeDeploymentAdapter`** (`intergrax/applications/_shared/reference_production_lifecycle.py`), which the module header explicitly excludes from **production deployment/readiness authority**.

There is **no** non-fake `RuntimeDeploymentAdapter` implementation in the repository. Physical readiness therefore cannot be evidenced on a production path: activation orchestration is structurally correct but **deployment truth is simulated**. Registry projection for production is constrained to **`MaterializationTopology.VENV_BUNDLE`** with **immutable on-disk artifacts** (`production_registry_projection_input_bundle.py`), while default in-tree **VENV_BUNDLE materialization** remains **unsupported** (`UnsupportedVenvBundleMaterializationAdapter` in `materialization_adapters.py`) — production assumes **externally produced** bundle directories referenced by `file://` (or qualification `test://`) locators.

Process ownership for strict GCA startup is partially qualified (**R5/R6**) via `open_reference_production_process_lifecycle`, but **deploy/activate** still uses fake deployment, and **GCA `main.run()`** is not a sanctioned production entry.

---

## 3. Closed-world inventory (contract → owner → implementation → caller → persistence)

### A. Deployment

| Mechanism | Contract | Semantic owner | Implementation / composition | Production caller (today) | Persistence / evidence |
|-----------|----------|----------------|------------------------------|---------------------------|-------------------------|
| Deployment port | `RuntimeDeploymentAdapter` (`intergrax/agent_distribution/deployment.py`) | **Agent Distribution** (§20) | **Protocol only**; sole impl: `FakeInMemoryRuntimeDeploymentAdapter` | `ActivationService.prepare_candidate` / drain / stop via `ReferenceProductionLifecycleLauncher.deploy_and_activate` | Adapter-internal dict (fake); **not durable** |
| Prepared unit | `PreparedDeployment`, `DrainStatus` | Agent Distribution | Pydantic models in `deployment.py` | `ActivationService` after `prepare` / `check_readiness` | `DeploymentInstanceRecord.readiness_evidence_ref`, `serving_unit_ref` |
| Instance state | `DeploymentInstanceState`, `DeploymentInstanceRecord` | Agent Distribution | `deployment.py` | `ActivationService`, rollback/drain paths | `DeploymentInstanceStore` |
| Instance store (protocol) | `DeploymentInstanceStore` (`stores.py`) | Agent Distribution | `InMemoryDeploymentInstanceStore`; durable: `SqliteDeploymentInstanceStore` (`sqlite_stores.py`) | Reference wiring uses **in-memory** per composition state; durable bundle exists for SQLite AP stores | SQLite when `SqliteAgentDistributionStoreBundle` used (`wire_durable_reference_production_lifecycle_services`) |
| Materialization topology enum | `MaterializationTopology` (`runtime_revision.py`) | Agent Distribution | `OCI_IMAGE`, `VENV_BUNDLE`, `SANDBOX_SIDECAR` | Revision admission + projection descriptor | On `RuntimeRevision` |
| In-tree materialization | `RuntimeMaterializationAdapter` (`materialization_adapters.py`) | Agent Distribution build coordinator | OCI adapter (Docker); VENV = **explicit unsupported** | Control-plane materialize flows (not production GCA path) | `materialization_store` on `ProductionAgentPlatformRuntime.stores` |
| Production artifact semantics | `resolve_production_artifact_root`, `assemble_production_registry_projection_input_bundle` (`production_registry_projection_input_bundle.py`) | Applications shared (AC-3) | **VENV_BUNDLE only**; `file://` / `test://` schemes; forbids `reference://` | `ReferenceProductionLifecycleLauncher.deploy_and_activate` (after external bundle assembly) | Immutable directory + digest on revision / projection input |
| Runtime factory for bundle | `build_production_runtime_agent_factory_resolver` (`venv_bundle_runtime_agent_factory_resolver.py`) | Applications shared | **VENV_BUNDLE: implemented**; OCI/sidecar: **deferred** (`PRODUCTION_RUNTIME_FACTORY_TOPOLOGY_STATUS`) | Projection input bundle assembly | Import-scope cache keyed by artifact digest (process memory; not deployment truth) |

### B. Activation

| Mechanism | Contract | Semantic owner | Implementation | Caller | Persistence |
|-----------|----------|----------------|----------------|--------|-------------|
| Activation orchestration | `ActivationService` (`activation.py`) | **Agent Distribution** | Class in `activation.py` | `ReferenceProductionLifecycleLauncher.deploy_and_activate` | `ApplicationEnvironmentActivationStore`, `ApplicationEnvironmentServingStore`, `DeploymentInstanceStore`, `RuntimeRevisionStore` |
| Prepare → ready | `prepare_candidate` | Agent Distribution | Calls `deployment_adapter.prepare` + `check_readiness`; persists `PREPARING` → `READY` | Launcher after governance admission + `mark_validated` | `Sqlite*` or in-memory stores |
| Traffic commit | `commit_activation` | Agent Distribution | `atomic_commit_activation` + projection `prepare_projection`; begins drain on prior unit | Launcher after `authorize_activation` | Serving pointer + revision state |
| Rollback / drain | `rollback_*`, `complete_drain` | Agent Distribution | Adapter drain/stop/resume_serving | Admin / rollback callers (tests, future ops) | Instance record transitions |
| Revision admission | `RuntimeRevisionService` | Agent Distribution | `runtime_revision_service.py` | Launcher: `persist_candidate_revision`, `mark_validated` | `RuntimeRevisionStore` |
| Projection coordination | `RuntimeServingProjectionCoordinator` | AP-10 boundary | `ApplicationRegistryProjectionCoordinator` (production); `FakeRuntimeServingProjectionCoordinator` (tests) | `ActivationService.commit_activation` | `RegistryProjectionInputStore`, `registry_projection_store` |
| Production projection inputs | `RegistryProjectionInputBundle` | Applications shared | `production_registry_projection_input_bundle.py` + authority resolver | Launcher registers input then commits | In reference path: `InMemoryRegistryProjectionInputStore` (qualification debt for durable inputs) |

### C. Governance

| Mechanism | Contract | Semantic owner | Implementation | Caller | Evidence |
|-----------|----------|----------------|----------------|--------|----------|
| Control-plane mutation auth | `ControlPlaneMutationAuthorizationBoundary` | **Governance** (runtime) | `build_reference_production_control_plane_mutation_boundary` via `reference_production_governance_wiring.py` | `authorize_scoped_control_plane_mutation` in launcher (`control_plane_governance.py`) | `ControlPlaneMutationAuthorizationEvidence` on denial |
| Tenant binding | `ApplicationEnvironmentTenantResolver` | Agent Distribution + Governance | `StaticApplicationEnvironmentTenantResolver` in reference wiring | Admission + activation authorization | Environment → tenant map |
| Principal | `RequestIdentity` | Contracts | Service principal from host profile | `deploy_and_activate(..., principal=...)` | Not deployment persistence |
| Reference-only policy | Harness mutation boundary builders | Reference / lab | `harness_control_plane_policy_wiring.py` | Reference production only | **Not** production-qualified without operator-approved production policy profile |

### D. Runtime / process

| Mechanism | Contract | Semantic owner | Implementation | Caller | Persistence |
|-----------|----------|----------------|----------------|--------|-------------|
| Composition root | `ProductionProcessComposition` (`production_process_composition.py`) | Applications shared | Dataclass: AP runtime + optional invocation + continuation stores | `open_reference_production_process_lifecycle`, GCA factories | Process-lifetime fields on composition |
| Process lifecycle opener | `open_reference_production_process_lifecycle` (`reference_production_process_lifecycle.py`) | Applications shared (reference name; **sanctioned production composition root until a renamed production opener exists**) | Opens continuation SQLite + optional invocation SQLite; builds launcher | **R6** tests, qualification harnesses | Files: continuation state file, invocation DB |
| Deploy / activate orchestration | `ReferenceProductionLifecycleLauncher` | Applications shared (**orchestration glue** — not semantic owner of activation) | `reference_production_lifecycle.py` | External operator / launcher CLI / tests before `create_governed_contractor_process_app` | AP stores on composition |
| GCA strict ASGI | `create_governed_contractor_process_app` (`governed_contractor_application/host/main.py`) | GCA application | `create_governed_contractor_backend_app` + `bootstrap_production_registry_projection` | After activation | Consumes **activated** serving revision from stores |
| GCA direct run | `run()` / module `app` | GCA | `StrictProductionAsgiPlaceholder` / `SystemExit(1)` | **Forbidden** unactivated | N/A |
| Execution continuation | `ExecutionContinuationStateStore` | Execution contracts | `open_production_execution_continuation_state_store` (ADR-02) | Factory via composition | Durable file |
| Provider invocation | `ProviderInvocationStore` | Execution / enterprise reliability | `SQLiteProviderInvocationStore` (PIS-ADR-03) | Composition + strict factory | SQLite |

### E. Supported topologies (evidence-bound)

| Topology | Artifact materialization (in-tree) | Production registry projection | Runtime agent factory | Deployment adapter expectation |
|----------|-----------------------------------|------------------------------|----------------------|--------------------------------|
| **VENV_BUNDLE** | **Unsupported** in `default_materialization_adapters()` | **Required** (`production_registry_projection_input_bundle.py`) | **Implemented** | **First production target** — pre-built directory + digest |
| **OCI_IMAGE** | Docker build adapter present | **Rejected** at projection assembly | **Deferred** | Not first production topology |
| **SANDBOX_SIDECAR** | No default adapter | **Rejected** at projection assembly | **Deferred** | Not first production topology |

---

## 4. Decision D1 — First supported production topology

**Decision (proposed):** **`MaterializationTopology.VENV_BUNDLE`** with **externally materialized immutable artifact directory** (`file://` locator), **co-located single-process serving unit** in the **same OS process** that will host strict GCA after activation.

**Evidence:**

| Criterion | VENV_BUNDLE (chosen) | OCI_IMAGE | SANDBOX_SIDECAR |
|-----------|----------------------|-----------|-----------------|
| Production registry projection | **Enforced** (`_validate_revision_roster_lock_authority` requires VENV_BUNDLE) | Fails assembly | Fails assembly |
| Runtime factory resolver | `implemented` | `deferred` | `deferred` |
| Artifact already materialized | **Yes** — operator/CI produces directory; locator in materialization store | Would require image registry + runtime — not projection-qualified | Not evidenced |
| Physical serving unit | **In-process slot** bound to artifact digest + import scope warmup (`build_production_runtime_agent_factory_resolver`) | Container/scheduling — no production resolver | Sidecar provider — deferred |
| Readiness without live cloud | **Yes** — filesystem + resolver + revision digest match | Needs Docker/registry infra | Needs sandbox provider |
| Deployment state across restart | **Durable** via `SqliteDeploymentInstanceStore` + serving stores (when wired); adapter state must not be sole truth | Same contract, more moving parts | Not evidenced |
| In-tree materialization | **Explicitly unsupported** — does **not** block first topology if artifact is pre-built | Supported for lab builds | Unsupported |

**Rejected for first production:**

- **OCI_IMAGE** — production projection path forbids it; factory deferred.
- **SANDBOX_SIDECAR** — deferred factory; no closed-world production caller.
- **Subprocess-per-revision VENV worker** — valid future extension of `RuntimeDeploymentAdapter`, but **not** narrowest: no canonical subprocess launcher contract exists; would duplicate process ownership with `open_reference_production_process_lifecycle` without ADR extension.
- **FakeInMemoryRuntimeDeploymentAdapter** — explicit non-production.

**Operator note:** External CI must produce digest-stable VENV bundle directories; in-tree `UnsupportedVenvBundleMaterializationAdapter` remains correct until a separate materialization qualification slice is opened.

---

## 5. Decision D2 — Canonical semantic ownership & composition root

**Semantic owners (unchanged):**

| Concern | Owner |
|---------|-------|
| Deployment orchestration semantics | **Agent Distribution** — `ActivationService` + `RuntimeDeploymentAdapter` port |
| Deployment instance facts | **Agent Distribution** — `DeploymentInstanceStore` |
| Activation / serving pointer | **Agent Distribution** — activation + serving stores |
| Registry projection truth | **AP-10** — coordinator + inputs from canonical revision authority |
| Control-plane mutations | **Governance** — `ControlPlaneMutationAuthorizationBoundary` |
| Execution / continuation / invocation | **Execution contracts** — stores on `ProductionProcessComposition` |
| Application HTTP host | **GCA** — consumer of activated composition |

**Exactly one sanctioned production composition root (proposed):**

```text
open_reference_production_process_lifecycle(...)   # rename optional later; behavior = production opener
  → ProductionProcessComposition (single AP store bundle + durable continuation + invocation)
  → wire_governed_reference_production_launcher(...)
       → ReferenceProductionLifecycleLauncher
            → deploy_and_activate (Governance + ActivationService + PRODUCTION deployment adapter)
  → create_governed_contractor_process_app(process_composition=composition)
```

**Before (current HEAD):** same graph with **`FakeInMemoryRuntimeDeploymentAdapter`** and reference governance boundary.

**After (target):** same graph with **`VenvBundleColocatedRuntimeDeploymentAdapter`** (name illustrative) injected at lifecycle wiring; **no** second activation authority in GCA.

`applications/governed_contractor_application/host/main.py` **must remain** a consumer — **not** deployment root. **`run()`** must **not** become deploy authority; external launcher opens lifecycle, activates, then starts ASGI (uvicorn/hypercorn) with `create_app(process_composition=...)`.

---

## 6. Decision D3 — Provider replaceability & injection

**Decision (proposed):**

- Keep **`RuntimeDeploymentAdapter`** provider-neutral; **no** vendor conditionals in `ActivationService` or GCA.
- Inject concrete adapter at **lifecycle service wiring** — extend `wire_reference_production_lifecycle_services` / `wire_durable_reference_production_lifecycle_services` (or production-named successors) with a **required** `deployment_adapter: RuntimeDeploymentAdapter` parameter; **forbid default** to fake in production mode.
- Configure adapter with **typed** config (e.g. artifact root policy, readiness timeouts) via constructor — **not** `dict`/`Any` registries.
- **No new global deployment registry** — composition parameter + existing `SqliteAgentDistributionStoreBundle` suffice.

Production qualification must prove **structural replaceability**: second test double implementing the same Protocol without changing `ActivationService`.

---

## 7. Decision D4 — Physical readiness vs execution readiness

| Layer | What can be proven | Owner of truth | Must not substitute |
|-------|-------------------|----------------|---------------------|
| **Physical deployment** | Artifact directory exists; digest matches revision; import-scope / factory resolver succeeds; optional policy cert refs on revision | **`RuntimeDeploymentAdapter.check_readiness`** → evidence ref stored on `DeploymentInstanceRecord` | Manual registry seeding; diagnostics HTTP 200 alone |
| **Activation gate** | Validated revision; READY instance; governance ALLOW | **`ActivationService.commit_activation`** | Skipping `prepare_candidate` |
| **Serving pointer** | Atomic pointer CAS | **`ApplicationEnvironmentActivationStore`** | In-memory test doubles on production path |
| **Registry projection** | Inputs match activated revision + digest | **`ApplicationRegistryProjectionCoordinator`** + `production_registry_projection_input_bundle` | Manifest defaults at host startup |
| **Application readiness** | Harness host built; strict admission satisfied | **GCA factory** (`build_harness_host_runtime`) after composition bound | Unactivated `create_app()` |
| **Execution admission** | Continuation durable; tool boundary materialized | **Execution** validation in factory | Lab profile in strict mode |

Deployment adapter **does not** prove business-level health endpoints; GCA readiness probes remain **downstream** and **non-authoritative** for deployment instance state.

---

## 8. Decision D5 — Process ownership

| Question | Decision (proposed) |
|----------|---------------------|
| Who owns OS process lifetime? | **External launcher** using `open_reference_production_process_lifecycle` context manager |
| Who owns deploy/activate? | **`ReferenceProductionLifecycleLauncher`** on the same composition (before ASGI serve) |
| Who owns ASGI app? | **GCA** `create_governed_contractor_process_app` after activation |
| Who owns continuation/invocation close? | **`ReferenceProductionProcessLifecycle.close_process_resources`** (R6) — **not** FastAPI shutdown alone |
| Circular ownership | **Forbidden:** GCA `run()` cannot deploy itself. Deploy/activate runs in launcher process **before** binding serving traffic to the strict host factory. Co-located adapter manages **in-process serving slot**, not a second GCA instance. |

Implementing **`main.run()`** as mandatory production entry: **rejected**. Production entry = **launcher + uvicorn factory** with explicit composition.

---

## 9. Decision D6 — Failure and recovery matrix

| Failure | Truth owner | Action owner | Serving permitted? |
|---------|-------------|--------------|-------------------|
| Invalid/missing artifact path | Adapter + projection assembly | Adapter `prepare` / assembly fail-closed | **No** |
| Missing digest on revision | `RuntimeRevision` store | `prepare_candidate` raises | **No** |
| Deployment prepare failure | `DeploymentInstanceStore` → FAILED | `ActivationService` persists FAILED | **No** |
| Readiness failure | Instance FAILED + evidence ref | Adapter + activation | **No** |
| Governance denial | Governance boundary | Launcher raises `ReferenceProductionLifecycleGovernanceBlockedError` | **No** |
| Activation conflict (stale pointer) | Serving store | `atomic_commit_activation` | **No** (prior pointer unchanged) |
| Projection prepare failure | Coordinator | `commit_activation` aborts before pointer commit | **No** |
| Commit failure after projection prepare | Serving store unchanged | Operator recovery / rollback slice | **No** |
| Startup crash before ASGI listen | Process lifecycle | `__exit__` closes continuation/invocation (R6); deployment instance may remain READY — **restart must reconcile** | **No** |
| Restart after activation | Serving + revision stores (durable) | Launcher skips re-deploy if READY instance valid; else re-prepare | **Only if** pointer + instance reconciled |
| Stale serving pointer | Serving store CAS | Activation conflict — operator refresh | **No** on failed commit |
| Wrong tenant/revision/env | Revision + governance scope | Admission/activation auth deny | **No** |
| Shutdown / drain timeout | Instance DRAINING/FAILED | `complete_drain` + adapter `stop` per policy | **Draining** — no new traffic on old unit |
| Drain cleanup failure | Instance evidence refs | Operator; adapter `stop` retry semantics | **Policy-bound** |
| Unresolved provider invocation | `ProviderInvocationStore` (PIS) | Execution recovery policies — **separate from deployment** | Deployment does not auto-serve |

**No exactly-once** claim for adapter `prepare` across crashes — idempotency via `DeploymentInstanceRecord` states and digest-keyed `serving_unit_ref` discipline.

---

## 10. End-to-end sequence (target production)

```text
1. External materialization → immutable VENV bundle directory + digest (out of band)
2. Control plane: admit revision (Governance) → VALIDATED in RuntimeRevisionStore
3. Register materialization record (artifact_locator) in materialization_store
4. assemble_production_registry_projection_input_bundle / register projection input
5. Launcher.deploy_and_activate:
     a. Governance: admission + activation mutations
     b. ActivationService.prepare_candidate → adapter prepare + check_readiness → READY instance
     c. ActivationService.commit_activation → projection prepare + serving pointer CAS
     d. begin_drain prior unit (if any)
5. open_reference_production_process_lifecycle (if not already open) with durable stores
6. create_governed_contractor_process_app(process_composition)
7. ASGI server serve (external)
8. Shutdown: drain/complete_drain (ops) → lifecycle.close_process_resources → adapter stop
```

---

## 11. Mandatory architecture invariants (disposition)

| Invariant | Satisfied by proposed decisions? |
|-----------|----------------------------------|
| Contracts over implementations | **Yes** — Protocol port retained |
| Layer boundaries | **Yes** — no GCA → adapter import required |
| Exactly-one semantic owner per concern | **Yes** — table §5 |
| Exactly-one composition root | **Yes** — lifecycle opener |
| Strong typing | **Yes** — Pydantic + Protocol; adapter config must stay typed |
| Pluginability / replaceability | **Yes** — injection at wiring |
| No fake on production path | **Pending R7-D1/D2** — **IN-SCOPE BLOCKER** today |
| No Governance bypass | **Yes** — launcher path preserved |
| Fail-closed activation | **Yes** — existing `ActivationService` |
| Diagnostics non-authoritative | **Yes** — §7 |

**Contract changes:** **None required** for first topology if co-located VENV adapter implements existing `RuntimeDeploymentAdapter`. Subprocess/OCI adapters later **must not** widen `ActivationService` semantics.

If co-located adapter needs `materialization_output` on `prepare`: optional parameter already exists — use typed `MaterializationOutput` from store, not ad-hoc dicts.

---

## 12. Minimal implementation slices (dependency order)

| Child | Owner | Scope | STOP condition |
|-------|-------|-------|----------------|
| **PROD-Q-R1-R7-D1** | Agent Distribution + Applications wiring | One production `RuntimeDeploymentAdapter` for VENV_BUNDLE co-located: `prepare`/`check_readiness`/`drain`/`stop` with real artifact + digest checks | Cannot prove readiness without filesystem test fixture |
| **PROD-Q-R1-R7-D2** | Applications shared lifecycle | Inject adapter + durable `SqliteDeploymentInstanceStore` bundle; remove fake default on production code path | Duplicate activation wiring in GCA |
| **PROD-Q-R1-R7-D3** | Applications + Agent Distribution | Production projection input store durability + post-restart reconciliation of serving pointer vs READY instance | Manual in-memory projection seeding required |
| **PROD-Q-R1-R7-D4** | GCA + lifecycle | Documented launcher → uvicorn factory; optional thin CLI — **no** `run()` deploy logic | GCA required to deploy itself |
| **PROD-Q-R1-R7-Q** | Qualification | End-to-end + adversarial proof matrix §13 | Any fake adapter reachable from strict production graph |

Consolidation: **D1+D2** inseparable for first production path — may merge if single PR boundary agreed.

---

## 13. Future proof matrix (implementation — not executed in this ADR)

Qualification must include **behavioral** tests (indicative paths):

- `tests/unit/agent_distribution/` — adapter + activation integration with real adapter (not class-name asserts)
- `tests/unit/applications/governed_contractor/` — lifecycle + strict factory fail-closed negatives
- `tests/qualification/prod_q/` — production reachability: **fake deployment adapter count = 0** on strict GCA graph

| Gate | Intent |
|------|--------|
| Physical prepare/readiness | Real artifact dir wrong digest → FAILED instance, no commit |
| Provider-neutral injection | Alternate stub adapter, same `ActivationService` |
| Authorized activation | Governance ALLOW → pointer bump |
| Denied / stale / cross-tenant | Governance DENY; CAS conflict |
| Failed readiness → zero serving | No `atomic_commit_activation` success |
| Activated revision → registry projection | Resolver sees activated revision id + digest |
| No manual test seeding | Forbidden `reference://` / harness registry shortcuts on strict path |
| Startup failure cleanup | R6-style lease release + deployment FAILED semantics |
| Restart / stale pointer | Durable stores + reconciliation tests |
| Drain/shutdown | `complete_drain` + `stop` evidence |
| Missing credentials/config | Fail-closed at governance or adapter |
| Unsupported topology | OCI revision → projection assembly error |
| No fake on production path | Static/AST reachability from `create_governed_contractor_process_app` strict graph |
| No silent fallback | Strict mode rejects missing adapter |
| Unresolved invocation | PIS store ambiguity — no duplicate external deploy effect |
| Replaceability | Second adapter implementation |
| Exact-SHA evidence | Qualification docs pin commit |

---

## 14. Applicable FRZ requirements (evidence contribution only — no PASS)

| ID | Relevance |
|----|-----------|
| **FRZ-PRD-04** | Startup lifecycle — blocked until D1–D4 + R1 admission gaps closed |
| **FRZ-PRD-05** | Shutdown/drain — adapter `stop` + R6 lifecycle close |
| **FRZ-PRD-01** | Separates reference fake adapter from production adapter |
| **FRZ-PRD-03** | Unsupported topology / missing adapter fail-closed |
| **FRZ-PRD-07** | Strict vs lab — fake adapter lab-only |
| **FRZ-PRD-08** | No production bypass of activation |
| **FRZ-TYP-*** | Typed Protocol/Pydantic boundaries |
| **FRZ-PLG-*** / **FRZ-RPL-*** | Adapter injection |
| **FRZ-CTR-*** | Deployment/activation contracts |
| **FRZ-TEN-*** | Governance tenant resolver on activation |
| **FRZ-SEC-04** | No unactivated GCA serve |

---

## 15. Enterprise audit matrix

| Category | Decision / finding |
|----------|-------------------|
| Boundaries | Tier-0/1 AD semantics stay in `intergrax/agent_distribution`; GCA consumes only |
| Communication | Launcher orchestrates; no hidden deploy in ASGI factory |
| Composition | Single `ProductionProcessComposition` root |
| Ownership | Tables §3, §5 — one owner per row |
| Contracts | `RuntimeDeploymentAdapter` unchanged |
| Abstraction | Adapter port isolates physical topology |
| Strong typing | Pydantic records; proposed adapter config typed |
| Pluginability | Inject adapter at wiring |
| Replaceability | Protocol + dual-adapter qualification |
| Bypass resistance | `main.run()` fail-closed; activation required |
| Governance | Admission + activation mutations preserved |
| Execution authority | Continuation/invocation unchanged |
| Persistence | Sqlite deployment store available; reference still in-memory — **debt** |
| Evidence / traceability | Instance records + readiness refs |
| Observability | Diagnostics non-authoritative |
| Fail-closed | Unactivated paths exit / error |
| Regression protection | §13 matrix |

---

## 16. Unresolved findings (classification)

| Finding | Classification | Child / action |
|---------|----------------|----------------|
| No production `RuntimeDeploymentAdapter` | **IN-SCOPE BLOCKER** | **R7-D1** |
| Fake adapter on lifecycle wiring | **IN-SCOPE BLOCKER** | **R7-D2** |
| In-memory deployment/projection input stores on reference path | **TRACKED FREEZE DEBT** | **R7-D3** |
| Full strict tool admission on all materialization paths | **IN-SCOPE BLOCKER** (existing **PROD-Q-R1**) | R1-R1 / follow-on |
| `UnsupportedVenvBundleMaterializationAdapter` vs external bundles | **TRACKED FREEZE DEBT** | Separate materialization qualification (not R7-D1) |
| Production governance policy vs reference harness boundary | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Operator-approved production mutation policy profile |
| GCA `run()` not production launcher | **TRACKED FREEZE DEBT** (documented) | **R7-D4** |
| **FRZ-PRD-04/05** global OPEN | **TRACKED FREEZE DEBT** | Post-implementation qualification |

---

## 17. Unresolved questions (operator ratification)

1. Accept **co-located in-process** VENV serving unit as first production topology (vs mandating subprocess worker in D1)?
2. Mandate **`SqliteAgentDistributionStoreBundle`** for all production deployment/serving persistence (vs hybrid in-memory projection inputs)?
3. Production **control-plane mutation policy** profile: extend reference boundary or new sanctioned production boundary module?
4. External **materialization pipeline** ownership (CI digest attestation) — platform vs application team?

---

## 18. Exit recommendation

**READY FOR ARCHITECTURE DECISION** — operator ratification of **D1–D6** and composition root §5. Upon ratification, implement **PROD-Q-R1-R7-D1** then **R7-D2**; keep **PROD-Q-R1** **BLOCKED** until independent GitHub audit of implementation slices and §13 proofs.

**Maximum outcome of this commit:** architecture proposal only — **no** production adapter, **no** FRZ PASS, **no** roadmap CLOSED.

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

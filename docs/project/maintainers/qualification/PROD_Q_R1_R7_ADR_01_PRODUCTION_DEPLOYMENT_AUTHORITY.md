# PROD-Q-R1-R7-ADR-01 — Production Deployment Authority, Topology & Process Ownership

| Field | Value |
|-------|-------|
| **Status** | **READY FOR ARCHITECTURE DECISION** (proposed — **not ratified**) |
| **Revision** | **R3** — boundary corrective (**B06–B07**); **R2** partially accepted (**B04/B05** direction); **R1 rejected** |
| **Date** | 2026-10-13 |
| **Baseline SHA** | `bc1f6926babf0b5fa499b1deebcda48c93cd4258` (prior ADR **R2** @ `480efbbb2ce18c17703c8fa06decc6d5821856fa`) |
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
| **Evidence lock** | `e4e82e356fbdd0a2a6dac8de3dfb17567e299f44` |
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

### 2.1 Independent-audit blockers (disposition)

| ID | Root cause (prior draft) | Corrective disposition (R1) |
|----|--------------------------|-----------------------------|
| **R7-ADR-B01 — Readiness conflation** | Prior text tied “physical readiness” to **artifact directory + digest + import-scope warmup** without a distinct **serving-unit** identity; `FakeInMemoryRuntimeDeploymentAdapter.check_readiness` can succeed from a dict key alone — indistinguishable from “artifact exists”. | **Decision A** defines `PreparedDeployment.serving_unit_ref` as an **adapter-issued opaque unit handle**, not `artifact_locator`, not filesystem path, not digest. **Artifact immutability** is proven upstream (`RuntimeRevision.materialization_artifact_digest`, `resolve_production_artifact_root`); **unit readiness** is proven only by `prepare` + `check_readiness` on that handle. Durable `DeploymentInstanceState.READY` **never** implies traffic eligibility. |
| **R7-ADR-B02 — Reference authority promotion** | Prior draft named `open_reference_production_process_lifecycle` / `ReferenceProductionLifecycleLauncher` as the **sanctioned production composition root** without classifying every non-adapter dependency; implied injection of a production adapter **qualifies** the reference graph. | **Decision B** separates **composition authority** (`wire_production_process_lifecycle` — proposed name) from **reference lab wiring**. Reference modules stay **REFERENCE-ONLY** or **REUSABLE CONTRACT / NOT YET QUALIFIED** until each dependency is explicitly sourced. Injecting `RuntimeDeploymentAdapter` **does not** production-qualify `ReferenceProductionLifecycleLauncher` or reference governance wiring. |
| **R7-ADR-B03 — In-process lifecycle ambiguity** | “Co-located single-process serving unit” was undefined for prepare/readiness vs **SERVING** vs crash/restart; risk of treating process boot as deployment truth. | **Decision A (A1)** + **Decision C** bind lifecycle to `DeploymentInstanceState` transitions (`PREPARING` → `READY` → `SERVING` via activation store commit), adapter drain/stop, and **post-crash reconciliation** (durable READY ≠ live unit). Process opener owns continuation/invocation only — **not** serving-unit identity. |
| **R7-ADR-B04 — No proven serving-unit handoff** | R1 assumed an in-process adapter “slot” prepared before activation is the unit that serves traffic after `create_governed_contractor_process_app`, but **no typed contract** transports `serving_unit_ref` (or equivalent runtime binding) from adapter → composition → strict host factory. Opaque string handle alone does not establish continuity. | **Decision A (R2/R3):** ratify **A1 co-located** only with mandatory **`ServingRuntimeHandoff`** on `ProductionProcessComposition` (identity + Applications-owned `RuntimeAgentFactoryResolver`) established by composition after **`attest_serving_unit`** (§4.5). **Reject** “prepare dict + independent `build_production_runtime_agent_factory_resolver`” without attestation-linked handoff. |
| **R7-ADR-B05 — Durable READY reused without liveness** | `ActivationService.prepare_candidate()` returns existing `READY` when `serving_unit_ref` and `readiness_evidence_ref` are non-null **without** adapter re-attestation (`activation.py` L174–177). After process crash, SQLite can retain READY while adapter state (and in-process import scopes) are gone. | **Decision B (R2, retained R3):** semantic owner **`ActivationService`** must call adapter **`attest_serving_unit`** before any READY reuse; stale/wrong digest → FAILED or controlled re-prepare under environment lock + `record_revision` CAS (§4.6). **No** independent recovery in GCA or SQLite store. |
| **R7-ADR-B06 — Cross-layer contract inversion** | R2 proposed `RuntimeDeploymentAdapter.materialize_serving_runtime` returning Applications `RuntimeAgentFactoryResolver`, making the Agent Distribution deployment port responsible for an application runtime factory — **violates** `intergrax/` MUST NOT depend on `applications/` at semantic boundaries (`AGENTS.md`). | **R3 rejected:** remove `materialize_serving_runtime` from `RuntimeDeploymentAdapter`. Physical attestation stays topology-neutral (§4.5A). **Applications** composition binds attested unit identity → `RuntimeAgentFactoryResolver` via `ServingRuntimeHandoff` (§4.5B). VENV in-process binding uses a **composition-injected** `ColocatedServingUnitRuntimeAccessor` (Applications Protocol), implemented by the production adapter at wiring time — **not** a method on the topology-neutral port. |
| **R7-ADR-B07 — Inconsistent runtime handoff owner** | R2 simultaneously forbade GCA from calling the deployment adapter **and** directed the GCA strict factory to call `materialize_serving_runtime` on that adapter — two owners for one handoff. | **R3 resolved:** **exactly one** composition-owned handoff during `deploy_and_activate` (launcher / production lifecycle): after `attest_serving_unit`, **`establish_serving_runtime_handoff`** writes `ProductionProcessComposition.serving_runtime_handoff`. GCA **only reads** the handoff; **never** imports or invokes `RuntimeDeploymentAdapter`. |

### 2.2 Code evidence (B04 / B05 — exact HEAD)

| Blocker | Evidence (file · lines · behavior) |
|---------|-------------------------------------|
| **B04** | `applications/governed_contractor_application/host/main.py` · `create_governed_contractor_process_app` accepts only `ProductionProcessComposition` — **no** `serving_unit_ref` / binding field (`production_process_composition.py` has AP runtime + continuation/PIS only). |
| **B04** | `reference_production_lifecycle.py` · `deploy_and_activate` calls `prepare_candidate` then `commit_activation` then launcher returns; GCA host built later via `create_governed_contractor_process_app` — **no** pass-through of prepared unit handle. |
| **B04** | `factory.py` · strict path calls `build_harness_host_runtime(...)` with registry projection from stores — constructs execution surface **without** reading `DeploymentInstanceRecord.serving_unit_ref`. |
| **B04** | `venv_bundle_runtime_agent_factory_resolver.py` · `_ArtifactScopeRegistry` / `_FactoryCache` are **process-global**; readiness in a future A1 adapter could warm scopes keyed by digest while `serving_unit_ref` remains an unrelated opaque string unless explicitly bound (dual authority risk). |
| **B05** | `activation.py` · L174–177: `if existing.instance_state is READY` → `return TransitionResult(value=existing)` with **zero** `deployment_adapter` calls. |
| **B05** | `activation.py` · `_require_ready_instance` (L691–715) checks durable fields only — **no** liveness attestation before `commit_activation`. |
| **B05** | `activation.py` · `_ensure_prior_instance_ready` (L611–616) treats durable `READY`/`SERVING`/… as sufficient to skip `prepare` except drain/stop resume paths — same stale-handle risk on rollback. |

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
| Reference-only policy | Harness mutation boundary builders | Reference / lab | `harness_control_plane_policy_wiring.py` | Reference production only | **Not** production-qualified without operator-approved production policy profile (**B02**) |

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

### E. Portability matrix (closed-world — HEAD)

| Contract | Owning layer | Semantic owner | Composition owner | Primary callers | VENV_BUNDLE | OCI_IMAGE | SANDBOX_SIDECAR |
|----------|--------------|----------------|-------------------|-----------------|-------------|-----------|-----------------|
| `RuntimeDeploymentAdapter` | Tier-1 AD | Agent Distribution | Lifecycle wiring | `ActivationService` | A1 colocated | Container ops | Sidecar ops |
| `PreparedDeployment` / `DeploymentInstanceRecord` | Tier-1 AD | Agent Distribution | — | `ActivationService`, stores | Portable | Portable | Portable |
| `ActivationService` | Tier-1 AD | Agent Distribution | Launcher orchestration | `ReferenceProductionLifecycleLauncher` | Portable | Portable | Portable |
| `ServingUnitAttestation` (proposed) | Tier-1 AD | Adapter observe / Activation orchestrate | — | `ActivationService` only | In-process liveness | Container health | Sidecar health |
| `RegistryProjectionInputBundle` | Tier-3 Apps | AP-10 inputs | Launcher register | `deploy_and_activate`, coordinator | **Required** (factory_resolver) | Assembly rejected today | Assembly rejected today |
| `RuntimeAgentFactoryResolver` | Tier-3 Apps | Applications shared | `establish_serving_runtime_handoff` | AP-10 projection, GCA strict | **Implemented** | Deferred | Deferred |
| `build_production_runtime_agent_factory_resolver` | Tier-3 Apps | VENV bundle module | Bundle assembly / accessor | `production_registry_projection_input_bundle` | **Implemented** | N/A | N/A |
| `ApplicationRegistryProjectionCoordinator` | Tier-1/3 seam | AP-10 | Wired on composition | `commit_activation` | Portable contract | Deferred inputs | Deferred inputs |
| `ProductionProcessComposition` | Tier-3 Apps | Process root | `open_*_production_process_lifecycle` | GCA, launcher | Portable | Portable | Portable |
| `ServingRuntimeHandoff` (proposed) | Tier-3 Apps | Applications | `establish_serving_runtime_handoff` | GCA strict factory | Colocated accessor | Remote resolver binder | Sidecar binder |
| `create_governed_contractor_process_app` | Tier-3 GCA | GCA consumer | External launcher | Post-activate | Portable consumer | Portable consumer | Portable consumer |
| `build_harness_host_runtime` | Tier-3 Apps | Harness spine | GCA factory | Strict/lab hosts | Portable | Portable | Portable |
| `open_reference_production_process_lifecycle` | Tier-3 Apps | Reference only | Reference opener | Tests, qual | Reference | Reference | Reference |

### F. Supported topologies (evidence-bound)

| Topology | Artifact materialization (in-tree) | Production registry projection | Runtime agent factory | Deployment adapter expectation |
|----------|-----------------------------------|------------------------------|----------------------|--------------------------------|
| **VENV_BUNDLE** | **Unsupported** in `default_materialization_adapters()` | **Required** (`production_registry_projection_input_bundle.py`) | **Implemented** | **First production target** — pre-built directory + digest |
| **OCI_IMAGE** | Docker build adapter present | **Rejected** at projection assembly | **Deferred** | Not first production topology |
| **SANDBOX_SIDECAR** | No default adapter | **Rejected** at projection assembly | **Deferred** | Not first production topology |

---

## 4. Decision A — Serving-unit identity and readiness (B01, B03, B04, B05)

### 4.1 Physical meaning of `PreparedDeployment.serving_unit_ref`

`serving_unit_ref` is an **opaque, adapter-allocated handle** naming one **runtime serving unit** (one revision binding in one environment context). It **must not** be:

- the `artifact_locator` string;
- a filesystem path or digest used as a stand-in for a unit;
- a fabricated token returned without adapter-side unit state.

The handle is created in `RuntimeDeploymentAdapter.prepare` and carried on `DeploymentInstanceRecord.serving_unit_ref`. **Artifact immutability** is established separately (`RuntimeRevision.materialization_artifact_digest`, `resolve_production_artifact_root`, `PreparedDeployment.materialization_artifact_digest` equality check in `ActivationService.prepare_candidate`). **Unit readiness** is established only by `check_readiness(revision, serving_unit_ref=…)` and stored as `readiness_evidence_ref`.

**Traffic eligibility** requires, in order: validated revision → READY instance (with evidence) → Governance-permitted `commit_activation` → serving pointer CAS → instance `SERVING` (activation store) → GCA strict factory admission. **READY without SERVING** explicitly means: *physical unit may exist and be validated, but production traffic is forbidden*.

### 4.2 Topology options (legitimate; not conflated with artifact checks)

| Option | Summary | When chosen |
|--------|---------|-------------|
| **A1 — Co-located, non-serving in-process unit** | Adapter allocates an **isolated in-process slot** (typed config + internal registry) bound to digest; slot is warmed/validated without admitting ASGI traffic; composition **must** bind attested unit → resolver via `ServingRuntimeHandoff` (§4.5B). | **First production topology** for VENV_BUNDLE (§5) — **not ratified without §4.5 contract delta**. |
| **A2 — Externally managed serving unit** | Adapter delegates to another OS process or supervisor; `prepare`/`check_readiness` observe that unit’s health boundary. | **Valid future** `RuntimeDeploymentAdapter` implementation; requires explicit supervisor contract in D1 follow-on, not reference launcher subprocess ambiguity. |

**Rejected for Decision A:** treating “immutable artifact directory exists” or “factory resolver import succeeded” as `serving_unit_ref` or as sufficient `check_readiness` — that is **artifact readiness**, not **serving-unit readiness** (B01).

### 4.3 Identity flow (closed-world — HEAD `bc1f692…`)

```text
revision (VALIDATED, digest)
  → artifact_locator (materialization store)
  → deployment_adapter.prepare → PreparedDeployment.serving_unit_ref
  → check_readiness → readiness_evidence_ref
  → DeploymentInstanceStore: PREPARING → READY  [durable]
  → commit_activation → serving pointer CAS + instance SERVING  [durable]
  → bootstrap_production_registry_projection(stores)  [activated revision id]
  → establish_serving_runtime_handoff on ProductionProcessComposition  [GAP: not implemented]
  → create_governed_contractor_process_app(process_composition)  [GAP: factory ignores handoff today]
  → build_harness_host_runtime(registry_projection + composition.serving_runtime_handoff.resolver)
```

| Transition | Typed contract | Semantic owner | Caller | Authoritative evidence |
|------------|----------------|----------------|--------|------------------------|
| Revision admission | `RuntimeRevision`, `RuntimeRevisionState` | Agent Distribution | Launcher / `RuntimeRevisionService` | Validated row + digests |
| Artifact immutability | `materialization_artifact_digest`, `resolve_production_artifact_root` | Agent Distribution + AC-3 bundle | Projection assembly | Digest match on revision |
| Unit allocation | `PreparedDeployment` | Agent Distribution (port) | `ActivationService.prepare_candidate` → `RuntimeDeploymentAdapter.prepare` | `serving_unit_ref` issued by adapter |
| Unit readiness | `check_readiness` → `str` evidence ref | Adapter (physical) / Activation (orchestration) | `ActivationService` | `readiness_evidence_ref` on instance record |
| Traffic gate | `atomic_commit_activation` | Agent Distribution | `ActivationService.commit_activation` | Serving store pointer + revision ACTIVE |
| Registry serve path | `RegistryProjection` / resolver | AP-10 + GCA consumer | `bootstrap_production_registry_projection` | Activated `runtime_revision_id` |
| **Execution surface (B04)** | *missing* `ServingRuntimeHandoff` | *unowned today* | GCA `build_harness_host_runtime` | **None** linking attested unit to resolver instance |

### 4.4 Architecture decision A — topology (A1 vs A2)

| Option | Verdict (R2) | Rationale on current code |
|--------|--------------|---------------------------|
| **A1 — Co-located in-process** | **Selected for first production VENV_BUNDLE** — **conditional** on typed handoff (§4.5) | Matches single launcher process + `open_reference_production_process_lifecycle`; production factory resolver already in-process (`venv_bundle_runtime_agent_factory_resolver.py`). **Insufficient alone:** opaque `serving_unit_ref` without binding to host factory. |
| **A2 — External supervised runtime** | **Deferred** (valid future adapter) | No canonical supervisor/route contract in repo; would require new process handle + traffic router + health boundary — **out of narrow R7-D1** unless operator expands scope. |

**Rejected alternatives:** token-only `serving_unit_ref`; global `_ArtifactScopeRegistry` as deployment authority; independent `build_harness_host_runtime` as proof of same unit without `ServingRuntimeHandoff`; **`materialize_serving_runtime` on `RuntimeDeploymentAdapter`** (B06 — cross-layer inversion).

**Mandatory acceptance (R3):** attested unit identity and activated execution share one **`ServingRuntimeHandoff`** on `ProductionProcessComposition`, established **once** by Applications composition after activation-path attestation — GCA **reads only**.

### 4.5 Typed contract delta (ratification required — not implemented in this ADR)

Two **separate** boundaries (A deployment liveness · B runtime handoff). No Applications types on Agent Distribution semantic ports.

#### 4.5A — Deployment state and liveness (topology-neutral)

**Semantic owner:** Agent Distribution (`deployment.py`, `activation.py`). **Port:** `RuntimeDeploymentAdapter` only.

| Addition | Module | Purpose |
|----------|--------|---------|
| `ServingUnitAttestationRequest` (frozen Pydantic) | `deployment.py` | Input to attestation |
| `ServingUnitAttestation` (frozen Pydantic) | `deployment.py` | Output evidence; **does not** authorize traffic |
| `RuntimeDeploymentAdapter.attest_serving_unit` | `deployment.py` | Liveness + identity binding |
| `ServingUnitAttestationError` | `deployment.py` | Fail-closed attestation failures |

**`ServingUnitAttestationRequest` fields (required):**

- `serving_unit_ref: str` — opaque adapter-issued handle (same as on `DeploymentInstanceRecord`);
- `runtime_revision_id: str` — revision identity under test;
- `materialization_artifact_digest: str` — must match `RuntimeRevision.materialization_artifact_digest`;
- `artifact_locator: str` — canonical locator from materialization store (not a substitute for `serving_unit_ref`).

**`ServingUnitAttestation` fields (required on success):**

- `serving_unit_ref`, `runtime_revision_id`, `materialization_artifact_digest` — **bound triple** (must match request);
- `physical_unit_live: bool` — adapter observes the unit responsive at attestation time;
- `evidence_ref: str` — opaque provenance handle (adapter-internal or structured ref string);
- `attested_at: datetime` — UTC timestamp of observation.

**Semantics:** attestation proves **physical unit liveness + artifact/revision binding** for one handle. It **does not** grant execution admission, serving pointer authority, or traffic. **`ActivationService`** is the sole consumer for READY reuse, `commit_activation` pre-checks, rollback `resume_serving`, and post-restart reconciliation — **not** GCA.

**`attest_serving_unit` signature (proposed):**

```python
def attest_serving_unit(
    self,
    revision: RuntimeRevision,
    *,
    request: ServingUnitAttestationRequest,
) -> ServingUnitAttestation:
    """Observe unit liveness and digest binding; pure read + evidence — no traffic."""
```

**Errors:** `ServingUnitAttestationError` when handle unknown, digest mismatch, revision scope mismatch, or unit not live. **ActivationService** maps to `FAILED` instance, controlled re-prepare, or `RuntimeActivationError` per §4.6 — never silent READY reuse.

**Failure impact:**

| Failure context | READY reuse | SERVING reconciliation | Rollback / resume | Recovery |
|-----------------|-------------|------------------------|-------------------|----------|
| Attest fail (stale unit) | Deny early return; re-prepare or FAILED | Block traffic until launcher reconciles | `resume_serving` → re-prepare prior unit | Operator + `ActivationService`; **no** GCA recovery |
| Wrong digest / revision | FAILED | No pointer widen | Deny resume | New prepare cycle |
| Concurrent attest + prepare | Env lock serializes | — | — | Single winner under `record_revision` CAS |
| Activation commit fail after attest | READY may remain | Pointer unchanged | — | No traffic |
| GCA startup fail after commit | Durable SERVING | Process exit | — | Restart attest path |

**Portability:** `attest_serving_unit` is **portable** across `VENV_BUNDLE`, `OCI_IMAGE`, `SANDBOX_SIDECAR` — each adapter interprets “live” for its topology (in-process slot, container health, sidecar RPC). No `RuntimeAgentFactoryResolver` in this boundary.

#### 4.5B — Runtime handoff to GCA (Applications-owned)

**Semantic owner:** Applications shared — `ServingRuntimeHandoff`. **Composition owner:** `establish_serving_runtime_handoff` in production lifecycle module (successor to reference launcher orchestration). **Consumer:** GCA strict factory (`factory.py` / `build_harness_host_runtime` wiring).

**Preserved ownership:** `RuntimeAgentFactoryResolver` remains defined in `intergrax/applications/_shared/runtime_agent_factory_resolver.py`. **No** move to Agent Distribution.

| Addition | Module | Purpose |
|----------|--------|---------|
| `ServingUnitIdentity` (frozen Pydantic) | `serving_runtime_handoff.py` | Attested identity triple + topology enum |
| `ServingRuntimeHandoff` (frozen dataclass) | `serving_runtime_handoff.py` | `identity` + `runtime_agent_factory_resolver` |
| `establish_serving_runtime_handoff(...)` | production lifecycle | **Single** write after attest + prepare success |
| `ColocatedServingUnitRuntimeAccessor` (Protocol) | `serving_runtime_handoff.py` | **VENV A1 only** — maps `serving_unit_ref` → resolver at composition boundary |
| `ProductionProcessComposition.serving_runtime_handoff` | `production_process_composition.py` | Process-lifetime carrier to GCA |

**Chain (mandatory):** `ServingUnitAttestation` (live + digest) + activated `DeploymentInstanceRecord` + canonical `RuntimeRevision` → `ServingUnitIdentity` → **same** `RuntimeAgentFactoryResolver` instance used for AP-10 projection materialization for that activation → GCA strict host.

**Existing carrier (HEAD):** `RegistryProjectionInputBundle.factory_resolver` already holds a `RuntimeAgentFactoryResolver` for projection (`registry_projection.py` L123–134). **R3 rule:** handoff must reference the **exact resolver instance** registered with projection inputs for the activated revision, or a resolver provably equivalent per attestation-linked binder — **no** second `build_production_runtime_agent_factory_resolver` call on the strict path without identity proof.

**`establish_serving_runtime_handoff` signature (proposed):**

```python
def establish_serving_runtime_handoff(
    *,
    composition: ProductionProcessComposition,
    attestation: ServingUnitAttestation,
    deployment_instance: DeploymentInstanceRecord,
    projection_input: RegistryProjectionInputBundle,
    colocated_accessor: ColocatedServingUnitRuntimeAccessor | None = None,
) -> ServingRuntimeHandoff:
    """Composition mutation: bind attested unit identity to resolver for this process."""
```

**VENV_BUNDLE (A1):** `colocated_accessor` is **required**. Production wiring passes the concrete adapter (or dedicated accessor object) **only** into lifecycle wiring — `intergrax/agent_distribution/deployment.py` Protocol **unchanged** beyond `attest_serving_unit`. Accessor method:

```python
def runtime_agent_factory_resolver_for_unit(
    self,
    *,
    serving_unit_ref: str,
    revision: RuntimeRevision,
) -> RuntimeAgentFactoryResolver:
    """Return resolver for warmed colocated slot; resource lifetime = process."""
```

**OCI_IMAGE / SANDBOX_SIDECAR (future):** `colocated_accessor` is `None`; `establish_serving_runtime_handoff` uses a topology-specific **Applications** binder (remote factory / RPC stub resolver) that still satisfies `RuntimeAgentFactoryResolver` without in-process Python import — **no** adapter return type leak.

**GCA consumption (D1b):** `create_governed_contractor_backend_app` / strict path validates `composition.serving_runtime_handoff.identity` matches `bootstrap_production_registry_projection` activated revision + digest, then passes `handoff.runtime_agent_factory_resolver` into registry projection / harness wiring — **forbidden:** GCA imports `RuntimeDeploymentAdapter`, calls `build_production_runtime_agent_factory_resolver` independently, string-keyed lookup, or dynamic type probing.

**R2 withdrawal:** `RuntimeDeploymentAdapter.materialize_serving_runtime` and `ServingUnitRuntimeBinding`-only handoff **superseded** by §4.5A/B.

Implementation belongs to **PROD-Q-R1-R7-D0** (contracts + activation attest) before **R7-D1** adapter + **D1b** GCA consumption.

#### 4.5C — API dependency graph (D0 — no cycles)

```text
intergrax/agent_distribution/deployment.py
  └─ attest models + RuntimeDeploymentAdapter (+ existing prepare/readiness/drain/stop/resume)
intergrax/agent_distribution/activation.py
  └─ imports deployment.py only (no applications/)
intergrax/applications/_shared/serving_runtime_handoff.py
  └─ imports runtime_agent_factory_resolver.py, deployment attestation types (read-only models)
intergrax/applications/_shared/production_process_composition.py
  └─ optional ServingRuntimeHandoff field
intergrax/applications/_shared/reference_production_lifecycle.py (or production successor)
  └─ establish_serving_runtime_handoff; imports handoff + composition
applications/governed_contractor_application/host/factory.py
  └─ reads composition.serving_runtime_handoff only
```

**Forbidden edges:** `deployment.py` → `applications/`; GCA → `RuntimeDeploymentAdapter`; cyclic handoff (GCA establishing handoff).

**D0 minimal scope:** land §4.5A/B **types and Protocol methods** + `ActivationService` attest gate + composition field + `establish_serving_runtime_handoff` **validation-only** (fail-closed if resolver instance ≠ `projection_input.factory_resolver` when both present). **D0 must not** ship a production adapter body or change strict GCA traffic (D1/D1b).

### 4.6 Architecture decision B — stale READY recovery

**Semantic owner:** `ActivationService`. **Remove:** READY early return at `activation.py` L174–177 without `attest_serving_unit`.

| Condition | Rule | Traffic |
|-----------|------|---------|
| READY + live attest + digest match | Allow idempotent return after attest | No until commit |
| READY + stale/absent unit | FAILED or re-prepare under env lock + `record_revision` CAS | **No** |
| READY + wrong digest/revision/env | FAILED | **No** |
| READY + unverifiable attest | FAILED (fail-closed) | **No** |
| SERVING after process restart | Reconcile via attest before ASGI; no infer from durable row | **No** until reconciled |
| Concurrent recovery | `_environment_lock`; no duplicate physical `prepare` without CAS | **No** |

Stale handle on rollback: failed `resume_serving` attest → `_prepare_prior_serving_unit` (new prepare), not dead ref reuse.

### 4.7 Identity / dependency graphs (before vs after)

**Before (HEAD):**

```text
RuntimeDeploymentAdapter ──prepare──► serving_unit_ref (opaque)
        │                                    │
        │                                    ▼
ActivationService ──► SqliteDeploymentInstanceStore (READY)
        │
        ▼
commit_activation ──► serving pointer
        │
        ▼
ProductionProcessComposition (no binding) ──► GCA ──► build_harness_host_runtime (independent factory path)
```

**After (target, post-ratification — R3):**

```text
RuntimeDeploymentAdapter ──prepare/check_readiness/attest──► unit table (AD types only)
        ▲ attest_serving_unit ◄── ActivationService (READY reuse / commit guards)
        │
ReferenceProductionLifecycleLauncher.deploy_and_activate
        │ establish_serving_runtime_handoff (Applications)
        │   ├─ ServingUnitAttestation + DeploymentInstanceRecord
        │   ├─ RegistryProjectionInputBundle.factory_resolver (same instance)
        │   └─ ColocatedServingUnitRuntimeAccessor (VENV A1 wiring only)
        ▼
ProductionProcessComposition.serving_runtime_handoff
        │
        ▼
create_governed_contractor_process_app (read handoff — no adapter import)
```

---

## 5. Decision D1 — First supported production topology (implements Decision A1)

**Decision (proposed):** **`MaterializationTopology.VENV_BUNDLE`** with **externally materialized immutable artifact directory** (`file://` locator), **co-located in-process serving unit (A1)** in the **same OS process** that will host strict GCA after activation.

**Evidence:**

| Criterion | VENV_BUNDLE (chosen) | OCI_IMAGE | SANDBOX_SIDECAR |
|-----------|----------------------|-----------|-----------------|
| Production registry projection | **Enforced** (`_validate_revision_roster_lock_authority` requires VENV_BUNDLE) | Fails assembly | Fails assembly |
| Runtime factory resolver | `implemented` | `deferred` | `deferred` |
| Artifact already materialized | **Yes** — operator/CI produces directory; locator in materialization store | Would require image registry + runtime — not projection-qualified | Not evidenced |
| Physical serving unit | **A1 in-process slot** (`serving_unit_ref` ≠ path); digest binding + resolver warmup are **inputs to** `check_readiness`, not substitutes for unit handle | Container/scheduling — no production resolver | Sidecar provider — deferred |
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

## 6. Decision B — Production composition authority (B02)

### 6.1 Semantic owners (exactly one per concern)

| Concern | Semantic owner | Notes |
|---------|----------------|-------|
| Deployment orchestration | **Agent Distribution** — `ActivationService` + `RuntimeDeploymentAdapter` | GCA must not call adapter; composition must not expose adapter to GCA |
| Runtime resolver handoff | **Applications** — `establish_serving_runtime_handoff` + `ServingRuntimeHandoff` | **Not** `RuntimeDeploymentAdapter`; not GCA-local factory rebuild |
| Activation / serving pointer commit | **Agent Distribution** — `ActivationService.commit_activation` | Governance does not mutate pointer |
| Governance mutation authorization | **Governance** — `ControlPlaneMutationAuthorizationBoundary` | Proposal ≠ permission |
| Deployment / serving durable facts | **Agent Distribution** — `DeploymentInstanceStore`, serving + activation stores | READY ≠ live unit |
| Registry projection | **AP-10** — `ApplicationRegistryProjectionCoordinator` + production input bundle | Non-authoritative for deployment instance |
| Runtime / process resources (OS) | **External launcher** + adapter-managed unit | Not GCA `run()` |
| Registry projection input assembly | **Applications shared** — `production_registry_projection_input_bundle.py` | AC-3 semantics |
| GCA | **Application consumer** — strict ASGI after activation | No deploy authority |

**Composition owner (sanctioned wiring only):** one **production composition function** (proposed: `wire_production_process_lifecycle` / `open_production_process_lifecycle` in `reference_production_process_lifecycle.py` or successor module) that **only** binds already-qualified ports. It **must not** introduce deployment or activation semantics beyond calling existing `ActivationService` / launcher methods.

**Not production-qualified by adapter injection:** `ReferenceProductionLifecycleLauncher`, `wire_reference_production_lifecycle_services`, `wire_governed_reference_production_launcher`, `build_reference_production_control_plane_mutation_boundary` remain **reference-named** until production wiring slice (**R7-D2**) proves reachability with **zero** fake adapter and **operator-approved** governance profile.

### 6.2 Minimum authorized production composition path (target)

```text
External launcher
  → open_production_process_lifecycle(...)          # NEW: composition owner; no FakeInMemory adapter default
       → ProductionProcessComposition
       → SqliteAgentDistributionStoreBundle (durable deployment + serving + revision)
       → wire_production_lifecycle_services(
            deployment_adapter=<PRODUCTION RuntimeDeploymentAdapter>,
            governance_boundary=<PRODUCTION ControlPlaneMutationAuthorizationBoundary>,
            projection stores = durable,
          )
       → ReferenceProductionLifecycleLauncher (or production-renamed type) — orchestration only
            → deploy_and_activate → ActivationService (unchanged semantics)
  → create_governed_contractor_process_app(process_composition=...)
  → ASGI server (external) — traffic only after activation + strict admission
```

**Before (current HEAD):** `open_reference_production_process_lifecycle` → `FakeInMemoryRuntimeDeploymentAdapter` + in-memory projection inputs + reference governance boundary.

**After (target):** production composition function with **explicit required** `deployment_adapter` and **no** fake default; reference entry points remain for lab/tests.

### 6.3 Dependency classification matrix (production path)

| Dependency | Classification | Valid production source |
|------------|----------------|-------------------------|
| `ActivationService` | **PRODUCTION-QUALIFIED** (orchestration contract) | `intergrax/agent_distribution/activation.py` — unchanged semantics |
| `RuntimeDeploymentAdapter` (production impl) | **MISSING** → **R7-D1** | New `VenvBundleColocated…` (illustrative); not fake |
| `FakeInMemoryRuntimeDeploymentAdapter` | **TEST-ONLY** | Unit/orchestration tests; forbidden on strict production graph |
| `SqliteAgentDistributionStoreBundle` / `SqliteDeploymentInstanceStore` | **PRODUCTION-QUALIFIED** (contract) | `sqlite_stores.py` — **R7-D2** must wire on production path |
| `InMemoryDeploymentInstanceStore` (reference wiring) | **REFERENCE-ONLY** | Lab single-process proofs only |
| `assemble_production_registry_projection_input_bundle` | **PRODUCTION-QUALIFIED** (contract) | `production_registry_projection_input_bundle.py` |
| `RegistryProjectionInputStore` (in-memory on reference path) | **REUSABLE CONTRACT / NOT YET QUALIFIED** | **R7-D3** — durable inputs required for restart |
| `build_production_runtime_agent_factory_resolver` | **PRODUCTION-QUALIFIED** (VENV_BUNDLE) | `venv_bundle_runtime_agent_factory_resolver.py` — used inside D1 readiness, not as deployment owner |
| Governance principal + policy | **REUSABLE CONTRACT / NOT YET QUALIFIED** | Operator production mutation profile — **not** `build_reference_production_control_plane_mutation_boundary` alone |
| `ControlPlaneMutationAuthorizationBoundary` | **PRODUCTION-QUALIFIED** (interface) | Implementation must be production profile |
| Serving pointer stores | **PRODUCTION-QUALIFIED** (contract) | SQLite bundle on production path |
| Continuation store (`open_production_execution_continuation_state_store`) | **PRODUCTION-QUALIFIED** | ADR-02 path — **R5/R6** |
| `SQLiteProviderInvocationStore` | **PRODUCTION-QUALIFIED** (contract) | PIS-ADR-03 — **R6** |
| `ReferenceProductionLifecycleLauncher.deploy_and_activate` | **REUSABLE CONTRACT / NOT YET QUALIFIED** | Callable orchestration; qualified only when **full** dependency column satisfied |
| `open_reference_production_process_lifecycle` | **REFERENCE-ONLY** | Must not be named production entry without `open_production_process_lifecycle` |
| `UnsupportedVenvBundleMaterializationAdapter` | **REFERENCE-ONLY** (explicit fail) | External CI produces bundles |
| GCA `create_governed_contractor_process_app` | **PRODUCTION-QUALIFIED** (consumer) | Requires activated composition |
| GCA `main.run()` | **REFERENCE-ONLY** / forbidden unactivated | **R7-D4** launcher documentation |
| Application process owner | **External launcher** | Owns `open_production_process_lifecycle` context + ASGI |

`applications/governed_contractor_application/host/main.py` **must remain** a consumer — **not** deployment root. **`run()`** must **not** become deploy authority.

### 6.4 Exactly-one ownership graph (R3 — typed edges)

```mermaid
flowchart LR
  AD[Agent Distribution ActivationService]
  RDA[RuntimeDeploymentAdapter]
  GOV[Governance mutation boundary]
  AP10[AP-10 ApplicationRegistryProjectionCoordinator]
  COMP[Applications composition lifecycle]
  GCA[GCA strict host consumer]
  PROC[Process lifecycle owned resources]

  AD -->|prepare check_readiness attest ServingUnitAttestationRequest| RDA
  GOV -->|ControlPlaneMutationAuthorizationEvidence ALLOW/DENY| COMP
  COMP -->|prepare_candidate commit_activation| AD
  COMP -->|RegistryProjectionInputBundle register| AP10
  AD -->|DeploymentInstanceRecord durable| PROC
  COMP -->|establish_serving_runtime_handoff ServingRuntimeHandoff| PROC
  PROC -->|ProductionProcessComposition| GCA
  GCA -->|MaterializedRegistryProjection read stores| AP10
  GCA -->|RuntimeAgentFactoryResolver from handoff only| PROC
```

| Edge | Typed contract | Layer leak check |
|------|----------------|------------------|
| Launcher → Composition | `open_production_process_lifecycle` → `ProductionProcessComposition` | Applications only |
| Composition → Activation | `ActivationService.prepare_candidate` / `commit_activation` | AD semantics; no GCA |
| Activation → Adapter | `RuntimeDeploymentAdapter` + `ServingUnitAttestation` | **No** Applications types on port |
| Composition → Handoff | `establish_serving_runtime_handoff` | **No** AD import of `RuntimeAgentFactoryResolver` |
| Handoff → GCA | `ServingRuntimeHandoff` | GCA reads composition; **no** adapter |
| AP-10 → Stores | `RegistryProjectionInputBundle` / `MaterializedRegistryProjection` | Non-authoritative for deployment instance |
| Governance → Launcher | `authorize_scoped_control_plane_mutation` | Governance ≠ execution |

**Negative proof (qualification):** static import graph must show **zero** `applications/` → `RuntimeDeploymentAdapter` from GCA modules; **zero** `intergrax/agent_distribution/deployment.py` imports from `intergrax/applications/`.

---

## 7. Decision D3 — Provider replaceability & injection

**Decision (proposed):**

- Keep **`RuntimeDeploymentAdapter`** provider-neutral; **no** vendor conditionals in `ActivationService` or GCA.
- Inject concrete adapter at **lifecycle service wiring** — extend `wire_reference_production_lifecycle_services` / `wire_durable_reference_production_lifecycle_services` (or production-named successors) with a **required** `deployment_adapter: RuntimeDeploymentAdapter` parameter; **forbid default** to fake in production mode.
- Configure adapter with **typed** config (e.g. artifact root policy, readiness timeouts) via constructor — **not** `dict`/`Any` registries.
- **No new global deployment registry** — composition parameter + existing `SqliteAgentDistributionStoreBundle` suffice.

Production qualification must prove **structural replaceability**: second test double implementing the same Protocol without changing `ActivationService`.

---

## 8. Decision D4 — Physical readiness vs execution readiness

| Layer | What can be proven | Owner of truth | Must not substitute |
|-------|-------------------|----------------|---------------------|
| **Artifact immutability** (pre-unit) | Directory exists; digest matches revision; locator allowed by `resolve_production_artifact_root` | Revision store + projection assembly | Equating path existence with `serving_unit_ref` |
| **Physical serving unit** | Adapter-allocated unit handle; slot checks + digest binding; factory resolver as **readiness input** only | **`RuntimeDeploymentAdapter.prepare` + `check_readiness`** → `serving_unit_ref` + `readiness_evidence_ref` | Manual registry seeding; diagnostics HTTP 200 alone |
| **Activation gate** | Validated revision; READY instance; governance ALLOW | **`ActivationService.commit_activation`** | Skipping `prepare_candidate` |
| **Serving pointer** | Atomic pointer CAS | **`ApplicationEnvironmentActivationStore`** | In-memory test doubles on production path |
| **Registry projection** | Inputs match activated revision + digest | **`ApplicationRegistryProjectionCoordinator`** + `production_registry_projection_input_bundle` | Manifest defaults at host startup |
| **Application readiness** | Harness host built; strict admission satisfied | **GCA factory** (`build_harness_host_runtime`) after composition bound | Unactivated `create_app()` |
| **Execution admission** | Continuation durable; tool boundary materialized | **Execution** validation in factory | Lab profile in strict mode |

Deployment adapter **does not** prove business-level health endpoints; GCA readiness probes remain **downstream** and **non-authoritative** for deployment instance state.

---

## 9. Decision D5 — Process ownership

| Question | Decision (proposed) |
|----------|---------------------|
| Who owns OS process lifetime? | **External launcher** using `open_reference_production_process_lifecycle` context manager |
| Who owns deploy/activate? | **`ReferenceProductionLifecycleLauncher`** on the same composition (before ASGI serve) |
| Who owns ASGI app? | **GCA** `create_governed_contractor_process_app` after activation |
| Who owns continuation/invocation close? | **`ReferenceProductionProcessLifecycle.close_process_resources`** (R6) — **not** FastAPI shutdown alone |
| Circular ownership | **Forbidden:** GCA `run()` cannot deploy itself. Deploy/activate runs in launcher process **before** binding serving traffic to the strict host factory. Co-located adapter manages **in-process serving slot**, not a second GCA instance. |

Implementing **`main.run()`** as mandatory production entry: **rejected**. Production entry = **launcher + uvicorn factory** with explicit composition.

---

## 10. Decision D — Recovery and operational invariants (negative matrix)

**Invariant:** A durable `DeploymentInstanceState.READY` or `SERVING` record **never** alone implies a currently alive serving unit after crash or adapter loss.

| # | Condition | Authoritative durable fact | Recovery action owner | Traffic permitted? | Evidence required | Regression proof (indicative) |
|---|-----------|---------------------------|----------------------|-------------------|-------------------|------------------------------|
| 1 | Artifact missing, corrupt, wrong digest | Revision / materialization record unchanged or FAILED instance | Projection assembly + adapter `prepare` fail-closed | **No** | Digest mismatch evidence on instance | Wrong digest → FAILED, no commit |
| 2 | Runtime factory construction failure | Instance → FAILED after readiness attempt | D1 adapter + `ActivationService` | **No** | `failure_evidence_ref` / readiness error | Factory throws → no READY |
| 3 | Serving unit not ready | PREPARING or FAILED instance | Adapter `check_readiness` | **No** | Absent/invalid `readiness_evidence_ref` | Readiness fail → no commit |
| 4 | Governance denial | No pointer change | Launcher / operator | **No** | `ControlPlaneMutationAuthorizationEvidence` | DENY → no activation |
| 5 | Activation failure after readiness | READY instance may exist; pointer unchanged | `ActivationService` / operator | **No** | Serving store revision unchanged | Commit error → pointer stable |
| 6 | Startup failure after activation | Pointer may be committed; instance SERVING in store | Launcher + process lifecycle close (R6) | **No** (process exit) | Serving record + failed host bootstrap | Strict factory fail-closed |
| 7 | Process crash after serving pointer commit | Durable pointer + SERVING/READY records | Restart reconciliation: adapter attest or re-prepare | **No** until reconciled | Live unit attestation or new prepare cycle | Restart test: no traffic on stale handle |
| 8 | Durable READY, physical unit absent | READY record with stale `serving_unit_ref` | Operator/launcher: mark failed or re-prepare | **No** | Adapter cannot attest handle | READY ≠ traffic gate alone |
| 9 | Restart with stale `serving_unit_ref` | Instance record vs adapter state mismatch | D1 adapter + `prepare_candidate` idempotency rules | **No** until new readiness | New evidence ref after re-prepare | Stale ref rejection |
| 10 | Drain timeout | DRAINING + policy snapshot | `complete_drain` + `DrainPolicy.action_on_timeout` | **No** new on drained unit | Drain `evidence_ref` | Timeout → STOP/MARK_FAILED |
| 11 | Failure during `stop` | DRAINING/FAILED instance | Adapter retry + operator | **Policy-bound** | `failure_evidence_ref` on stop | Stop error surfaced |
| 12 | Wrong tenant/environment/revision | Revision scope mismatch | Governance + `ActivationService` guards | **No** | Scope denial evidence | Cross-tenant activation denied |
| 13 | Concurrent activation | Serving store CAS | `atomic_commit_activation` conflict | **No** on loser | `RuntimeActivationConflict` | Concurrent commit test |
| 14 | Previous revision rollback/resume | Prior instance STOPPED/DRAINING | `resume_serving` + rollback paths | **Only** after rollback commit | New readiness ref on resume | Rollback qualification |
| 15 | Unresolved external `ProviderInvocation` | PIS durable ambiguity | Execution recovery (PIS) — not deployment | Deployment does not widen authority | PIS store state | No duplicate deploy side effect |
| 16 | Unsupported topology (OCI/sidecar) | Revision rejected at assembly | `production_registry_projection_input_bundle` | **No** | Assembly error | OCI revision rejected |
| 17 | Missing production Governance configuration | No ALLOW for mutations | Fail-closed at boundary | **No** | Deny evidence | Unconfigured policy → deny |

**No exactly-once** for adapter `prepare` across crashes — idempotency via `DeploymentInstanceRecord` states and digest-keyed unit discipline.

### Architectural STOP conditions (recovery)

- Inferring liveness from `readiness_evidence_ref` or artifact path without adapter attestation → **STOP**.
- Promoting reference governance boundary as production without operator profile → **STOP** (B02).
- GCA or ASGI factory performing `prepare_candidate` → **STOP** (authority widening).

---

## 11. Decision C — Lifecycle completeness (state transitions)

| Phase | Durable state | Physical unit | Permitted op | Owner | CAS / concurrency | Evidence | Traffic admission |
|-------|---------------|---------------|--------------|-------|-------------------|----------|-------------------|
| prepare | PREPARING → READY | slot warming | `prepare` + `check_readiness` | ActivationService + adapter | env `RLock`; instance `record_revision` | `serving_unit_ref`, `readiness_evidence_ref` | **Deny** |
| readiness reuse | READY | must be live | `prepare_candidate` idempotent path | ActivationService | attest before return | `ServingUnitLivenessAttestation` | **Deny** |
| activation | SERVING + pointer | same bound unit | `commit_activation` | ActivationService | `atomic_commit_activation` CAS | pointer revision bump | **Deny** until GCA strict admission |
| serving | SERVING | live + binding on composition | ASGI listen | External ASGI + GCA | — | binding + activated projection | **Allow** after strict factory |
| drain | DRAINING | draining | `begin_drain`, `check_drain`, `complete_drain` | ActivationService + adapter | instance state CAS | drain `evidence_ref` | **Deny** new work on drained unit |
| stop | STOPPED | terminated | `stop` after drain policy | Adapter | — | stop completion | **Deny** |
| rollback/resume | READY/SERVING swap | attest or re-prepare prior | `rollback`, `resume_serving` | ActivationService | rollback atomic CAS | new readiness ref if re-prepared | **Policy-bound** after rollback commit |
| crash/restart | READY/SERVING rows may persist | **absent** | launcher reconciliation attest | ActivationService + launcher | no traffic until attest | failed attest → FAILED | **Deny** |
| ASGI startup failure post-commit | pointer committed | partial host | lifecycle `close_process_resources` (R6) | Launcher | — | serving record + failed bootstrap | **Deny** (process exit) |
| drain timeout | DRAINING | policy action | `DrainPolicy.action_on_timeout` | ActivationService | — | timeout evidence | **Deny** on timed-out unit |
| abrupt termination | unknown cleanup | lost | **no assumption** of `stop` | Operator recovery | — | durable rows only | **Deny** until §4.6 reconcile |

### 11.1 State / restart matrix (R3 — testable invariants)

| Scenario | Durable truth | Liveness truth | Owner | Permissible traffic | Testable invariant |
|----------|---------------|----------------|-------|---------------------|-------------------|
| Fresh PREPARING → READY | Instance `PREPARING`→`READY`; evidence refs set | Adapter unit live after `check_readiness` | `ActivationService` + adapter | **Deny** | No `commit_activation` without READY |
| Valid READY reuse | READY row unchanged if attest passes | `attest_serving_unit.physical_unit_live` | `ActivationService` | **Deny** | Early return only after attest |
| Stale READY after restart | READY row persists | Attest **fail** | `ActivationService` | **Deny** | FAILED or re-prepare; no GCA bypass |
| Lost SERVING unit | SERVING + pointer | Attest fail / drain unknown | Launcher + `ActivationService` | **Deny** | Reconcile before ASGI |
| Wrong revision or digest | Revision or instance FAILED | N/A | `ActivationService` | **Deny** | Attest request mismatch → error |
| Concurrent attest / re-prepare | Single CAS winner | Serialized by env lock | `ActivationService` | **Deny** | No duplicate live units |
| Activation commit failure | READY; pointer unchanged | Unit may be live | `ActivationService` | **Deny** | Serving store revision stable |
| GCA startup failure post-commit | SERVING committed | Partial host | Launcher `close_process_resources` | **Deny** (exit) | No auto-deploy in GCA |
| Rollback / resume | Prior instance row | `resume_serving` + attest | `ActivationService` | Policy after rollback CAS | Failed attest → `_prepare_prior_serving_unit` |
| Drain / stop | DRAINING→STOPPED | Adapter drain complete | `ActivationService` + adapter | **Deny** new on drained unit | Evidence refs required |
| Graceful shutdown | STOPPED or DRAINING complete | Unit stopped | Adapter + lifecycle close | **Deny** | R6 close ordering |
| Crash without cleanup | READY/SERVING rows | Unknown | Operator + attest path | **Deny** | No infer liveness from SQLite |

**`prepare_candidate` READY early return (HEAD L174–177):** must gain **`attest_serving_unit`** gate (B05); on failure, transition to FAILED or full re-prepare — **no** separate GCA recovery authority.

## 12. Decision C — Process lifecycle sequence (no circular dependencies)

**Separation (mandatory):** **Physical preparation ≠ Governance permission ≠ activation ≠ execution admission ≠ serving traffic.**

| Phase | Allowed before activation? | Owner | Restrictions |
|-------|---------------------------|-------|--------------|
| Immutable artifact preparation (out of band) | Yes | External CI / operator | No platform traffic |
| Revision admission + `mark_validated` | Yes (control plane) | Agent Distribution + Governance admission | No serving pointer |
| Governance permission for activation mutations | Required before commit | Governance | Not execution |
| `prepare_candidate` (physical unit) | Yes | `ActivationService` + adapter | **No** serving pointer; instance READY only |
| Projection input registration + `prepare_projection` | Only inside `commit_activation` after ready | Coordinator | Fail-closed on mismatch |
| `commit_activation` (pointer CAS) | N/A — defines activation | `ActivationService` | Promotes instance SERVING in store |
| `open_production_process_lifecycle` / composition open | Yes, but strict GCA **not** serving until activated | Launcher | May open continuation/invocation stores |
| `create_governed_contractor_process_app` | After composition bound to **activated** stores | GCA | Fail-closed if unactivated |
| ASGI listen / traffic | After activation + strict admission | External ASGI | Diagnostics non-authoritative |

```text
1. External materialization → immutable VENV bundle + digest (out of band)
2. Governance: admit revision → VALIDATED in RuntimeRevisionStore
3. Register materialization record (artifact_locator)
4. Register projection input (production bundle assembly)
5. open_production_process_lifecycle (durable Sqlite bundle + production adapter + production governance)
6. Launcher.deploy_and_activate:
     a. Governance: activation-scoped mutations (ALLOW)
     b. ActivationService.prepare_candidate → adapter prepare + check_readiness → READY (no traffic); attest on reuse
     c. establish_serving_runtime_handoff → ProductionProcessComposition.serving_runtime_handoff
     d. ActivationService.commit_activation → prepare_projection + serving pointer CAS → SERVING instance
     e. begin_drain prior unit (if any)
7. create_governed_contractor_process_app(process_composition) — consumes serving_runtime_handoff
8. ASGI server serve (external) — traffic enablement
9. Serving lifetime — execution admission via strict factory (downstream)
10. Shutdown: complete_drain (ops) → lifecycle.close_process_resources (R6) → adapter stop
11. Restart: reconcile pointer vs adapter attestation (Decision D §7–9) before traffic
```

**Forbidden:** deployment authority inside GCA; `run()` deploy; opening ASGI before step 6c completes.

---

## 13. Mandatory architecture invariants (disposition)

| Invariant | Satisfied by proposed decisions? |
|-----------|----------------------------------|
| Contracts over implementations | **Yes** — Protocol port retained |
| Layer boundaries | **Yes (R3)** — no GCA → adapter; no Applications types on `RuntimeDeploymentAdapter` |
| Exactly-one semantic owner per concern | **Yes** — §6.1 |
| Exactly-one sanctioned composition owner | **Yes** — `open_production_process_lifecycle` (proposed) |
| Strong typing | **Yes** — Pydantic + Protocol; adapter config must stay typed |
| Pluginability / replaceability | **Yes** — injection at wiring |
| No fake on production path | **Pending R7-D1/D2** — **IN-SCOPE BLOCKER** today |
| No Governance bypass | **Yes** — launcher path preserved |
| Fail-closed activation | **Yes** — existing `ActivationService` |
| READY ≠ live serving unit | **Yes (R2)** — Decision B + `attest_serving_unit`; **not satisfied on HEAD** (B05) |
| Serving-unit handoff to GCA | **Pending §4.5B** — **not satisfied on HEAD** (B04); **B06/B07** resolved in ADR text only |
| Diagnostics non-authoritative | **Yes** — §8 |

**Contract changes (R3):** **Required** — §4.5A (`attest_serving_unit`, attestation models) + §4.5B (`ServingRuntimeHandoff`, `establish_serving_runtime_handoff`, composition field) + `ActivationService` READY reuse fix. **Explicitly excluded:** `materialize_serving_runtime` on deployment port. Ratify before **R7-D1** implementation.

If co-located adapter needs `materialization_output` on `prepare`: optional parameter already exists — use typed `MaterializationOutput` from store, not ad-hoc dicts.

---

## 14. Minimal implementation slices (dependency order)

| Child | Owner | Scope | STOP condition |
|-------|-------|-------|----------------|
| **PROD-Q-R1-R7-D0** | Agent Distribution + Applications contracts | §4.5A attestation models + `attest_serving_unit` on Protocol; `ActivationService` attest gate (B05); §4.5B `ServingRuntimeHandoff` + `establish_serving_runtime_handoff` stub; composition field; `FakeInMemory` attest; **no** `materialize_serving_runtime` | READY reuse without attest; Applications type on AD port; GCA calls adapter |
| **PROD-Q-R1-R7-D1** | Agent Distribution + Applications wiring | Production `RuntimeDeploymentAdapter` (A1): unit table, `attest_serving_unit`, drain/stop/resume; `ColocatedServingUnitRuntimeAccessor` on same concrete class at wiring only | Readiness = path only; resolver via deployment port |
| **PROD-Q-R1-R7-D1b** | GCA + Applications shared | Strict factory **reads** `serving_runtime_handoff`; identity match vs activated projection — **no** parallel `build_production_runtime_agent_factory_resolver` on strict graph | Independent factory path; adapter import in GCA |
| **PROD-Q-R1-R7-D2** | Applications shared lifecycle | `open_production_process_lifecycle` + required production adapter; durable `SqliteAgentDistributionStoreBundle`; **no** fake default | Duplicate activation wiring in GCA |
| **PROD-Q-R1-R7-D3** | Applications + Agent Distribution | Production projection input store durability + post-restart reconciliation of serving pointer vs READY instance | Manual in-memory projection seeding required |
| **PROD-Q-R1-R7-D4** | GCA + lifecycle | Documented launcher → uvicorn factory; optional thin CLI — **no** `run()` deploy logic | GCA required to deploy itself |
| **PROD-Q-R1-R7-Q** | Qualification | End-to-end + adversarial proof matrix §15; Decision D rows 1–17 | Any fake adapter reachable from strict production graph |

Consolidation: **D1+D2** inseparable for first production path — may merge if single PR boundary agreed.

---

## 15. Future proof matrix (implementation — not executed in this ADR)

Qualification must include **behavioral** tests (indicative paths):

- `tests/unit/agent_distribution/` — adapter + activation integration with real adapter (not class-name asserts)
- `tests/unit/applications/governed_contractor/` — lifecycle + strict factory fail-closed negatives
- `tests/qualification/prod_q/` — production reachability: **fake deployment adapter count = 0** on strict GCA graph

| Gate | Intent |
|------|--------|
| Physical prepare/readiness | Wrong digest → FAILED; readiness cannot pass on path-as-ref alone |
| Serving unit identity | `serving_unit_ref` ≠ `artifact_locator`; adapter attestation after crash |
| Provider-neutral injection | Alternate stub adapter, same `ActivationService` |
| Authorized activation | Governance ALLOW → pointer bump |
| Denied / stale / cross-tenant | Governance DENY; CAS conflict |
| Failed readiness → zero serving | No `atomic_commit_activation` success |
| Activated revision → registry projection | Resolver sees activated revision id + digest |
| No manual test seeding | Forbidden `reference://` / harness registry shortcuts on strict path |
| Startup failure cleanup | R6-style lease release + deployment FAILED semantics |
| Restart / stale pointer | Durable stores + reconciliation tests |
| Stale READY after crash | SQLite READY + empty adapter → attest fails → FAILED or re-prepare (**B05**) |
| Identity continuity | Same attested `serving_unit_ref` → `ServingRuntimeHandoff.resolver` used by strict GCA (**B04**) |
| Cross-layer inversion | `intergrax/agent_distribution/deployment.py` has **no** import from `applications/` (**B06**) |
| GCA bypass | GCA modules do not reference `RuntimeDeploymentAdapter` (**B07**) |
| Drain/shutdown | `complete_drain` + `stop` evidence |
| Missing credentials/config | Fail-closed at governance or adapter |
| Unsupported topology | OCI revision → projection assembly error |
| No fake on production path | Static/AST reachability from `create_governed_contractor_process_app` strict graph |
| No silent fallback | Strict mode rejects missing adapter |
| Unresolved invocation | PIS store ambiguity — no duplicate external deploy effect |
| Replaceability | Second adapter implementation |
| Exact-SHA evidence | Qualification docs pin commit |

---

## 16. Applicable FRZ requirements (evidence contribution only — no PASS)

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

## 17. Enterprise audit matrix

| Category | Decision / finding |
|----------|-------------------|
| Boundaries | Tier-0/1 AD semantics stay in `intergrax/agent_distribution`; GCA consumes only |
| Communication | Launcher orchestrates; no hidden deploy in ASGI factory |
| Composition | Single `ProductionProcessComposition` root |
| Ownership | Tables §3, §6.1 — one owner per row |
| B01–B07 corrective | R3 boundary handoff; B06/B07 **resolved in ADR**; B04/B05 **pending code**; R1 “no port change” **withdrawn** |
| Contracts | §4.5A/B delta **proposed** — ratification required |
| Cross-layer | **PASS (design)** — no Applications return type on deployment port |
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

## 18. Unresolved findings (classification)

| Finding | Classification | Child / action |
|---------|----------------|----------------|
| B04 — no GCA handoff contract | **IN-SCOPE BLOCKER** | **R7-D0** + **R7-D1b** |
| B05 — READY reuse without attest | **IN-SCOPE BLOCKER** | **R7-D0** |
| B06 — cross-layer `materialize_serving_runtime` | **RESOLVED (ADR R3)** — implement guard in D0 | AST: no Applications imports in `deployment.py` |
| B07 — GCA vs composition handoff contradiction | **RESOLVED (ADR R3)** — implement in D1b | Import graph gate GCA ↔ adapter |
| No production `RuntimeDeploymentAdapter` | **IN-SCOPE BLOCKER** | **R7-D1** |
| Fake adapter on lifecycle wiring | **IN-SCOPE BLOCKER** | **R7-D2** |
| In-memory deployment/projection input stores on reference path | **TRACKED FREEZE DEBT** | **R7-D3** |
| Full strict tool admission on all materialization paths | **IN-SCOPE BLOCKER** (existing **PROD-Q-R1**) | R1-R1 / follow-on |
| `UnsupportedVenvBundleMaterializationAdapter` vs external bundles | **TRACKED FREEZE DEBT** | Separate materialization qualification (not R7-D1) |
| Production governance policy vs reference harness boundary | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Operator-approved production mutation policy profile |
| GCA `run()` not production launcher | **TRACKED FREEZE DEBT** (documented) | **R7-D4** |
| **FRZ-PRD-04/05** global OPEN | **TRACKED FREEZE DEBT** | Post-implementation qualification |

---

## 19. Unresolved questions (operator ratification)

1. Ratify **§4.5A/B typed contract delta** (incl. **rejection** of `materialize_serving_runtime` on deployment port) and **Decision B** stale-READY rules before any D1 adapter code?
2. Ratify **A1 conditional** (co-located + binding) vs defer production until **A2** supervisor contract exists?
3. Mandate **`SqliteAgentDistributionStoreBundle`** for all production deployment/serving persistence (vs hybrid in-memory projection inputs)?
4. Production **control-plane mutation policy** profile: extend reference boundary or new sanctioned production boundary module?
5. External **materialization pipeline** ownership (CI digest attestation) — platform vs application team?
6. Approve proposed names **`open_production_process_lifecycle`** / **`wire_production_lifecycle_services`** as composition owner API?

---

## 20. Exit recommendation

**READY FOR ARCHITECTURE DECISION** — operator ratification of **§4.4–4.6** (A1 conditional + **§4.5A/B** + stale READY), **§6.4** ownership graph, **Decision D1**, and **§6** composition path. **First implementation child:** **PROD-Q-R1-R7-D0** — exit: attestation Protocol + `ServingRuntimeHandoff` types landed; `ActivationService` READY path calls `attest_serving_unit`; **no** `materialize_serving_runtime` on `RuntimeDeploymentAdapter`; negative import tests for B06/B07. Then **R7-D1** / **D1b** / **D2**. Exit: B04/B05 closed in code + §15 matrix green on exact SHA. Keep **PROD-Q-R1** **BLOCKED**; **FRZ-PRD-04/05** **OPEN**.

**Maximum outcome of this commit:** architecture proposal only — **no** production adapter, **no** FRZ PASS, **no** roadmap CLOSED.

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

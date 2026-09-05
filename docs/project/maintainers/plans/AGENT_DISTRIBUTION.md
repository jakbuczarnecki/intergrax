# Agent Distribution and Management - Plan

**Status:** Active (architecture frozen - AGENT-PLATFORM-2 + ARCH-AGENT-ACTIVATION-1)  
**Architecture (1:1):** [`architecture/AGENT_DISTRIBUTION.md`](../../architecture/AGENT_DISTRIBUTION.md)  
**ADR:** [`adr/entries/2026-08-12/ADR-AGENT-004.md`](../../technical/adr/entries/2026-08-12/ADR-AGENT-004.md)  
**Evidence:** [`audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md`](../audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md)  
**Last updated:** 2026-09-05

---

## Goal

Implement the Tier-0 Agent Distribution domain so operators can discover, install, bind, configure, enable, upgrade, rollback, and uninstall agents **without** hot-loading Python, **without** LKW-local stores, and **without** breaking deterministic runtime graphs or Nexus capability routing.

### Protocol v2 remediation - TIER_LAYER_BOUNDARIES (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md`](../../audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings - **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-TIER-LAYER-PERSIST.

| Block | Status | Findings | Acceptance intent |
|-------|--------|----------|-------------------|
| **TL-FIX-B** | ACCEPTED / PLANNED | [`AUDIT-20260818-TIER_LAYER_BOUNDARIES-02`](../../audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md) | Exactly one canonical `echo@1.0.0` concrete production implementation; no colliding Tier-1 concrete copy; package/import/registry tests prove canonical resolution; clean-cut removal preferred (no legacy alias unless real dependency demonstrated during remediation); revalidate current `development` before implementation |

**Remediation rules:**

- Revalidate finding against then-current `development` HEAD before implementation.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).

### Protocol v2 remediation - AGENT_SYSTEM (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/AGENT_SYSTEM.md`](../../audit_results/2026-08-18/AGENT_SYSTEM.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings - **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-AGENT-SYSTEM-PERSIST.

| Block | Status | Findings | Acceptance intent |
|-------|--------|----------|-------------------|
| **AGSYS-IDENTITY-PROJECTION** | ACCEPTED / PLANNED | [`AUDIT-20260818-AGENT_SYSTEM-04`](../../audit_results/2026-08-18/AGENT_SYSTEM.md) | Registry projection preserves canonical package/contract identity; `AgentRegistry.from_agents` dict-key rewrite fails closed or uses explicit typed alias contract; clean-cut removal preferred if no required consumer; distinguish bootstrap compatibility from activated runtime projection truth; cross-ref [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) registry bootstrap and **TL-FIX-B** single-authority invariants |

**Remediation rules:**

- Revalidate finding against then-current `development` HEAD before implementation.
- Contract/routing integrity findings 01–03, 05–06 owned by **AGSYS-CONTRACT-INTEGRITY** in [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) - cross-reference only; do not duplicate.
- **TL-FIX-B** remains separate ownership for colliding Tier-1/Tier-2 concrete implementations; AGSYS-IDENTITY-PROJECTION addresses registry bootstrap identity rewrite - explicitly cross-link where bootstrap could admit competing identities.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).

## Architecture delivery (AGENT-PLATFORM-2) - Done

| Item | Status |
|------|--------|
| Canonical `AGENT_DISTRIBUTION.md` | done |
| Deterministic `MaterializedRuntimeLock` model | done (architecture) |
| `RuntimeRevision` + activation semantics | done (architecture - ARCH-AGENT-ACTIVATION-1 frozen) |
| Effective roster merge specification | done (architecture) |
| Cross-link from agent execution hub | done |
| Plan pair (this file) | done |

## Implementation waves (AP-3+)

| Wave | Deliverable | Depends on |
|------|-------------|------------|
| AP-3 | Tier-0 contracts: identity, catalog IF, installation/binding records | AP-2 arch |
| AP-4 | Store interfaces + transactional domain services | AP-3 |
| AP-5 | `AgentPackageTrust` coordinator | AP-3, plugin evidence patterns |
| AP-6 | Effective roster merge + `CandidateDependencySpecification` builder | AP-4 |
| AP-7 | `MaterializedRuntimeLock` producer + graph simulation gates | AP-6, runtime graph util |
| AP-8 | Materialization adapters (OCI, venv bundle) | AP-7 |
| AP-9 | `RuntimeRevision` activation + rollback orchestration (implements §20 zero-downtime model) | AP-8 |
| AP-10 | `build_application_registry` extension + snapshot fields | AP-9 |
| AP-11 | Generic Tier-3 harness admin API routes | AP-4..AP-9 |
| AP-12 | LKW consumer proof wiring | AP-11 |

## Non-goals (program)

- Marketplace billing, reviews, publisher portal
- Second Nexus or registry
- Runtime hot-load
- LKW-specific installer or persistence
- Mandating Docker for all topologies

## Verification intent

> **Historical (2026-08):** Original verification intent listed digest-pinned install → activate → registry as future LKW proof. AC-3 canonical lifecycle E2E and AC-4 production-composition E2E now provide **platform proof** under reference production V1 (§35 architecture). LKW consumer wiring (AP-12) remains separate.

- Digest-pinned install → lock → graph → activate → registry → routable capability — **proven** (AC-3 E2E + AC-4 E2E)
- Rollback restores prior `RuntimeRevision` and lock digest — AP-9 tests
- Catalog outage does not affect active revision reproducibility — architecture invariant
- Concurrent install/upgrade serialization on installation slot — AP-4 concurrency tests

## AGENT-PLATFORM-3 gate

**READY_FOR_CLOSE** (2026-08-12) - Tier-0 contracts and store ports under `intergrax/agent_distribution/`; AP-3-FIX-1 hardened namespace alignment, deep immutability, recursive secret rejection, typed ports, and shared digest validation.

| Deliverable | Status |
|-------------|--------|
| `AgentPackageIdentity` / catalog contracts | done |
| `CatalogSourceProvider` port | done |
| `AgentCapabilityMetadataProvider` port (architecture projection) | done (AGENT-CONSOLIDATION-2-FIX-1: package pyproject → descriptor; no central inventory) |
| Trust/provenance evidence surface | done |
| Installation / binding contracts | done |
| Effective roster projection models | done |
| Dependency + `MaterializedRuntimeLock` contracts | done |
| `RuntimeRevision` + materialization I/O contracts | done |
| Store ports (installation, binding, lock, revision, artifact metadata) | done |
| Focused unit tests + tier-boundary check | done |
| AP-3-FIX-1 contract hardening | done |

**Next:** AP-4 (transactional domain services).

## AGENT-PLATFORM-4 gate

**READY_FOR_CLOSE** (2026-08-13) - transactional domain services with explicit store-port atomic operations; no runtime capability discovery in AP-4 services.

| Deliverable | Status |
|-------------|--------|
| `InstallationService` lifecycle (candidate → verified → active → rollback/revoke/tombstone) | done |
| `BindingService` lifecycle (create/update/enable/disable/tombstone + slot-anchored upgrade survival) | done |
| `RuntimeRevisionService` (candidate → validated → active + rollback pointer) | done |
| `InMemory*Store` + `AgentDistributionStoreState` durable backing | done |
| Typed conflicts (`InstallationSlotConflict`, `BindingRevisionConflict`, `RuntimeRevisionConflict`) | done |
| Bounded domain events on `TransitionResult` | done |
| Focused AP-4 tests (installation, binding, runtime revision, concurrency/atomicity) | done |
| Tier-boundary check (no `agents/` / `applications/` imports) | done |
| AP-4-FIX-1 explicit port atomic ops (`atomic_promote_active_installation`, `atomic_activate_revision`, `list_bindings_for_slot`) | done |
| AP-4-FIX-1 service boundary (no `getattr`/`hasattr`/`.state` introspection) | done |

**Evidence:** `tests/unit/agent_distribution/test_agent_distribution_services.py`, `test_agent_distribution_runtime_revision_services.py`, `test_agent_distribution_concurrency.py`, `test_agent_distribution_contracts.py` (`test_ap4_services_forbid_runtime_store_introspection`)

**Next:** AP-5 (`AgentPackageTrust` coordinator).

## AGENT-PLATFORM-5 gate

**READY_FOR_CLOSE** (2026-08-13) - fail-closed `AgentPackageTrustCoordinator` with typed policy,
revocation evaluation, deterministic reason codes, digest-pinned trust records, and installation
verification gate.

| Deliverable | Status |
|-------------|--------|
| `AgentPackageTrustPolicy` / `AgentPackageTrustRevocationState` contracts | done |
| `AgentPackageTrustCoordinator` (`package_trust.py`) | done |
| `AgentPackageTrustDecision` + stable `AgentPackageTrustReasonCode` | done |
| `AgentInstallationTrustRecord.package_digest` digest-pinned trust authority | done |
| Required `evidence_package_digest` binding on ALLOW path | done |
| `assert_installation_trust_record_acceptable` digest mismatch gate | done |
| Focused AP-5 trust tests | done |
| AP-3/AP-4 regression (trust evidence refs on verification) | done |
| Tier-boundary check | done |

**Evidence:** `tests/unit/agent_distribution/test_agent_distribution_package_trust.py`

**Next:** AP-7 may begin.

## AGENT-PLATFORM-6 gate

**READY_FOR_CLOSE** (2026-08-13) - deterministic effective roster merge with durable default-agent override, secret-safe manifest defaults, L2 installed-agent requirement set, and L3 `CandidateDependencySpecification` builder under `intergrax/agent_distribution/`.

| Deliverable | Status |
|-------------|--------|
| `ManifestDefaultAgentDeclaration` neutral manifest input | done |
| `ApplicationAgentBinding.default_agent` durable override (`bool \| None`) | done |
| `BindingService.update_default_agent` revision-safe path | done |
| `EffectiveRosterBuilder` canonical §13.2 default merge | done |
| `EffectiveRosterEntry.effective_default_agent` | done |
| Shared `validate_non_secret_distribution_config` (binding + manifest) | done |
| `InstalledAgentRequirementSetBuilder` | done |
| `build_candidate_dependency_specification` (L1 + L2 → L3) | done |
| `AgentArtifactMetadata.agent_project_metadata_ref` | done |
| `CandidateDependencySpecification` release-id invariant | done |
| Focused AP-6 / AP-6-FIX-1 tests | done |
| AP-3/AP-4/AP-5 regression | done |
| Tier-boundary check | done |

**Evidence:** `tests/unit/agent_distribution/test_agent_distribution_effective_roster.py`, `tests/unit/agent_distribution/test_agent_distribution_dependency_specification.py`, `tests/unit/agent_distribution/test_agent_distribution_services.py`, `tests/unit/agent_distribution/test_agent_distribution_contracts.py`

**Next:** AP-7 may begin.

## AGENT-PLATFORM-7 gate

**READY_FOR_CLOSE** (2026-08-13) - deterministic `MaterializedRuntimeLock` producer, resolver port,
in-memory lock store, and candidate runtime graph simulation gates under `intergrax/agent_distribution/`.

| Deliverable | Status |
|-------------|--------|
| `DependencyResolver` port + `ResolvedDependencyClosure` | done |
| `MaterializedRuntimeLockProducer` / `MaterializedRuntimeLockService` | done |
| Deterministic package/agent closure canonicalization | done |
| `created_at` excluded from lock content digest | done |
| Content-addressed lock identity invariant (`lock_id == lock_digest == compute_lock_digest()`) | done |
| `InMemoryMaterializedRuntimeLockStore` canonicalization + idempotency | done |
| `AgentProjectMetadataProvider` port | done |
| `CandidateRuntimeGraphBuilder` + `CandidateRuntimeGraphValidator` | done |
| `CandidateApplicationRuntimeGraph` content-addressed digest | done |
| Neutral `intergrax/runtime_graph_semantics.py` shared by legacy + AP-7 | done |
| Focused AP-7 lock + graph tests | done |
| AP-3..AP-6 regression | done |
| Tier-boundary check | done |

**Evidence:** `tests/unit/agent_distribution/test_agent_distribution_runtime_lock.py`, `test_agent_distribution_runtime_graph.py`, `tests/unit/test_runtime_graph_semantics.py`

**Next:** AP-8 may begin (physical materialization).

## AGENT-PLATFORM-7-FIX-1 gate

**READY_FOR_CLOSE** (2026-08-13) - enforce true content-addressed lock identity and consolidate
runtime-graph semantics into neutral `intergrax/runtime_graph_semantics.py`.

| Deliverable | Status |
|-------------|--------|
| `MaterializedRuntimeLock` forged-identity fail-closed validation | done |
| Store canonicalization before persist (no caller-trusted IDs) | done |
| Shared dependency parsing / taxonomy / cycle formatting | done |
| Legacy `application_runtime_graph` consumes neutral helpers | done |
| AP-7 `runtime_graph_service` duplicate semantics removed | done |
| Lock + graph regression tests | done |
| Tier-boundary check | done |

**Evidence:** `intergrax/runtime_graph_semantics.py`, `tests/unit/test_runtime_graph_semantics.py`

**Next:** AP-8 may begin (physical materialization).

## AGENT-PLATFORM-8 gate

**READY_FOR_CLOSE** (2026-08-13) - topology-agnostic materialization coordinator, explicit adapters,
graph-authoritative staging, lock-driven OCI install manifest, digest-verified package artifact
staging (`.intergrax-artifacts/<digest>/`), and explicit VENV unsupported port.

| Deliverable | Status |
|-------------|--------|
| `ApplicationBuildContext` Tier-0 physical-build contract | done |
| `RuntimeMaterializationAdapter` explicit topology port | done |
| `RuntimeMaterializationService` consistency gates + output validation | done |
| Graph-authoritative `runtime_context_staging` helpers | done |
| `PackageArtifactProvider` + `MetadataBackedPackageArtifactProvider` artifact authority | done |
| Digest-verified `.intergrax-artifacts/` staging keyed by `package_digest` | done |
| Lock-driven `.intergrax-runtime-install.txt` from `MaterializedRuntimeLock` | done |
| Third-party production install requires `package_digest` + `--hash=` enforcement | done |
| OCI Dockerfile uses lock install manifest - not repository `uv.lock` authority | done |
| `DockerBuildRunner` explicit `image_ref` contract + RepoDigest/image-ID inspect | done |
| `FakeRuntimeMaterializationAdapter` deterministic test adapter | done |
| `OciImageMaterializationAdapter` production staging + docker boundary | done |
| `UnsupportedVenvBundleMaterializationAdapter` explicit deferral | done |
| `MaterializationLockArtifactLocationBlocked` fail-closed for missing artifact bytes | done |
| `MaterializationLockArtifactIdentityBlocked` fail-closed for missing package digest | done |
| AP-8 focused tests + AP-3..AP-7 regression | done |
| Tier-boundary check (no `applications/` imports) | done |

**Evidence:** `intergrax/agent_distribution/materialization_service.py`, `materialization_adapters.py`, `runtime_context_staging.py`, `package_artifact_provider.py`, `errors.py`, `tests/unit/agent_distribution/test_agent_distribution_materialization.py`, `test_agent_distribution_materialization_adapter.py`

**Next:** AP-11 delivered - generic V1 admin control plane. AP-12 is LKW consumer proof wiring.

### AP-10 evidence (registry projection)

| Item | Status |
|------|--------|
| `build_application_registry(..., effective_roster=...)` extension | done |
| `ApplicationRegistryProjectionCoordinator` + projection store | done |
| `RegistryProjectionEvidence` / audit snapshot fields | done |
| Legacy manifest-only behavior preserved | done |
| `RuntimeAgentFactoryResolver` port + in-memory test adapter | done |
| Revision-bound projection uses exact `(package_digest, factory_reference)` resolver | done |
| Host builders map is not production authority for AP-10 projection | done |
| `VenvBundleRuntimeAgentFactoryResolver` for `VENV_BUNDLE` artifact authority | done |
| `PRODUCTION_RUNTIME_FACTORY_ADAPTER_DEFERRED` | partial - `VENV_BUNDLE` implemented; OCI/sidecar deferred |
| AP-10 focused tests + AP-9 activation regression | done |

**Evidence:** `intergrax/applications/_shared/runtime_agent_factory_resolver.py`, `wiring.py`, `registry_projection.py`, `tests/unit/applications/test_registry_projection_ap10.py`

**Next:** AP-12 may begin after AP-11 close - LKW consumer proof wiring.

## AGENT-PLATFORM-11 gate

**READY_FOR_REVIEW** (2026-08-14) - generic V1 Agent Platform administration control plane:
typed `AgentPlatformAdminService` facade over AP-3..AP-10 services, shared FastAPI routes
under `/v1/agent-platform`, reused `require_harness_api_key` authorization, desired vs serving
read models, and enable/disable that never mutate the serving RuntimeRevision.

| Deliverable | Status |
|-------------|--------|
| `AgentPlatformAdminService` typed orchestration facade | done |
| Explicit V1 REST operations (no generic action dispatcher) | done |
| Environment scope `(application_id, application_environment_id)` | done |
| Desired-state vs serving-state response split | done |
| Build/apply → AP-6/7/8 candidate (no auto-activate) | done |
| Activate/rollback → AP-9 `ActivationService` CAS | done |
| Catalog list via injected `CatalogSourceProvider` | done (blocked 501 if missing) |
| Reused harness admin API-key boundary | done |
| Domain events preserved from delegated services | done |
| Secret-like binding config rejected via existing validators | done |
| Focused facade + HTTP tests + AP-9/AP-10 regression | done |

**Evidence:** `intergrax/agent_distribution/admin_service.py`, `admin_models.py`, `intergrax/applications/_shared/agent_platform_admin_routes.py`, `tests/unit/agent_distribution/test_agent_platform_admin_service.py`, `tests/unit/applications/test_agent_platform_admin_routes.py`

**Next:** AP-12 may begin - LKW consumer proof wiring of this admin control plane. Production topology factory loaders remain a separate host/runtime adapter.

### AP-9 evidence (activation orchestration)

| Item | Status |
|------|--------|
| `DeploymentInstanceState` + `DeploymentInstanceRecord` | done |
| `ApplicationEnvironmentServingRecord` CAS serving pointer | done |
| `RuntimeDeploymentAdapter` port + `FakeInMemoryRuntimeDeploymentAdapter` | done |
| `ActivationService` PREPARE → READY → COMMIT + drain + rollback | done |
| `RuntimeServingProjectionCoordinator` boundary + fake | done |
| AP-9 focused tests + AP-8 regression | pending validation |

**Evidence:** `intergrax/agent_distribution/activation.py`, `deployment.py`, `stores.py`, `in_memory_stores.py`, `tests/unit/agent_distribution/test_agent_distribution_activation.py`, `test_agent_distribution_rollback.py`

---

## AGENT-CONSOLIDATION-4 — Dynamic Capability Discovery & Acquisition

**Architecture (1:1):** [`architecture/AGENT_DISTRIBUTION.md` §35](../../architecture/AGENT_DISTRIBUTION.md#35-ac-4-dynamic-capability-discovery--acquisition-architecture-freeze)
**Status:** Phases 1–9 **CLOSED** · Phase 10 documentation freeze **COMPLETE**
**AC-4 closure:** **READY_FOR_FINAL_AUDIT** — independent audit required after Phase 10 commit; **AC-4 NOT CLOSED** until audit completes.

### Phase delivery record

| Phase | Deliverable | Module(s) | Status |
|-------|-------------|-----------|--------|
| **1** | Capability identity, requirement, deterministic matching | `capability_matching.py` | **CLOSED** |
| **2** | Pluggable discovery contracts + catalog projection | `agent_discovery.py`, `catalog.py` | **CLOSED** |
| **3** | Pluggable selection above matching | `agent_selection.py` | **CLOSED** |
| **4** | Federated discovery composition | `federated_discovery.py` | **CLOSED** |
| **5** | Task → capability resolution | `task_capability_resolution.py` | **CLOSED** |
| **6** | Dynamic acquisition → AC-3 lifecycle bridge | `dynamic_acquisition.py` | **CLOSED** |
| **7** | Task-scoped lease ownership | `task_scoped_agents.py` | **CLOSED** |
| **8** | Delegated subtask orchestration + child execution | `delegated_subtasks.py`, `active_execution_task_scope.py`, `active_task_registry.py`, `delegated_subtask_child_port.py` | **CLOSED** |
| **9** | Production composition wiring | `production_agent_capability_runtime.py`, `production_process_composition.py`, `production_delegated_subtask_plans.py`, `catalog_discovery.py` | **CLOSED** |
| **10** | Architecture documentation freeze | `architecture/AGENT_DISTRIBUTION.md` §35, this section | **COMPLETE** |

### AC-4 gate summary

| Deliverable | Status |
|-------------|--------|
| Capability model + matcher | done |
| Source-qualified discovery | done |
| Federated discovery | done |
| Pluggable selection | done |
| Task capability resolution (deterministic reference) | done |
| Dynamic acquisition (exact identity, no rediscovery) | done |
| Task-scoped / ephemeral ownership leases | done |
| Delegated subtask discovery/execution | done |
| Production composition (single AC-3 universe) | done |
| Canonical AC-3 lifecycle handoff | done |
| Reference Production V1 limitations documented | done |
| Platform proof E2E | done |

**Evidence (by phase):** `tests/unit/agent_distribution/test_capability_matching.py`, `test_agent_discovery.py`, `test_agent_selection.py`, `test_federated_discovery.py`, `test_task_capability_resolution.py`, `test_dynamic_acquisition.py`, `test_task_scoped_agents.py`, `test_delegated_subtasks.py`, `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py`, `tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py`, `tests/unit/runtime/task/test_active_task_registry.py`

### Explicit non-goals (AC-4 — unchanged)

- Public commercial marketplace product
- Remote publisher onboarding backend
- Billing / settlement / usage accounting
- Durable multi-instance lease recovery
- Generic universal `Registry → Agent.run` adapter
- LLM semantic capability resolver (protocol extension only)
- Second lifecycle or trust engine

### Historical implementation waves (AP-3..AP-12)

The AP-3+ table above remains the **historical delivery chronology** for AC-3 lifecycle infrastructure. AC-4 (AGENT-CONSOLIDATION-4) is a **separate consolidation program** layered on AC-3 — do not merge gate records.

**Stale "Next" notes (historical):** Prior gate sections retain their original **Next:** lines as delivery chronology. Current follow-on for marketplace productization and LKW consumer proof (AP-12) remains open; AC-4 platform proof does not replace AP-12 LKW wiring.

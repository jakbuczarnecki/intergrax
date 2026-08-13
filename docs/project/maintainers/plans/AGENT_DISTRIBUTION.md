# Agent Distribution and Management — Plan

**Status:** Active (architecture frozen — AGENT-PLATFORM-2 + ARCH-AGENT-ACTIVATION-1)  
**Architecture (1:1):** [`architecture/AGENT_DISTRIBUTION.md`](../../architecture/AGENT_DISTRIBUTION.md)  
**ADR:** [`adr/entries/2026-08-12/ADR-AGENT-004.md`](../../technical/adr/entries/2026-08-12/ADR-AGENT-004.md)  
**Evidence:** [`audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md`](../audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md)  
**Last updated:** 2026-08-13

---

## Goal

Implement the Tier-0 Agent Distribution domain so operators can discover, install, bind, configure, enable, upgrade, rollback, and uninstall agents **without** hot-loading Python, **without** LKW-local stores, and **without** breaking deterministic runtime graphs or Nexus capability routing.

## Architecture delivery (AGENT-PLATFORM-2) — Done

| Item | Status |
|------|--------|
| Canonical `AGENT_DISTRIBUTION.md` | done |
| Deterministic `MaterializedRuntimeLock` model | done (architecture) |
| `RuntimeRevision` + activation semantics | done (architecture — ARCH-AGENT-ACTIVATION-1 frozen) |
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
| AP-9 | `RuntimeRevision` activation + rollback orchestration (implements §20 zero-downtime model) | in progress |
| AP-10 | `build_application_registry` extension + snapshot fields | AP-9 |
| AP-11 | Generic Tier-3 harness admin API routes | AP-4..AP-9 |
| AP-12 | LKW consumer proof wiring | AP-11 |

## Non-goals (program)

- Marketplace billing, reviews, publisher portal
- Second Nexus or registry
- Runtime hot-load
- LKW-specific installer or persistence
- Mandating Docker for all topologies

## Verification intent (future)

- Digest-pinned install → lock → graph → activate → registry → routable capability (LKW proof)
- Rollback restores prior `RuntimeRevision` and lock digest
- Catalog outage does not affect active revision reproducibility
- Concurrent install/upgrade serialization on installation slot

## AGENT-PLATFORM-3 gate

**READY_FOR_CLOSE** (2026-08-12) — Tier-0 contracts and store ports under `intergrax/agent_distribution/`; AP-3-FIX-1 hardened namespace alignment, deep immutability, recursive secret rejection, typed ports, and shared digest validation.

| Deliverable | Status |
|-------------|--------|
| `AgentPackageIdentity` / catalog contracts | done |
| `CatalogSourceProvider` port | done |
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

**READY_FOR_CLOSE** (2026-08-13) — transactional domain services with explicit store-port atomic operations; no runtime capability discovery in AP-4 services.

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

**READY_FOR_CLOSE** (2026-08-13) — fail-closed `AgentPackageTrustCoordinator` with typed policy,
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

**READY_FOR_CLOSE** (2026-08-13) — deterministic effective roster merge with durable default-agent override, secret-safe manifest defaults, L2 installed-agent requirement set, and L3 `CandidateDependencySpecification` builder under `intergrax/agent_distribution/`.

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

**READY_FOR_CLOSE** (2026-08-13) — deterministic `MaterializedRuntimeLock` producer, resolver port,
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

**READY_FOR_CLOSE** (2026-08-13) — enforce true content-addressed lock identity and consolidate
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

**READY_FOR_CLOSE** (2026-08-13) — topology-agnostic materialization coordinator, explicit adapters,
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
| OCI Dockerfile uses lock install manifest — not repository `uv.lock` authority | done |
| `DockerBuildRunner` explicit `image_ref` contract + RepoDigest/image-ID inspect | done |
| `FakeRuntimeMaterializationAdapter` deterministic test adapter | done |
| `OciImageMaterializationAdapter` production staging + docker boundary | done |
| `UnsupportedVenvBundleMaterializationAdapter` explicit deferral | done |
| `MaterializationLockArtifactLocationBlocked` fail-closed for missing artifact bytes | done |
| `MaterializationLockArtifactIdentityBlocked` fail-closed for missing package digest | done |
| AP-8 focused tests + AP-3..AP-7 regression | done |
| Tier-boundary check (no `applications/` imports) | done |

**Evidence:** `intergrax/agent_distribution/materialization_service.py`, `materialization_adapters.py`, `runtime_context_staging.py`, `package_artifact_provider.py`, `errors.py`, `tests/unit/agent_distribution/test_agent_distribution_materialization.py`, `test_agent_distribution_materialization_adapter.py`

**Next:** AP-10 may begin after AP-9 close — registry projection population from traffic-serving `RuntimeRevision`.

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

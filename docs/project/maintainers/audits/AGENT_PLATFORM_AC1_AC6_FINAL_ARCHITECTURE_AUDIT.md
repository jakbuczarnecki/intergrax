# Agent Platform AC-1–AC-6 Final Architecture Audit

**Audit ID:** FINAL-AGENTS-ARCHITECTURE-AUDIT-AC1-AC6  
**Date:** 2026-09-06  
**Branch:** `development`  
**Audited SHA:** `67f57331835c247716ff028c0029157ab3f53207`  
**origin/development at audit:** `67f57331835c247716ff028c0029157ab3f53207`  
**Mode:** Independent evidence-based architecture gate (read/trace/test; no production code changes)

**Canonical architecture:** [`architecture/AGENT_DISTRIBUTION.md`](../../architecture/AGENT_DISTRIBUTION.md)  
**Prior closure:** [STAGE_16_AGENT_ARCHITECTURE_AUDIT.md](STAGE_16_AGENT_ARCHITECTURE_AUDIT.md) · [AGENT_PLATFORM_FINAL_CLOSURE.md](AGENT_PLATFORM_FINAL_CLOSURE.md)

---

## Architecture verdict

**AGENT PLATFORM AC-1–AC-6 ARCHITECTURE: FROZEN — Reference Production V1**

**READY FOR APPLICATION-INTEGRATION-AND-PROOF-VALIDATION:** YES

No architecture blockers discovered. No STOP conditions triggered.

---

## AC phase matrix

| AC | Capability | Authority | Production implementation | Tests | Docs | Known limitations | Verdict |
|----|------------|-----------|---------------------------|-------|------|-------------------|---------|
| **AC-1** | Foundational lifecycle/domain contracts (identity, orthogonal states, store ports) | Tier-0 contracts + `intergrax/agent_distribution/` models | `identity.py`, `catalog.py`, `trust.py`, `stores.py`, `installation_service.py`, `binding_service.py` | `tests/unit/agent_distribution/test_*` (contracts, installation, binding) | `AGENT_DISTRIBUTION.md` §3–§9; ADR-AGENT-004 | Process-local reference stores; no distributed durability | **CLOSED** |
| **AC-2** | Distribution ↔ application binding boundaries; no central inventory bypass | `BindingService`, `InstallationService`, manifest/roster separation | `binding_service.py`, `effective_roster.py`, `wiring.build_application_registry` revision-bound path | `test_registry_projection_ap10.py`, binding/roster unit tests | `AGENT_DISTRIBUTION.md` §11–§14; `AGENT_CONTRACTS_AND_ASSEMBLY.md` | Lab manifest assembly via explicit `build_manifest_development_registry` only | **CLOSED** |
| **AC-3** | Production lifecycle authority (N/N+1, rollback, historical replay) | `ProductionProcessComposition` → `ActivationService` / `RuntimeRevisionService` | `production_process_composition.py`, `production_host_composition.py`, `activation.py`, `admin_service.py` | `test_ac3_phase5_production_lifecycle_e2e.py`, `test_canonical_agent_lifecycle_e2e.py`, activation/rollback suites | `AGENT_DISTRIBUTION.md` §20, §34; ADR-AGENT-005/006/007 | Single-process V1; in-memory stores; no distributed consensus | **CLOSED** |
| **AC-4** | Capability plane: discover → match → select → acquire → delegate | `TaskCapabilityResolver`, `DiscoveryStrategy`, `CapabilityMatcher`, `SelectionStrategy`, `DynamicAcquisitionService` | `dynamic_acquisition.py`, `production_agent_capability_runtime.py`, `delegated_subtasks.py` | `test_ac4_phase9_production_composition_e2e.py`, `test_dynamic_acquisition.py`, `test_delegated_subtasks.py`, `test_task_scoped_agents.py` | `AGENT_DISTRIBUTION.md` §35 | Billing/settlement not implemented; marketplace optional source only | **CLOSED** |
| **AC-5** | Canonical revision-bound factory invocation | `RuntimeAgentFactoryResolver` → `invoke_canonical_agent_factory` | `runtime_agent_factory_resolver.py`, `venv_bundle_runtime_agent_factory_resolver.py`, `registry_projection.py` | `test_ac5_phase3_factory_e2e.py`, `test_production_factory_invocation_ac5.py` | `AGENT_DISTRIBUTION.md` §21 (AC-5 freeze) | OCI/sidecar topologies deferred; `VENV_BUNDLE` implemented | **CLOSED** |
| **AC-6** | Trust/certification/revocation/freshness/active emergency response | `AgentPackageTrustCoordinator` (sole ALLOW/DENY) | `package_trust.py`, `emergency_revocation_response.py`, Ed25519 attestation adapters | `test_ac6_trust_lifecycle_e2e.py`, `test_ac6_architecture_gates.py`, `test_emergency_revocation_response.py` | `AGENT_DISTRIBUTION.md` §10.4–§10.7 | No background revocation watcher; no Sigstore; process-local snapshot | **CLOSED** |

---

## Canonical lifecycle (frozen chain)

```text
DISCOVER → RESOLVE → VERIFY → TRUST → INSTALL → BIND → ENABLE
  → SNAPSHOT → LOCK → MATERIALIZE → VALIDATE → PROJECT
  → PREPARE → READY → COMMIT → ACTIVE → ROUTE → DRAIN/SUPERSEDE
```

Dynamic acquisition enters at **DISCOVER/RESOLVE** and converges through the same install/bind/build/activate chain (`dynamic_acquisition.py` → `AgentPlatformAdminService`).

---

## Authority matrix

| Concern | Sole authority |
|---------|----------------|
| Catalog discovery | `CatalogSourceProvider` / `CatalogSourceProviderRegistry` |
| Capability discovery | `DiscoveryStrategy` / federation |
| Functional match | `CapabilityMatcher` |
| Selection | `SelectionStrategy` |
| Trust ALLOW/DENY | `AgentPackageTrustCoordinator` |
| Crypto authenticity | `AgentPackageAttestationVerifier` (+ publisher key provider) |
| Installation mutation | `InstallationService` (via admin facade) |
| Binding mutation | `BindingService` |
| Effective roster snapshot | `EffectiveRosterAuthorityService` |
| Dependency closure / lock | canonical resolver → `MaterializedRuntimeLock` |
| Materialization | `RuntimeMaterializationService` |
| Factory resolution | `RuntimeAgentFactoryResolver` |
| Registry projection | `RegistryProjectionAuthority` / revision-bound `build_application_registry` |
| Traffic activation | `ActivationService` |
| Runtime capability routing | Nexus |
| Active revocation response | `AgentEmergencyRevocationService` over trust + `ActivationService.rollback()` |
| Process composition root | `ProductionProcessComposition` |

### Negative authority (frozen)

| Not authority | Meaning |
|---------------|---------|
| Marketplace | ≠ trust, ≠ runtime, ≠ lifecycle |
| AgentRegistry | ≠ installation, ≠ activation |
| Nexus | ≠ installation |
| Application / scenario | ≠ private lifecycle (private agents OK) |
| Manifest | ≠ active runtime authority in STRICT production |
| Historical trust record | ≠ current ALLOW |
| Valid signature | ≠ ALLOW |
| Qualification | ≠ ALLOW |
| Installed | ≠ admissible |
| Enabled | ≠ ACTIVE |
| ACTIVE revision | ≠ safe forever |

---

## Store ownership (AC-3)

| Store | Owner service | Mutation authority | Semantics | Duplicate? |
|-------|---------------|-------------------|-----------|------------|
| Installation | `InstallationService` | `persist_installation` via service only | process-local durable (in-memory ref impl) | No |
| Binding | `BindingService` | binding store via service | same | No |
| Effective roster snapshot | `EffectiveRosterAuthorityService` | immutable snapshots | historical authority | No |
| Runtime revision | `RuntimeRevisionService` | candidate/validated/active transitions | revision-bound | No |
| Materialized lock | lock producer + store | deterministic from roster | immutable per revision | No |
| Materialization | `RuntimeMaterializationService` | artifact records | immutable | No |
| Deployment instance | `ActivationService` / adapter | PREPARE/COMMIT/DRAIN | per revision instance | No |
| Serving pointer | `ActivationService` | `atomic_commit_activation` / `atomic_commit_rollback` CAS | traffic authority | No |
| Registry projection | projection coordinator | at traffic commit | derived | No |
| Task-scoped lease | `TaskScopedAgentLeaseStore` | lease acquire/release | task overlay; lifecycle remains platform-owned | No |

---

## Bypass audit

| Vector | Finding | Classification |
|--------|---------|----------------|
| **AgentRegistry production construction** | Only in `build_application_registry` (revision-bound projection), `build_manifest_development_registry` (explicit lab), `build_scenario_lab_agent_registry` (scenario lab), tests | A derived / B lab — no D suspicious production |
| **Trust bypass** | `AgentPackageTrustCoordinator` referenced from admin, dynamic acquisition, emergency revocation; AST gates forbid parallel trust engines | PASS |
| **Factory bypass** | Revision-bound path requires `RuntimeAgentFactoryResolver`; `invoke_canonical_agent_factory` strict `(ctx, binding)`; legacy probing isolated to `invoke_legacy_compatible_agent_factory` | PASS |
| **Activation bypass** | Serving mutations only in `ActivationService` + `InMemoryApplicationEnvironmentActivationStore.atomic_commit_*` | PASS |
| **Dynamic acquisition** | `DynamicAcquisitionService` delegates install/bind/build/activate to admin lifecycle | PASS |
| **Application/scenario** | Production apps use `bootstrap_production_registry_projection` + `ProductionProcessComposition`; lab demos use `build_manifest_development_registry`; scenario baseline explicitly lab | PASS |
| **Marketplace** | Tool/skill registry registrations in `platform_proofs` — not agent lifecycle | PASS |
| **Core → concrete imports** | One CLI lazy import (`intergrax/cli/external_work.py` → governed contractor boundary); no `intergrax/` production import of concrete `agents.*` or `applications.*` host paths | PASS (CLI boundary only) |

---

## Historical authority / N+1 / rollback

- Historical reconstruction path: `runtime_revision_id` → `RuntimeRevision` → roster snapshot → lock → materialization → projection — verified by `test_phase5_historical_n_resolvable_after_n_plus_one_commit` and canonical lifecycle E2E.
- N serves while N+1 prepares; failed N+1 does not mutate N (`test_phase5_failed_prepare_preserves_active_n`).
- Rollback uses exact prior revision with current trust revalidation (`AgentEmergencyRevocationService`, `test_emergency_revocation_response.py`).

---

## Layering matrix

| Layer | Production core imports concrete apps/agents/scenarios? |
|-------|------------------------------------------------------|
| `intergrax/agent_distribution/` | No |
| `intergrax/applications/_shared/` production composition | No (`test_production_composition_architecture_gate.py`) |
| `intergrax/runtime/` (agent paths) | No concrete Tier-2/3 |
| Applications STRICT hosts | Consume platform via `ProductionProcessComposition` |

---

## Test evidence matrix

| Property | Test(s) | Boundaries crossed | Status |
|----------|---------|-------------------|--------|
| Canonical lifecycle | `test_canonical_agent_lifecycle_e2e.py` | full Tier-0 → projection | PASS |
| AC-3 production lifecycle | `test_ac3_phase5_production_lifecycle_e2e.py` (8 tests) | composition, activation, Nexus | PASS |
| N/N+1 safety | `test_phase5_n_plus_one_switch_preserves_n_until_commit` | activation CAS | PASS |
| Rollback | `test_agent_distribution_rollback.py`, emergency revocation suite | trust + activation | PASS |
| Dynamic acquisition | `test_dynamic_acquisition.py`, AC-4 Phase 9 E2E | discovery → lifecycle | PASS |
| Task-scoped agents | `test_task_scoped_agents.py` | lease overlay on platform lifecycle | PASS |
| Delegated subtasks | `test_delegated_subtasks.py`, AC-4 Phase 9 | platform delegation port | PASS |
| Canonical factory | `test_ac5_phase3_factory_e2e.py`, `test_production_factory_invocation_ac5.py` | revision-bound resolver | PASS |
| Crypto trust / freshness | AC-6 unit + `test_ac6_trust_lifecycle_e2e.py` | attestation → coordinator | PASS |
| Active revocation | `test_emergency_revocation_response.py` | serving → rollback | PASS |
| Architecture gates | `test_ac6_architecture_gates.py`, `test_canonical_agent_lifecycle_architecture_gate.py`, `test_production_composition_architecture_gate.py`, `test_agent_lifecycle_bypass_ast.py` | AST/static enforcement | PASS |
| Cross-layer acceptance | `test_agent_platform_cross_layer_acceptance.py` | Tier-0 ↔ Tier-3 | PASS |

**Test runs (audited SHA):**

```text
tests/unit/agent_distribution/ + tests/integration/agent_distribution/
+ AC-3/4/5/6 E2E + architecture gates + extended proofs
→ 182 passed, 0 failed (4 pydantic warnings in AC-3 E2E only)
```

Log: `.tmp/session/final-agents-audit-ac1-ac6/pytest-agent-platform.log`, `pytest-extended.log`

---

## Known debt (non-blocking)

| Item | Evidence | Blocks freeze? |
|------|----------|----------------|
| AC-3 diagnostic `RuntimeEvent persistence unavailable` (historical) | **Not reproduced** on current `development`; all 8 `test_ac3_phase5_*` tests PASS | **NO** — verified unrelated baseline debt / resolved |
| AP-12 LKW consumer proof wiring | Plan open | NO — product integration stage |
| OCI/sidecar factory topologies | Deferred per AC-5 docs | NO — explicit non-claim |
| AGSYS-IDENTITY-PROJECTION (audit remediation) | Planned; bootstrap vs activated distinction documented | NO — does not reopen lifecycle authority |
| `intergrax/cli/external_work.py` lazy app import | CLI-only; not production host path | NO |
| Pydantic `BudgetReactionProfile` serialization warnings | AC-3 E2E warnings only | NO |

---

## Reference Production V1 limits (explicit non-claims)

- Single process; process-local stores where documented
- Restart may lose in-memory state
- No distributed consensus / leader election
- No full commercial marketplace
- No universal remote agent execution topology
- No arbitrary historical safe revision search
- No zero-serving quarantine unless explicitly implemented
- No Sigstore transparency log
- No background revocation watcher

---

## Final architecture score

| Category | Score |
|----------|-------|
| Lifecycle authority | PASS |
| Runtime immutability | PASS |
| Historical replay | PASS |
| Trust/security | PASS |
| Dynamic acquisition | PASS |
| Factory authority | PASS |
| Activation safety | PASS |
| Registry projection | PASS |
| Layering | PASS |
| Pluginability | PASS |
| Modularity | PASS |
| Type contracts | PASS |
| Test evidence | PASS |
| Documentation consistency | PASS WITH DEBT → corrected in this audit |

---

## Next stage

**APPLICATION-INTEGRATION-AND-PROOF-VALIDATION** — use frozen canonical platform from real applications/scenarios without architecture changes. New platform gaps get bounded follow-on tasks; no AC-7 lifecycle phase.

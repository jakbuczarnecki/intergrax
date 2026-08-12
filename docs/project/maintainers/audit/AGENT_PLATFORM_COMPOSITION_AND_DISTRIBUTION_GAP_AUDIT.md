# AGENT-PLATFORM-0 — Enterprise Agent Platform, Application Composition and Distribution Gap Audit

**Task:** AGENT-PLATFORM-0
**Status:** Discovery / architecture evidence gate
**Branch:** `development`
**Date:** 2026-08-12
**Scope:** Tier-2 Agents platform · Tier-3 application consumption · LKW as primary proof workload
**Non-goals:** Marketplace design · production install mechanisms · LKW/product architecture changes

---

## 1. Executive conclusion

Intergrax already has a **production-grade agent execution and governance stack** at Tier-1–2: `AgentContract`, in-process `AgentRegistry`, capability-based Nexus routing, ACP/UAEP step loop, lifecycle/certification evaluators, capability graph, and declarative `ApplicationManifest` / `AgentBinding` composition. Agents are correctly packaged as independent Tier-2 workspace distributions (`intergrax-*-agent`) and consumed by Tier-3 via `pyproject.toml` + manifest roster + `build_application_registry`.

What **does not exist** is a neutral **distribution and installation plane** that separates:

- package/catalog identity from runtime contract,
- installation state from application binding,
- durable operator-managed roster from static Python manifests,
- catalog source providers (built-in, org, marketplace) from Nexus execution.

Today, **install an agent = declare a Python package dependency and rebuild the application runtime graph / image.** `AgentBinding.enabled` is a static manifest flag, not a persisted operator toggle. `AgentRegistry` is an ephemeral runtime index, not an installation store.

**LKW proves agent execution and multi-agent orchestration strongly** but **cannot prove** install → configure → enable → disable → upgrade/rollback → uninstall without new platform mechanisms. Its `GET /agents` surface is read-only introspection of the already-wired registry.

**Verdict:** **Architecture decision gate required before implementation.** Extend existing registries, manifests, runtime graph, and Platform Plugin qualification patterns — do **not** add a second Nexus, LKW-specific agent store, or marketplace execution path.

---

## 2. Current architecture map

```text
Tier-0  intergrax/          tools · skills · integrations · prompts · core/plugins
Tier-1  intergrax/runtime/  NexusLoop · AgentRegistry · capability graph · lifecycle evaluators
Tier-2  agents/<slug>/      Agent classes · AgentContract · pyproject.toml packages
Tier-3  applications/<app>/ ApplicationManifest · AgentBinding · host factories · serving

Composition path (today):
  agents/<slug>/pyproject.toml
    → applications/<app>/pyproject.toml (direct agent deps)
    → ApplicationRuntimeGraph (transitive Tier-2 closure)
    → build_application_image.py (minimal Docker context + .intergrax-runtime-graph.json)
    → manifest.py (AgentBinding.mount + factories)
    → wire_application_environment → ApplicationBuildContext
    → build_application_registry → AgentRegistry.register (per enabled binding)
    → NexusLoop(registry) → find_by_capability / TaskClassifier → agent.run()

Routing path:
  Task.context.capability
    → AgentRegistry.find_by_capability(capability, production_mode=…)
    → evaluate_agent_routing(contract) filters lifecycle/ownership
    → Nexus task planner / handoff / graph runner
```

**Canonical references:** [`agents/README.md`](../../../agents/README.md) · [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §12–§21 · [`APPLICATION_RUNTIME_GRAPH_MODEL.md`](../../architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md) · [`APPLICATION_DEPENDENCY_MODEL.md`](../../architecture/APPLICATION_DEPENDENCY_MODEL.md)

---

## 3. Existing capabilities inventory

| Capability | Location | Role |
|------------|----------|------|
| Agent metadata contract | `intergrax/contracts/agent_contract_meta.py` · `AgentContract` | Declarative id, capabilities, skills, budgets, lifecycle, ownership |
| Agent base + run facade | `intergrax/agents/agent_contract.py` · §13 ACP hub | `run()` → step loop / UAEP |
| Runtime agent index | `intergrax/runtime/registry/agent_registry.py` · `AgentRegistry` | Register instances, capability lookup, routing policy |
| Assembly validation | `intergrax/runtime/registry/agent_assembly_resolver.py` | Register-time contract validation |
| Routing policy | `intergrax/runtime/registry/agent_routing_policy.py` | Lifecycle + production_eligible gates |
| Capability routing | `intergrax/runtime/nexus/task_classifier.py` · `task_planner.py` · `handoff/coordinator.py` | `find_by_capability` |
| Tier-3 composition | `intergrax/applications/contracts/manifest.py` | `ApplicationManifest`, `AgentBinding` |
| Materialization | `intergrax/applications/_shared/wiring.py` | `build_application_registry`, `build_agent_from_binding` |
| Host runtime | `intergrax/applications/_shared/harness_host_runtime.py` | Single path: env → registry → NexusLoop |
| Runtime graph | `intergrax/applications/_shared/application_runtime_graph.py` | Transitive Tier-2 closure from pyproject |
| Image isolation | `scripts/build/build_application_image.py` (per arch doc) | Minimal context, graph manifest v2 |
| Lifecycle governance | `intergrax/runtime/architecture/agent_lifecycle_governance.py` | Deprecation/retirement contracts |
| Certification / promotion | `agent_certification.py`, `agent_promotion.py` | Evidence bundles, gate evaluators |
| Tier-3 certification wiring | `intergrax/applications/_shared/agent_certification_wiring.py` | STRICT roster certification materialization |
| Agent governance profile | `intergrax/applications/contracts/agent_governance.py` | `AgentCertificationRecord`, approval policy |
| Capability graph | `intergrax/runtime/architecture/capability_graph.py` + app wiring | Blast-radius, deploy gates |
| Registry snapshots | `intergrax/applications/_shared/registry_snapshot_store.py` | Durable id-set audit (not install state) |
| Platform Plugin package contract | `intergrax/core/plugins/package_contract.py` | Plugin manifest, compatibility, qualification |
| Scaffold agent catalog | `intergrax/scaffold/agent_catalog.py` | Code-gen reference list (not runtime catalog) |
| Fleet inventory / CI gates | `agents/README.md` · `scripts/gates/check_agent_production_readiness.py` | Production readiness scoreboard |

---

## 4. Current agent lifecycle

| Stage | Mechanism today | Persistence |
|-------|-----------------|-------------|
| Authoring | Tier-2 package + `AgentContract` on agent class | Source repo |
| Package declare | `agents/<slug>/pyproject.toml` workspace member | Monorepo / uv.lock |
| Application select | `applications/<app>/pyproject.toml` direct deps | Build metadata |
| Manifest bind | `AgentBinding.mount` in `manifest.py` | Static Python module |
| Certification evidence | `AgentCertificationRecord` in `AgentGovernanceProfile` | Environment profile (deploy-time) |
| Promotion eval | `evaluate_agent_promotion` | CI / release process |
| Runtime register | `AgentRegistry.register` at host startup | In-memory only |
| Routing filter | `AgentLifecycleState` + `evaluate_agent_routing` | Contract fields on registered agent |
| Deprecation / retirement | `agent_lifecycle_governance.py` + routing policy | Contract state; CI metadata gates |
| Uninstall | Remove pyproject dep + rebuild image | Deploy pipeline |

**Gap:** No platform-owned **installation record** or **operator-driven enable/disable** independent of code deploy. Lifecycle fields live on `AgentContract` and governance profiles, not on a durable per-application installation table.

---

## 5. Application composition model

### 5.1 `ApplicationManifest`

- Stable `app_id`, profile (`LAB` / `PRODUCT`), route/env prefixes, integration profile, optional `ApplicationEnvironmentProfile`, ownership metadata.
- `agents: list[AgentBinding]` — full roster.
- `enabled_agents()`, `default_agent()`, `require_enabled_agents()` — static roster helpers.

**Evidence:** `intergrax/applications/contracts/manifest.py` — `ApplicationManifest`, `AgentBinding`.

### 5.2 `AgentBinding`

| Field | Purpose |
|-------|---------|
| `agent_type` / `import_path` | Tier-2 class reference |
| `factory` / `factory_path` / `builder_key` | Tier-3 instantiation |
| `config` | Lightweight binding options (not secrets) |
| `contract_id` | Optional registry id override |
| `capabilities` | Documented routing hints (not enforced at binding layer) |
| `enabled` | Static roster inclusion |
| `default` | Product default agent |
| `memory_scope_override`, `rag_collection_override` | Per-roster resource binding |
| `tool_allowlist_extra`, `tool_denylist` | Per-roster tool policy |
| `org_role_id`, `budget_slice` | Org policy + token budget |

`AgentBinding.reference(contract_id=…)` supports harness catalogs without importing agent classes.

### 5.3 Materialization order (`build_agent_from_binding`)

1. Typed `factory` on binding
2. `builders[agent_type]` map
3. Legacy `factory_path` import
4. Zero-arg `agent_type()`

Then `AgentRegistry.register` resolves skills/tools into `allowed_tools`.

---

## 6. LKW consumption map

| Agent | Package | Capability | Manifest / factory | Execution |
|-------|---------|------------|-------------------|-----------|
| `local_indexer` | `intergrax-local-indexer-agent` | `local.workspace.index` | `manifest.py` → `build_local_workspace_local_indexer_from_context` | Nexus task / pipeline |
| `local_search` | `intergrax-local-search-agent` | `local.workspace.search` (default) | `build_local_workspace_local_search_from_context` | Primary ask/search path |
| `local_synthesizer` | `intergrax-local-synthesizer-agent` | `local.workspace.synthesize` | `build_local_workspace_local_synthesizer_from_context` | Shadow artifact drafts |

**Wiring chain:**

1. `LOCAL_WORKSPACE_APPLICATION_MANIFEST` — `applications/local_workspace_application/manifest.py`
2. `create_local_workspace_backend_app` — `host/factory.py` → `build_harness_host_runtime(manifest, env)`
3. `build_application_registry` — enabled bindings only
4. `LocalWorkspaceTaskExecutor` — Nexus-backed runs
5. `GET /v1/local_workspace/agents` — `nexus_loop.registry.list_agent_ids()` (read-only)

**Product architecture:** LKW ARCHITECTURE §5–§6 documents four-layer stack and agent roster; composition is manifest + environment profile, not LKW-local agent registry.

---

## 7. Static vs dynamic composition analysis

| Concern | Static (build/deploy time) | Dynamic (runtime) |
|---------|---------------------------|-------------------|
| Agent package on filesystem / image | `pyproject.toml` + `ApplicationRuntimeGraph` + Docker context | **Not supported** |
| Transitive agent closure | Graph resolver from lockfile | — |
| Roster membership | `manifest.agents` Python module | — |
| Binding `enabled` | Manifest field (requires redeploy to change) | No hot toggle API |
| Factory / config | `AgentBinding.config`, env profile | Per-request task context only |
| `AgentRegistry` contents | Built once at startup | In-memory; no add/remove API |
| Certification records | `AgentGovernanceProfile` on environment | — |
| Capability routing | Registry snapshot at startup | Per-task capability on `Task` |
| Registry snapshot audit | `SqliteRegistrySnapshotStore` | Post-materialization capture |

**Conclusion:** Application→agent composition is **overwhelmingly static**. Runtime dynamism is limited to Nexus task routing among **already registered** agents.

---

## 8. Enterprise-management capability matrix

| Capability | Status | Notes |
|------------|--------|-------|
| Reusable Tier-2 agent packages | **EXISTS** | `agents/<slug>/pyproject.toml`, tier boundaries enforced |
| Declarative application roster | **EXISTS** | `ApplicationManifest` + `AgentBinding` |
| Capability-based Nexus routing | **EXISTS** | `find_by_capability`, §16 routing invariant |
| Agent contract metadata | **EXISTS** | `AgentContract` superset of §12 |
| Runtime registration | **EXISTS** | `AgentRegistry` — scope is execution index |
| Lifecycle states | **EXISTS** | `AgentLifecycleState` + routing policy |
| Certification / promotion evaluators | **EXISTS** | Release/CI oriented |
| Ownership / on-call on contract | **EXISTS** | `owner_team`, `on_call_contact`, runbook |
| Per-agent budgets / memory / tools | **EXISTS** | Binding overrides + contract + skill resolution |
| Observability (dual planes) | **EXISTS** | §31 architecture; harness wiring |
| Minimal runtime graph / image isolation | **EXISTS** | `APPLICATION_RUNTIME_GRAPH_MODEL` |
| Transitive agent dependency resolution | **EXISTS** | `application_runtime_graph.py` |
| Registry snapshot audit | **PARTIAL** | Id sets only; not install roster |
| Operator enable/disable without redeploy | **MISSING** | `enabled` is manifest-static |
| Durable installed-agent persistence | **MISSING** | No installation store |
| Neutral agent catalog abstraction | **MISSING** | Scaffold list ≠ catalog |
| Catalog source providers (org/marketplace) | **MISSING** | Platform Plugin is not agent catalog |
| Package install/upgrade/rollback at platform layer | **PARTIAL** | uv + image rebuild; not agent-specific API |
| Agent package signing / publisher metadata | **MISSING** | Plugin qualification partial analog |
| Runtime agent install without image rebuild | **MISSING** | Would break graph isolation |
| LKW install/configure/lifecycle APIs | **MISSING** | Read-only `/agents` only |
| Marketplace as routing source only | **NOT REQUIRED** (yet) | Requires catalog provider abstraction first |
| Second AgentRegistry for execution | **NOT REQUIRED** | Extend existing registry + installation plane |

---

## 9. Marketplace-readiness analysis

### 9.1 Can the platform eventually support multiple agent sources?

| Source | Feasibility with current stack | Blocker |
|--------|-------------------------------|---------|
| Built-in Intergrax agents | **Yes today** | Monorepo workspace packages |
| Organization / private catalog | **Partial** | No catalog API; private PyPI + pyproject is workaround |
| Local / developer package | **Yes today** | Workspace / path deps + manifest |
| Official Intergrax marketplace | **No** | No catalog entry + install record model |
| Governed third-party source | **Partial** | Platform Plugin trust model exists for **plugins**, not Tier-2 agents |

### 9.2 Neutral abstractions required (marketplace = one provider)

```text
AgentPackageIdentity          # distribution name + version (pyproject / wheel)
AgentCatalogEntry             # display metadata, publisher, categories, compatibility
AgentInstallationRecord       # installed version, source URI, install timestamp, health
ApplicationAgentBinding       # app_id + agent_id + binding config (extends AgentBinding semantics)
BindingConfigurationState     # validated config blob, secrets refs
BindingEnablementState        # operator toggle independent of package presence
RuntimeRegistrationView       # materialized AgentRegistry slice (derived, not authoritative)
CatalogSourceProvider         # built-in | org_registry | marketplace | local_path
```

Marketplace-specific UI and billing stay **outside** Nexus; only `CatalogSourceProvider` feeds installation records.

### 9.3 Reuse vs new

- **Reuse:** `AgentContract`, `AgentBinding` fields, `AgentCertificationRecord`, `PlatformPluginTrustModel` / qualification evidence shapes, `ApplicationRuntimeGraph`, capability graph deploy gates.
- **New (GAP):** installation persistence, catalog entry model, source provider interface, materialization bridge from installation record → `build_application_registry` input.

---

## 10. Reuse map — extend, do not duplicate

| Mechanism | Extend for agent distribution | Do not duplicate |
|-----------|------------------------------|------------------|
| `AgentRegistry` | Optional dynamic register after install materialization | Second runtime registry |
| `AgentBinding` / `ApplicationManifest` | Serialize from installation + binding store | LKW-specific binding types |
| `AgentContract` | Remains author-time truth on package | Catalog marketing copy only in `AgentCatalogEntry` |
| `AgentGovernanceProfile` | Link certification to installation version | Per-app certification silo |
| `ApplicationRuntimeGraph` | Pre-install compatibility check | Manual graph bypass |
| Platform Plugin manifest / qualification | Pattern for publisher + trust evidence | Conflate plugins with agents |
| `registry_snapshot_store` | Snapshot includes installation ids | Treat snapshots as install DB |
| Nexus `find_by_capability` | Unchanged routing spine | Marketplace routing branch |
| `build_application_registry` | Accept roster from merged static + dynamic sources | Per-app registry builders |

---

## 11. Gap register

| ID | Severity | Gap | Evidence |
|----|----------|-----|----------|
| GAP-AP0-01 | **Critical** | No durable **installation state** model | No `AgentInstallationRecord`; registry in-memory only |
| GAP-AP0-02 | **Critical** | Agent **catalog entry** separate from contract/package | Only `ScaffoldAgentSpec` / monorepo tree |
| GAP-AP0-03 | **Critical** | **Runtime install** without image rebuild incompatible with minimal graph | `APPLICATION_RUNTIME_GRAPH_MODEL` §1–§3 |
| GAP-AP0-04 | **High** | No **operator enable/disable** without manifest redeploy | `AgentBinding.enabled` static in `manifest.py` |
| GAP-AP0-05 | **High** | No **catalog source provider** interface | Platform Plugin targets Tier-0 extensions, not Tier-2 agents |
| GAP-AP0-06 | **High** | **Agent package trust** metadata (publisher, signature) missing | `platform_qualification.py` is plugin-scoped |
| GAP-AP0-07 | **Medium** | LKW **no lifecycle APIs** beyond read-only listing | `fastapi_router.py` `GET /agents` only |
| GAP-AP0-08 | **Medium** | **Upgrade/rollback** is deploy pipeline only | uv + docker; no platform agent version manager |
| GAP-AP0-09 | **Medium** | Binding **configuration** not persisted separately from code | `config` dict on static binding |
| GAP-AP0-10 | **Low** | `AgentBinding.capabilities` documented as non-enforced hints | `manifest.py` field description |

---

## 12. Architecture decisions required

| ADR topic | Question | Options (indicative) |
|-----------|----------|----------------------|
| **AD-AP0-1 Installation plane ownership** | Where do installation records live? | Tier-1 platform service vs Tier-0 core vs relational store behind Tier-3 host |
| **AD-AP0-2 Runtime vs build-time install** | Allow runtime package load? | (A) build-time only + operator binding toggles (B) dynamic venv extension (high risk to graph isolation) |
| **AD-AP0-3 Catalog vs contract** | Canonical discovery surface | `AgentCatalogEntry` indexed by package id; `AgentContract` loaded from installed package |
| **AD-AP0-4 Enablement authority** | Who toggles enablement? | Operator API → installation store → filtered roster before `build_application_registry` |
| **AD-AP0-5 Marketplace boundary** | Provider contract | `CatalogSourceProvider` returns entries; install command materializes pyproject/graph change or pre-built image layer |
| **AD-AP0-6 Trust model for third-party agents** | Extend Platform Plugin qualification or parallel AgentPackageTrust? | Prefer extend with `delivery_source` + evidence kinds for agents |
| **AD-AP0-7 LKW proof scope** | What LKW should prove | Generic Tier-3 admin routes calling platform install API — not LKW-local store |

**Gate:** Resolve AD-AP0-1, AD-AP0-2, AD-AP0-4 before implementation waves.

---

## 13. Recommended target conceptual model

```text
AgentPackage (wheel/workspace dist)
  └── contains AgentContract defaults + agent class

AgentCatalogEntry (source-specific metadata)
  └── package_id, publisher, channels, compatibility spec, trust badge

AgentInstallationRecord (per host or per tenant-environment)
  └── package_id, version, source_provider, installed_at, status

ApplicationAgentBinding (per application environment)
  └── installation_ref, binding config, enablement, overrides
      (maps to today's AgentBinding fields)

Materialization (startup or admin-triggered)
  └── merge static manifest + installation records
  └── build_application_registry → AgentRegistry

Nexus (unchanged spine)
  └── capability → AgentRegistry.find_by_capability
```

**Validated separation** (compare to existing mechanisms):

| Layer | Today | Target |
|-------|-------|--------|
| Package/distribution identity | `intergrax-*-agent` pyproject | **EXISTS** — keep |
| Catalog metadata | Absent | **GAP** — `AgentCatalogEntry` |
| Installation state | pyproject + image | **GAP** — `AgentInstallationRecord` |
| Application binding | `AgentBinding` in manifest | **PARTIAL** — persist + merge |
| Configuration state | `binding.config` + env profile | **PARTIAL** — durable config store |
| Enablement state | `binding.enabled` static | **GAP** — operator toggle |
| Runtime registration | `AgentRegistry` | **EXISTS** — derived view |

---

## 14. Recommended implementation sequencing

1. **ADRs** for installation plane, build-time vs runtime install, enablement authority (AD-AP0-1/2/4).
2. **Platform contracts** — `AgentCatalogEntry`, `AgentInstallationRecord`, `CatalogSourceProvider` (no marketplace UI).
3. **Persistence** — installation + binding store (relational); migration from manifest-only hosts.
4. **Materialization bridge** — `build_application_registry` accepts merged roster; feature-flag static-only fallback.
5. **Trust extension** — agent package qualification aligned with `PlatformPluginQualificationSubject`.
6. **Graph guard extension** — pre-install compatibility via `ApplicationRuntimeGraph` simulation.
7. **Generic Tier-3 admin surface** — install/enable/configure routes in shared harness (not LKW-specific).
8. **LKW proof** — consume generic admin API; verify capability routing unchanged.
9. **Marketplace provider** — optional `CatalogSourceProvider` implementation (out of scope for first wave).

---

## 15. Explicit non-goals

- LKW-specific agent store or marketplace
- Second Nexus or second execution AgentRegistry
- Marketplace checkout, billing, or publisher portal
- Runtime hot-load of arbitrary agent code without graph/isolation review
- Replacing `AgentContract` or Nexus capability routing
- Modifying canonical architecture hubs in this task
- Duplicating certification systems (extend existing evaluators)

---

## 16. Evidence — file and symbol references

| Topic | Path | Symbol / section |
|-------|------|------------------|
| Tier-2 role | `agents/README.md` | Agent roster, LKW trio table |
| Agent contract §12 | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §12 `AgentContract` fields |
| Registry §15 | same | §15 Agent Registry responsibilities |
| Capability routing §16 | same | §16 routing invariant |
| Lifecycle §20 | same | §20 certification/promotion/deprecation |
| ACP §21 | same | §21 design invariants |
| Runtime graph model | `docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md` | §1–§3 declaration, build context |
| Dependency model | `docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md` | Tier declaration table |
| `AgentContract` model | `intergrax/contracts/agent_contract_meta.py` | `class AgentContract` |
| Lifecycle enum | `intergrax/contracts/agent_lifecycle_state.py` | `AgentLifecycleState` |
| `AgentRegistry` | `intergrax/runtime/registry/agent_registry.py` | `register`, `find_by_capability`, `is_routable` |
| Assembly validation | `intergrax/runtime/registry/agent_assembly_resolver.py` | `validate_contract_metadata` |
| Routing policy | `intergrax/runtime/registry/agent_routing_policy.py` | `evaluate_agent_routing` |
| Task classification | `intergrax/runtime/nexus/task_classifier.py` | `TaskClassifier.classify` |
| Capability match usage | `intergrax/runtime/nexus/planning/task_planner.py` | `find_by_capability` |
| `AgentBinding` | `intergrax/applications/contracts/manifest.py` | `AgentBinding.mount`, `enabled` |
| `ApplicationManifest` | same | `ApplicationManifest.product` |
| Registry build | `intergrax/applications/_shared/wiring.py` | `build_application_registry` |
| Host runtime | `intergrax/applications/_shared/harness_host_runtime.py` | `build_harness_host_runtime` |
| Runtime graph | `intergrax/applications/_shared/application_runtime_graph.py` | `ApplicationRuntimeGraph` |
| Certification wiring | `intergrax/applications/_shared/agent_certification_wiring.py` | `materialize_roster_certifications_for_agents` |
| Agent governance | `intergrax/applications/contracts/agent_governance.py` | `AgentCertificationRecord` |
| Registry snapshot | `intergrax/applications/_shared/registry_snapshot_store.py` | `SqliteRegistrySnapshotStore` |
| Platform plugin manifest | `intergrax/core/plugins/package_contract.py` | `PlatformPluginManifest` |
| Plugin qualification | `intergrax/core/plugins/platform_qualification.py` | `PlatformPluginTrustModel` |
| Scaffold catalog | `intergrax/scaffold/agent_catalog.py` | `BUILTIN_AGENTS` |
| LKW manifest | `applications/local_workspace_application/manifest.py` | `LOCAL_WORKSPACE_APPLICATION_MANIFEST` |
| LKW factory | `applications/local_workspace_application/host/factory.py` | `create_local_workspace_backend_app` |
| LKW agent factories | `applications/local_workspace_application/host/agent_factories.py` | `build_local_workspace_*_from_context` |
| LKW agent listing | `applications/local_workspace_application/serving/fastapi_router.py` | `list_agents` |
| LKW architecture | `docs/project/technical/applications/local_workspace_application/ARCHITECTURE.md` | §5.1, §6 agent roster |
| Agent package example | `agents/local_search/pyproject.toml` | `intergrax-local-search-agent` |
| Lifecycle governance | `intergrax/runtime/architecture/agent_lifecycle_governance.py` | `AgentLifecycleTransitionRequest` |
| Certification eval | `intergrax/runtime/architecture/agent_certification.py` | `evaluate_agent_certification` |
| Promotion eval | `intergrax/runtime/architecture/agent_promotion.py` | `evaluate_agent_promotion` |

---

## Mandatory questions — consolidated answers

1. **Production-grade architecture:** Tier-2 packages + `AgentContract` + Tier-3 `ApplicationManifest` → `build_application_registry` → `AgentRegistry` → Nexus capability routing + ACP step loop (§2–§4).
2. **AgentContract models:** Full metadata per §12 + skills, memory, cognitive pattern, lifecycle, ownership — `intergrax/contracts/agent_contract_meta.py`.
3. **AgentRegistry owns:** In-process agent instances and contracts, capability lookup, routability, skill/tool resolution at register. **Does not own:** package install, catalog, binding persistence, cross-host fleet catalog.
4. **Discovery / register / instantiate / route:** pyproject graph (discovery) → manifest bindings → factories → `register` → Nexus `find_by_capability` / `TaskClassifier`.
5. **Capabilities for routing:** `contract.capabilities` matched in registry; production filters via lifecycle policy; §16 normative invariant.
6. **Lifecycle mechanisms:** `AgentLifecycleState`, certification/promotion/deprecation evaluators, STRICT deploy gates, CI metadata scripts — release-oriented, not operator install API.
7. **Governance:** Contract ownership, binding budgets/memory/tools, skill resolution, dual observability, organizational policy §39 — **EXISTS** for wired agents.
8. **Tier-2 packaging:** `agents/<slug>/pyproject.toml`, `intergrax-<slug>-agent` distribution, workspace member.
9. **Dependencies:** Direct in app pyproject; transitive in agent pyproject; `ApplicationRuntimeGraph` resolves closure.
10. **Manifest / binding composition:** `AgentBinding.mount` + `ApplicationManifest.agents` list; optional `reference()` for harness.
11. **Static vs dynamic:** Predominantly static build/deploy; runtime routing only among pre-registered agents (§7).
12. **Persistence for roster:** Manifest Python modules + optional governance profile; registry snapshots audit ids — **no** install roster DB.
13. **Generic catalog/plugin reuse:** Platform Plugin system for Tier-0 extensions; scaffold agent catalog for codegen — **not** agent distribution catalog.
14. **Install/upgrade/uninstall:** uv lock + runtime graph image build — generic Python packaging, not agent-lifecycle API.
15. **Provenance/trust for third-party agents:** Plugin qualification + capability graph provenance — **not** agent package signing.
16. **Runtime graph interaction:** Installed agents must be in resolved graph before image build; runtime add violates minimal isolation model.
17. **Endangered invariants:** Tier boundaries, acyclic graph, image minimal context, capability graph conformance, single Nexus routing spine.
18. **LKW consumes LKW trio:** Via manifest mounts + host factories + `build_harness_host_runtime` (§6).
19. **LKW discovery/config/lifecycle surfaces:** Read-only `GET /agents` only — no install/configure/enable API.
20. **LKW gaps vs Applications proof:** Applications layer proves runtime graph, image isolation, deploy gates; agents lack install lifecycle proof path (§8, GAP-AP0-07).

---

**Implementation proceed?** **No — architecture decision gate required first** (§12 AD-AP0-1, AD-AP0-2, AD-AP0-4).

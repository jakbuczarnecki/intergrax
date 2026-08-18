# ADR-AGENT-004: Agent Distribution, Installation and Enablement Architecture

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture only) |
| **Date** | 2026-08-12 |
| **Task** | AGENT-PLATFORM-1 — architecture decision gate |
| **Evidence** | [`AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md`](../../../../audit_results/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md) (AGENT-PLATFORM-0) |
| **Deciders** | Platform architecture (Harness AI) |
| **Related** | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §15–§16 · [`APPLICATION_RUNTIME_GRAPH_MODEL.md`](../../architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md) · [`APPLICATION_DEPENDENCY_MODEL.md`](../../architecture/APPLICATION_DEPENDENCY_MODEL.md) · [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) · [ADR-AGENT-001](entries/2026-06-11/ADR-AGENT-001.md) · [ADR-AGENT-002](entries/2026-06-11/ADR-AGENT-002.md) · [ADR-AGENT-003](entries/2026-06-11/ADR-AGENT-003.md) · [ADR-HOST-001](entries/2026-07-13/ADR-HOST-001.md) |

---

## 1. Context

AGENT-PLATFORM-0 established that Intergrax has a production-grade **agent execution and governance stack** (Tier-1–2: `AgentContract`, `AgentRegistry`, Nexus capability routing, ACP/UAEP, lifecycle/certification evaluators, `ApplicationManifest` / `AgentBinding` composition) but **no neutral distribution and installation plane**.

Today **install an agent = declare a Python package dependency and rebuild the application runtime graph / image**. `AgentBinding.enabled` is a static manifest flag. `AgentRegistry` is an ephemeral runtime index, not an installation store.

Intergrax must eventually support operator UX resembling VS Code extension management (discover → install → configure → enable/disable → upgrade/rollback → uninstall) while preserving deterministic runtime graphs, dependency isolation, reproducibility, certification, provenance, security, auditability, capability-based Nexus routing, tier boundaries, and rollback safety.

**Install must not mean hot-loading arbitrary Python code into a running production process.**

This ADR resolves AD-AP1-01 through AD-AP1-10 and gates AGENT-PLATFORM-2 canonical architecture work.

---

## 2. Existing constraints from AGENT-PLATFORM-0

| Constraint | Evidence |
|------------|----------|
| Tier boundaries: `intergrax/` MUST NOT import `agents/` or `applications/` | `AGENTS.md`, `APPLICATION_RUNTIME_GRAPH_MODEL.md` §1 |
| Minimal transitive runtime graph is mandatory for images | `APPLICATION_RUNTIME_GRAPH_MODEL.md` §1, §3; GAP-AP0-03 |
| `uv.lock` is canonical third-party closure | `APPLICATION_DEPENDENCY_MODEL.md` §1; `APPLICATION_RUNTIME_GRAPH_MODEL.md` §2 |
| `AgentRegistry` is execution index only | `agent_registry.py`; audit §3, §4 |
| Nexus routes by capability, not class name | `AGENT_CONTRACTS_AND_ASSEMBLY.md` §16 routing invariant |
| Platform Plugin system is Tier-0 extension coordination, not Tier-2 agent distribution | `PLATFORM_PLUGINS.md`; audit §9.3 |
| No second Nexus or second execution registry | Audit §10 reuse map |
| Certification / promotion evaluators exist but are release-oriented | `agent_certification.py`, `agent_promotion.py`; audit §4 |
| Registry snapshots audit id sets, not install roster | `registry_snapshot_store.py`; GAP-AP0-01 |
| LKW proves execution only; read-only `/agents` today | Audit §6, §7 |

**Gate verdict from AGENT-PLATFORM-0:** architecture decision required before implementation (AD-AP0-1, AD-AP0-2, AD-AP0-4). This ADR satisfies that gate.

---

## 3. Decision matrix

| Decision | Options evaluated | Selected |
|----------|-------------------|----------|
| AD-AP1-01 Installation plane ownership | Tier-3 app · Tier-1 runtime · Tier-0 core domain · extend Platform Plugins only | **Tier-0 Agent Distribution domain** (new, pattern-aligned with Platform Plugins) |
| AD-AP1-02 Physical installation model | A hot in-process · B immutable materialization · C isolated sidecar | **B primary**; **C bounded optional** for future high-risk tiers |
| AD-AP1-03 Application binding authority | A immutable manifest only · B merge defaults + durable · C projection only · D other | **B** — manifest defaults merged with durable operator bindings |
| AD-AP1-04 Enablement authority | Manifest static · persisted per binding · registry-only | **Persisted per binding**; production policy may override |
| AD-AP1-05 Configuration authority | Manifest only · catalog · binding store · env profile only | **Layered** — contract / catalog / binding / secrets refs / env |
| AD-AP1-06 Runtime materialization | Registry as SoT · manifest as SoT · installation as SoT | **Installation + binding durable**; registry derived |
| AD-AP1-07 Version semantics | floating latest · digest-pinned · channel-based | **Digest-pinned immutable identity**; no authoritative `latest` in prod |
| AD-AP1-08 Trust boundary | reuse plugins only · parallel agent trust · no trust | **Parallel `AgentPackageTrust`** reusing plugin evidence patterns |
| AD-AP1-09 Catalog abstraction | monorepo-only · provider interface | **`CatalogSourceProvider` neutral boundary** |
| AD-AP1-10 LKW proof boundary | LKW-local store · generic platform API | **Generic platform APIs**; LKW as consumer only |
| UX hypothesis | hot install UX · dynamic UX + immutable runtime | **Dynamic UX over immutable runtime materialization** |

---

## 4. AD-AP1-01 — Installation plane ownership

### Decision

Introduce a **Tier-0 Agent Distribution domain** under `intergrax/core/agent_distribution/` (exact module tree deferred to AGENT-PLATFORM-2) owning:

| Artifact / concern | Owner |
|--------------------|-------|
| `AgentPackageIdentity` | Tier-0 Agent Distribution contracts |
| `AgentCatalogEntry` | Tier-0 Agent Distribution contracts |
| `AgentInstallationRecord` | Tier-0 contracts + durable store interface |
| Installation state transitions | Tier-0 domain service (interface); host invokes via Tier-3 harness admin surface |
| Installation persistence | Tier-0 store interface; **relational implementation** behind application host environment (not LKW, not `AgentRegistry`) |
| Package verification | Tier-0 verification coordinator (digest, signature, compatibility hooks) |
| Dependency compatibility | Tier-0 + `ApplicationRuntimeGraph` simulation (reuse `application_runtime_graph.py`) |
| Install health state | Tier-0 installation record fields; post-activation health validated at materialization |

**`AgentRegistry` does NOT own installation persistence.** It remains the Tier-1 execution index populated from materialization projections.

### Extend vs new

| Mechanism | Verdict |
|-----------|---------|
| Platform Plugin package/trust contracts | **Extend patterns only** — do not conflate plugins with agents |
| `AgentRegistry` | **Do not extend** for install state |
| `ApplicationRuntimeGraph` | **Extend** for pre-install compatibility simulation |
| `registry_snapshot_store` | **Extend** to include installation/binding ids in audit snapshots (not as install DB) |
| New generic Agent Distribution domain | **Required** — no existing subsystem owns installation records |

### Rejected

- **Tier-3 application ownership (LKW):** violates platform neutrality and tier boundaries.
- **Tier-1 `AgentRegistry` as install store:** conflates execution index with distribution state (audit GAP-AP0-01).
- **Platform Plugin subsystem as sole owner:** plugins coordinate Tier-0 extensions; Tier-2 agents have different delivery graph, certification, and runtime graph semantics.

---

## 5. AD-AP1-02 — Physical installation model

### Models compared

| Criterion | Model A — hot in-process | Model B — immutable materialization | Model C — isolated sidecar |
|-----------|--------------------------|-------------------------------------|----------------------------|
| `ApplicationRuntimeGraph` | **Violates** — runtime graph mutation | **Aligned** — graph resolved before build | Partial — separate graph per sidecar |
| Minimal image isolation | **Breaks** | **Core design** | Strong isolation |
| `uv.lock` determinism | **Weak** — runtime pip/uv into live venv | **Strong** — lock resolved at build | Per-unit lock |
| Supply-chain security | **High risk** | Verified artifact before activation | Strong boundary |
| Third-party dep conflicts | Runtime discovery | Pre-build fail closed | Isolated |
| Arbitrary code execution risk | **Highest** | Gated materialization | Lowest in-process |
| Certification | Hard to gate mid-flight | Pre-activation gates | Per-sidecar qual |
| Startup / runtime perf | Fast toggle illusion | Rebuild + activation latency | Sidecar overhead |
| Rollback | Unclear / unsafe | Atomic activation swap | Sidecar version swap |
| Upgrade | In-place import | New artifact + activation | New sidecar revision |
| Horizontal deployment | Inconsistent nodes | Immutable image per revision | Per-replica sidecars |
| Self-hosted topology | Tempting but unsafe | Image / venv artifact pipeline | Ops complexity |
| Hosted / SaaS | Unsafe multi-tenant | Standard deploy pattern | K8s sidecar pattern |
| Enterprise private deploy | Airgap artifact risk | Signed immutable bundles | Possible |
| Marketplace future | UX-friendly trap | Catalog → artifact pipeline | Premium isolation tier |
| Operational complexity | Low short-term | Medium — familiar deploy ops | High |
| Developer experience | Misleading simplicity | Matches current monorepo + image path | Extra boundary learning |
| Current architecture fit | **Contradicts** GAP-AP0-03 | **Extends** today's pyproject + graph + image | Future optional |

### Decision

**Primary: Model B — managed immutable runtime materialization.**

**Bounded hybrid:** Model C reserved as an **optional execution trust tier** for future third-party / high-risk agents (governed runtime boundary). It is **not** the default install path and does not replace Model B for built-in and org-trusted agents.

**Rejected: Model A** — hot in-process installation. Selecting it would sacrifice minimal runtime graph isolation, reproducibility, and certification gates documented in `APPLICATION_RUNTIME_GRAPH_MODEL.md` and AGENT-PLATFORM-0 GAP-AP0-03. UX label "Install" does not mandate Model A.

### Selected target model — dynamic UX over immutable runtime

Operators experience Install / Upgrade / Disable as **normal product operations**. Internally the platform:

```text
install request
  → resolve package (catalog + identity)
  → verify trust / provenance / digest
  → dependency resolution (uv.lock + ApplicationRuntimeGraph)
  → candidate ApplicationRuntimeGraph + compatibility / certification gates
  → build / materialize new immutable runtime artifact (image layer or isolated venv bundle)
  → health validation
  → atomic activation (with rollback pointer)
```

No arbitrary code import into an already-running production process.

---

## 6. AD-AP1-03 — Application binding authority

### Decision

**Option B — static manifest defaults merged with durable operator bindings.**

| Source | Role | Authority |
|--------|------|-----------|
| `ApplicationManifest.agents` (`AgentBinding`) | Built-in / authoring-time **default roster** and factory wiring | Immutable per application **release artifact** (Python module or serialized scaffold output) |
| `ApplicationAgentBinding` (durable) | Operator-managed bindings per application environment | **Authoritative** for operator-added/removed agents and overrides |
| Effective roster | `merge_manifest_defaults(durable_bindings)` | **Single derived roster** — sole input to materialization |

Rules:

1. Manifest entries provide **defaults** (factory paths, builder keys, default config skeleton, default enabled flags for first boot).
2. Durable bindings reference `AgentInstallationRecord` (or built-in package identity for monorepo agents).
3. Operator may add binding without editing Python source once installation exists.
4. Operator may override manifest defaults (config, enablement) via durable store.
5. **One unambiguous effective roster** — no parallel runtime truth.

`AgentBinding` semantics (fields, validation, `mount()` authoring) are **reused** for durable binding payloads where possible.

### Rejected

- **A only:** blocks operator install/bind without redeploy (GAP-AP0-04).
- **C projection-only manifest:** factories still need Tier-3 wiring; manifest remains necessary as default/template layer.

---

## 7. AD-AP1-04 — Enablement authority

### Decision

| Concern | Owner |
|---------|-------|
| Durable enable/disable state | `ApplicationAgentBinding.enablement` (persisted) |
| Transition authority | Operator via generic Tier-3 admin API → Agent Distribution service |
| Production override | `AgentGovernanceProfile` / `ApplicationEnvironmentProfile` / deploy policy — may **force disable** in production regardless of operator toggle |
| Registry projection | Disabled bindings **excluded from `AgentRegistry.register`** at materialization (preferred) OR registered with `evaluate_agent_routing` → not routable (secondary path for in-flight semantics) |

Semantics:

| Requirement | Behavior |
|-------------|----------|
| Enable/disable without editing Python | Durable binding enablement field |
| Disabling stops future routing | Agent not registered or not routable; `find_by_capability` excludes |
| In-flight executions | **Continue** under started agent contract; no mid-step kill (align with Nexus task lifecycle) |
| Restart preserves state | Enablement read from durable store at materialization |
| Production policy override | Policy may deny enablement even when operator enabled — fail closed |
| Enablement ≠ routability | Lifecycle (`AgentLifecycleState`), certification, and routing policy still apply |

Audit evidence: enablement transition events on observability spine; binding revision id in registry snapshot.

---

## 8. AD-AP1-05 — Configuration authority

| Layer | Owner | Notes |
|-------|-------|-------|
| Agent author-time contract | `AgentContract` in Tier-2 package | Capabilities, skills, budgets, lifecycle defaults |
| Catalog metadata | `AgentCatalogEntry` | Display, publisher, categories, compatibility spec — **no secrets** |
| Application binding configuration | `ApplicationAgentBinding.config` | Reuse `AgentBinding.config` semantics — lightweight options |
| Secret references | Binding `secret_refs` / integration profile | **Never** in catalog or ordinary config blobs |
| Tenant / workspace overrides | Optional `WorkspaceAgentBindingOverlay` | Only when org policy permits; merged at materialization |
| Environment / runtime-derived | `ApplicationEnvironmentProfile`, `ApplicationBuildContext` | Integrations, skill profiles, governance profile |

Validation: binding config validated against agent config schema (future contract hook) before CONFIGURED state.

---

## 9. AD-AP1-06 — Runtime materialization

### Authoritative chain

```text
AgentPackageIdentity + AgentCatalogEntry        [catalog view — AVAILABLE]
        ↓ install
AgentInstallationRecord                         [durable SoT — INSTALLED]
        ↓ bind
ApplicationAgentBinding                         [durable SoT — BOUND / CONFIGURED / ENABLED]
        ↓ merge with manifest defaults
EffectiveRoster                                 [derived — materialization input]
        ↓ build_application_registry (extended)
Runtime materialization / host startup            [process boundary]
        ↓
AgentRegistry                                     [derived projection — REGISTERED]
        ↓ evaluate_agent_routing + capability match
Nexus routing                                     [ROUTABLE subset]
```

| Stage | Durable SoT? | Derived? |
|-------|--------------|----------|
| Catalog entry | Metadata index per provider | Yes — catalog is index, not install |
| Installation record | **Yes** | — |
| Application binding | **Yes** | — |
| Manifest defaults | Release artifact (versioned with app) | Partial SoT for defaults only |
| Effective roster | **No** | **Yes** — computed |
| `AgentRegistry` | **No** | **Yes** — execution projection |
| Routable set | **No** | **Yes** — registry + routing policy |

`build_application_registry` accepts **effective roster** (merged bindings) instead of raw manifest-only path; static-only fallback feature-flagged for migration.

---

## 10. AD-AP1-07 — Version / upgrade / rollback semantics

| Concept | Rule |
|---------|------|
| Version identity | PEP 440 version **plus** immutable `package_digest` (wheel/sdist hash) |
| Active version | Exactly one `active_installation_ref` per logical agent package slot per environment |
| Previous version | `previous_installation_ref` on installation record for rollback |
| Upgrade eligibility | Compatibility evidence + `ApplicationRuntimeGraph` simulation + certification gate |
| Compatibility check | Platform version spec, dependency closure, capability graph edge check |
| Migration requirements | Agent-authored migration hooks (future); recorded on installation record |
| Rollback target | Previous digest-pinned installation record — **not** floating version label |
| Failed activation | Retain previous active; mark candidate `activation_failed`; fail closed for routing |
| Production | **No authoritative `latest`** — operator or catalog channel selects explicit digest |

---

## 11. AD-AP1-08 — Trust and provenance boundary

### Decision

Introduce **`AgentPackageTrust`** parallel to `PlatformPluginTrustModel` / `PluginQualificationSubject` — **reuse patterns, separate subject type**.

| Concern | Reuse from Platform Plugins | Agent-specific |
|---------|----------------------------|----------------|
| Publisher identity | Pattern | `AgentPublisherIdentity` |
| Source identity | `PluginDeliverySource` pattern | `AgentDeliverySource` (+ marketplace, org registry, workspace) |
| Immutable digest | Evidence kind | Required on `AgentPackageIdentity` |
| Signature / verification | Evidence pipeline pattern | Agent package signing contract |
| Certification status | `PluginQualificationStatus` shape | `AgentPackageQualificationResult` |
| Compatibility evidence | `PLATFORM_COMPATIBILITY` kind | Graph + Intergrax version spec |
| Revocation | Rejection / deny policy pattern | Global revocation list + per-org deny |
| Enterprise allow/deny | Policy bundle pattern | Org agent allowlist policy |

**Do not** equate Platform Plugins and Tier-2 agents. Shared: evidence kinds, qualification levels, production admission shape. Separate: subject identity, graph impact, runtime materialization path.

---

## 12. AD-AP1-09 — Catalog source abstraction

### Decision

Define **`CatalogSourceProvider`** (Tier-0 interface):

```text
CatalogSourceProvider
  → list_entries(filters) → AgentCatalogEntry[]
  → resolve_package(entry) → AgentPackageIdentity + artifact locator
```

Supported sources (future implementations — not in scope now):

| Provider | Purpose |
|----------|---------|
| `builtin` | Monorepo / built-in Intergrax agents |
| `local_developer` | Path / workspace dev packages |
| `enterprise_private` | Org registry, airgap bundles |
| `official_catalog` | Future Intergrax marketplace index |
| `governed_third_party` | Trusted external catalogs |

**Execution runtime does not care** which provider supplied the agent. Installation records store `source_provider_id` + `source_entry_ref` for audit.

Marketplace billing, reviews, recommendations — **out of scope**.

---

## 13. AD-AP1-10 — LKW proof boundary

### LKW must NOT own

- `AgentInstallationStore`
- `CatalogSourceProvider` implementations
- Package installer / materialization engine
- Agent certification system
- Runtime graph mutation framework

### LKW may

- Expose **generic agent-management capabilities** in frontend/API by calling **platform-owned** Tier-3 harness admin routes (shared, not LKW-local).

### Future LKW proof journey

```text
discover (catalog provider)
  → install (platform API → installation record + materialization job)
  → bind/configure (application binding store)
  → enable (binding enablement)
  → invoke by capability (Nexus — unchanged)
  → disable (binding enablement)
  → upgrade/rollback (installation version swap + activation)
  → uninstall (installation removal + binding cleanup)
```

LKW proves **consumption** of generic platform mechanisms, not ownership.

---

## 14. Selected target model (summary)

| Dimension | Selection |
|-----------|-----------|
| Installation plane | Tier-0 Agent Distribution domain |
| Physical model | Model B — immutable runtime materialization |
| UX hypothesis | Dynamic UX over immutable runtime |
| Binding authority | Manifest defaults + durable operator bindings → effective roster |
| Enablement | Durable per-binding; policy may override in production |
| Registry role | Derived execution projection only |

---

## 15. Rejected alternatives and reasons

| Alternative | Reason rejected |
|-------------|-----------------|
| Model A — hot in-process install | Breaks runtime graph isolation, reproducibility, certification; GAP-AP0-03 |
| LKW-local agent store | Tier violation; not reusable across Tier-3 apps |
| `AgentRegistry` as install DB | Conflates execution index with distribution state |
| Platform Plugin subsystem owns agents | Different tier, graph, and trust semantics |
| Manifest-only binding (no durable store) | Cannot enable/disable or install without redeploy |
| Authoritative `latest` in production | Non-reproducible; rollback unsafe |
| Second Nexus / second registry | Audit §10 — extend existing spine |
| Marketplace execution path | Catalog provider only; Nexus unchanged |

---

## 16. Source-of-truth matrix

| Entity | Authoritative owner | Persistence | Consumers |
|--------|---------------------|-------------|-----------|
| `AgentContract` | Tier-2 agent package | Package source | Registry, certification |
| `AgentCatalogEntry` | Catalog provider index | Provider + optional local cache | Install UI, discovery |
| `AgentPackageIdentity` | Agent Distribution | Installation + catalog | Verification, graph |
| `AgentInstallationRecord` | Agent Distribution | Relational store (host-scoped) | Bindings, materialization |
| `ApplicationAgentBinding` | Agent Distribution | Relational store | Effective roster merge |
| `ApplicationManifest.agents` | Tier-3 app release | Python module / scaffold artifact | Default roster template |
| `AgentGovernanceProfile` | Tier-3 environment profile | Deploy config | Certification, policy |
| Effective roster | — (derived) | Transient | `build_application_registry` |
| `AgentRegistry` | Tier-1 runtime | In-memory per process | Nexus |
| Routable agents | — (derived) | Transient | `find_by_capability` |

---

## 17. State machine

States remain **conceptually distinct**:

```text
AVAILABLE          catalog entry visible; not on host
    │ install (+ verify)
INSTALLED            installation record active on host/environment
    │ bind to application
BOUND_TO_APPLICATION durable binding created (enablement may be false)
    │ validate config
CONFIGURED           binding config validated
    │ enable (operator; policy may block)
ENABLED              enablement true (still may not be routable)
    │ host startup / activation materialization
REGISTERED_IN_RUNTIME  agent instance in AgentRegistry
    │ routing policy + capability + lifecycle
ROUTABLE             Nexus may select for capability
```

| State | Authoritative owner | Persistence owner | Transition authority | Failure semantics | Audit evidence |
|-------|---------------------|-------------------|----------------------|-------------------|----------------|
| AVAILABLE | Catalog provider | Provider index | Provider publish | Entry stale → hide; no install | Catalog sync audit |
| INSTALLED | Agent Distribution | Installation store | Install API / admin | Fail closed — no binding without install | `installation.created`, digest, trust result |
| BOUND | Agent Distribution | Binding store | Bind API | Missing install → reject | `binding.created`, app_id, installation_ref |
| CONFIGURED | Agent Distribution | Binding store | Config validate API | Invalid config → remain BOUND, not ENABLED | `binding.config_validated` or error |
| ENABLED | Agent Distribution | Binding store | Enable API; policy gate | Policy deny → remain disabled | `binding.enablement_changed` |
| REGISTERED | Tier-1 materialization | Process memory | Host startup | Materialization fail → previous activation | Registry snapshot, startup event |
| ROUTABLE | Tier-1 routing policy | Derived | Lifecycle/cert gates | Not routable → skip in `find_by_capability` | Routing decision events |

---

## 18. Ownership matrix

| Concern | Tier-0 | Tier-1 | Tier-2 | Tier-3 |
|---------|--------|--------|--------|--------|
| Distribution contracts | **Owner** | Consumer | — | Consumer via host |
| Catalog providers | **Interface** | — | — | — |
| Installation store | **Interface** | — | — | Host mounts impl |
| Package verification | **Owner** | — | — | — |
| Runtime graph simulation | Shared util | — | — | Pre-build gate |
| `AgentContract` | — | Consumer | **Owner** | — |
| `AgentRegistry` | — | **Owner** | — | Consumer |
| Nexus routing | — | **Owner** | — | Consumer |
| Manifest defaults | — | — | — | **Owner** (release) |
| Admin API surface | Contracts | — | — | **Host routes** |
| LKW product UI | — | — | — | **Consumer only** |

---

## 19. Runtime materialization flow

```text
Host startup (or post-activation hook)
  1. Load installation records for environment
  2. Load durable application bindings for app_id
  3. Load manifest defaults (ApplicationManifest.agents)
  4. merge → EffectiveRoster
  5. wire_application_environment → ApplicationBuildContext
  6. build_application_registry(effective_roster, ctx)
       for binding in enabled bindings:
         build_agent_from_binding
         registry.register(agent, contract=...)
  7. Optional: registry snapshot store capture
  8. build_harness_host_runtime → NexusLoop(registry)
```

Activation swap (upgrade): materialize new artifact → health check → atomic pointer swap on `active_installation_ref` → process restart or rolling deploy.

---

## 20. Install flow

```text
Operator: Install agent X for environment E
  1. CatalogSourceProvider.resolve → AgentCatalogEntry
  2. Fetch artifact; compute digest
  3. AgentPackageTrust.verify(publisher, signature, digest, revocation)
  4. Simulate ApplicationRuntimeGraph with candidate package
  5. evaluate_agent_certification / compatibility (as required by policy)
  6. Materialize immutable runtime artifact (image build or venv bundle job)
  7. Health validation on candidate
  8. Create AgentInstallationRecord (INSTALLED)
  9. Emit audit events; return installation_ref
```

Fail closed at any verification / graph / certification step — no partial INSTALLED without artifact integrity.

---

## 21. Enable / disable flow

```text
Operator: Disable agent for app A
  1. Authorize operator on app A
  2. Load ApplicationAgentBinding
  3. Policy check (production may deny re-enable later)
  4. Set enablement=false; persist
  5. Emit binding.enablement_changed
  6. Next materialization: skip register OR mark not routable
  7. In-flight Nexus tasks on that agent: continue to completion
```

Enable: reverse with policy + certification gates in production mode.

---

## 22. Upgrade / rollback flow

```text
Upgrade:
  1. Resolve target entry (explicit version + digest — not latest)
  2. Verify trust + graph + certification for target
  3. Materialize candidate runtime
  4. Health validation
  5. Set previous_installation_ref = current active
  6. Atomic activation → new active_installation_ref
  7. Rolling restart / deploy
  8. On health failure post-activation → automatic rollback attempt

Rollback:
  1. Validate previous_installation_ref still trusted and present
  2. Atomic activation swap to previous
  3. Deploy / restart
  4. If rollback fails → fail closed; alert; retain last known good if possible
```

---

## 23. Uninstall flow

```text
  1. Check bindings — if bound to any application → reject OR require unbind first (fail closed)
  2. Revoke active installation record
  3. Remove artifact from environment store (retain audit tombstone)
  4. Emit installation.removed
  5. Rebuild effective roster on next deploy (agent absent)
```

---

## 24. Failure semantics

| Failure | Behavior | Fail mode |
|---------|----------|-----------|
| Package unavailable | Install aborts; no record | Closed |
| Untrusted publisher | Install rejected | Closed |
| Signature / digest mismatch | Install rejected; quarantine artifact | Closed |
| Incompatible Intergrax version | Install rejected with compatibility evidence | Closed |
| Dependency conflict | Graph simulation fails; install rejected | Closed |
| Certification failure | Install or enable rejected in production | Closed |
| Invalid binding configuration | Binding stays unconfigured; enable blocked | Closed |
| Missing integration / tool / skill | Materialization fails at register | Closed |
| Failed runtime materialization | No activation; previous remains active | Closed |
| Failed activation | Rollback to previous; mark candidate failed | Closed |
| Runtime health failure after activation | Auto rollback attempt; alert | Closed |
| Rollback failure | Alert; manual intervention; routing fail closed if ambiguous | Closed |
| Disabling agent with active runs | Disable persisted; in-flight continue; no new routing | Open for in-flight only |
| Uninstall while bound | Reject uninstall until unbind | Closed |
| Revocation of installed agent | Block new enables; flag installation; policy may force disable on next materialization | Closed |

---

## 25. Trust / provenance implications

- Every production installation record carries: `package_digest`, `source_provider_id`, `trust_evidence_refs`, `qualification_status`.
- Enterprise deployments may require org allowlist intersection before install.
- Revocation list checked at install, enable, and materialization.
- Agent packages from marketplace follow same trust pipeline as org-private bundles — provider differs, not verification shape.
- Platform Plugin qualification code paths may be shared at evidence evaluation layer; subjects remain distinct.

---

## 26. Impact on `ApplicationManifest` / `AgentBinding`

| Artifact | Change |
|----------|--------|
| `ApplicationManifest` | Retained; `agents` become **default roster template** for first boot and authoring |
| `AgentBinding` | Retained; durable `ApplicationAgentBinding` reuses field semantics |
| `AgentBinding.enabled` | Default only; operator enablement in durable store supersedes on merge |
| `enabled_agents()` | Computed on **effective roster**, not raw manifest |
| Scaffold / codegen | May emit manifest defaults; not operator lifecycle authority |

**No manifest schema change required in AGENT-PLATFORM-1** — behavioral authority shift documented for AGENT-PLATFORM-2.

---

## 27. Impact on `ApplicationRuntimeGraph`

| Aspect | Impact |
|--------|--------|
| Pre-install simulation | **New gate** — candidate package must resolve into acyclic graph |
| Image build | Still canonical materialization path for production |
| `.intergrax-runtime-graph.json` | Includes installed agent closure, not only static pyproject |
| `uv.lock` | Remains third-party closure authority |
| Runtime hot-add | **Not supported** — graph changes require rebuild |

---

## 28. Impact on `AgentRegistry`

| Aspect | Impact |
|--------|--------|
| Ownership | Unchanged — execution index |
| Population | From effective roster at materialization |
| Dynamic register API | **Not required** for v1 — restart/redeploy after install |
| `find_by_capability` | Unchanged spine |
| Install state | **Never stored** in registry |

Optional future: explicit `unregister` for dev Lab profile only — not production path.

---

## 29. LKW proof implications

- LKW adds **no** agent installation persistence.
- Proof uses shared harness admin routes + LKW UI wiring.
- `GET /v1/local_workspace/agents` remains introspection of materialized registry.
- New proof APIs: install/bind/enable/disable via platform contracts (AGENT-PLATFORM-3+).
- Nexus capability routing proof unchanged — validates ROUTABLE state end-to-end.

---

## 30. Marketplace-readiness implications

| Ready | Not ready (by design) |
|-------|----------------------|
| `CatalogSourceProvider` abstraction | Billing, reviews, checkout |
| Digest-pinned install records | Publisher portal |
| Trust / revocation pipeline | Recommendation engine |
| Org private catalog provider | Marketplace-specific Nexus branch |
| Neutral installation plane | LKW-specific store |

Marketplace is a **catalog provider implementation**, not a runtime fork.

---

## 31. Explicit non-goals

- Production code implementation (this ADR only)
- LKW-local agent store or marketplace
- Second Nexus or execution registry
- Runtime hot-load of arbitrary agent code
- Marketplace billing / commercial workflows
- Replacing `AgentContract` or capability routing model
- Modifying `AgentRegistry` implementation in this phase
- Duplicating certification evaluators (extend instead)

---

## 32. Open questions

| ID | Question | Defer to |
|----|----------|----------|
| OQ-1 | Exact relational schema for installation + binding stores | AGENT-PLATFORM-2 |
| OQ-2 | Image vs isolated venv bundle as default artifact per deployment topology | AGENT-PLATFORM-2 / HOST |
| OQ-3 | Sidecar trust tier criteria and protocol | AGENT-PLATFORM-4+ |
| OQ-4 | Agent config schema validation contract on Tier-2 packages | AGENT-PLATFORM-3 |

No open question blocks AGENT-PLATFORM-2 canonical architecture.

---

## 33. Recommended next architecture task

**AGENT-PLATFORM-2 — Canonical Agent Distribution & Management architecture**

Deliver:

1. Tier-0 contract modules (`AgentPackageIdentity`, `AgentCatalogEntry`, `AgentInstallationRecord`, `ApplicationAgentBinding`, `CatalogSourceProvider`)
2. Store interfaces and state transition service
3. Effective roster merge specification (manifest + durable)
4. Extended `build_application_registry` input contract
5. Trust / qualification subject mapping (`AgentPackageTrust`)
6. Architecture hub section in `AGENT_CONTRACTS_AND_ASSEMBLY.md` or dedicated `AGENT_DISTRIBUTION.md` pair

---

## Compliance

- Tier boundaries preserved — distribution in Tier-0; execution in Tier-1; agents Tier-2; host admin in Tier-3
- `intergrax/` does not import `applications/` or `agents/`
- Nexus capability routing invariant preserved (§16)
- `ApplicationRuntimeGraph` minimal isolation preserved
- Platform Plugin patterns reused without conflating plugins and agents
- Fail-closed security and execution integrity throughout

## Consequences

### Positive

- Clear separation: distribution vs binding vs execution vs routing
- Operator UX can evolve without sacrificing enterprise invariants
- Reuses proven monorepo + image + graph pipeline
- LKW and future apps share one platform mechanism
- Marketplace becomes a catalog provider, not architectural fork

### Negative

- Install/upgrade latency tied to materialization jobs (honest ops model)
- New Tier-0 domain and persistence layer to implement
- Migration path needed for manifest-only hosts
- Sidecar path deferred — some high-risk scenarios wait for Model C

---

**AGENT-PLATFORM-2 may begin.** Implementation of this ADR remains gated on AGENT-PLATFORM-2 architecture + subsequent implementation waves.

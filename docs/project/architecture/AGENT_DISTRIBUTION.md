# Agent Distribution and Management

**Status:** Canonical architecture (AGENT-PLATFORM-2 frozen — documentation only)  
**Plan (1:1):** [`plan/AGENT_DISTRIBUTION.md`](../maintainers/plans/AGENT_DISTRIBUTION.md)  
**ADR:** [`adr/entries/2026-08-12/ADR-AGENT-004.md`](../technical/adr/entries/2026-08-12/ADR-AGENT-004.md)  
**Evidence gate:** [`audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md`](../maintainers/audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md) (AGENT-PLATFORM-0)  
**Execution hub (do not duplicate):** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §15–§16  
**Runtime graph:** [`APPLICATION_RUNTIME_GRAPH_MODEL.md`](APPLICATION_RUNTIME_GRAPH_MODEL.md)  
**Packaging:** [`APPLICATION_DEPENDENCY_MODEL.md`](APPLICATION_DEPENDENCY_MODEL.md)  
**Trust patterns (reuse only):** [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) §16–§18  

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Implement / audit default:** §1–§8 (purpose, invariants, vocabulary, identity, state, catalog, package resolution).
- **Materialization chain:** §9–§16 (trust, installation, binding, roster, dependency lock, runtime revision, materialization, activation).
- **Operations:** §17–§22 (registry projection, persistence, concurrency, failure, topology, LKW boundary).
- **Program:** §23–§26 (marketplace, security, observability, migration, non-goals, sequencing).

---

## Table of contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Relationship to ADR-AGENT-004](#2-relationship-to-adr-agent-004)
3. [Architecture invariants](#3-architecture-invariants)
4. [Layer and ownership map](#4-layer-and-ownership-map)
5. [Domain vocabulary](#5-domain-vocabulary)
6. [Identity model](#6-identity-model)
7. [State model](#7-state-model)
8. [Catalog architecture](#8-catalog-architecture)
9. [Package identity and resolution](#9-package-identity-and-resolution)
10. [Trust and provenance](#10-trust-and-provenance)
11. [Installation model](#11-installation-model)
12. [Application binding model](#12-application-binding-model)
13. [Manifest and default merge semantics](#13-manifest-and-default-merge-semantics)
14. [Effective roster model](#14-effective-roster-model)
15. [Dependency-resolution architecture](#15-dependency-resolution-architecture)
16. [Materialized runtime lock / deterministic closure](#16-materialized-runtime-lock--deterministic-closure)
17. [Candidate runtime graph](#17-candidate-runtime-graph)
18. [Runtime revision model](#18-runtime-revision-model)
19. [Materialization model](#19-materialization-model)
20. [Activation and rollback model](#20-activation-and-rollback-model)
21. [AgentRegistry projection](#21-agentregistry-projection)
22. [Nexus routing boundary](#22-nexus-routing-boundary)
23. [Persistence and source-of-truth matrix](#23-persistence-and-source-of-truth-matrix)
24. [Concurrency and transaction semantics](#24-concurrency-and-transaction-semantics)
25. [Failure and recovery semantics](#25-failure-and-recovery-semantics)
26. [Self-hosted / hosted / enterprise topology treatment](#26-self-hosted--hosted--enterprise-topology-treatment)
27. [LKW proof boundary](#27-lkw-proof-boundary)
28. [Marketplace readiness](#28-marketplace-readiness)
29. [Security implications](#29-security-implications)
30. [Observability and audit evidence](#30-observability-and-audit-evidence)
31. [Migration from manifest-only applications](#31-migration-from-manifest-only-applications)
32. [Explicit non-goals](#32-explicit-non-goals)
33. [Implementation dependency graph / recommended AP-3+ sequencing](#33-implementation-dependency-graph--recommended-ap-3-sequencing)

---

## 1. Purpose and scope

This document is the **canonical architecture** for the Intergrax **Agent Distribution and Management** platform — the Tier-0 plane that separates **catalog availability**, **installation**, **application binding**, **deterministic runtime dependency closure**, **immutable materialization**, and **activation** from Tier-1 **execution** (`AgentRegistry`, Nexus capability routing).

**In scope (architecture only):**

- Platform-neutral chain from catalog discovery through routable agents.
- Deterministic dependency closure for operator-installed agents not present in application source `pyproject.toml`.
- Identity, state, trust, merge, persistence, failure, and topology semantics.
- LKW as future **consumer** proof boundary — not owner.

**Out of scope (this task):**

- Production code, contracts, persistence schemas, APIs, materializer implementation, resolver implementation, `AgentRegistry` changes, LKW changes, marketplace product code.

**Primary outcome:** freeze architecture so **AGENT-PLATFORM-3** may implement Tier-0 contracts and store interfaces without reopening distribution vs execution boundaries.

---

## 2. Relationship to ADR-AGENT-004

[ADR-AGENT-004](../technical/adr/entries/2026-08-12/ADR-AGENT-004.md) (AGENT-PLATFORM-1) is **accepted** and gates this document. AGENT-PLATFORM-2 **instantiates** ADR decisions into a single canonical reference:

| ADR decision | AGENT-PLATFORM-2 resolution |
|--------------|----------------------------|
| AD-AP1-01 Tier-0 Agent Distribution domain | §4 ownership map; `intergrax/core/agent_distribution/` (deferred module tree) |
| AD-AP1-02 Model B immutable materialization | §19 materialization abstraction; §20 activation |
| AD-AP1-03 Manifest defaults + durable bindings | §12–§14 binding and effective roster |
| AD-AP1-04 Persisted enablement | §7 state model; §13 merge precedence |
| AD-AP1-05 Layered configuration | §12 binding config layers |
| AD-AP1-06 Installation + binding durable; registry derived | §21 registry projection |
| AD-AP1-07 Digest-pinned identity | §6 identity model; §16 runtime lock |
| AD-AP1-08 `AgentPackageTrust` parallel to plugins | §10 trust architecture |
| AD-AP1-09 `CatalogSourceProvider` | §8 catalog architecture |
| AD-AP1-10 LKW consumer only | §27 LKW proof boundary |

ADR open questions OQ-1 (schema) and OQ-2 (default artifact topology per deploy) are **resolved at architecture level** in §12, §18–§19, §23; concrete schema and host defaults remain **AP-3+ implementation**.

---

## 3. Architecture invariants

### 3.1 Orthogonal lifecycle dimensions (normative)

These dimensions MUST remain **conceptually and operationally distinct**:

```text
AVAILABLE
  ≠ INSTALLED
  ≠ BOUND_TO_APPLICATION
  ≠ CONFIGURED
  ≠ ENABLED
  ≠ REGISTERED_IN_RUNTIME
  ≠ ROUTABLE
```

| Dimension | Meaning | Authoritative owner |
|-----------|---------|---------------------|
| **AVAILABLE** | Catalog entry visible; package may be resolved | Catalog provider (+ optional cache) |
| **INSTALLED** | Digest-pinned artifact verified and persisted on host/environment | Tier-0 Agent Distribution |
| **BOUND_TO_APPLICATION** | Durable `ApplicationAgentBinding` exists for app + slot | Tier-0 Agent Distribution |
| **CONFIGURED** | Binding config validated against agent config contract | Tier-0 Agent Distribution |
| **ENABLED** | Operator enablement true (policy may override) | Tier-0 Agent Distribution |
| **REGISTERED_IN_RUNTIME** | Agent instance present in `AgentRegistry` after materialization | Tier-1 runtime (derived) |
| **ROUTABLE** | Nexus may select agent for capability | Tier-1 routing policy (derived) |

**Normative rule:** no persistence layer may claim a later dimension is true when an earlier required dimension failed (e.g. never `INSTALLED` without verified artifact; never `REGISTERED` without successful activation materialization).

### 3.2 Platform invariants (preserved)

| Invariant | Enforcement |
|-----------|-------------|
| Tier-0 distribution ownership | Contracts, verification, installation/binding stores, lock production |
| Tier-1 `AgentRegistry` execution ownership | Population from materialization only; no install state |
| Tier-2 reusable agent packages | `AgentContract` + `pyproject.toml` metadata in package |
| Tier-3 application defaults / admin hosting | Manifest defaults; harness admin API surface |
| Capability-based Nexus routing | Unchanged — [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §16 |
| Immutable production runtime | Model B materialization + activation swap |
| Minimal runtime graph | [`APPLICATION_RUNTIME_GRAPH_MODEL.md`](APPLICATION_RUNTIME_GRAPH_MODEL.md) |
| Deterministic dependency closure | §15–§16 — **every activated runtime revision** |
| Fail-closed trust / certification | §10; production gates before activation |
| No hot arbitrary Python installation | No runtime `pip install` into live production process |

### 3.3 Canonical platform chain

```text
CatalogSourceProvider
  → AgentCatalogEntry
  → AgentPackageIdentity
  → package resolution + AgentPackageTrust verification
  → AgentInstallation
  → ApplicationAgentBinding
  → EffectiveRoster
  → deterministic dependency resolution
  → MaterializedRuntimeLock (immutable dependency artifact)
  → CandidateApplicationRuntimeGraph
  → RuntimeMaterialization
  → Activation (RuntimeRevision)
  → AgentRegistry (projection)
  → Nexus capability routing (ROUTABLE subset)
```

**Marketplace** is one **future** `CatalogSourceProvider` implementation only — not a runtime fork.

---

## 4. Layer and ownership map

```text
Tier-0  intergrax/core/agent_distribution/   distribution contracts, trust, stores (interfaces),
                                              dependency lock producer, catalog provider interfaces
Tier-1  intergrax/runtime/registry/          AgentRegistry, routing policy, Nexus (unchanged spine)
Tier-2  agents/<slug>/                       AgentContract, package metadata, agent pyproject
Tier-3  applications/<app>/                ApplicationManifest defaults, host admin routes, env profiles
```

| Concern | Owner tier | Notes |
|---------|------------|-------|
| `CatalogSourceProvider` | Tier-0 interface | Implementations: builtin, local, enterprise, official, governed third party |
| `AgentCatalogEntry`, `AgentPackageIdentity` | Tier-0 contracts | Catalog is index, not execution truth |
| Installation / binding persistence | Tier-0 store interfaces | Relational impl behind host environment — **not LKW** |
| Package verification / trust | Tier-0 | Reuses plugin evidence **patterns** only |
| Dependency resolution + lock | Tier-0 coordinator | Consumes effective roster + declarations |
| `CandidateApplicationRuntimeGraph` | Shared util (`application_runtime_graph.py` extended) | Pre-activation simulation |
| `RuntimeRevision` / activation | Tier-0 + Tier-3 host orchestration | Atomic from application perspective |
| `AgentRegistry` | Tier-1 | Derived execution index |
| Nexus routing | Tier-1 | `find_by_capability` unchanged |
| `ApplicationManifest.agents` | Tier-3 release artifact | Default roster template only |
| Admin API routes | Tier-3 host | Calls Tier-0 services — shared across apps |

**Tier boundary:** `intergrax/` MUST NOT import `agents/` or `applications/`.

---

## 5. Domain vocabulary

| Term | Definition |
|------|------------|
| **Logical agent** | Stable product/agent identity (`logical_agent_id` / roster slot) independent of package version |
| **Agent package** | Tier-2 installable distribution (`intergrax-*-agent` or external equivalent) |
| **Catalog entry** | Provider-indexed discoverable metadata — not installed |
| **Installation** | Host-scoped record that a digest-pinned package artifact is verified and stored |
| **Installation slot** | Stable logical install identity for one agent package line on an environment |
| **Binding** | Application-scoped durable link from roster slot to installation target + config |
| **Effective roster** | Single derived merge of manifest defaults + durable bindings |
| **Materialized runtime lock** | Immutable resolved dependency closure artifact for one candidate/active revision |
| **Runtime revision** | Complete identity of one materialized application runtime |
| **Activation** | Atomic promotion of a validated runtime revision to **active** for an application environment |
| **Materialization** | Physical build of runtime bundle (image, venv bundle, or future sandbox unit) |

---

## 6. Identity model

### 6.1 Identity types

| Identity | Stable vs revision | Purpose |
|----------|-------------------|---------|
| `logical_agent_id` | **Stable** | Roster / product identity; merge key for manifest + bindings |
| `distribution_package_id` | **Stable** (normalized PyPI name) | Package line identity (`intergrax-local-search-agent`) |
| `package_version` | **Revision** (PEP 440) | Human-selectable version label — **not** production authority alone |
| `package_digest` | **Immutable revision** | Content-addressed artifact hash (wheel/sdist/OCI layer) — **production authority** |
| `catalog_entry_id` | **Provider-scoped stable** | Provider's entry key (may map many versions) |
| `catalog_source_id` | **Stable** | Provider type + instance (`builtin`, `enterprise:acme`, `official`) |
| `installation_id` | **Immutable revision** | One digest-pinned installation record |
| `installation_slot_id` | **Stable** | Logical install slot per environment + package line |
| `application_binding_id` | **Stable** | Durable binding row identity |
| `application_environment_id` | **Stable** | Deploy target (app + env profile + host scope) |
| `effective_roster_revision_id` | **Content hash revision** | Hash of merged roster inputs |
| `runtime_revision_id` | **Immutable revision** | Activated materialized runtime identity |
| `materialization_artifact_digest` | **Immutable revision** | Physical bundle digest (image manifest, venv tree hash, etc.) |
| `runtime_dependency_lock_digest` | **Immutable revision** | Hash of `MaterializedRuntimeLock` blob |
| `runtime_graph_digest` | **Immutable revision** | Hash of `CandidateApplicationRuntimeGraph` canonical JSON |

### 6.2 `AgentPackageIdentity` (canonical shape)

```text
AgentPackageIdentity:
  distribution_package_id     # stable package name
  package_version             # PEP 440 (informational)
  package_digest              # required immutable authority
  artifact_locator            # provider-specific fetch ref (not runtime truth after install)
  contract_id                 # optional AgentContract.id default
  platform_compatibility_spec # Intergrax version range
  python_requires             # optional
```

### 6.3 `AgentCatalogEntry` (catalog view only)

```text
AgentCatalogEntry:
  catalog_entry_id
  catalog_source_id
  display_name, publisher, categories   # metadata only — no secrets
  package_id_line                       # distribution_package_id
  version_channel_refs[]                # pointers to resolvable versions — not "latest" in prod
  compatibility_summary
  trust_labels                          # display hints only
```

Catalog metadata MUST NOT become execution truth. Disappearance of a catalog entry MUST NOT affect reproducibility of digest-pinned installations already on the host.

---

## 7. State model

### 7.1 Global dimension state machine

```text
AVAILABLE ──install──► INSTALLED ──bind──► BOUND_TO_APPLICATION
                              │                    │
                              │                    ├──validate config──► CONFIGURED
                              │                    │
                              │                    └──enable (+ policy)──► ENABLED
                              │                                          │
                              │                                          ▼
                              │                              REGISTERED_IN_RUNTIME
                              │                                          │
                              │                                          ▼
                              │                                    ROUTABLE
```

### 7.2 Installation record substates

| Substate | Meaning | `INSTALLED` flag |
|----------|---------|------------------|
| **candidate** | Install requested; resolving catalog + fetching artifact | **false** |
| **verified** | Digest + trust checks passed; artifact staged | **false** |
| **installed_active** | Artifact in environment store; record committed | **true** |
| **installed_previous** | Superseded digest retained for rollback | **true** (historical) |
| **failed_candidate** | Verification, materialization, or activation failed | **false** |
| **revoked** | Trust revocation flagged; block new enable/activation | **true** but **blocked** |
| **removed_tombstone** | Uninstalled; audit tombstone only | **false** |

### 7.3 When `INSTALLED` becomes true (normative)

`INSTALLED` becomes **true** only when **all** hold atomically in one transaction boundary:

1. `AgentPackageTrust.verify` succeeded for the target digest.
2. Package artifact is **persisted** in the environment artifact store (not merely downloaded to temp).
3. `AgentInstallationRecord` is committed with `installation_state = installed_active` (or `installed_previous` for rollback retention).

`INSTALLED` is **not** set when:

- catalog resolution alone succeeded;
- trust passed but artifact persistence failed;
- candidate runtime graph simulation failed;
- application binding exists but installation does not.

**Activation** (`RuntimeRevision` active) is a **separate** later step. An agent may be `INSTALLED` on the host but not `REGISTERED` / `ROUTABLE` for any application.

### 7.4 Runtime revision lifecycle

```text
candidate ──validate──► validated ──activate──► active ──supersede──► superseded
     │                      │                      │
     └────fail──────────────┴──────fail────────────┴──► failed
```

Only **one** `active` `RuntimeRevision` per `application_environment_id` at a time.

---

## 8. Catalog architecture

### 8.1 `CatalogSourceProvider` (conceptual contract)

```text
CatalogSourceProvider:
  catalog_source_id: str

  list_entries(filters) -> list[AgentCatalogEntry]
  resolve_package(entry, version_selector) -> AgentPackageIdentity + artifact_locator
  health() -> ProviderHealth                  # optional
```

**Provider kinds (future implementations):**

| Provider ID | Purpose |
|-------------|---------|
| `builtin` | Monorepo / first-party Intergrax agents |
| `local_developer` | Path / workspace dev packages |
| `enterprise_private` | Org registry, airgap bundles |
| `official_catalog` | Future Intergrax marketplace index |
| `governed_third_party` | Trusted external catalogs |

### 8.2 Catalog vs installation

| Property | Catalog | Installation store |
|----------|---------|-------------------|
| Authority for routing | No | No (bindings + revision) |
| Authority for digests | No — resolves candidates | **Yes** |
| Survives provider outage | N/A | **Yes** — digest-pinned artifacts local |
| Optional cache | Yes | No — durable SoT |

Execution runtime does not branch on provider type after installation — only `installation_ref`, `package_digest`, and trust evidence matter.

---

## 9. Package identity and resolution

### 9.1 Resolution pipeline

```text
1. Operator selects catalog_entry + explicit version (or digest)
2. CatalogSourceProvider.resolve_package → AgentPackageIdentity candidate
3. Fetch artifact bytes; compute/confirm package_digest
4. AgentPackageTrust.verify(publisher, signature, digest, revocation, org policy)
5. Persist artifact → installation record (see §11)
```

Production MUST reject floating `latest` as sole selector. Channel labels may map to digests at **selection time**; activated revisions store **digest only**.

### 9.2 Built-in monorepo agents

`builtin` provider resolves workspace members to `AgentPackageIdentity` with:

- `distribution_package_id` from agent `pyproject.toml`
- `package_digest` from **built artifact** or **workspace content hash policy** (implementation choice in AP-3 — architecture requires *some* immutable digest per activation)
- `catalog_source_id = builtin`

Built-in agents may skip external fetch but **never** skip trust/compatibility simulation in production profiles.

---

## 10. Trust and provenance

### 10.1 `AgentPackageTrust` (parallel to Platform Plugins)

Reuse **evidence pipeline patterns** from [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) §16–§18. **Do not** reuse plugin subject identity.

| Concern | Plugin pattern reused | Agent-specific subject |
|---------|----------------------|------------------------|
| Publisher identity | Evidence kind shape | `AgentPublisherIdentity` |
| Delivery source | `PluginDeliverySource` pattern | `AgentDeliverySource` (+ marketplace, org registry, workspace) |
| Immutable digest | Required evidence | On `AgentPackageIdentity` |
| Signature verification | Evidence pipeline | Agent package signing contract |
| Qualification | `PluginQualificationStatus` shape | `AgentPackageQualificationResult` |
| Platform compatibility | `PLATFORM_COMPATIBILITY` kind | + runtime graph simulation |
| Revocation | Deny policy pattern | Global list + org deny |
| Org allow/deny | Policy bundle | Org agent allowlist |

### 10.2 Trust evaluation points (fail closed)

Trust and qualification MUST be re-evaluated at:

- install (before `INSTALLED`);
- bind/enable (production profile);
- candidate runtime revision validation;
- activation;
- rollback target validation.

### 10.3 Trust record on installation

Every production `AgentInstallationRecord` carries:

```text
trust_evidence_refs[]
qualification_status
publisher_identity_ref
source_provider_id + source_entry_ref   # audit only
revocation_checked_at
org_policy_decision_ref                 # optional
```

---

## 11. Installation model

### 11.1 `AgentInstallationRecord` (conceptual)

```text
AgentInstallationRecord:
  installation_id              # immutable
  installation_slot_id         # stable per env + package line
  environment_id
  package_identity               # AgentPackageIdentity (digest required)
  installation_state             # substates §7.2
  active_for_slot: bool          # exactly one true per slot
  previous_installation_ref      # rollback pointer on slot
  artifact_store_ref             # where bytes live on host
  materialization_evidence_ref   # optional pre-app bundle proof
  trust_evidence_refs[]
  created_at, superseded_at, tombstoned_at
```

### 11.2 Installation slot semantics

- **`installation_slot_id`** is the stable anchor for upgrades: one slot per `(environment_id, distribution_package_id)` unless policy allows multiple slots (advanced — default **one**).
- **Upgrade** creates new `installation_id`, sets prior to `installed_previous`, moves `active_for_slot`.
- **Rollback** reactivates `previous_installation_ref` if still trusted and present.

### 11.3 Install flow (reference)

```text
Operator: Install package P for environment E
  1. CatalogSourceProvider.resolve → AgentCatalogEntry + AgentPackageIdentity candidate
  2. Fetch artifact; confirm package_digest
  3. AgentPackageTrust.verify
  4. Stage artifact → environment artifact store
  5. Commit AgentInstallationRecord (installed_active)     ← INSTALLED true here
  6. Emit audit: installation.created
```

Pre-install **application** graph simulation may run at step 3b or at bind/activate time; failure prevents bind/activate, not necessarily host install if org policy allows install-without-bind.

---

## 12. Application binding model

### 12.1 Binding target (architecture decision)

`ApplicationAgentBinding` references:

| Field | Target | Rationale |
|-------|--------|-----------|
| **Primary** | `installation_slot_id` | Stable across digest upgrades; config anchor |
| **Resolved** | `active_installation_id` | Current digest-pinned record (denormalized cache) |
| **Roster** | `logical_agent_id` | Merge key with manifest defaults |
| **Built-in fallback** | `builtin_package_ref` | Monorepo agents without formal install row (migration) |

**Normative:** bindings MUST NOT reference only a floating `package_version`. They MUST resolve through `installation_slot_id` → active `installation_id` → `package_digest`.

### 12.2 `ApplicationAgentBinding` (conceptual)

```text
ApplicationAgentBinding:
  application_binding_id
  application_id
  application_environment_id
  logical_agent_id                 # roster merge key
  installation_slot_id             # stable upgrade anchor
  active_installation_id           # denormalized; refreshed on upgrade
  enablement: bool
  config                           # AgentBinding.config semantics
  secret_refs
  policy_overrides                 # tool allow/deny, budget, etc.
  manifest_origin_ref              # optional link to manifest default key
  tombstone: bool                  # operator removal of manifest default
  binding_revision                 # monotonic for audit
```

### 12.3 Data surviving version changes

| Survives upgrade | Re-validated on upgrade | Slot-scoped only |
|------------------|-------------------------|------------------|
| `logical_agent_id`, `application_binding_id` | Config schema vs new `AgentContract` | `installation_slot_id` |
| `enablement` (unless policy blocks) | Capability / tool policy compatibility | |
| Config keys still in schema | Certification / qualification | |
| `secret_refs` (refs, not values) | Factory wiring if import paths change | |
| Policy overrides (tool lists) | Graph simulation | |

Upgrade MUST NOT delete durable binding rows — only update `active_installation_id` and increment `binding_revision`.

### 12.4 Bind flow

```text
  1. Verify installation_slot has installed_active record
  2. Create/update ApplicationAgentBinding (BOUND; enablement may be false)
  3. Validate config → CONFIGURED or remain BOUND with error
  4. Enable → ENABLED (subject to AgentGovernanceProfile)
```

---

## 13. Manifest and default merge semantics

### 13.1 Inputs

| Source | Role |
|--------|------|
| `ApplicationManifest.agents` (`AgentBinding`) | Release-scoped **default roster template** |
| Durable `ApplicationAgentBinding` records | Operator authority for add/remove/override |

### 13.2 Merge algorithm (deterministic)

```text
effective_roster = merge_manifest_defaults(manifest.agents, durable_bindings)
```

**Identity key:** `logical_agent_id` derived as:

```text
logical_agent_id = binding.contract_id
                 ?? manifest AgentBinding.contract_id
                 ?? stable slug from agent_type / distribution_package_id
```

**Precedence (highest wins unless noted):**

| Concern | Precedence |
|---------|------------|
| Roster membership | Durable binding exists → included; durable `tombstone=true` → **excluded** even if manifest lists agent |
| Operator-added agent | Durable binding with no `manifest_origin_ref` → included |
| Default bootstrap | Manifest-only entries → included when no tombstone and no conflicting durable row |
| `enablement` | Durable `enablement` **overrides** manifest `AgentBinding.enabled` |
| `config` | Deep merge: manifest defaults **under** durable overrides (durable wins per key) |
| `policy_overrides` | Durable only (manifest tool lists seed defaults if no durable row) |
| `default` agent flag | Manifest provides default; durable may override if explicit `default=true` on one binding — **conflict → fail closed** at merge |
| Version selection | **Never** from manifest — always `active_installation_id` / builtin digest |
| Factory wiring | Manifest `factory` / `builder_key` / `factory_path` unless durable specifies override |

### 13.3 Conflict behavior

| Conflict | Behavior |
|----------|----------|
| Duplicate `logical_agent_id` in durable store | Reject write — fail closed |
| Two `default=true` after merge | Merge fails; activation blocked |
| Duplicate capability across enabled agents | Allowed; Nexus routing uses registry + policy — document in capability graph; merge emits **warning** evidence |
| Manifest agent + tombstone | Excluded from effective roster |
| Enabled binding → missing installation | Merge fails closed for activation |

### 13.4 Output

Exactly **one** `EffectiveRoster` per `(application_id, application_environment_id, merge_inputs_revision)`:

```text
EffectiveRoster:
  effective_roster_revision_id    # hash(manifest_release_id, binding_revisions[], tombstones)
  entries: EffectiveRosterEntry[]
```

Each `EffectiveRosterEntry` carries resolved `package_digest`, factory wiring, merged config, effective enablement, and `installation_slot_id`.

---

## 14. Effective roster model

The effective roster is **derived only** — never a durable SoT. It is the sole input to:

1. dependency resolution (§15);
2. `CandidateApplicationRuntimeGraph` (§17);
3. `build_application_registry` (extended input contract — AP-3).

**Recompute triggers:**

- manifest release change;
- any binding CRUD or enablement change;
- installation slot active digest change;
- policy profile change affecting enablement.

---

## 15. Dependency-resolution architecture

### 15.1 Problem statement (monorepo vs operator install)

**Today (monorepo):**

```text
application pyproject.toml
+ agent pyproject.toml (transitive Tier-2)
+ shared monorepo uv.lock
  → ApplicationRuntimeGraph
```

**Future (operator install):**

```text
catalog package → installed agent NOT in application source pyproject
```

The architecture MUST produce a **complete immutable closure** for every activated runtime revision **without assuming** the shared monorepo `uv.lock` remains authoritative for third-party packages introduced after application release.

### 15.2 Dependency metadata layers

| Layer | Name | Source | Role |
|-------|------|--------|------|
| L1 | **RepositoryDependencyDeclaration** | Application release artifact (`pyproject.toml` or extracted release metadata) | Dev/monorepo baseline; app direct deps |
| L2 | **InstalledAgentRequirementSet** | `EffectiveRoster` → active `AgentInstallationRecord` → package metadata | Digest-pinned agent packages + their declared deps |
| L3 | **CandidateDependencySpecification** | Deterministic merge of L1 + L2 + platform baseline | Resolver input document |
| L4 | **DependencyResolverInput** | L3 + platform version + policy constraints + lock policy | Fed to resolver |
| L5 | **MaterializedRuntimeLock** | Resolver output | **Immutable closure artifact** |
| L6 | **CandidateApplicationRuntimeGraph** | Graph builder over L5 + contracts | Structural Tier-2 closure |
| L7 | **RuntimeRevision** | Activation bundles L5 + L6 + physical artifact | Production authority |

### 15.3 Merge into `CandidateDependencySpecification`

```text
CandidateDependencySpecification:
  application_release_id
  platform_version
  repository_declaration: RepositoryDependencyDeclaration
  agent_packages[]:           # from EffectiveRoster — each entry:
      distribution_package_id
      package_digest
      agent_project_metadata_ref   # extracted from installed package
  platform_extras[]             # from ApplicationEnvironmentProfile
  policy_constraints[]          # deny packages, pin overrides, Python version
  repository_lock_hint_ref      # optional — monorepo uv.lock slice for dev only
```

**Normative rules:**

1. Every agent in `EffectiveRoster` with `enablement=true` MUST contribute its **installed** package metadata — not catalog metadata.
2. Transitive Tier-2 agent deps come from **installed agent** `pyproject.toml` embedded in artifact metadata extraction — same semantic as today, but source is installation store not workspace path.
3. Third-party closure is produced by the resolver into `MaterializedRuntimeLock` — not by ad hoc union of floating requirements.

### 15.4 Resolver responsibilities (implementation-agnostic)

The resolver MUST:

- accept fully pinned **direct** agent digests;
- resolve transitive Python dependencies deterministically given same input bytes;
- detect conflicts (`AGENT_DEPENDENCY_CYCLE`, tier violations, incompatible pins) → fail closed;
- emit reproducible lock bytes (canonical JSON / TOML — exact encoding AP-3);
- record resolver algorithm version in lock for audit.

**No concrete package-manager implementation is mandated.** The architecture requires **deterministic input → deterministic output** semantics equivalent in evidence to today's `uv export --frozen` behavior.

### 15.5 Relationship to monorepo `uv.lock`

| Phase | `uv.lock` role | `MaterializedRuntimeLock` role |
|-------|----------------|--------------------------------|
| Monorepo dev | Shared workspace authority for declared members | May be **derived from** uv export for convenience |
| Release build | Input hint + CI gate | **Becomes authoritative** when operator agents added post-release |
| Production activation | Not sufficient alone if roster ⊄ workspace declaration | **Always authoritative** for active revision |

**Critical invariant:** post-release operator-installed third-party packages MUST appear only via `MaterializedRuntimeLock` on the active `RuntimeRevision`, not by mutating application source `pyproject.toml` or relying on a stale shared lock.

---

## 16. Materialized runtime lock / deterministic closure

### 16.1 `MaterializedRuntimeLock` (canonical artifact)

```text
MaterializedRuntimeLock:
  lock_id                         # content-addressed
  lock_digest
  resolver_algorithm_id + version
  created_at
  inputs_digest                   # hash(DependencyResolverInput canonical form)

  platform:
    intergrax_version
    python_version
    platform_extras[]

  packages[]:                     # complete closure — direct + transitive
    distribution_name
    version                       # resolved pin
    package_digest                # when available (wheels)
    dependency_of                 # parent edges for audit

  agent_closure[]:
    distribution_package_id
    package_digest
    role: direct|transitive

  repository_lock_hint_digest     # optional — traceability to monorepo uv.lock slice
  reproducibility_evidence:
    resolver_log_ref
    input_snapshot_ref

  rollback_evidence:
    supersedes_lock_id            # prior active lock
    rollback_eligible: bool
```

### 16.2 Immutability

- Once referenced by an **`active`** `RuntimeRevision`, a lock is **immutable**.
- New installs/upgrades produce **new** `lock_id`; activation swaps pointer.
- Rollback reactivates prior revision's lock by digest — validates trust + artifact presence.

### 16.3 Reproducibility evidence

For every active revision, the platform MUST persist evidence sufficient to answer:

- which exact package bytes were included;
- which resolver input produced the lock;
- which effective roster revision was used;
- which installation digests were active.

This satisfies audit without live catalog access.

---

## 17. Candidate runtime graph

`CandidateApplicationRuntimeGraph` extends [`APPLICATION_RUNTIME_GRAPH_MODEL.md`](APPLICATION_RUNTIME_GRAPH_MODEL.md) semantics:

```text
CandidateApplicationRuntimeGraph:
  schema_version
  application_id
  runtime_graph_digest
  materialized_runtime_lock_id      # required — graph ⊆ lock closure
  direct_agents[]
  transitive_agents[]
  direct_third_party_distributions[]  # app-declared only (unchanged rule)
  tier_violations[]                 # empty or fail
```

**Gates before activation:**

1. Acyclic Tier-2 closure.
2. Every agent ⊆ installed digests from effective roster.
3. Lock closure contains all agent-declared deps.
4. Certification / compatibility evaluators pass (production).

Output serializes to `.intergrax-runtime-graph.json` (schema v3+ — version bump AP-3) inside materialization context.

---

## 18. Runtime revision model

### 18.1 `RuntimeRevision` (canonical)

Identifies the **complete materialized application runtime**:

```text
RuntimeRevision:
  runtime_revision_id
  application_environment_id
  application_release_id            # app version / image tag lineage
  platform_version
  effective_roster_revision_id
  installed_agent_package_digests[] # from roster
  materialized_runtime_lock_id
  materialized_runtime_lock_digest
  runtime_graph_digest
  materialization_artifact_digest
  materialization_topology          # oci_image | venv_bundle | sandbox_sidecar
  policy_certification_evidence_refs[]
  revision_state                    # candidate|validated|active|superseded|failed
  supersedes_revision_id
  rollback_target_revision_id
  activated_at
```

### 18.2 Lifecycle

```text
candidate
  → validated     (lock + graph + trust + certification OK)
  → active        (atomic activation — application-visible)
  → superseded    (replaced by newer active)
  → failed        (validation or activation failure)
```

**Atomicity (application perspective):** routing and registry observe either the **prior** active revision or the **new** active revision — never a mixed agent closure.

---

## 19. Materialization model

### 19.1 Logical contract (topology-agnostic)

All deployment topologies implement the same **logical materialization contract**:

```text
MaterializationInput:
  RuntimeRevision (validated)
  MaterializedRuntimeLock
  CandidateApplicationRuntimeGraph
  EffectiveRoster
  ApplicationBuildContext

MaterializationOutput:
  materialization_artifact_digest
  artifact_locator                  # image ref, venv path, sidecar spec
  health_check_evidence_ref
  runtime_graph_manifest_path       # .intergrax-runtime-graph.json
```

### 19.2 Supported topologies (abstract)

| Topology | Physical output | Notes |
|----------|-----------------|-------|
| **OCI container image** | Image manifest digest | Current `build_application_image.py` path |
| **Isolated venv / app bundle** | Directory tree hash | Self-hosted without Docker |
| **Sandbox / sidecar** (future) | Sidecar unit digest | Model C trust tier — optional |

**Normative:** Docker is **not** required globally. Every topology MUST produce `materialization_artifact_digest` + embedded runtime graph manifest satisfying the same validation gates.

### 19.3 Materialization flow

```text
  1. Compute EffectiveRoster
  2. Build CandidateDependencySpecification → resolve → MaterializedRuntimeLock
  3. Build CandidateApplicationRuntimeGraph
  4. Run certification / trust / health pre-checks
  5. Materialize physical bundle (topology-specific adapter)
  6. Health validation on candidate bundle
  7. Create RuntimeRevision (candidate → validated)
  8. Activation (§20)
```

---

## 20. Activation and rollback model

### 20.1 Activation

```text
activate(runtime_revision_id):
  1. Assert revision_state == validated
  2. Begin activation transaction
  3. Mark prior active → superseded (retain rollback pointer)
  4. Mark candidate → active
  5. Commit durable activation record
  6. Host process restart OR rolling deploy OR hot-swap registry on startup hook
  7. build_application_registry(EffectiveRoster) → AgentRegistry
  8. Capture registry snapshot audit
```

Failure after step 5 but before healthy registry → **failed** activation; attempt automatic rollback to `rollback_target_revision_id`.

### 20.2 Rollback

```text
rollback(application_environment_id):
  1. Load rollback_target_revision_id (previous active)
  2. Re-verify trust + artifact presence + lock digest
  3. Atomic pointer swap to prior revision
  4. Redeploy / restart
  5. On failure → fail closed; alert; retain last known good if ambiguous
```

Rollback uses **digest-pinned** prior installation records — never floating version labels.

---

## 21. AgentRegistry projection

[`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §15 defines execution registry semantics. Distribution architecture adds:

| Rule | Detail |
|------|--------|
| Population source | `EffectiveRoster` entries with `enablement=true` after successful activation |
| Input contract | `build_application_registry(manifest, env, effective_roster=...)` (AP-3) |
| Install state | **Never** stored in registry |
| Disabled agents | Excluded from register (preferred) or registered not routable |
| Dynamic register API | **Not required** v1 — restart/redeploy after binding changes |

Registry snapshots (`registry_snapshot_store`) SHOULD include `effective_roster_revision_id`, `runtime_revision_id`, and installation/binding ids for audit — not as install DB.

---

## 22. Nexus routing boundary

Unchanged normative invariant from §16:

- Nexus selects by **capability** → `AgentRegistry.find_by_capability`.
- **ROUTABLE** ⊆ **REGISTERED** agents passing `evaluate_agent_routing` (lifecycle, certification, production mode).

Distribution plane MUST NOT add marketplace-specific routing branches or second Nexus.

```text
Task.required_capability
  → AgentRegistry.find_by_capability
  → evaluate_agent_routing(contract)
  → agent.run()
```

Install/enable/disable affects routing only after **next materialization + activation** (or startup hook on active revision).

---

## 23. Persistence and source-of-truth matrix

| Store (conceptual) | Owner | Durable? | Contents |
|--------------------|-------|----------|----------|
| Catalog provider index | Provider + optional **catalog cache** | Cache optional | `AgentCatalogEntry` |
| **Installation store** | Tier-0 | **Yes** | `AgentInstallationRecord`, artifact metadata |
| **Binding store** | Tier-0 | **Yes** | `ApplicationAgentBinding` |
| **Runtime revision store** | Tier-0 | **Yes** | `RuntimeRevision`, activation pointers |
| **Artifact metadata store** | Tier-0 | **Yes** | Digests, locators, tombstones |
| **Lock artifact store** | Tier-0 | **Yes** | `MaterializedRuntimeLock` blobs |
| Manifest defaults | Tier-3 release | Versioned with app | `ApplicationManifest.agents` |
| `AgentRegistry` | Tier-1 | **No** (process) | Execution projection |
| Effective roster | — | **No** | Derived per activation |

### Transaction / atomicity boundaries

| Boundary | Must be atomic |
|----------|----------------|
| Artifact persist + `INSTALLED` record | **Yes** |
| Binding write + revision bump | **Yes** |
| Lock persist + graph validation result | **Yes** |
| Activation pointer swap | **Yes** |
| Registry population | Process startup — consistent with active revision |
| Catalog cache refresh | **No** — best effort |

---

## 24. Concurrency and transaction semantics

| Scenario | Behavior |
|----------|----------|
| Concurrent installs same slot | Serialize on `installation_slot_id`; one wins, other gets conflict |
| Concurrent binding mutations | Optimistic locking on `binding_revision` |
| Install during activation | Activation uses roster snapshot taken at validation start; concurrent install requires re-validation |
| Runtime restart during activation | On startup, load **durable active** `RuntimeRevision` only; incomplete candidate ignored |
| Partial persistence failure | Roll back transaction; no `INSTALLED` / no `active` revision |

---

## 25. Failure and recovery semantics

| Failure | Behavior | Fail mode |
|---------|----------|-----------|
| Package resolution failure | Abort install; no record | Closed |
| Trust failure | Reject; quarantine artifact | Closed |
| Dependency conflict | Lock/graph simulation fails | Closed |
| Candidate lock failure | No revision promotion | Closed |
| Runtime graph failure | Activation blocked | Closed |
| Certification failure | Install/enable/activate rejected (prod) | Closed |
| Materialization failure | No activation; prior active remains | Closed |
| Health validation failure | Candidate failed; no activation | Closed |
| Activation failure | Rollback attempt to previous revision | Closed |
| Partial persistence failure | Transaction rollback | Closed |
| Active package revocation | Block new enables; flag install; policy may force disable on next materialization | Closed |
| Rollback failure | Alert; manual intervention | Closed |
| Removal with active bindings | Reject uninstall until unbind | Closed |
| Catalog unavailable after prior install | No impact on active digest-pinned revision | Open for discovery only |
| Disable with in-flight runs | Disable persisted; in-flight continue | Open for in-flight only |

---

## 26. Self-hosted / hosted / enterprise topology treatment

| Topology | Catalog | Artifact store | Materialization | Lock authority |
|----------|---------|----------------|-----------------|----------------|
| **Monorepo dev** | `builtin` + `local_developer` | Local / workspace | venv bundle or image | uv.lock hint + derived lock |
| **Self-hosted prod** | `enterprise_private` + `builtin` | Customer artifact registry | Image or venv bundle | `MaterializedRuntimeLock` on revision |
| **Hosted SaaS** | `official_catalog` + enterprise | Multi-tenant object store | OCI image per revision | Same lock semantics |
| **Airgap enterprise** | `enterprise_private` bundles | On-prem store | Image/bundle sideload | Pre-signed lock in bundle |

**Uniform logical contract** — topology affects **adapters only**, not state machine or identity model.

---

## 27. LKW proof boundary

LKW (`local_workspace_application`) remains **consumer only**.

| LKW MAY | LKW MUST NOT |
|---------|--------------|
| Present Discover / Install / Configure / Enable / Disable / Upgrade / Rollback / Uninstall UX | Own `AgentInstallationStore` |
| Call **generic** platform harness admin APIs | Implement `CatalogSourceProvider` |
| Prove end-to-end capability routing after platform materialization | Own materializer or resolver |
| Keep `GET /v1/local_workspace/agents` as registry introspection | Fork Nexus or registry |

**Future proof journey:**

```text
discover → install → bind/configure → enable → invoke (Nexus) → disable → upgrade/rollback → uninstall
```

All transitions invoke **platform-owned** capabilities — identical API surface for other Tier-3 apps.

---

## 28. Marketplace readiness

| Ready (architecture) | Explicitly not ready |
|----------------------|----------------------|
| `CatalogSourceProvider` with `official_catalog` | Billing, reviews, checkout |
| Digest-pinned install + lock | Publisher portal |
| Trust / revocation pipeline | Recommendation engine |
| Org private catalog provider | LKW-specific store |
| Neutral installation plane | Marketplace Nexus branch |

Marketplace = **catalog provider + publisher onboarding** — not execution fork.

---

## 29. Security implications

- Installation equals **deploying trusted code** — same trust posture as Platform Plugins §16.
- Production requires digest + trust evidence on every active revision.
- Org allow/deny intersects before `INSTALLED`.
- Revocation re-check at enable and activation.
- Secrets never in catalog or lock artifacts — binding `secret_refs` only.
- Materialization fail-closed on secret-like payloads (existing graph builder behavior).
- No hot arbitrary Python in production process.
- Audit tombstones retained on uninstall — artifacts removed per policy, records persist.

---

## 30. Observability and audit evidence

| Event | Minimum evidence |
|-------|------------------|
| `installation.created` | `installation_id`, `package_digest`, `catalog_source_id`, trust result |
| `binding.*` | `application_binding_id`, `logical_agent_id`, `binding_revision` |
| `enablement_changed` | prior/new, policy decision ref |
| `lock.produced` | `lock_digest`, `inputs_digest`, resolver version |
| `runtime_revision.activated` | full `RuntimeRevision` identity tuple |
| `activation.failed` / `rollback.*` | prior/new revision ids, reason |
| Registry snapshot | `runtime_revision_id`, `effective_roster_revision_id`, agent id set |

Align with observability spine — distribution events on Plane B; routing on Plane A.

---

## 31. Migration from manifest-only applications

| Phase | Behavior |
|-------|----------|
| **M0 (today)** | Manifest + pyproject only; graph from workspace |
| **M1** | Introduce durable bindings mirroring manifest; merge produces same roster |
| **M2** | Install API for non-workspace agents; lock produced on activate |
| **M3** | Operator enable/disable without redeploy; manifest defaults remain bootstrap |

**Feature flag:** `build_application_registry` may accept manifest-only fallback when no binding store mounted (dev/lab).

Built-in monorepo agents map to `builtin_package_ref` until explicit install records are required by policy.

---

## 32. Explicit non-goals

- Production code (AP-3+)
- LKW-local stores or marketplace product
- Second Nexus or execution registry
- Runtime hot-load of arbitrary agent code
- Marketplace billing / commercial workflows
- Replacing `AgentContract` or capability routing
- Concrete ORM schemas (deferred AP-3)
- Mandating Docker for all deployments
- Choosing uv/pip/poetry as permanent resolver implementation

---

## 33. Implementation dependency graph / recommended AP-3+ sequencing

```text
AP-3  Tier-0 contracts (identity, catalog provider IF, installation/binding records)
  → AP-4  Store interfaces + transactional services
  → AP-5  Trust coordinator (AgentPackageTrust) + verification
  → AP-6  Effective roster merge + DependencyResolverInput builder
  → AP-7  MaterializedRuntimeLock producer + graph simulation gates
  → AP-8  Materialization adapters (image, venv bundle)
  → AP-9  RuntimeRevision activation + rollback orchestration
  → AP-10 build_application_registry extension + registry snapshot fields
  → AP-11 Tier-3 harness admin API routes (generic)
  → AP-12 LKW consumer proof wiring
```

**AGENT-PLATFORM-3 may begin** with AP-3 Tier-0 contracts and store interfaces — this architecture document is complete and contains no blocking open questions for implementation start.

### Unresolved architecture blockers

**None.** Deferred implementation choices (exact lock file encoding, built-in digest policy for workspace agents, default topology per host profile) are explicitly bounded to AP-3/AP-8 without reopening distribution vs execution boundaries.

---

## Compliance checklist

- [x] Platform-neutral chain documented end-to-end
- [x] State dimensions remain orthogonal
- [x] Deterministic dependency closure independent of floating catalog state
- [x] `uv.lock` relationship explicit — not sole post-release authority
- [x] Binding targets `installation_slot_id` with digest resolution
- [x] `RuntimeRevision` identifies complete materialized runtime
- [x] Materialization topology-abstract
- [x] Marketplace = catalog provider only
- [x] LKW consumer boundary explicit
- [x] Cross-link from agent execution hub (see hub header)

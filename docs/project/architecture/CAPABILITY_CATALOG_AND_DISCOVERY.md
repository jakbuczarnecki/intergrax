# Capability Catalog and Discovery

**Intergrax Capability Catalog and Discovery** is the cross-domain **federated read and discovery plane** for V1 capability types — **Agent**, **Skill**, and **Tool** — that aggregates discoverable candidates from existing domain sources, supports query/filter/rank/recommendation, and hands off to domain-owned lifecycle authorities. It is **not** a universal runtime engine, **not** a merged registry, and **not** lifecycle authority.

The plane sits **above** domain catalogs and indexes and **below** governance, selection, and domain materialization: it answers *what capability candidates exist and match a need*; domain authorities answer *what is installed, enabled, materialized, and executable*.

## Why it matters

Today, an autonomous worker or agent should not receive a huge static bundle of every Tool and Skill the host could ever expose. Work is staged: goals decompose into steps, each step may surface a different capability need, and policy plus tenant scope further narrow what is appropriate.

The platform needs a path where runtime intelligence can reason about capability needs without bypassing governance:

```text
Goal
  ↓
Current work step
  ↓
Capability need
  ↓
Discovery
  ↓
Candidates
  ↓
Governance / policy
  ↓
Selection
  ↓
Existing runtime authority
  ↓
Execution
```

**Dynamic discovery does not mean dynamic bypass of governance.** Discovery surfaces candidates; it does not install, enable, materialize, grant permissions, or mutate registries. Durable production change always flows through the owning domain authority and, for Autonomous Work, through governance first.

## Current reality / maturity boundary

Read this hub in four layers — do not merge them into a single “shipped” headline.

**A. Frozen architecture (this document).** Capability Catalog is a **pure federating consumer**. AC-4 (agent acquisition) and AW-7A (worker capability recovery) remain **separate mechanisms** that may share lower-level primitives only where reuse is real. No `UniversalCapabilityEngine`, no `UniversalRegistry`, no merged `AgentRegistry` / `SkillRegistry` / `ToolRegistry`.

**B. Existing reusable implementation.** Platform Plugins (packaging, discovery primitives, trust/qualification vocabulary), Agent Distribution discovery/acquisition (AC-4), Tool selection layers, SkillResolver, domain registries, `wire_application_environment()` Tier-3 composition, AW-7A policy adapters (in progress).

**C. Missing / planned.** Federated Capability Catalog read model across Agent + Skill + Tool, cross-domain ranking utilities, Tools/Skills typed bootstrap evidence in application evidence aggregates, private enterprise catalog sources for Tool/Skill, third-party isolation beyond trusted in-process, monetization/metering product surfaces.

**D. Future product surfaces.** Public/private Marketplace is a **product layer above** federated discovery — presentation, publisher metadata, pricing metadata, availability — not runtime or lifecycle authority.

> [!NOTE]
> **Maturity boundary:** Frozen architecture documentation is **not** equivalent to shipped federated catalog federation or Marketplace product rollout. AC-4 agent discovery/acquisition is **implemented and frozen** for reference production V1 under Agent Distribution; cross-domain Capability Catalog federation is **planned**.

**Primary audience:** CTOs, principal/staff engineers, software architects, and AI platform engineers evaluating how Intergrax separates capability discovery from domain lifecycle and execution.

**Related canon:** [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) · [`AGENT_DISTRIBUTION.md`](AGENT_DISTRIBUTION.md) · [`TOOLS.md`](TOOLS.md) · [`SKILLS.md`](SKILLS.md) · [`AUTONOMOUS_WORK.md`](AUTONOMOUS_WORK.md) · [`APPLICATION_RUNTIME_GRAPH_MODEL.md`](APPLICATION_RUNTIME_GRAPH_MODEL.md) · [plan](../maintainers/plans/CAPABILITY_CATALOG_AND_DISCOVERY.md)

## At a glance

| Concern | Responsibility / current boundary |
| -------- | ----------------------------------- |
| **Catalog role** | Federated read model over domain sources — **pure federating consumer** |
| **Discovery role** | Query, filter, rank, recommend candidates — **read-only** |
| **V1 capability types** | Agent, Skill, Tool only |
| **Agent lifecycle** | Agent Distribution → RuntimeRevision → AgentRegistry → Nexus — **unchanged** |
| **Skill lifecycle** | SkillProfile → SkillRegistry → SkillResolver — **domain-owned** |
| **Tool lifecycle** | ToolProfile → ToolRegistry → governed Tool execution — **domain-owned** |
| **Tier-3 composition** | `wire_application_environment()` remains canonical entry |
| **Platform Plugins** | Package/plugin coordination — **not replaced** |
| **AC-4** | Agent acquisition discovery plane — **separate from AW-7A** |
| **AW-7A** | Worker obstacle → tool/skill discovery → bounded decision — **separate from AC-4** |
| **Marketplace** | Product surface above catalog — **not runtime** |
| **Maturity** | Architecture frozen; federation and cross-domain maturity **planned** — see [Current reality](#current-reality--maturity-boundary) |
| **Go deeper** | [Core mental model](#core-mental-model) · [§Hard invariants](#hard-invariants-normative) · [§Forbidden flows](#forbidden-flows) · [plan](../maintainers/plans/CAPABILITY_CATALOG_AND_DISCOVERY.md) |

## Core mental model

Capability Catalog **reads** existing sources, **aggregates** candidates, and **enables discovery**. It does **not** install, activate, materialize, execute, mutate registries, or replace domain lifecycle authority.

```text
        CATALOG SOURCES

Public Marketplace
Enterprise Private
Local / Built-in
Application-visible sources
        │
        ▼
Capability Catalog
federated read model
        │
        ▼
Capability Discovery
query / filter / rank
        │
        ▼
Candidates / Recommendation
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Agent  Skill   Tool
 │      │       │
domain authorities
 │      │       │
runtime registries / runtime
```

**Normative separation:**

```text
DISCOVERY ≠ SELECTION ≠ ENABLEMENT ≠ MATERIALIZATION ≠ ACTIVATION ≠ EXECUTION
```

## Scope

### In scope (V1)

```text
Capability Ecosystem V1:
  Agent
  Skill
  Tool
```

Cross-domain federation, discovery vocabulary, effective availability model, governance boundaries, evidence requirements, and integration points with AC-4 and AW-7A.

### Out of scope (V1)

This document does **not** own runtime semantics for Integrations, Memory, RAG, Context, Policy, or Models except where a **boundary relation** is required. It does **not** define a universal capability engine, merged registry, or Marketplace billing/settlement implementation.

---

## Engineering canon

| Topic | Section |
|-------|---------|
| Platform Plugins relation | [§Platform Plugins relation](#platform-plugins-relation) |
| Domain ownership | [§Domain ownership](#domain-ownership) |
| Discovery states | [§Discovery states and separation](#discovery-states-and-separation) |
| Federated catalog model | [§Federated catalog model](#federated-catalog-model) |
| Dynamic discovery use cases | [§Dynamic discovery](#dynamic-discovery) |
| Effective availability | [§Effective capability availability](#effective-capability-availability) |
| Governance | [§Governance](#governance) |
| Version / provenance | [§Version and provenance](#version-and-provenance) |
| Trust / qualification | [§Trust and qualification](#trust-and-qualification) |
| Multi-tenancy / scopes | [§Multi-tenancy and scopes](#multi-tenancy-and-scopes) |
| Marketplace | [§Marketplace relation](#marketplace-relation) |
| Monetization | [§Monetization boundary](#monetization-boundary) |
| Enterprise deployment | [§Enterprise deployment](#enterprise-deployment) |
| Security / third-party | [§Security and third-party code](#security-and-third-party-code) |
| Observability / evidence | [§Observability and evidence](#observability-and-evidence) |
| Hard invariants | [§Hard invariants](#hard-invariants-normative) |
| Forbidden flows | [§Forbidden flows](#forbidden-flows) |
| Maturity boundary detail | [§Maturity boundary detail](#maturity-boundary-detail) |

---

## Platform Plugins relation

```text
Platform Plugins
=
package / plugin coordination

Capability Catalog & Discovery
=
capability-level federated read / discovery plane
```

| Layer | Owns |
|-------|------|
| **Platform Plugins** | Packaging, setuptools discovery primitives, manifest metadata, config/secrets/DI conventions, trust vocabulary, qualification vocabulary, compatibility, admission/evidence patterns at the **package boundary** |
| **Capability Catalog & Discovery** | Cross-capability-type **candidate aggregation**, federated query, ranking/recommendation **read semantics** |

Capability Catalog **consumes** Platform Plugins primitives where applicable (e.g. entry-point discovery, trust vocabulary). It **does not** replace Platform Plugins and **does not** create a second plugin engine or global plugin lifecycle authority.

Tier-3 composition remains:

```text
ApplicationEnvironmentProfile
  ↓
wire_application_environment()
  ↓
ToolRegistry / SkillRegistry
  (+ other domain wiring)
```

See [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) §Tier-3 host composition.

---

## Domain ownership

Each V1 capability type retains **its own** runtime path. Do not generalize these into a single lifecycle engine.

| Capability | Lifecycle / runtime authority |
| ---------- | ----------------------------- |
| **Agent** | Agent Distribution → dependency closure → materialization → `RuntimeRevision` → `AgentRegistry` → Nexus |
| **Skill** | `SkillProfile` → `SkillRegistry` → `SkillResolver` (composition into agent contract; not direct execution) |
| **Tool** | `ToolProfile` → `ToolRegistry` → governed `ToolRuntime` execution |

**Agent canonical chain (unchanged):**

```text
catalog
  ↓
installation
  ↓
binding
  ↓
dependency closure
  ↓
materialization
  ↓
RuntimeRevision
  ↓
AgentRegistry
  ↓
Nexus
```

Capability Catalog **must not** short-circuit this chain. See [`AGENT_DISTRIBUTION.md`](AGENT_DISTRIBUTION.md).

---

## Discovery states and separation

The platform uses a **shared vocabulary** across domains. Not every domain implements every state; states are **not** a shared lifecycle engine.

```text
AVAILABLE
  ≠ DISCOVERED
  ≠ SELECTED
  ≠ INSTALLED
  ≠ ENABLED
  ≠ MATERIALIZED
  ≠ ACTIVE
  ≠ EXECUTABLE / ROUTABLE / RESOLVABLE
```

| State (conceptual) | Typical meaning | Authority |
|------------------|-----------------|-----------|
| **AVAILABLE** | Listed in a catalog source; may be queried | Catalog source / federated read model |
| **DISCOVERED** | Returned by a discovery query for a need | Discovery plane (read-only) |
| **SELECTED** | Chosen candidate for downstream action | Selection policy / operator / AW decision |
| **INSTALLED** | Durable artifact on host (agents) or package present | Domain distribution / host install |
| **ENABLED** | Host or binding enablement | Domain profile / Agent Distribution |
| **MATERIALIZED** | Immutable runtime artifact produced | Domain materialization (agents) |
| **ACTIVE** | Serving revision / routable runtime subset | `RuntimeRevision`, registry projection |
| **EXECUTABLE / ROUTABLE / RESOLVABLE** | May be invoked or resolved for work | ToolRuntime, Nexus, SkillResolver |

---

## Federated catalog model

Capability Catalog is **not** a central database of truth. It is a **federated read model** over domain-owned and product-owned sources.

```text
Agent catalogs ──┐
Skill catalogs ──┤
Tool catalogs ───┤
Private sources ─┤
                 ▼
       federated discovery
```

**Requirements:**

- **Source identity** — every candidate retains which provider/source produced it.
- **Provenance** — publisher, package identity, version or digest where the domain defines them.
- **Read-only federation** — aggregation and query only; no write-through to registries.
- **Deterministic conflict handling** — identity or evidence conflicts **fail closed**; no silent merge of incompatible candidates.

Agent federation patterns (`FederatedAgentDiscoveryStrategy`, source-qualified `AgentDiscoveryCandidateIdentity`) are **reference implementations** for the Agent slice; Skill and Tool federation must follow the same principles without merging domain registries.

---

## Dynamic discovery

Two primary V1 use cases illustrate **separate** mechanisms sharing only justified lower-level primitives.

### Agent acquisition (AC-4)

```text
need specialist
  ↓
agent discovery (AC-4)
  ↓
matching
  ↓
selection
  ↓
Agent Distribution (AC-3 lifecycle)
```

AC-4 is **frozen** under Agent Distribution §35. It owns task capability resolution, agent discovery, matching, selection, and acquisition handoff to AC-3. It is **not** the Capability Catalog subsystem and **must not** be merged with AW-7A.

### Worker capability recovery (AW-7A)

```text
worker obstacle
  ↓
capability need
  ↓
tool / skill discovery
  ↓
bounded decision (A0–A4)
  ↓
governed downstream action
```

AW-7A is **in progress** under Autonomous Work. It may discover Tools and Skills (and related surfaces per AW plan) but **cannot** directly install, mutate registries, or elevate authority. Durable change:

```text
AW decision
  ↓
governance
  ↓
existing domain authority
  ↓
controlled lifecycle
```

### AC-4 vs AW-7A (frozen)

| Aspect | AC-4 | AW-7A |
|--------|------|-------|
| **Primary need** | Specialist agent for delegated work | Recover from missing capability during worker execution |
| **V1 types** | Agent | Tool, Skill (ordered search per AW plan) |
| **Lifecycle handoff** | `DynamicAgentAcquisitionService` → AC-3 | Governance → domain authority |
| **Merge rule** | **MUST NOT** merge into one subsystem |

**Shared lower-level primitives (allowed when justified):** candidate identity patterns, evidence semantics, disposition vocabulary where semantically aligned, conflict handling, federation pattern, ranking utilities. **Forbidden:** one universal discovery port that collapses AC-4 and AW-7A responsibilities.

---

## Effective capability availability

Logical model for what a worker or agent can **consider** at a work step (not a new runtime registry):

```text
Host availability
∩ Application configuration
∩ Agent / Worker scope
∩ Policy
∩ Tenant / organization constraints
∩ Current Unit of Work
=
Effective Capability Set
```

Implementation may compute this as a query-time view. **Do not** introduce a new global runtime inventory authority for this intersection; domain registries and profiles remain authoritative for what is actually wired and executable.

Tool selection layers ([`TOOLS.md`](TOOLS.md)) and Skill host enablement ([`SKILLS.md`](SKILLS.md)) already narrow host-visible sets; policy and worker scope further narrow at execution time.

---

## Governance

**Normative rule:** each layer may **narrow** authority; no layer may **unilaterally extend** it.

| Action | Grants permission? |
|--------|-------------------|
| Discovery | **No** |
| Ranking / recommendation | **No** |
| Selection | **No** (selection ≠ authorization) |
| Catalog listing | **No** |

Governance and policy bundles may exclude candidates discovered from private or public sources. Fail-closed dispositions apply when evidence or identity conflicts with policy.

Autonomous Work durable mutations **must** cross governance before domain lifecycle ([`AUTONOMOUS_WORK.md`](AUTONOMOUS_WORK.md) capability acquisition model).

---

## Version and provenance

Candidates and selections should carry, where the domain defines them:

| Field | Agent | Skill | Tool |
|-------|-------|-------|------|
| **Source identity** | `AgentDiscoveryCandidateIdentity` | Catalog/source id | Catalog/bundle source |
| **Package / artifact identity** | `AgentPackageIdentity`, digest | `skill_id`, bundle | `tool_id`, contract |
| **Version** | Agent package version | Skill manifest version | Tool contract version |
| **Runtime revision** | `RuntimeRevision` when active | N/A (resolver-time) | Registry bootstrap revision |
| **Immutable artifact digest** | Required for install | Where packaged | Where packaged |

### Skill version pinning (Stage 6 — closed)

Enterprise skill version correctness is owned by the Skill domain (`SkillResolver`, `ResolvedSkillPack`, `AgentRegistry.get_resolved_skill_pack`). Root agent declarations are **PINNED**; transitive `requires_skills` are **MATERIALIZED** with exact versions captured in immutable snapshots. Capability Catalog discovery projects catalog `version_label` and `SkillVersionBindingDisposition` only — it does not resolve or override runtime bind versions. See [SKILLS.md](SKILLS.md) and Stage 6 in the [plan](../maintainers/plans/CAPABILITY_CATALOG_AND_DISCOVERY.md).

---

## Trust and qualification

Reuse Platform Plugins trust and qualification **vocabulary** and domain admission patterns. Do **not** create a second global trust engine for Capability Catalog.

| Concern | Owner |
|---------|-------|
| Package trust evaluation | Domain-specific (e.g. `AgentPackageTrust`) + plugin evidence |
| Production qualification | Domain-owned admission reports |
| Discovery trust surfacing | Read-only metadata on candidates |

**Discoverable ≠ production-qualified.** Installation does not imply activation. Catalog visibility does not imply trust allowance.

---

## Multi-tenancy and scopes

Discovery queries must be scope-aware. Scoping is part of the query, governance filter, and existing domain authority — **not** a single global store.

| Scope | Role in discovery |
|-------|-------------------|
| **Organization** | Private catalog visibility, enterprise policy |
| **Tenant** | Data and capability isolation boundaries |
| **Application** | `ApplicationEnvironmentProfile`, manifest defaults |
| **Worker / agent** | Task or worker-specific capability need |
| **Unit of Work** | Stage-specific rediscovery and effective set |

---

## Marketplace relation

Marketplace is a **product** built on top of federated catalog sources.

**Marketplace may:** present listings, search, filter, show versions and publishers, expose pricing metadata and availability signals.

**Marketplace must not:** install, activate, mutate registries, execute tools/skills/agents, or become lifecycle authority.

Public Marketplace is **optional** for platform operation (see Enterprise deployment).

---

## Monetization boundary

Billing and metering are **separate subsystems**.

- Runtime may emit **typed usage events** with source-qualified identity.
- **Do not** embed prices in `ToolRegistry`, `SkillRegistry`, or `AgentRegistry`.
- **Skill is not** a direct executable invocation surface for metering — tools and agent delegations carry execution semantics.

Selection evidence from AC-4 already anticipates future usage accounting; implementation is **planned**, not shipped.

---

## Enterprise deployment

Architecture must support:

- public cloud
- private cloud
- on-premises
- air-gapped environments

Therefore **public Marketplace is not a mandatory platform dependency**. Federated catalog must operate from local, enterprise-private, and built-in sources alone.

### Private Tool and Skill catalog sources (Stage 7)

Private Tool and Skill catalogs implement the same read-only `CapabilityCatalogSource` port as built-in bundle adapters. Entries use `CapabilitySourceKind.ENTERPRISE_PRIVATE` with a stable per-source `source_id` (for example `enterprise.acme.tools`, `enterprise.acme.skills`).

**Catalog presence does not imply installation, profile enablement, registry materialization, entitlement, trust, or runtime routability.**

Operator flow (Tool and Skill):

```text
1. discover private capability via federated catalog query
2. operator/admin chooses acquisition path (Platform Plugin package availability)
3. domain plugin registered through existing catalog/bootstrap paths
4. ToolProfile or SkillProfile updated by operator
5. wire_application_environment()
6. ToolRegistry or SkillRegistry materializes the capability
7. availability evidence may then classify HOST_AVAILABLE separately from CATALOG_AVAILABLE
```

Discovery **must not** execute steps 2–6.

### Adaptive work-stage discovery (Stage 8)

Workers rediscover capabilities per work stage as goals and current steps evolve — not once at bootstrap.

```text
Goal (durable objective)
        ↓
Current work stage (step identity + stage objective)
        ↓
WorkStageCapabilityNeed (wraps CapabilityDiscoveryQuery)
        ↓
Stage-3 discovery → Stage-4 ranking → Stage-5 governance
        ↓
EffectiveCapabilitySet (HOST_AVAILABLE ∩ governed allowed)
        ↓
WorkStageCapabilityDiscoveryEvidence
```

**`EffectiveCapabilitySet` is a deterministic query result — not runtime inventory authority, not lifecycle authority, not permission authority.** Catalog-only (`CATALOG_AVAILABLE`) candidates may appear in governed discovery evidence but do not become executable effective members until host/profile availability evidence classifies them as `HOST_AVAILABLE`. The public contract is self-validating: every effective member must be an exact `governed_result.allowed` candidate with `HOST_AVAILABLE` availability and unique identity; derived catalog-only and transition evidence cannot diverge from canonical result state.

Capability Catalog Stage 8 provides stage-scoped adaptive discovery; it does **not** perform Autonomous Work recovery, acquisition, or registry mutation.

### Stage 9 — Autonomous Work bridge (implemented)

AW-7A consumes Tool/Skill discovery through thin AW adapters over governed catalog discovery (`intergrax/autonomous_work/capability_catalog_discovery_adapters.py`):

```text
WorkerCapabilityNeed
        ↓ map_worker_capability_need_to_discovery_query
Stage-3 discovery → Stage-4 ranking → Stage-5 governance
        ↓ HOST_AVAILABLE executable narrowing
WorkerCapabilityCandidate projection
        ↓
WorkerCapabilityAcquisitionDecisionService (A0–A4 unchanged)
```

AW retains A0–A4 decision authority. Catalog does not execute acquisition. Stage 8 (`WorkStageCapabilityNeed`) remains for callers with real work/stage context; AW recovery uses the Stage 3–5 plane (PATH B) because `WorkerCapabilityDiscoveryRequest` does not carry canonical work/stage identity.

Registry-backed Tool/Skill adapters are **not** canonical AW production discovery after Stage 9; no silent catalog→registry fallback.

Contracts: `intergrax/contracts/autonomous_work/capability_acquisition.py`. Adapters: `intergrax/autonomous_work/capability_catalog_discovery_adapters.py`.

---

## Security and third-party code

**Current fact:** Platform Plugins use a **trusted in-process** Python extension model ([`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md)). Installing third-party packages deploys code into the host process; Platform Plugins provide admission and qualification controls, not arbitrary-code sandboxing.

Public marketplace growth may require **stronger isolation** (remote execution, isolation providers). That is **future roadmap work** — do not design a new execution engine in Capability Catalog V1. Record as Stage 12 in the [plan](../maintainers/plans/CAPABILITY_CATALOG_AND_DISCOVERY.md).

---

## Observability and evidence

Required evidence classes:

| Evidence | Purpose |
|----------|---------|
| **Provenance** | Which source produced the candidate |
| **Discovery evidence** | Query, filters, candidate set |
| **Selection evidence** | Why a candidate was chosen |
| **Usage evidence** | Execution-time consumption (downstream) |
| **Domain lifecycle evidence** | Install, bind, activate, resolve (domain-owned) |

**Normative rule:**

```text
evidence ≠ runtime source of truth
```

Evidence supports audit and governance; registries and `RuntimeRevision` remain execution truth.

**Roadmap requirement:** typed Tools and Skills bootstrap evidence aggregated into application evidence (alongside Security, Policy, Context, Memory today) — evidence only, not registry authority. See [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) observability chain.

---

## Hard invariants (normative)

1. Discovery **MUST NOT** mutate runtime state or registries.
2. Discovery **MUST NOT** be lifecycle authority.
3. Discovery **MUST NOT** be runtime source of truth.
4. Discovery **MUST NOT** grant permissions.
5. Selection **MUST** remain separate from discovery.
6. Selection **MUST NOT** imply authorization.
7. Capability Catalog **MUST** remain a federating consumer (read/aggregate/query).
8. Capability Catalog **MUST NOT** become a global runtime registry.
9. `AgentRegistry`, `SkillRegistry`, and `ToolRegistry` **MUST** remain domain-owned.
10. Platform Plugins **MUST** remain canonical package/plugin coordination.
11. `wire_application_environment()` **MUST** remain canonical Tier-3 composition entry.
12. Agent Distribution **MUST** remain sole agent lifecycle authority.
13. AW durable changes **MUST** cross governance and domain authority.
14. AC-4 and AW-7A **MUST NOT** be merged into one lifecycle or universal discovery subsystem.
15. Federated discovery **MUST** preserve source identity and provenance on every candidate.
16. Conflict handling **MUST** fail closed where identity or evidence conflicts.
17. Marketplace **MUST NOT** bypass domain lifecycle.
18. Evidence **MUST NOT** become runtime inventory authority.
19. V1 **MUST** remain Agent + Skill + Tool only for catalog federation scope.
20. New shared abstractions **MUST** be justified by real cross-domain reuse — no speculative generalization.

---

## Forbidden flows

### Forbidden 1 — Marketplace install bypass

```text
Worker
  ↓
Marketplace
  ↓
pip install
  ↓
ToolRegistry
  ↓
execute
```

### Forbidden 2 — Catalog mutates registry

```text
Capability Catalog
  ↓
modify ToolRegistry
```

### Forbidden 3 — Marketplace registers agents

```text
Marketplace
  ↓
AgentRegistry.register()
```

### Forbidden 4 — Discovery as authority

```text
Discovery result
=
runtime authority
```

### Forbidden 5 — Universal registry

```text
Universal Capability Registry
(merging AgentRegistry + SkillRegistry + ToolRegistry)
```

### Forbidden 6 — Universal capability engine

```text
UniversalCapabilityEngine
(controlling install + execute across domains)
```

---

## Maturity boundary detail

### A. Frozen architecture

- Pure federating consumer model
- V1 types: Agent, Skill, Tool
- Domain lifecycle ownership unchanged
- AC-4 / AW-7A separation
- Platform Plugins coordination unchanged
- `wire_application_environment()` unchanged
- Marketplace and monetization boundaries
- Evidence ≠ runtime truth

### B. Existing reusable implementation

| Component | Reuse |
|-----------|-------|
| Platform Plugins | Discovery loader, trust/qualification vocabulary, Tier-3 evidence patterns |
| AC-4 | Agent discovery, federation, matching, selection, acquisition handoff |
| Agent Distribution | `CatalogSourceProvider`, trust, AC-3 lifecycle |
| Tools | `ToolProfile`, selection layers, semantic/hierarchical selection |
| Skills | `SkillProfile`, `SkillRegistry`, `SkillResolver` |
| AW-7A | Capability need classification, ordered search policy (in progress) |
| Tier-3 | `wire_application_environment()` |

### C. Missing / planned

| Gap | Status |
|-----|--------|
| Cross-domain Capability Catalog federation | Planned |
| Cross-domain ranking shared utilities | Planned (only if reuse proven) |
| Skill version pinning | **Implemented** — Skill domain + discovery projection (Stage 6) |
| Tools/Skills typed bootstrap evidence | Planned |
| Private enterprise catalog for Tool/Skill | Planned |
| Third-party isolation beyond in-process | Future |
| Monetization / metering consumer | Future |
| Marketplace product surface | Future |

---

## Compliance checklist

- [x] Federated read model documented; no central lifecycle authority
- [x] V1 scope Agent + Skill + Tool explicit
- [x] Platform Plugins relation explicit
- [x] Domain ownership table preserves separate runtime paths
- [x] Discovery state vocabulary frozen
- [x] AC-4 and AW-7A separation documented
- [x] Effective capability availability as logical model only
- [x] Governance narrowing rules documented
- [x] Skill version pinning closed in Skill domain + discovery projection (Stage 6)
- [x] Marketplace / monetization / enterprise boundaries documented
- [x] Hard invariants and forbidden flows listed
- [x] Maturity boundary A/B/C explicit
- [x] Plan pair cross-linked

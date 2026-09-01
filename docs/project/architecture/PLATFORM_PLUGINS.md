# Platform Plugins

**Platform Plugins** is Intergrax's canonical coordination layer for third-party extension packages: shared packaging, discovery, manifest metadata, configuration, secrets/DI conventions, trust, qualification, compatibility, and lifecycle vocabulary — without replacing domain-owned runtime semantics for Tools, Skills, Integrations, RAG, context, or agents.

## Why it matters

Without a common extension model, each domain can invent its own packaging, discovery, and trust story:

- installable packages get confused with runtime authority,
- third-party extensions can bypass domain-owned contracts,
- trust, config, secrets, and lifecycle drift apart across surfaces,
- a plugin layer risks becoming a monolithic second runtime.

Platform Plugins coordinates **packaging, discovery, manifest/metadata, configuration, secrets/DI, trust, qualification, compatibility, and lifecycle** at the package boundary. It does **not** own runtime semantics that belong to domain architecture documents.

## Current reality / maturity boundary

Read this hub in four layers — do not merge them into a single “shipped” headline.

**A. Canonical / frozen architecture.** PLATFORM-PLUGIN-2 freezes taxonomy, platform-vs-domain boundaries, DO-NOT-UNIFY decisions, and cross-cutting coordination rules. Domain documents ([`INTEGRATIONS.md`](INTEGRATIONS.md), [`TOOLS.md`](TOOLS.md), [`SKILLS.md`](SKILLS.md), [`RAG.md`](RAG.md), [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md), etc.) remain **authoritative for runtime semantics**.

**B. Implemented slices (capability-specific).** PLATFORM-PLUGIN-3..9 program stages delivered package-level contracts, shared discovery primitives, config/secrets/DI conventions (PLUGIN-5 **Done**), lifecycle/compatibility/trust/qualification vocabulary, reference external and host-embedded examples, and a dual-mode Tools E2E proof (`tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py`). These are **partial, surface-specific** outcomes — not a universal plugin runtime.

**C. Program closeout vs ongoing domain work.** The PLATFORM-PLUGIN-1..9 program is **closed** per maintainer plan; residual Protocol v2 audit findings (§Protocol v2 platform extensibility target invariants) remain **planned, not implemented**. Domain programs continue to own capability behavior independently.

**D. Not yet established.** A **complete third-party install-to-runtime E2E proof** across all public extension surfaces is **not** implied. Discoverable ≠ production-qualified. Installation does not imply activation. Platform Plugins ≠ universal execution engine.

> [!NOTE]
> **Maturity boundary:** Canonical architecture is frozen. Implementation maturity is slice-specific. Frozen architecture documentation is not equivalent to universal production rollout or complete third-party lifecycle qualification.

**Primary audience:** CTOs, principal/staff engineers, and extension authors evaluating how Intergrax coordinates packages without collapsing domain contracts.

**Related canon:** [`EXTENSION_AUTHOR_GUIDE.md`](../technical/guides/EXTENSION_AUTHOR_GUIDE.md) · [`AGENT_DISTRIBUTION.md`](AGENT_DISTRIBUTION.md) (agent-specific distribution — parallel concern)

## At a glance

| Concern | Boundary |
| -------- | -------- |
| **Package model** | Installable Python distribution; may expose multiple capabilities |
| **Discovery** | Shared setuptools loader (`intergrax/core/plugins/discovery.py`) — partial adoption; opt-in flags |
| **Registration** | Domain-owned catalogs and host composition — no global registry merge |
| **Manifest** | Optional `[tool.intergrax.plugin]` metadata; entry points remain required |
| **Config** | Host/domain resolves before materialization; PLUGIN-5 matrix (§12.3) |
| **Secrets** | Host/domain credential bindings — no global secret API |
| **DI** | Domain-owned injection shapes — no global Platform Plugin container |
| **Trust** | `platform_qualification.py` vocabulary; trusted in-process model |
| **Qualification** | Domain- and capability-specific; package ≠ blanket production admission |
| **Compatibility** | `platform_semantics.py` explicit-version checks |
| **Lifecycle** | Vocabulary frozen; no global lifecycle engine |
| **Domain ownership** | Integrations, Tools, Skills, RAG, context, VK, security, agents — domain docs own semantics |
| **Maturity** | Architecture frozen; slice-specific implementation; universal E2E proof not established — see [Current reality](#current-reality--maturity-boundary) and §26 |
| **Go deeper** | [Engineering canon](#engineering-canon) · [§6 Platform Plugin definition](#6-platform-plugin-definition) · [§7 responsibility model](#7-platform-vs-domain-responsibility-model) · [plan](../maintainers/plans/PLATFORM_PLUGINS.md) |

## Core mental model

```text
third-party package
  → plugin metadata / manifest
  → discovery
  → trust + compatibility + config
  → domain-owned registration / composition
  → domain runtime
```

**Platform Plugins ≠ universal execution engine.** Platform Plugins ≠ replacement for Tools / Skills / Integrations / RAG / Agent Distribution. Platform Plugins coordinates packaging, discovery, and trust; **domain contracts own semantics**.

```text
COMMON PLATFORM COORDINATION
        +
DOMAIN-OWNED CAPABILITY CONTRACTS
```

Third-party authors install Python packages. Hosts and applications decide which discovered capabilities are enabled, configured, injected, and qualified for production.

## Engineering canon

**Status:** Canonical architecture (PLATFORM-PLUGIN-2 frozen) · PLUGIN-5 config/secrets/DI conventions **Done** · PLATFORM-PLUGIN-1..9 program **closed**
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Program roadmap:** [`plan/PLATFORM_PLUGINS.md`](../maintainers/plans/PLATFORM_PLUGINS.md)
**Audit evidence:** [`plan/PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](../maintainers/plans/PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md)
**Author guide:** [`technical/guides/EXTENSION_AUTHOR_GUIDE.md`](../technical/guides/EXTENSION_AUTHOR_GUIDE.md)
**Target:** [`technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Implement / audit default:** §1–§8 (purpose, taxonomy, platform definition, responsibility model, package/capability model, discovery, registration).
- **Cross-cutting decisions:** §9–§20 (manifest, config/secrets/DI, lifecycle, compatibility, trust, conflicts, qualification, observability, public API, failure model).
- **Program boundaries:** §21–§27 (DO-NOT-UNIFY, invariants, future stages, non-goals).
- **Max reads:** at most **one** related domain architecture file per session unless RESUME cites more.

---

## Table of contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Architectural context](#2-architectural-context)
3. [Current-state baseline](#3-current-state-baseline)
4. [Canonical terminology](#4-canonical-terminology)
5. [Canonical taxonomy](#5-canonical-taxonomy)
6. [Platform Plugin definition](#6-platform-plugin-definition)
7. [Platform vs domain responsibility model](#7-platform-vs-domain-responsibility-model)
8. [Package and capability model](#8-package-and-capability-model)
9. [Discovery architecture](#9-discovery-architecture)
10. [Registration and composition architecture](#10-registration-and-composition-architecture)
11. [Manifest and metadata model](#11-manifest-and-metadata-model)
12. [Configuration and secrets](#12-configuration-and-secrets)
    - [12.3 Cross-surface configuration, secrets and DI matrix](#123-cross-surface-configuration-secrets-and-di-matrix)
    - [12.4 Canonical configuration flow](#124-canonical-configuration-flow)
13. [Dependency injection](#13-dependency-injection)
14. [Lifecycle model](#14-lifecycle-model)
15. [Compatibility and versioning](#15-compatibility-and-versioning)
16. [Trust and security model](#16-trust-and-security-model)
17. [Conflict semantics](#17-conflict-semantics)
18. [Qualification model](#18-qualification-model)
    - [18.3 Provider-scoped qualification (PROVIDER-QUAL-1)](#183-provider-scoped-qualification-provider-qual-1)
19. [Observability expectations](#19-observability-expectations)
20. [Third-party public API boundary](#20-third-party-public-api-boundary)
21. [Multi-capability package model](#21-multi-capability-package-model)
22. [Failure model](#22-failure-model)
23. [DO-NOT-UNIFY decisions](#23-do-not-unify-decisions)
24. [Backward compatibility and migration principles](#24-backward-compatibility-and-migration-principles)
25. [Architecture invariants](#25-architecture-invariants)
26. [Future implementation responsibilities (PLATFORM-PLUGIN-3..9)](#26-future-implementation-responsibilities-platform-plugin-39)
27. [Explicit non-goals](#27-explicit-non-goals)
28. [Evidence and references](#28-evidence-and-references)
29. [Open architecture questions](#29-open-architecture-questions)

---

## 1. Purpose and scope

This document is the **canonical architecture decision** for Intergrax platform extensibility. It freezes taxonomy, platform/domain boundaries, and cross-cutting coordination rules established by **PLATFORM-PLUGIN-1** and decided here in **PLATFORM-PLUGIN-2**.

**In scope:**

- What “Platform Plugin” means at the architecture level.
- How third-party Python distributions relate to domain capability contracts.
- Shared vocabulary for discovery, qualification, trust, conflicts, and lifecycle.
- What may be harmonized in future implementation stages vs what must remain domain-owned.

**Out of scope for this document and program:**

- Implementing loaders, registries, manifest parsers, qualification engines, or SDK code.
- Refactoring RAG, Vendor Knowledge, Integrations, Tools, Skills, LKW, security, or RuntimePlugin internals.
- Replacing domain contracts with one monolithic runtime plugin abstraction.

**Relationship to domain architecture:**

Domain documents ([`INTEGRATIONS.md`](INTEGRATIONS.md), [`TOOLS.md`](TOOLS.md), [`SKILLS.md`](SKILLS.md), [`RAG.md`](RAG.md), [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md), etc.) remain **authoritative for runtime semantics**. This document coordinates **packaging, discovery, trust, and author experience** across those domains.

---

## 2. Architectural context

Intergrax is a four-tier Agent OS:

```text
Tier-0  intergrax/           — catalogs, contracts, shared loaders
Tier-1  intergrax/runtime/   — Nexus, policy, hooks, composition
Tier-2  agents/              — domain agents (host-wired)
Tier-3  applications/        — product hosts and environment profiles
```

Extension today is **not one framework**. It is a **constellation of domain capability contracts** plus host composition. PLATFORM-PLUGIN-2 adopts the audit conclusion:

```text
COMMON PLATFORM COORDINATION
        +
DOMAIN-OWNED CAPABILITY CONTRACTS
```

Third-party authors install Python packages. Hosts and applications decide which discovered capabilities are enabled, configured, injected, and qualified for production.

---

## 3. Current-state baseline

**FACT (PLATFORM-PLUGIN-1):** Intergrax exposes **22+ materially distinct extension surfaces** with **no** global plugin manifest, **no** shared lifecycle engine, and **no** unified sandbox.

| Mechanism class | Examples |
|-----------------|----------|
| Tier-0 setuptools entry-point catalogs | integrations, tools, skills, context, memory stores, RAG chunkers/retrievers/rerankers |
| Domain-specific entry-point catalogs | vendor knowledge providers, security defenses, policy rules, tool invocation patterns |
| Shipped first-party bootstrap | `register_default_*`, RAG defaults, integration manifests |
| Host-composed wiring | `ApplicationEnvironmentProfile`, `RuntimePlugin`, `AgentRegistry` |
| Internal registries | embedding providers, document handlers, integration registry v2 |
| Descriptor-only contracts | token optimization plugin descriptor |

**FACT:** `intergrax/core/plugins/discovery.py` provides a unified setuptools loader used by several Tier-0 groups but **not all** surfaces use it.

**FACT:** `INTERGRAX_DISCOVER_PLUGINS` gates Tier-0 discovery in wiring helpers (default **off**). Vendor Knowledge and other domains use additional opt-in flags or explicit composition parameters.

**FACT:** Third-party code is **trusted in-process Python** after installation. Discoverable ≠ production-qualified.

**TARGET (this architecture):** Preserve intentional domain separation; harmonize only cross-cutting author/platform concerns approved below. Implementation of target behavior begins in PLATFORM-PLUGIN-3+.

---

## 4. Canonical terminology

| Term | Meaning |
|------|---------|
| **Platform Plugin** | A **package-level coordination concept** — not a single runtime class. See §6. |
| **Plugin package** | An installable Python distribution that may contribute one or more capabilities. |
| **Capability** | A domain-typed extension unit governed by a domain contract (e.g. integration provider, tool bundle, skill bundle, RAG retriever). |
| **Entry-point group** | Setuptools namespace (e.g. `intergrax.tools`) mapping EP names to loadable callables/classes. |
| **Catalog** | Tier-0 slug registry for integrations, tools, skills, context plugins. |
| **Profile** | Host/application selection object (`IntegrationProfile`, `ToolProfile`, `SkillProfile`, …) that gates materialization. |
| **Host composition** | Explicit Tier-3 wiring of registries, plugins, and profiles without setuptools discovery. |
| **Qualification** | Domain- or program-defined evidence that a capability is fit for a target environment (dev/staging/prod). |
| **Materialization** | Building runtime registries/instances from profiles and discovered plugins. |

**Vocabulary rule:** The overloaded word “plugin” in code (`IntegrationPlugin`, `RuntimePlugin`, `SecurityDefensePlugin`, …) remains valid **within its domain**. Platform Plugin refers only to the cross-cutting coordination layer defined here.

---

## 5. Canonical taxonomy

PLATFORM-PLUGIN-2 **freezes** the PLATFORM-PLUGIN-1 taxonomy. Names are unchanged.

### 5.1 Taxonomy summary

| Code | Name | Author | Enters system via | Pip install | Host composition required | Production qualification | Public compatibility promise |
|------|------|--------|-------------------|-------------|---------------------------|--------------------------|------------------------------|
| **PEP** | `PUBLIC_EXTERNAL_PLUGIN` | Third-party or first-party package authors | Setuptools entry points and/or documented registration APIs | **Yes** (intended) | **Yes** for activation in a host | **Yes** before production reliance | **Yes** — EP groups and domain plugin protocols listed in §20 |
| **IP** | `INTEGRATION_PROVIDER` | Integration provider authors | `intergrax.integrations` EP or shipped manifest path (first-party only for manifest) | Yes for EP path | Yes — `IntegrationProfile` / wiring | Yes — integration domain gates | Yes for EP model; manifest path is first-party internal |
| **HCE** | `HOST_COMPOSED_EXTENSION` | Application/host authors | Explicit Python wiring, registries, profile tuples | Optional (library dep) | **Always** | Host/application responsibility | **No** third-party EP promise — host integration contract only |
| **IEP** | `INTERNAL_EXTENSION_POINT` | Core maintainers or host code | Internal `register(...)` APIs, bootstrap, YAML/config | Not as public EP | Usually yes | Internal/domain CI | **No** — not a supported third-party surface unless promoted to PEP |
| **NE** | `NOT_EXTENSIBLE` | Core only | Source change | N/A | N/A | N/A | N/A |

### 5.2 Category definitions

#### `PUBLIC_EXTERNAL_PLUGIN` (PEP)

**Definition:** A setuptools entry-point group (or equivalent documented loader target) intended for third-party pip-installable packages, governed by a **domain capability contract**.

**Who authors it:** External package maintainers and first-party provider packages.

**How it enters:** Package installed into the environment; entry points discovered when host/domain loaders run with opt-in discovery enabled.

**Examples:** tools, skills, context, memory stores, RAG components, security defenses, policy rules, vendor knowledge providers, tool invocation patterns.

**Production qualification:** Required before treating as production-safe; discovery alone is insufficient.

**Public promise:** Entry-point group name, domain plugin protocol, and domain architecture rules are compatibility-owned surfaces (§20).

#### `INTEGRATION_PROVIDER` (IP)

**Definition:** A **PEP specialization** for `IntegrationCategory` backends — `IntegrationPlugin` + `IntegrationManifest` via `intergrax.integrations`.

**Who authors it:** Integration provider package authors.

**How it enters:** Third-party: setuptools EP. First-party: shipped `manifest.py` + factory bootstrap (internal scale path).

**Package installation:** Allowed for EP model.

**Host composition:** Required — integrations are not globally active after discovery; `IntegrationProfile` and application wiring select providers.

**Dual model note:** Shipped manifest registration is **IEP/first-party** at scale; it is **not** a third-party compatibility surface.

#### `HOST_COMPOSED_EXTENSION` (HCE)

**Definition:** Extensions wired explicitly by a Tier-3 host or application factory without setuptools discovery as the primary activation path.

**Who authors it:** Application developers and platform integrators building a product host.

**Examples:** `RuntimePlugin`, `AgentRegistry` registration, `ApplicationEnvironmentProfile`, observability extension SDK schema registration, task execution registry handlers.

**Package installation:** May depend on installed libraries, but **activation is always host code**.

**Public compatibility promise:** No setuptools EP contract. Application hosting patterns are documented but not versioned as third-party plugin APIs.

#### `INTERNAL_EXTENSION_POINT` (IEP)

**Definition:** Registry or bootstrap hook that exists in code but is **not** a supported third-party extension surface.

**Who authors it:** Core maintainers; occasionally application code in monorepo.

**Examples:** embedding provider registry, document handler registry, integration registry v2 (metadata), token optimization descriptor without loader, LLM model catalog YAML overlay.

**Third-party extension:** Not supported or not documented for external authors.

**Promotion path:** An IEP may become PEP only through an explicit architecture + implementation program with compatibility ownership — not by incidental EP addition.

#### `NOT_EXTENSIBLE` (NE)

**Definition:** Closed implementation; extension requires core or application source change.

**Use:** Core algorithms, closed security internals, non-pluggable registries without public register API.

---

## 6. Platform Plugin definition

**Decision:** “Platform Plugin” means **B + Cₐ** — not A.

| Option | Description | Decision |
|--------|-------------|----------|
| **A** | Executable universal wrapper type | **Rejected** — would absorb domain contracts and duplicate loaders |
| **B** | Package-level coordination contract | **Adopted** — primary meaning |
| **Cₐ** | Taxonomy/metadata layer over domain contracts | **Adopted** — shared vocabulary and optional packaging metadata |
| **Cᵦ** | Broad wrapper registering all domain contributions | **Rejected** — audit Option C |

A Platform Plugin is therefore:

1. An **installable Python distribution** (plugin package) that may declare one or more **capabilities**.
2. A **coordination story** for identity, discovery, compatibility metadata, trust classification, and qualification hooks at the **package boundary**.
3. **Not** a replacement for `IntegrationPlugin`, `ToolPlugin`, `SkillPlugin`, `ContextPlugin`, RAG component plugins, VK contributions, security defenses, or other domain protocols.

Runtime execution always flows through **domain contracts and host composition**. There is no single `PlatformPlugin.execute()`.

---

## 7. Platform vs domain responsibility model

```text
┌─────────────────────────────────────────────────────────────┐
│  Platform Plugin layer (coordination — this document)       │
│  identity · EP conventions · trust vocabulary ·           │
│  optional package manifest · shared discovery utility ·     │
│  qualification metadata hooks · conflict vocabulary         │
└──────────────────────────┬──────────────────────────────────┘
                           │ does not replace
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Domain capability contracts (authoritative runtime)        │
│  IntegrationPlugin · ToolPlugin · SkillPlugin ·             │
│  ContextPlugin · RAG plugins · VK contribution ·            │
│  SecurityDefensePlugin · PolicyRuleHandler · …              │
└──────────────────────────┬──────────────────────────────────┘
                           │ materialized by
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Host / application (Tier-3)                              │
│  profiles · wiring · enablement · secrets · DI ·            │
│  qualification gates · production selection                 │
└─────────────────────────────────────────────────────────────┘
```

| Concern | Platform owns | Domain owns |
|---------|---------------|-------------|
| Entry-point group naming for PEP surfaces | Conventions + index | Semantics of loaded type |
| Catalog slug / bundle_id rules | Shared vocabulary only | Conflict policy defaults and registration |
| Capability contract validation | — | Full contract |
| Profile materialization | — | Domain profiles and builders |
| Security fail-open/closed | Trust statement | Security defense semantics |
| VK publication / LKW qualification | — | Vendor Knowledge program |
| Hook attachment | — | UAEP / hook registry semantics |
| Production qualification evidence | Package-level hooks | Domain test gates and live qual |

**Rule:** The Platform Plugin layer **must not bypass** domain validation, policy, or security gates.

**Tier-3 adoption (APP-ADOPTION-1 / APP-ADOPTION-1A):** `wire_application_environment()` collects per-domain `DomainPluginLoadReport` evidence from the same domain bootstrap pass into `ApplicationPlatformPluginEvidence` on `ApplicationEnvironmentWiring.platform_plugin_evidence` (Memory, Policy when declarative policy participates, Context). Applications consume resolved capabilities and this bootstrap snapshot; they **must not** run duplicate discovery or maintain a global installed-plugin inventory. Evidence is discovery/admission only — not `PRODUCTION_QUALIFIED` (package gate 10 remains separate).

---

## 8. Package and capability model

- **Plugin package:** One Python distribution (one wheel/sdist name) installed into the runtime environment.
- **Capability:** A logically separable extension unit typed by a **domain contract** and advertised through a **domain entry-point group** and/or domain manifest type.
- **Capability identity:** Domain-scoped — e.g. `integration_slug`, `bundle_id`, `tool_id`, `skill_id`, `plugin_id`, VK contribution name, RAG component id. Identity format remains **domain-owned**.
- **Package identity:** Distribution name + version (PyPI/normalized name). Used for trust, observability, and compatibility metadata.

A single plugin package **may** expose zero, one, or many capabilities across one or more domains (§21).

---

## 9. Discovery architecture

### 9.1 Target principles

| Principle | Decision |
|-----------|----------|
| Shared low-level EP utility | **Yes** — `intergrax/core/plugins/discovery.py` (or successor) is the canonical setuptools entry-point **scan/load helper** |
| Single global entry-point group | **Rejected** — domain-specific groups retained |
| Domain-specific groups | **Retained** — each PEP surface keeps its group (§20) |
| One loader to rule all domains | **Rejected** — domain orchestration stays separate |

### 9.2 Who uses the shared utility

**MAY use** shared discovery utility without losing domain semantics:

- Tier-0 catalogs currently using `discovery.py` (integrations, tools, skills, context, memory stores).
- RAG component bootstraps (chunkers, retrievers, rerankers) — load via shared helper, register into RAG registries.
- Security defense loader (`intergrax.security_defenses`) — shared scan/load primitives; domain registration and override policy unchanged.
- Policy rule handler loader (`intergrax.policy_rules`) — shared scan/load primitives; YAML + EP merge remains domain-owned.
- Tool invocation pattern loader (`intergrax.tool_invocation_patterns`) — shared scan/load primitives; lazy lookup by `pattern_id` unchanged.
- Future harmonization of bespoke loaders **if** they only need setuptools scan + import — remaining candidates are Vendor Knowledge (composition model retained).

**MUST remain separate** (orchestration, not just scan):

- Vendor Knowledge `VendorKnowledgeContributionCatalog` composition — instance-local, publication snapshot, explicit `discover_entry_points` on builders.
- Host-composed bootstrap (`register_default_*`, explicit plugin tuples).
- Internal registries (`AgentRegistry`, embedding registry, task execution registry).
- Shipped first-party integration manifest registration at scale.

### 9.3 Installed vs discovered vs enabled

| State | Meaning |
|-------|---------|
| **Installed** | Python distribution present in environment (pip/uv install). |
| **Discovered** | Loader found entry point(s) or registration callable for a domain group. |
| **Loadable** | Import/instantiation succeeded; domain contract validation may still fail. |
| **Enabled** | Host/application/profile explicitly selects the capability for use in this process. |
| **Active** | Materialized into runtime registry and invokable per domain rules. |

**Architecture rule:** `installed ≠ discovered ≠ enabled ≠ qualified ≠ active`.

Default discovery for Tier-0 wiring helpers remains **opt-in** (`INTERGRAX_DISCOVER_PLUGINS`) unless a domain document specifies otherwise. Harmonization of flags is allowed in PLATFORM-PLUGIN-4 **only** as additive aliases — not silent behavior change.

---

## 10. Registration and composition architecture

PLATFORM-PLUGIN-1 identified five composition models; all remain valid:

| Model | Role | Platform action |
|-------|------|-----------------|
| **R1** Tier-0 catalog slug registration | integrations, tools, skills, context | Terminology alignment only |
| **R2** Profile-gated materialization | profiles select subsets | Unchanged — domain-owned |
| **R3** Bootstrap composition pipelines | RAG, VK, policy wiring | Unchanged — domain-owned |
| **R4** Runtime hook/event attachment | RuntimePlugin, defenses, hooks | Unchanged — HCE / domain |
| **R5** Instance-local contribution catalog | VK | Unchanged — DO-NOT-UNIFY |

**Registration rule:** Discovery registers **candidates** into domain catalogs or registries. **Activation** for production paths requires host profiles, wiring, and qualification.

---

## 11. Manifest and metadata model

### 11.1 Decision

| Layer | Required? | Role |
|-------|-----------|------|
| Python package metadata (`pyproject.toml`) | **Required** (Python packaging) | Distribution identity, dependencies, entry points |
| Setuptools entry points | **Required** for PEP setuptools surfaces | Machine discovery |
| Domain manifests (`IntegrationManifest`, `ToolBundleManifest`, …) | **Required by domain contracts** where specified | Runtime capability semantics |
| **Platform Plugin package manifest** (sidecar or `pyproject` tool table) | **Optional** | Author-facing coordination only |

**Rejected:** A single mandatory cross-domain runtime manifest that duplicates domain manifests or replaces entry points.

### 11.2 Optional Platform Plugin manifest (TARGET)

When present, an optional package-level manifest may declare **only**:

- Normalized plugin package id and version
- Intergrax platform compatibility range (see §15)
- List of **provided capability descriptors** (domain, entry-point group, entry-point name, optional domain ids)
- Trust / qualification metadata hooks (labels, not secrets)
- Author and documentation URIs

It **must not** duplicate full domain contract payloads (tool schemas, integration category bindings, VK publication rules, etc.).

**CURRENT STATE:** No platform manifest parser exists. Domain manifests and entry points are authoritative.

---

## 12. Configuration and secrets

### 12.1 Configuration ownership

| Layer | Owner | Examples |
|-------|-------|----------|
| **Platform package** | Package author + platform conventions | Package id, compatibility range, declared capabilities |
| **Domain capability** | Domain contract + profile | `IntegrationManifest.env_prefix`, tool/skill bundle config, RAG profile sections |
| **Application** | Tier-3 host | `ApplicationEnvironmentProfile`, `.env`, feature flags, which capabilities enabled |

**FROZEN principle (PLUGIN-5):** Capability implementations should receive **resolved configuration** from host/domain mechanisms (profile, wiring context, injected config objects). Direct `os.environ` reads are **discouraged** as the primary configuration path except where a domain contract explicitly standardizes env prefixes (integrations today). See §12.3 matrix and §12.4 flow.

**Optional dependencies:** Declared in package metadata; domains define graceful absence behavior. Platform does not force a global optional-deps resolver.

### 12.2 Credentials and secrets

| | CURRENT STATE (PLUGIN-5) | TARGET ARCHITECTURE |
|---|---------------|---------------------|
| Secret storage | Integration `env_prefix`; host secret stores; ad hoc per domain | Unchanged storage diversity, clarified ownership (§12.3) |
| Plugin access | Full process privileges; no platform sandbox | **Capabilities receive only required credentials/bindings** resolved by host/domain |
| Metadata | Secrets must not appear in manifests or EP targets; `PLATFORM_PLUGIN_MANIFEST_SECRET_POLICY` on the canonical secret-safe engine enforces the package manifest | **Frozen** |
| Default | No unrestricted global secret API for plugins | **Frozen** |

Installation of a plugin package is a **trust decision** (§16). Secret scope limiting is a **host/domain responsibility**, not something plugin metadata can safely enforce alone.

### 12.3 Cross-surface configuration, secrets and DI matrix

**PLATFORM-PLUGIN-5 outcome:** Representative public PEP surfaces already expose **domain-owned** configuration and injection shapes. No universal runtime configuration object or global Platform Plugin DI container is required. The matrix below is the canonical reference for PLUGIN-8 author docs and PLUGIN-9 closeout.

| Domain surface | PLUGIN-5 class | Configuration owner | Credentials / secrets owner | DI / materialization | Direct `os.environ` in capability code | Host resolves before materialization | Reusable primitive | PLUGIN-5 action |
|----------------|----------------|---------------------|-----------------------------|----------------------|----------------------------------------|--------------------------------------|--------------------|-----------------|
| **Integrations** (`IntegrationPlugin`) | **B** — document convention | Application host + `IntegrationProfile`; domain `IntegrationManifest` | Host / `IntegrationProfile` + integration `env_prefix` (domain contract) | `IntegrationProfile.resolve(category)` → `create_integration(**kwargs)` | **Allowed** when manifest declares `env_prefix` (domain-owned compatibility) | Yes — profile selects provider before factory | `IntegrationProfile`, `IntegrationManifest.env_prefix` | Document `env_prefix` as integration exception; no migration |
| **Tools** (`ToolPlugin`) | **A** — already aligned | Application host + `ToolProfile` | Host via `ToolWiringContext` integration slots (`secrets_store`, …) | `register_tools(registry, ctx: ToolWiringContext)` | Discouraged for portable plugins | Yes | `ToolWiringContext` | Document only |
| **Skills** (`SkillPlugin`) | **B** — document convention | Application host + `SkillProfile` (enablement) | Runtime credentials stay with tools/host | `register_skills(SkillRegistry)` — declarative manifests | Not applicable (declarative) | Yes — profile gates bundles | `SkillProfile` | Document only; runtime deps via tools |
| **Context** (`ContextPlugin`) | **B** — document convention | Host / `ContextProfile` in `ApplicationEnvironmentProfile` | Typically none at plugin boundary | `register(registry: ContextPluginRegistry)` | Discouraged for portable plugins | Yes | `ContextPluginRegistry` | Document only |
| **RAG components** (chunkers / retrievers / rerankers) | **A** — already aligned | Host + `RagProfile`; bootstrap kwargs | Via integration bindings passed into bootstrap | `BaseRetrieverPlugin.create(vector_store=…, profile=…)` etc. | `RagProfile` may use domain env helpers when host passes profile from env | Yes — bootstrap receives resolved managers | `RagProfile`, per-component bootstrap | Document only; no generic RAG context |
| **Memory stores** (`intergrax.memory_stores`) | **B** — document convention | Host + `MemoryProfile` | Host passes kwargs / integration refs to factories | `create_user_profile_store(**kwargs)` / `create_session_storage(**kwargs)` | Discouraged unless kwargs explicitly document env ownership | Yes — host invokes factories | `UserProfileStorePlugin` / `SessionStoragePlugin` | Document factory kwargs ownership |
| **Security defenses** (`SecurityDefensePlugin`) | **A** — already aligned | Host security wiring / `ApplicationSecurityProfile` | None via entry point; inspect `HookContext` only | `PluginSecurityDefenseMiddleware(plugin, event_bus=…)` | Discouraged | Yes | `HookContext`, `SecurityFailMode` | Document only |
| **Policy rules** (`PolicyRuleHandler`) | **A** — already aligned | Host + `PolicyRulesProfile` / YAML bundle | None in handler EP | `evaluate(rule, context=dict[str, str])` via `PolicyRuleRegistry` | Discouraged | Yes | `PolicyRuleRegistry` | Document only |
| **Tool invocation patterns** (`ToolInvocationPattern`) | **A** — already aligned | Nexus runtime config (`ToolInvocationMode`) | None | `execute(state, invoker, planner, …)` | Not applicable (stateless orchestration) | Yes — mode selects pattern | `ToolInvocationMode` → pattern | Document only |
| **Vendor Knowledge** (`VendorKnowledgeProviderContribution`) | **D** — defer (DO-NOT-UNIFY) | Host builder + `KnowledgeSourceBinding` | `credential_ref` scoped per binding | Instance-local contribution catalog / builders | Not a portable EP config surface | Yes — tenant-scoped composition | VK builders, `KnowledgeSourceBinding` | Document host/domain boundary only |

**Classification key:** **A** = already aligned · **B** = documentation/convention only · **C** = small additive wiring contract (none required in PLUGIN-5) · **D** = defer — domain architecture must not change under Platform Plugin banner.

**Rejected in PLUGIN-5:** global Platform Plugin DI container, global secrets store, universal plugin configuration schema, service locator, global `get_secret()` API, new application configuration framework.

### 12.4 Canonical configuration flow

Cross-cutting invariant (all PEP surfaces):

```text
application config / environment / secret source
        ↓
host / domain resolver (profile, wiring, enablement)
        ↓
typed profile / resolved config / wiring context
        ↓
domain materializer (registry, bootstrap, factory)
        ↓
plugin capability (consumes explicit bindings only)
```

| Layer | Configuration responsibility | Secret responsibility |
|-------|------------------------------|----------------------|
| **Platform Plugin package manifest** (`[tool.intergrax.plugin]`) | Coordination metadata only — package id, compatibility, capability pointers | **Prohibited** — `PLATFORM_PLUGIN_MANIFEST_SECRET_POLICY` via `intergrax.core.security` (PLUGIN-3). Detection only; not a secret manager. |
| **Setuptools entry points** | Identify import target only | **Prohibited** in EP values |
| **Domain manifests** (`IntegrationManifest`, tool/skill bundles, …) | Domain-specific non-secret fields (e.g. `env_prefix` name, not value) | **Prohibited** as manifest field values |
| **Application / host** | Selects capabilities, environment/profile, feature flags | Resolves credentials into domain bindings |
| **Domain profile / wiring** | Validates and materializes domain config shape | Scopes credentials to least privilege for the capability |
| **Capability implementation** | Consumes resolved config objects / constructor parameters | Receives only explicitly passed credential bindings |

**Environment access rule:** Direct `os.environ` reads are **not** a portable Platform Plugin author contract. They remain **supported** where a domain contract explicitly owns env-prefix semantics (integrations `env_prefix` today). New portable plugin code should prefer host-resolved profiles and wiring contexts.

**Configuration resolution rule:** Parsing profiles, manifests, or Platform Plugin metadata **must not** register plugins or mutate global catalogs as a side effect. Discovery and configuration resolution are separate phases (§9.3).

---

## 13. Dependency injection

**Host-owned DI principles (FROZEN — PLUGIN-5):**

1. Plugin package **declares** capability via entry point / domain plugin type.
2. Host/domain **decides** runtime dependencies (integrations, vector stores, event bus, policy engine, wiring contexts).
3. Capability implementation **must not** invent hidden global bindings where injection is available for that domain.
4. Domain-specific injection shapes remain **domain-owned** (e.g. RAG retriever receives vector store + embedding manager; tools receive `ToolWiringContext`; integrations receive profile-resolved factories).
5. **Least-credential rule:** a capability receives only the credential bindings required for its domain operation — not an application-wide secret dictionary.
6. **No hidden service locator:** plugin implementations must not depend on module-level mutable global dependency registries, generic `get_service(...)`, global platform secret accessors, or implicit application singletons where explicit domain wiring exists.

**Explicit non-goal:** A new global Platform Plugin DI container. Reuse existing wiring contexts and profile builders (see §12.3 matrix).

**CURRENT STATE (PLUGIN-5):** Domain-owned injection shapes are authoritative across inspected PEP surfaces. Optional `[tool.intergrax.plugin]` parsing and profile construction are side-effect free. Some process-scoped globals remain (catalog snapshots, shipped bootstrap); new work must not expand implicit globals or introduce cross-domain service locators.

---

## 14. Lifecycle model

### 14.1 Shared vocabulary (conceptual)

| Phase | Meaning |
|-------|---------|
| **discovered** | Entry point or registration target found |
| **validated** | Import succeeded; domain contract checks passed |
| **enabled** | Selected by host/profile for this runtime |
| **materialized** | Registry/builder constructed runtime object |
| **active** | Participating in requests/hooks as domain defines |
| **stopping** | Shutdown signal (where supported) |
| **stopped** | Resources released |
| **failed** | Terminal error in load, validation, or activation |

### 14.2 Applicability

| Scope | Decision |
|-------|----------|
| Vocabulary | **Mandatory** for documentation, observability, and PLATFORM-PLUGIN-3+ APIs |
| Runtime lifecycle interface | **Selectively applicable** — not every domain implements unload/reload |
| Forced unload/reload | **Not required** — process-scoped catalogs remain valid |

`RuntimePlugin` retains its **domain-specific** `on_shutdown` lifecycle (HCE). Tier-0 catalogs remain predominantly **process-scoped register without unload**.

**CURRENT STATE (PLUGIN-6):** Shared enum `PlatformPluginLifecycleState` in `intergrax/core/plugins/platform_semantics.py` — values `discovered`, `validated`, `enabled`, `materialized`, `active`, `stopping`, `stopped`, `failed`. Vocabulary only; **no** global lifecycle manager or transition engine. Qualification states (`qualified`, `production-qualified`, `live-qualified`) and `installed` remain **outside** this enum (§18).

## 15. Compatibility and versioning

### 15.1 Compatibility layers

| Layer | What it versions | Owned by |
|-------|------------------|----------|
| **Package platform compatibility** | Plugin package vs Intergrax platform release | Platform Plugin manifest / metadata (PLATFORM-PLUGIN-3) |
| **Capability contract compatibility** | Domain protocol version (`IntegrationPlugin`, `ToolPlugin`, …) | Domain architecture |
| **Runtime compatibility** | Host/runtime features (e.g. `compatible_runtime` on `RuntimePlugin`) | Host/runtime domain |
| **Qualification status** | Evidence of fit for environment | Domain + program gates |

### 15.2 Rules

- Platform compatibility metadata is **checked** by PLUGIN-6 tooling (`check_platform_compatibility`, `require_platform_compatibility`) using standard Python `packaging` specifier semantics.
- **Compatible ≠ qualified** — compatibility check does not imply trust or production qualification (§18, PLUGIN-7).
- Domain contracts keep **their own** version fields and breaking-change policy.
- **No** single global semver for all extension surfaces.

**CURRENT STATE (PLUGIN-6):** `intergrax/core/distribution/platform_compatibility.py` exposes deterministic, explicit-version compatibility checking against `PlatformCompatibility.intergrax_version`. Returns immutable `PlatformCompatibilityResult` (`declared_specifier`, `tested_platform_version`, `compatible`, `reason`). Invalid platform version → `InvalidPlatformVersionError` (via `require_platform_compatibility`) or `reason=invalid_platform_version` (via `check_platform_compatibility`). **Activation blocking** at host boundaries is deferred to PLUGIN-8 reference host — callers must pass platform version explicitly; no authoritative runtime Intergrax distribution version helper exists yet (`importlib.metadata` distribution name not standardized for gating).

---

## 16. Trust and security model

**Default trust statement (FROZEN):**

- Installed Python plugins execute as **trusted in-process code** in the host process.
- There is **no sandbox guarantee** for PEP surfaces today.
- **Installation is a trust decision** equivalent to deploying application code.
- **Discovery is not qualification.**
- **Qualification is not sandboxing.**

**CURRENT STATE (PLUGIN-7):** Shared enum `PlatformPluginTrustModel` (`trusted_in_process` only) in `intergrax/core/plugins/platform_qualification.py`. Optional audit distinction `PluginTrustOrigin` (`host_local_code`, `installed_third_party_package`). No verified/signed/sandboxed claims.

| Topic | Decision |
|-------|----------|
| Code signing | Not required by platform architecture (future product decision) |
| Isolated plugin execution | **Optional future architecture** — out of scope for PLATFORM-PLUGIN-2..7 unless explicitly programed |
| Network/filesystem access | Full process privileges unless domain/host restricts higher layers |
| Security defenses | Domain-owned fail-open/fail-closed per plugin |

Future isolated execution (subprocess, WASM, remote worker) would be a **separate architecture program** and must not be implied by current loaders.

---

## 17. Conflict semantics

### 17.1 Conflict classes

| Conflict type | Shared vocabulary | Resolution policy owner |
|---------------|-------------------|-------------------------|
| Duplicate **package/plugin identity** | Platform | Platform documentation; enforcement PLATFORM-PLUGIN-6 |
| Duplicate **entry-point name** within group | Platform + domain | Domain loader (VK: error; security: override; Tier-0: `ConflictPolicy`) |
| Duplicate **capability identity** (slug/bundle_id/tool_id) | Domain | Domain catalog `ConflictPolicy` |
| Duplicate **domain resource ID** | Domain | Domain registry rules |

### 17.2 Platform policy

**TARGET:** Shared **conceptual** conflict vocabulary across domains. **Unified default policy across all surfaces is rejected** — security override semantics and VK publication conflicts must remain domain-owned.

**CURRENT STATE (PLUGIN-6):** Shared enum `PlatformPluginConflictKind` (`package_identity`, `entry_point_name`, `capability_identity`, `domain_resource_id`) in `platform_semantics.py`. Tier-0 duplicate entry-point errors attach `conflict_kind=entry_point_name` to `PluginConflictError` without changing `ConflictPolicy` behavior (`error` / `skip` / `override` / `warn_override`). Package identity conflict helper `package_identities_conflict` lives in `intergrax/core/distribution/package_identity.py`; capability and domain-resource conflicts remain domain-owned.

---

## 18. Qualification model

### 18.1 Status distinction (FROZEN)

```text
installed → discovered → loadable → contract-valid → enabled → qualified → production-qualified
                                                              ↘ live-qualified (domain-specific, e.g. VK/LKW)
```

| Status | Platform meaning |
|--------|------------------|
| **installed** | Package in environment |
| **discovered** | EP visible to loader |
| **loadable** | Import OK |
| **contract-valid** | Domain contract validation passed |
| **enabled** | Host selected for use |
| **qualified** | Domain/program evidence threshold met |
| **production-qualified** | Approved for production host profiles |
| **live-qualified** | Domain-specific runtime evidence (e.g. live VK rollout) |

### 18.2 Granularity

**Decision:** Qualification is **combination**:

- **Package-level** — trust/compatibility metadata, CI install checks, optional platform manifest
- **Capability-level** — per integration slug, tool bundle, skill bundle, RAG component, etc.
- **Domain-level** — domain test harness, live qual, security review

A package may be installed but carry **mixed** qualification across capabilities.

**CURRENT STATE (PLUGIN-7):** Shared contracts in `intergrax/core/plugins/platform_qualification.py`:

- `PluginQualificationLevel` — `package`, `capability`, `domain`
- `QualificationStatus` (`intergrax/core/qualification/status.py`) — `not_qualified`, `qualified`, `production_qualified`, `rejected` (distinct from lifecycle states in §14)
- `PluginQualificationSubject` + `PluginQualificationResult` + `PluginQualificationEvidence` — immutable audit records; no persistence or global registry
- `PluginDeliverySource` — `external_package` vs `host_embedded_extension` (both converge on domain qualification; wheel/entry-point not required for host-embedded path)
- `require_production_qualification` / `evaluate_package_production_admission` — pure production gates; compatible/enabled alone insufficient
- `live-qualified` — optional domain label via `domain_qualification_label` / evidence kind `live_qualification`; not a mandatory platform state

**Delivery modes (FROZEN):**

| Mode | Entry | Package metadata | Qualification path |
|------|-------|------------------|-------------------|
| **A. External package** | setuptools entry point → discovery | `[project]` + optional `[tool.intergrax.plugin]` | Package + capability + domain evidence; PLUGIN-6 compatibility applies at package boundary |
| **B. Host-embedded extension** | explicit `register_*_plugin()` / host wiring | Not required | Same capability/domain qualification model; `host_registration_path` identity; compatibility N/A at package boundary |

PLUGIN-8 proves executable E2E for both modes. PLUGIN-7 does not wire gates into every host bootstrap.

### 18.3 Provider-scoped qualification (PROVIDER-QUAL-1)

**Status:** Architecture freeze — **READY_FOR_REVIEW** (contract design only; no runtime implementation).

**Decision (PROVIDER-QUAL-0):** **EXTEND_EXISTING** — reuse `intergrax/core/qualification/`, PLUGIN-7 coordination, `QualificationEvidence`, `ProofReceipt`, and domain-owned suites. **No** parallel `ProviderQualificationEngine`.

**Canonical satellite:** [`satellites/PLATFORM_PLUGINS_provider_qualification.md`](satellites/PLATFORM_PLUGINS_provider_qualification.md)

**Frozen highlights:**

| Topic | Freeze |
|-------|--------|
| **Subject** | `ProviderQualificationSubject` — `provider_id`, `provider_version`, `capability_id`, `domain`, adapter identity, `intergrax_revision`, `qualification_suite_id` / `version`, `environment_id`; string/data-driven `provider_id` only |
| **Run** | `ProviderQualificationRun` — immutable executed historical fact; `qualification_run_id` created at execution (persistence preserves identity); outcome, executor-neutral metadata, evidence refs, reproducibility, limitations, `source_revision`; **no** embedded `validity` field |
| **Outcome vs validity** | `QualificationStatus` = immutable historical outcome (`NOT_QUALIFIED` … `REJECTED` only); `QualificationEvidenceValidity` = separate current admission view/record (append-only or derived latest); compatibility separate |
| **Admission** | `PRODUCTION_QUALIFIED` + `CURRENT` validity + compatibility + domain policy; no admission policy engine in PROVIDER-QUAL-1 |
| **Evidence** | `QualificationEvidence` in-run canonical; optional `ProofReceipt` persistence via `ref` mapping — no second vocabulary |
| **Capability scope** | Never globally qualified per provider; identity is provider + version + capability + suite + environment |
| **Suite semantics** | Domain-owned; platform coordinates identity/index only |
| **CI** | Contract/schema/harness tests only — not live all-vendor-on-every-PR |
| **Scale** | 5/20/50+ providers via shared contracts + linear per-vendor adapter/setup/evidence |

**Implementation:** PROVIDER-QUAL-2 (typed contracts only); PROVIDER-QUAL-3C (**READY_FOR_REVIEW**) persists `ProviderQualificationRun` through existing `ProofReceipt` / `DocumentStore` mechanics with lookup by `qualification_run_id`; PROVIDER-QUAL-3C-R1 (**READY_FOR_REVIEW**) hardens durability (persistent store reopen) and rejects credential-bearing evidence before persistence via platform `secret_safety`; broader discovery/staleness/runner remain out of scope.

## 19. Observability expectations

Minimum **TARGET** observability for third-party plugin packages and capabilities (without replacing [`OBSERVABILITY.md`](OBSERVABILITY.md)):

| Signal | Required attribution |
|--------|----------------------|
| Plugin package name + version | Yes |
| Capability domain + identity | Yes |
| Entry-point group + name | On discovery/load |
| Registration/load failures | Logged with domain context |
| Qualification state | When exposed by host/domain |
| Runtime invocation | Where domain supports attribution (tools, integrations, hooks) |

Platform Plugin program adds **documentation and optional metadata hooks** in PLATFORM-PLUGIN-3+; it does not replace the observability extension SDK (schema registration ≠ plugin loading).

---

## 20. Third-party public API boundary

### 20.1 Supported third-party surfaces (compatibility-owned)

| Entry-point group | Domain doc | Plugin protocol |
|-------------------|------------|-----------------|
| `intergrax.integrations` | INTEGRATIONS | `IntegrationPlugin` |
| `intergrax.tools` | TOOLS | `ToolPlugin` |
| `intergrax.skills` | SKILLS | `SkillPlugin` |
| `intergrax.context` | CONTEXT_ENGINEERING | `ContextPlugin` |
| `intergrax.memory_stores` | MEMORY | duck-typed factory methods |
| `intergrax.rag.chunkers` | RAG | chunker plugin protocol |
| `intergrax.rag.retrievers` | RAG | retriever plugin protocol |
| `intergrax.rag.rerankers` | RAG | reranker plugin protocol |
| `intergrax.vendor_knowledge.providers` | Vendor Knowledge guides | `VendorKnowledgeProviderContribution` |
| `intergrax.security_defenses` | UNIFIED_EXECUTION_RUNTIME / security | `SecurityDefensePlugin` |
| `intergrax.policy_rules` | policy domain | `PolicyRuleHandler` |
| `intergrax.tool_invocation_patterns` | Nexus tools | `ToolInvocationPattern` |

**Also public:** [`EXTENSION_AUTHOR_GUIDE.md`](../technical/guides/EXTENSION_AUTHOR_GUIDE.md), domain author guides (VK), documented conflict/discovery env conventions once harmonized.

### 20.2 Internal / host-only (not third-party API)

| Surface | Taxonomy |
|---------|----------|
| Shipped integration manifest bootstrap | IEP (first-party) |
| Integration registry v2 | IEP |
| Embedding / document handler registries | IEP |
| `RuntimePlugin` | HCE |
| `AgentRegistry` | HCE |
| Task execution registry | HCE |
| Observability extension SDK | HCE |
| Token optimization descriptor (no loader) | IEP |
| Hook registry internals | IEP |
| LLM model catalog YAML overlay | IEP |

**Rule:** Public contracts require **explicit compatibility ownership** in domain or platform program stages. Undocumented `register()` calls are not public API.

### 20.3 Public extension author matrix (PLUGIN-8)

Author-facing summary of all canonical setuptools entry-point surfaces (§20.1). **Tools** is the executable dual-mode reference (external wheel + host-embedded). Local explicit-registration paths are documented only where repository evidence exists today — gaps are recorded for PLUGIN-9.

| Extension surface | Public contract | External EP group | Local explicit registration | Domain doc | Config / DI | Author guide / example |
|-------------------|-----------------|-------------------|----------------------------|------------|-------------|------------------------|
| Integrations | `IntegrationPlugin` | `intergrax.integrations` | `register_integration_plugin()` from host composition | [INTEGRATIONS.md](INTEGRATIONS.md) | `IntegrationProfile` + `IntegrationManifest.env_prefix` | [EXTENSION_AUTHOR_GUIDE §2](../technical/guides/EXTENSION_AUTHOR_GUIDE.md#2-external-integration-plugin) · `intergrax/integrations/examples/custom_memory_kv/` |
| **Tools** | **`ToolPlugin`** | **`intergrax.tools`** | **`register_tool_plugin()`** — scaffold `extensions/` + `host/tool_wiring.py` | [TOOLS.md](TOOLS.md) | **`ToolWiringContext`** | [EXTENSION_AUTHOR_GUIDE §3, §16](../technical/guides/EXTENSION_AUTHOR_GUIDE.md#16-dual-mode-developer-quickstarts-platform-plugin-8) · **External:** `examples/platform_plugins/intergrax_reference_tool_plugin/` · **Local:** `examples/platform_plugins/local_embedded_tool_extension/` |
| Skills | `SkillPlugin` | `intergrax.skills` | `register_skill_plugin()` from host composition | [SKILLS.md](SKILLS.md) | `SkillProfile`; runtime deps via tools | [EXTENSION_AUTHOR_GUIDE §4](../technical/guides/EXTENSION_AUTHOR_GUIDE.md#4-external-skill-plugin) · `intergrax/skills/examples/custom_pack/` |
| Context | `ContextPlugin` | `intergrax.context` | `register_context_plugin()` from host — **scaffold hook not yet documented** | [CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md) | `ContextProfile` | [CONTEXT_PLUGIN_AUTHOR_GUIDE.md](../technical/guides/CONTEXT_PLUGIN_AUTHOR_GUIDE.md) · **External (multi-cap):** `examples/platform_plugins/intergrax_reference_enterprise_plugin/` |
| Memory stores | `UserProfileStorePlugin` / `SessionStoragePlugin` factories | `intergrax.memory_stores` | Host invokes factory callables — **no documented local-plugin helper** | [MEMORY.md](MEMORY.md) | Factory kwargs from host | [EXTENSION_AUTHOR_GUIDE §9](../technical/guides/EXTENSION_AUTHOR_GUIDE.md#9-memory-store-plugins-phase-mem) · `tests/fixtures/plugin_packages/` |
| RAG chunkers | `BaseChunkingStrategy` | `intergrax.rag.chunkers` | RAG bootstrap/registry — **local explicit-registration path not yet documented** | [RAG.md](RAG.md) | `RagProfile` + bootstrap kwargs | [RAG_EXTENSION_GUIDE.md](../technical/guides/RAG_EXTENSION_GUIDE.md) |
| RAG retrievers | `BaseRetriever` | `intergrax.rag.retrievers` | Same as chunkers | [RAG.md](RAG.md) | `RagProfile` + vector store bindings | [RAG_EXTENSION_GUIDE.md](../technical/guides/RAG_EXTENSION_GUIDE.md) |
| RAG rerankers | `BaseReranker` | `intergrax.rag.rerankers` | Same as chunkers | [RAG.md](RAG.md) | `RagProfile` + bootstrap kwargs | [RAG_EXTENSION_GUIDE.md](../technical/guides/RAG_EXTENSION_GUIDE.md) |
| Vendor Knowledge | `VendorKnowledgeProviderContribution` | `intergrax.vendor_knowledge.providers` | Host builder composition — **not Tier-0 catalog registration** | [KNOWLEDGE_SOURCE_INTEGRATIONS.md](KNOWLEDGE_SOURCE_INTEGRATIONS.md) | `KnowledgeSourceBinding` + tenant scope | [VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md](../technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md) · **External:** `examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/` |
| Security defenses | `SecurityDefensePlugin` | `intergrax.security_defenses` | `register_security_defense_plugin()` + profile ids — advanced host composition | [UNIFIED_EXECUTION_RUNTIME.md](UNIFIED_EXECUTION_RUNTIME.md) | `ApplicationSecurityProfile` | [SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md](../technical/guides/SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md) · `tests/fixtures/plugin_packages/intergrax_security_defense_fixture/` |
| Policy rules | `PolicyRuleHandler` | `intergrax.policy_rules` | `PolicyRuleRegistry.register()` + explicit `load_policy_rule_plugins()` | [UNIFIED_EXECUTION_RUNTIME.md](UNIFIED_EXECUTION_RUNTIME.md) | `PolicyRulesProfile` / YAML bundle | [POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md](../technical/guides/POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md) |
| Tool invocation patterns | `ToolInvocationPattern` | `intergrax.tool_invocation_patterns` | `RuntimeConfig.tool_invocation_pattern` instance override | [TOOLS.md](TOOLS.md) | `ToolInvocationMode` | [TOOL_INVOCATION_PATTERN_AUTHOR_GUIDE.md](../technical/guides/TOOL_INVOCATION_PATTERN_AUTHOR_GUIDE.md) |

**PLUGIN-8 convergence evidence:** both delivery modes for Tools use the same `ToolPlugin` contract, `register_tool_plugin` / `bootstrap_catalogs` catalog materialization, `ToolWiringContext`, and `RuntimeToolInvoker` execution path. No parallel local-plugin framework was introduced.

### 20.4 PLUGIN-8 executable evidence

| Artifact | Path |
|----------|------|
| External reference package (outside `intergrax/`) | `examples/platform_plugins/intergrax_reference_tool_plugin/` |
| Distribution name / version | `intergrax-reference-tool-plugin` / `0.1.0` |
| Entry point | `intergrax.tools:reference_prefix_echo` |
| Host-embedded example | `examples/platform_plugins/local_embedded_tool_extension/` |
| Application scaffold hook | `extensions/` + `register_<app>_local_tool_extensions(...)` after `require_production_qualification` in generated `host/tool_wiring.py` |
| Executable E2E proof | `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py` |
| Wheel build | `uv build --wheel` on reference package (no new build dependency) |
| Isolated install | `uv pip install <wheel> --target <tmpdir> --no-deps` |
| Discovery | `iter_entry_point_specs` / `load_entry_point_plugins` + `bootstrap_catalogs(discover_entry_points=True)` |
| Qualification | `evaluate_package_production_admission` + `require_production_qualification` |
| Platform version for compatibility | Explicit host input (`0.1.0` in E2E) — no global runtime version authority |
| Known gaps | Local explicit-registration documented for Tools/Integrations/Skills only; other surfaces remain external-EP-first (acceptable per PLUGIN-9 closeout) |

---

## 21. Multi-capability package model

### 21.1 Decision

**Multi-capability packages are ALLOWED.**

One external Python distribution may expose multiple capabilities across domains (evidence: `tests/fixtures/plugin_packages/intergrax_catalog_fixture` and **`examples/platform_plugins/intergrax_reference_enterprise_plugin/`**).

### 21.2 Rules

| Question | Decision |
|----------|----------|
| Allowed? | **Yes** |
| Capability identity | **Domain-scoped** ids; optional package manifest lists descriptors |
| Discoverability | **Each capability separately discoverable** via its domain entry-point group |
| Import failure | Package import failure blocks **that package**; partial domain failure should be **bounded per capability** where loaders support it |
| Qualification | **Both** package-level and capability-level |
| Activation | Host may enable subset of capabilities from one package |

**Not required:** One entry-point group for all capabilities. **Rejected.**

---

## 22. Failure model

| Scenario | TARGET behavior |
|----------|-----------------|
| Plugin package fails to import | Fail **that package**; other packages/domains continue where loaders isolate imports |
| One capability in multi-capability package fails validation | **Bounded** — other capabilities from same package may still load if domain loader supports per-EP isolation |
| Duplicate identity conflict | Domain/policy resolves per §17; startup may fail or skip per `ConflictPolicy` |
| Process isolation | **Not promised** |

**CURRENT STATE:** Import failures often surface at bootstrap; isolation varies by loader. PLATFORM-PLUGIN-4 may improve per-EP isolation for shared utility paths.

---

## 23. DO-NOT-UNIFY decisions

FROZEN from PLATFORM-PLUGIN-1 §P:

| Mechanism | Decision | Rationale |
|-----------|----------|-----------|
| Vendor Knowledge contribution catalog | **KEEP DOMAIN-SPECIFIC** | Publication snapshot, LKW qualification, tenant semantics |
| Security defense plugins | **KEEP DOMAIN-SPECIFIC** | Hook-point model, override policy, fail modes |
| `RuntimePlugin` | **KEEP DOMAIN-SPECIFIC** | Tier-3 lifecycle, event bus — HCE not catalog discovery |
| `AgentRegistry` | **KEEP DOMAIN-SPECIFIC** | Tier-2 assembly; no third-party EP by design |
| RAG component registries | **KEEP DOMAIN-SPECIFIC** | Per-component DI (vector store, embeddings) |
| Integration registry v2 | **KEEP DOMAIN-SPECIFIC** | Metadata-only transitional layer — not author surface |
| Policy YAML + EP handlers | **KEEP DOMAIN-SPECIFIC** | Declarative + imperative merge — policy domain owns |
| Observability extension SDK | **HARMONIZE TERMINOLOGY/METADATA ONLY** | Schema registration ≠ plugin loading |
| Task execution registry | **KEEP DOMAIN-SPECIFIC** | Worker-local handlers |
| Shipped integration manifest path | **KEEP DOMAIN-SPECIFIC** | First-party scale (~167+ slugs), performance, ownership |
| Tier-0 catalog registries (integration/tool/skill/context) | **KEEP DOMAIN-SPECIFIC** | Intentional domain separation — shared loader utility only |
| RAG per-type EP groups vs single RAG group | **KEEP DOMAIN-SPECIFIC** | Component injection differs per type |
| `core/plugins/discovery.py` vs bespoke loaders | **SHARE LOW-LEVEL UTILITY ONLY** | Harmonize scan/load helper in PLATFORM-PLUGIN-4 where approved |
| Context EP vs author guide drift | **HARMONIZE TERMINOLOGY/METADATA ONLY** | Doc alignment — CONTEXT_ENGINEERING owns runtime |
| Token optimization plugin | **FUTURE REVIEW** | Descriptor exists; loader path not established |

---

## 24. Backward compatibility and migration principles

1. **Additive first** — new entry-point groups, metadata fields, and discovery flags must not break existing packages.
2. **Deprecation over replacement** — supported EP groups remain until explicit deprecation release notes and migration window.
3. **No silent activation** — harmonized discovery flags must not enable third-party plugins by default in production hosts.
4. **Domain contracts stable** — platform coordination changes must not require changes to domain plugin class shapes without domain-owned major versions.
5. **First-party manifest path preserved** — shipped integration manifests remain for core tree scale.
6. **Tests as evidence** — `intergrax_catalog_fixture`, `intergrax_reference_enterprise_plugin`, external integration EP tests, VK reference plugin remain compatibility witnesses.

Migration **code** belongs to PLATFORM-PLUGIN-4 and PLATFORM-PLUGIN-9, not this stage.

---

## 25. Architecture invariants

Testable statements for audits and PLATFORM-PLUGIN-9 closeout:

1. **Domain capability contracts remain authoritative** for runtime behavior.
2. **Platform Plugin coordination must not bypass** domain validation, policy, or security gates.
3. **Installed Python plugin code is trusted in-process** unless a separate isolation architecture is explicitly in effect.
4. **Installation does not imply activation.**
5. **Discovery does not imply qualification.**
6. **Qualification may be capability- and domain-specific.**
7. **Third-party extensions must not require core source modification** for PEP surfaces.
8. **Domain-specific semantics must not be erased** for global uniformity.
9. **Public contracts require explicit compatibility ownership** (§20).
10. **Future harmonization preserves supported packages** through additive compatibility and deprecation strategy.
11. **No single global entry-point group** replaces domain groups.
12. **Secrets are not plugin metadata** and are resolved by host/domain (§12.3–§12.4).
13. **Multi-capability packages are allowed**; capabilities are separately discoverable.
14. **No global Platform Plugin DI container or global secret API** — domain wiring contexts and profiles remain authoritative (§13).

---

## 26. Future implementation responsibilities (PLATFORM-PLUGIN-3..9)

| Stage | Allowed to build (from this architecture) |
|-------|-------------------------------------------|
| **PLUGIN-3** | Author contract docs; optional package manifest schema; capability descriptor format; packaging rules for multi-capability wheels; **no** new runtime wrapper class |
| **PLUGIN-4** | Shared discovery utility adoption; additive discovery flags; per-EP import isolation improvements; **no** global catalog merge |
| **PLUGIN-5** | **Done** — §12.3 cross-surface config/secrets/DI matrix; §12.4 canonical flow; author guide §14; domain DI preserved; no global container |
| **PLUGIN-6** | **Done** — `platform_semantics.py`: explicit-version compatibility check API; `PlatformPluginLifecycleState` vocabulary; `PlatformPluginConflictKind` vocabulary; `package_identities_conflict` helper; EP conflict classification on `PluginConflictError`; **no** global lifecycle engine, conflict policy, or qualification gates |
| **PLUGIN-7** | **Done** — `platform_qualification.py`: trust model (`PlatformPluginTrustModel`); qualification level/status/evidence/subject/result contracts; delivery source (`external_package`, `host_embedded_extension`); pure production gates (`require_production_qualification`, `evaluate_package_production_admission`); PLUGIN-6 compatibility consumed as evidence; no sandbox/signing claims; no global registry or persistence |
| **PLUGIN-8** | **Done** — reference external wheel (`examples/platform_plugins/intergrax_reference_tool_plugin/`); host-embedded example (`examples/platform_plugins/local_embedded_tool_extension/`); application scaffold `extensions/` hook; executable E2E (`tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py`); §20.3 public extension matrix; author guide §16 |
| **PLUGIN-9** | **Done** — cross-stage conformance suite (`tests/contract/core/plugins/test_platform_plugin_contract.py`); CI gate in `.github/workflows/unit-tests.yml`; deprecation audit; [`PLATFORM_PLUGIN_9_CLOSEOUT.md`](../maintainers/plans/PLATFORM_PLUGIN_9_CLOSEOUT.md) |

**Explicitly not authorized before architecture amendment:** monolithic `PlatformPlugin` runtime type, mandatory global manifest replacing EPs, merging VK catalog into Tier-0 integration catalog, AgentRegistry setuptools discovery, sandbox claims without isolation implementation.

---

## 27. Explicit non-goals

- One universal plugin class replacing domain protocols
- Global plugin registry subsuming integration/tool/skill catalogs
- Forced unification of conflict policies (especially security override semantics)
- Process sandboxing as part of PLATFORM-PLUGIN-3..7
- Opening IEP registries (embedding, document handlers, registry v2) as public third-party APIs without domain programs
- LKW / application binding generalized as “plugin loading”
- RAG, VK, or Integrations feature refactors under the Platform Plugin banner

---

<a id="protocol-v2-platform-extensibility-target-invariants-2026-08-18"></a>

## Protocol v2 platform extensibility target invariants (2026-08-18)

Accepted [`PLATFORM_EXTENSIBILITY`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md) findings **01–06** (audit unit **PLATFORM_EXTENSIBILITY**, owning program **PLATFORM_PLUGINS**, 2026-08-21). Remediation **ACCEPTED / PLANNED** — **not implemented** by audit persistence.

1. **Evidence-derived qualification authority** — production qualification status is derived/validated against a versioned qualification policy and required evidence set; callers cannot create authoritative production qualification merely by setting `QualificationStatus.PRODUCTION_QUALIFIED`. Reuse `intergrax.core.qualification`; **no** second qualification engine ([`AUDIT-20260818-PLATFORM_EXTENSIBILITY-01`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md)).
2. **Package vs capability/domain qualification scope** — package qualification may be a prerequisite, but capability/domain admission binds separately where the domain requires it: distribution identity + domain + exact capability/entry point + qualification policy + evidence. Package-level bundle lookup must not blanket-apply one qualification to every entry point in a multi-capability distribution ([`AUDIT-20260818-PLATFORM_EXTENSIBILITY-03`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md)).
3. **Exact manifest capability ↔ admission binding** — production admission binds exact distribution + manifest identity/hash + exact `CapabilityDescriptor` / entry point + qualification result. Undeclared capabilities cannot inherit production admission from another capability in the same distribution ([`AUDIT-20260818-PLATFORM_EXTENSIBILITY-04`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md)).
4. **Cross-domain admission coverage** — each supported public PEP domain retains its domain loader/registry but consumes one shared pre-registration production-admission contract in strict/product profiles. Policy Rule/Definition loader is the reference partial implementation today; **no** global runtime plugin loader ([`AUDIT-20260818-PLATFORM_EXTENSIBILITY-02`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md)).
5. **Typed manifest-resolution evidence** — installed manifest resolution returns a typed result (VALID / ABSENT / INVALID / UNREADABLE) with safe reason codes; absent and invalid manifests must not collapse to indistinguishable `compatibility=None` diagnostics ([`AUDIT-20260818-PLATFORM_EXTENSIBILITY-05`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md)).
6. **Explicit process-lifetime discovery/cache lifecycle** — freeze policy **A** (installed plugin set immutable for process lifetime; package changes require restart) **or** **B** (controlled versioned rediscovery/cache invalidation). Do not leave lifecycle semantics as incidental `_EP_SPECS_CACHE` behavior ([`AUDIT-20260818-PLATFORM_EXTENSIBILITY-06`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md)).

**Preserved:** package-level coordination concept; domain contracts as runtime owners; taxonomy PEP/IP/HCE/IEP/NE; **no** universal `PlatformPlugin.execute()`; **no** unified sandbox/DI/registry; trusted-in-process model; DO-NOT-UNIFY list (§23); historical PLATFORM-PLUGIN-1..9 completion facts; current **PROVIDER-QUAL** track state (§18.3) — cross-link for provider-scoped evidence, do not duplicate. Protocol-v2 FAIL documents residual gaps only; it does **not** erase program history or claim remediation is implemented.

---

## 28. Evidence and references

| Artifact | Role |
|----------|------|
| [`PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](../maintainers/plans/PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md) | Inventory, taxonomy proposal, DO-NOT-UNIFY evidence |
| [`PLATFORM_PLUGINS.md`](../maintainers/plans/PLATFORM_PLUGINS.md) | Program roadmap |
| [`EXTENSION_AUTHOR_GUIDE.md`](../technical/guides/EXTENSION_AUTHOR_GUIDE.md) | Current author-facing EP index + PLUGIN-8 dual-mode quickstarts (§16) |
| `examples/platform_plugins/intergrax_reference_tool_plugin/` | PLUGIN-8 external wheel reference package |
| `examples/platform_plugins/local_embedded_tool_extension/` | PLUGIN-8 host-embedded Tools example |
| `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py` | PLUGIN-8 executable dual-mode E2E proof |
| `intergrax/core/plugins/discovery.py` | Unified EP loader (partial adoption) |
| `intergrax/core/plugin_env.py` | `INTERGRAX_DISCOVER_PLUGINS` |
| `tests/fixtures/plugin_packages/intergrax_catalog_fixture/` | Multi-capability package evidence |

---

## 29. Open architecture questions

| Question | Status | Notes |
|----------|--------|-------|
| Token optimization loader path | **FUTURE REVIEW** | Descriptor exists; no production loader — decide in token optimization program or PLUGIN-3+ |
| MCP as extension surface vs tool export | **Out of scope** | Application host wiring; not decided here |
| Default `INTERGRAX_DISCOVER_PLUGINS` in production hosts | **Operational** | Per-application; platform mandates opt-in, not default-on |
| Full promotion of Context plugins to public qualified surface | **Domain-owned** | CONTEXT_ENGINEERING + doc alignment; EP already exists |
| Agent EP discovery | **Closed** | Remains HCE — host registers agents |
| Isolated plugin execution | **Optional future** | Not required for program closeout |

No material unresolved issue blocks PLATFORM-PLUGIN-3 from starting author contract work.

---

## Canonical Platform Plugin Contract — summary

**Exists:** **Yes** — as a **package-level coordination contract** (metadata and vocabulary), not an executable wrapper.

**Scope (FROZEN):**

| In scope | Out of scope |
|----------|--------------|
| Plugin/package identity and version | Domain capability runtime behavior |
| Intergrax platform compatibility range | Integration category contracts |
| Provided capability descriptors (pointers) | Tool/skill schemas and manifests content |
| Optional author/trust/qualification metadata hooks | Secret values |
| Shared discovery terminology | VK publication rules |
| | RAG DI wiring |
| | Security fail modes |

Implementation of this contract in Python belongs to **PLATFORM-PLUGIN-3**.

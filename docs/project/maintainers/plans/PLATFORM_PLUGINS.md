# Platform Plugins — Maintainer Roadmap

**Program:** Platform Plugin architecture  
**Status:** PLATFORM-PLUGIN-1 **Done** (audit/evidence) · architecture **not frozen** until PLATFORM-PLUGIN-2  
**Audit evidence:** [`PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md)  
**Future architecture (PLATFORM-PLUGIN-2):** [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) — **does not exist yet**

**Last updated:** 2026-08-10

---

## Purpose

Establish a **platform-wide** extension model for Intergrax without prematurely unifying domain-specific mechanisms that differ for valid reasons.

This program answers:

- what extension surfaces exist today;
- which are public, host-composed, or internal;
- where duplication and inconsistency are accidental vs intentional;
- whether a canonical Platform Plugin Contract is justified.

**Out of scope for this program:** implementing RAG, Vendor Knowledge, LKW, Integrations, Tools, or Skills features. Those domains remain **audit evidence** and **consumers** of extension mechanisms.

---

## Current-state summary (PLATFORM-PLUGIN-1)

**FACT:** Intergrax today has **no single global plugin framework**. Extension is implemented through **multiple coexisting models**:

| Class | Examples |
|-------|----------|
| Tier-0 setuptools entry-point catalogs | integrations, tools, skills, context, memory stores, RAG chunkers/retrievers/rerankers |
| Domain-specific entry-point catalogs | vendor knowledge providers, security defenses, policy rules, tool invocation patterns |
| Shipped first-party bootstrap | `register_default_integrations`, `register_default_tools`, `register_default_skills`, RAG defaults |
| Host-composed wiring | `ApplicationEnvironmentProfile`, profiles, `RuntimePlugin`, `AgentRegistry` |
| Internal registries without third-party EP | embedding providers, document handlers, integration registry v2 (metadata), task execution registry |
| Descriptor-only / planned contracts | token optimization plugins |

**FACT:** Unified Tier-0 discovery loader exists (`intergrax/core/plugins/discovery.py`) but **not all surfaces use it**. Conflict policies, opt-in flags, and registration targets differ per domain.

**FACT:** Third-party installation is **discoverable and loadable** for several entry-point groups when `INTERGRAX_DISCOVER_PLUGINS=true` (or domain-specific opt-in). This is **not** production qualification.

See audit document for inventory, taxonomy proposal, gaps, and evidence matrix.

---

## Relationship to existing extension systems

| Domain program | Relationship to Platform Plugin |
|----------------|------------------------------|
| **Integrations** | Largest Tier-0 catalog; dual shipped manifest vs `IntegrationPlugin` model; registry v2 metadata track |
| **Tools / Skills** | Tier-0 catalog plugins; profile-gated materialization |
| **Context Engineering** | `ContextPlugin` catalog; entry points wired but author guide still marks partial rollout |
| **RAG** | Separate EP groups per component type; bootstrap-time registry, not global catalog |
| **Vendor Knowledge** | Separate contribution catalog and EP group; host composition required |
| **Security / Policy** | Separate EP groups; hook/policy integration |
| **Runtime (Nexus)** | `RuntimePlugin` host-composed; distinct from Tier-0 catalogs |
| **Observability** | Extension SDK for payload schemas; not a plugin loader |
| **Agents** | `AgentRegistry` — host registration only; no setuptools discovery |

**Principle:** Platform Plugin program **coordinates** cross-cutting discovery, trust, lifecycle, and author experience. It must **not** replace domain contracts (integration category contracts, tool contracts, VK contributions, etc.) without PLATFORM-PLUGIN-2 architecture decision.

---

## Roadmap stages

| Stage | Name | Status | Depends on | Exit criteria (summary) |
|-------|------|--------|------------|-------------------------|
| **PLATFORM-PLUGIN-1** | Global extension surface inventory & architecture audit | **Done** | — | Audit/evidence doc; this roadmap; no production code changes |
| **PLATFORM-PLUGIN-2** | Architecture decision & canonical domain doc | **Planned** | PLUGIN-1 | Create/finalize `architecture/PLATFORM_PLUGINS.md`; taxonomy decision; unify/do-not-unify boundaries |
| **PLATFORM-PLUGIN-3** | Author contract & packaging model | **Planned** | PLUGIN-2 | Single external package / multi-capability rules; manifest vs EP decision |
| **PLATFORM-PLUGIN-4** | Discovery & registration harmonization (where approved) | **Planned** | PLUGIN-2, PLUGIN-3 | Implement only approved unifications; preserve domain-specific loaders where rejected |
| **PLATFORM-PLUGIN-5** | Configuration, secrets & DI conventions | **Planned** | PLUGIN-2 | Cross-surface config matrix; host injection rules |
| **PLATFORM-PLUGIN-6** | Lifecycle, compatibility & conflict policy | **Planned** | PLUGIN-2 | Version/compatibility engine scope; duplicate-ID semantics |
| **PLATFORM-PLUGIN-7** | Trust, qualification & production gates | **Planned** | PLUGIN-2, PLUGIN-6 | discoverable vs production-qualified separation; security boundary doc |
| **PLATFORM-PLUGIN-8** | Third-party developer experience | **Planned** | PLUGIN-3, PLUGIN-7 | Scaffold, guides, reference plugin packaging (if justified) |
| **PLATFORM-PLUGIN-9** | Qualification, rollout & deprecation | **Planned** | PLUGIN-4–8 | Contract tests, CI gates, legacy path deprecation plan |

### PLATFORM-PLUGIN-2 expected output

**PROPOSAL:** PLATFORM-PLUGIN-2 creates the canonical architecture hub:

`docs/project/architecture/PLATFORM_PLUGINS.md`

equivalent to `RAG.md`, `TOOLS.md`, `SKILLS.md`, `INTEGRATIONS.md`, etc.

That document will freeze (for implementation) taxonomy, unification boundaries, and whether a **canonical Platform Plugin Contract** exists.

**Until PLATFORM-PLUGIN-2 completes, no architecture in this roadmap or audit should be treated as final.**

---

## Dependencies between stages

```text
PLUGIN-1 (audit)
    → PLUGIN-2 (architecture decision)
        → PLUGIN-3 (author contract)
        → PLUGIN-5 (config/secrets)
        → PLUGIN-6 (lifecycle/conflicts)
            → PLUGIN-4 (harmonization — only approved items)
            → PLUGIN-7 (trust/qualification)
                → PLUGIN-8 (DX)
                    → PLUGIN-9 (rollout)
```

PLUGIN-4 is intentionally **after** architecture decision: harmonization without PLUGIN-2 risks accidental centralization.

---

## Changes discovered during PLATFORM-PLUGIN-1

1. **Unified discovery loader is partial** — `core/plugins/discovery.py` covers Tier-0 groups; VK, security, policy, tool-invocation patterns use separate loaders.
2. **Integrations dual registration model** — shipped `manifest.py` + factory vs third-party `IntegrationPlugin` is intentional but increases author cognitive load.
3. **Context entry points exist in code** — `EXTENSION_AUTHOR_GUIDE` still lists Context as "Planned"; doc drift recorded in audit.
4. **Token optimization plugin descriptor** — contract exists; no setuptools loader or production registration path found.
5. **Integration registry v2** — additive metadata registry; not an extension surface for third parties.
6. **No `docs/audit_results/` tree on current `development`** — audit evidence placed alongside domain audits under `maintainers/plans/`.

---

## Explicit boundaries

| Track | Boundary |
|-------|----------|
| RAG | RAG component EP groups remain domain-owned; Platform Plugin may harmonize discovery flags only if PLUGIN-2 approves |
| Vendor Knowledge | Contribution catalog and LKW qualification remain VK program; not subsumed by Tier-0 catalog |
| LKW | Application binding and live capability rollout are not generalized plugin loading |
| Integrations | Category contracts and provider packages remain authoritative in INTEGRATIONS domain |
| Tools / Skills | ToolContract / SkillManifest contracts remain authoritative |
| Security / Policy | Fail-open/closed semantics stay security-domain owned |

---

## References

- Extension author guide: [`technical/guides/EXTENSION_AUTHOR_GUIDE.md`](../../technical/guides/EXTENSION_AUTHOR_GUIDE.md)
- Vendor Knowledge plugin guide: [`technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md`](../../technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md)
- Provider category contracts plan: [`PROVIDER_CATEGORY_CONTRACTS.md`](PROVIDER_CATEGORY_CONTRACTS.md)
- Platform foundation: [`architecture/PLATFORM_FOUNDATION.md`](../../architecture/PLATFORM_FOUNDATION.md)

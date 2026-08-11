# Platform Plugins — Maintainer Roadmap

**Program:** Platform Plugin architecture  
**Status:** PLATFORM-PLUGIN-1 **Done** · PLATFORM-PLUGIN-2 **Done** (canonical architecture frozen)
**Audit evidence:** [`PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md)  
**Canonical architecture:** [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

**Last updated:** 2026-08-11

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

**Principle (frozen in PLATFORM-PLUGIN-2):** Platform Plugin program **coordinates** cross-cutting discovery, trust, lifecycle, and author experience at the **package boundary**. It **must not** replace domain contracts (integration category contracts, tool contracts, VK contributions, etc.). See [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) §6–§7.

---

## Roadmap stages

| Stage | Name | Status | Depends on | Exit criteria (summary) |
|-------|------|--------|------------|-------------------------|
| **PLATFORM-PLUGIN-1** | Global extension surface inventory & architecture audit | **Done** | — | Audit/evidence doc; this roadmap; no production code changes |
| **PLATFORM-PLUGIN-2** | Architecture decision & canonical domain doc | **Done** | PLUGIN-1 | [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) — taxonomy, platform/domain boundary, DO-NOT-UNIFY, contract scope |
| **PLATFORM-PLUGIN-3** | Author contract & packaging model | **Done** | PLUGIN-2 | Package-level Platform Plugin contract in `intergrax/core/plugins/`; optional `[tool.intergrax.plugin]` manifest parsing; multi-capability metadata rules; entry points remain required for discovery |
| **PLATFORM-PLUGIN-4** | Discovery & registration harmonization (where approved) | **Done** | PLUGIN-2, PLUGIN-3 | Shared `discovery.py` scan/load primitives adopted by security, policy, tool-invocation loaders; RAG already on shared utility; VK composition unchanged |
| **PLATFORM-PLUGIN-5** | Configuration, secrets & DI conventions | **Done** | PLUGIN-2 | Cross-surface config matrix (architecture §12.3); canonical flow §12.4; host-resolved config and credential bindings documented; integration `env_prefix` preserved; no global DI/secrets API |
| **PLATFORM-PLUGIN-6** | Lifecycle, compatibility & conflict policy | **Planned** | PLUGIN-2 | Shared lifecycle vocabulary enforcement in tooling; platform compatibility metadata; shared conflict vocabulary — **domain policies remain** |
| **PLATFORM-PLUGIN-7** | Trust, qualification & production gates | **Planned** | PLUGIN-2, PLUGIN-6 | Installed/discoverable/qualified separation; package- and capability-level qualification; explicit in-process trust statement |
| **PLATFORM-PLUGIN-8** | Third-party developer experience & executable E2E proof | **Planned** | PLUGIN-3, PLUGIN-7 | Scaffold and guides; **third-party reference package** (genuine external Python wheel, structurally representative of an external author package — **multi-capability allowed** per architecture §21; exact layout defined in PLUGIN-3); **executable E2E proof** covering pip install → entry-point / approved discovery → plugin/capability discovery → configuration → host dependency injection → runtime invocation → cleanup/shutdown **without modifying Intergrax core** |
| **PLATFORM-PLUGIN-9** | Qualification, rollout, deprecation & program closeout | **Planned** | PLUGIN-4–8 | Contract tests, CI gates, additive deprecation plan for legacy paths; **final platform-level closeout audit** (see § Program closeout criteria) |

### PLATFORM-PLUGIN-2 output (complete)

Canonical architecture hub: [`docs/project/architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

**Frozen decisions (summary):**

- Taxonomy: PEP, IP, HCE, IEP, NE — unchanged from audit.
- Platform Plugin = **package-level coordination** + shared vocabulary — **not** a universal runtime wrapper.
- **Canonical Platform Plugin Contract:** **yes** — package metadata scope only; domain contracts unchanged.
- Multi-capability packages: **allowed**; capabilities separately discoverable via domain EP groups.
- Discovery: shared `discovery.py` utility **yes**; single global EP group **no**; domain groups retained.
- Manifest: **optional** coordination layer; domain manifests + entry points remain authoritative.
- DO-NOT-UNIFY list: frozen in architecture §23.

Implementation stages PLUGIN-3..9 must conform to this document.

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

## Audit evidence placement

**FACT:** [`docs/audit_results/`](../../audit_results/README.md) exists on `development`. It stores **dated outputs from orchestrated harness architecture audits** ([`ORCHESTRATOR.md`](../../audit/ORCHESTRATOR.md)): `YYYY-MM-DD/` folders with `progress.json`, `RUN_SUMMARY.md`, and per-domain `<DOMAIN>.md` results, initialized and validated via `scripts/audit/` tooling.

**FACT:** PLATFORM-PLUGIN-1 evidence lives in [`PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md) under `maintainers/plans/` — **retained here** because it is **program-specific** extension-surface inventory and architecture audit evidence, not an orchestrated per-domain run in the harness audit workflow. Domain programs persist Mode A2 results under `docs/audit_results/`; the Platform Plugin program coordinates cross-cutting extension architecture and keeps its audit alongside this roadmap.

---

## Program closeout criteria (PLATFORM-PLUGIN-9)

Before the Platform Plugin program can be marked **CLOSED**, a final audit must confirm:

1. Canonical [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) matches implementation.
2. Maintainer and author documentation matches executable behavior.
3. The **third-party reference package** (PLATFORM-PLUGIN-8) installs, discovers, and runs successfully.
4. The full **install → discovery → configuration → DI → runtime → cleanup** extension path works without core changes.
5. Trust, compatibility, and conflict rules are represented correctly in docs and runtime.
6. No accidental competing plugin architectures were introduced.
7. Approved **DO-NOT-UNIFY** boundaries from PLATFORM-PLUGIN-1 remain preserved.

---

## Changes discovered during PLATFORM-PLUGIN-1

1. **Unified discovery loader is partial** — `core/plugins/discovery.py` covers Tier-0 groups; VK, security, policy, tool-invocation patterns use separate loaders.
2. **Integrations dual registration model** — shipped `manifest.py` + factory vs third-party `IntegrationPlugin` is intentional but increases author cognitive load.
3. **Context entry points exist in code** — `EXTENSION_AUTHOR_GUIDE` still lists Context as "Planned"; doc drift recorded in audit.
4. **Token optimization plugin descriptor** — contract exists; no setuptools loader or production registration path found.
5. **Integration registry v2** — additive metadata registry; not an extension surface for third parties.
6. **`docs/audit_results/` exists** for orchestrated harness domain audits — PLATFORM-PLUGIN-1 program evidence correctly placed under `maintainers/plans/` (see § Audit evidence placement).

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

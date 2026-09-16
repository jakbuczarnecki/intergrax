# Capability Marketplace Engine

**Status:** ARCHITECTURE FROZEN (ME-RB1)  
**Audience:** Principal/staff engineers, platform architects  
**Authority:** Current production code, contracts, and architecture gates — not historical audit prose alone.

The **Capability Marketplace Engine** is the single enterprise **discovery / product / acquisition plane** for Agent, Tool, and Skill capabilities. It composes **Capability Catalog** (federated read, query, rank, govern) with **Marketplace** (product listing, publisher/commercial presentation, availability projection). It is **not** lifecycle authority, **not** execution, and **not** a Decision System.

```text
ONE COMMON CAPABILITY MARKETPLACE ENGINE
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
     AGENT     TOOL     SKILL
    vertical  vertical  vertical
       │          │          │
 Agent Dist.  Tool domain  Skill domain
```

**Related canon:** [`CAPABILITY_CATALOG_AND_DISCOVERY.md`](CAPABILITY_CATALOG_AND_DISCOVERY.md) · [`AGENT_DISTRIBUTION.md`](AGENT_DISTRIBUTION.md) · [`TOOLS.md`](TOOLS.md) · [`SKILLS.md`](SKILLS.md) · [ME-RB1 gap register](../maintainers/audits/COMMON_MARKETPLACE_ENGINE_RB1.md)

---

## 1. Purpose

Provide one reusable plane for:

- Federated capability catalog read and discovery (Agent + Tool + Skill).
- Marketplace product surface (listings, publisher/commercial metadata, visibility).
- Governed candidate narrowing (via Capability Catalog governance — not marketplace billing).
- Typed handoff **intent** to domain lifecycle authorities (future ME-RB4 contract).

Support **human** consumers (browse, search, inspect) and **machine** consumers (capability need → discover → rank → govern → recommend → handoff) without duplicating engines per vertical.

---

## 2. Non-goals

The Common Marketplace Engine does **not**:

- Install, activate, enable, or mutate runtime registries.
- Materialize agents, invoke tools, or execute skills.
- Act as trust authority, permission grant, billing settlement, or governance engine of record.
- Import or expose **Nexus** or Execution Engine private orchestration.
- Implement AC-4 dynamic agent acquisition lifecycle (Agent Distribution owns that path).
- Replace Integration registry semantics (see IntegrationMarketplaceCatalog classification in RB1 audit).

---

## 3. Common Marketplace Core

Normative composition (implemented slices in parentheses):

| Concern | Owner module | Notes |
| -------- | ------------- | ----- |
| Catalog federation | `intergrax.capability_catalog.federation` | `FederatedCapabilityCatalog`, `CapabilityCatalogSource` |
| Discovery query | `intergrax.capability_catalog.discovery` | `CapabilityDiscoveryQuery` |
| Filtering | Discovery + query contracts | Typed filters, not ad-hoc dict bags |
| Ranking strategy host | `intergrax.capability_catalog.ranking` | `CapabilityRanker` protocol |
| Governed narrowing | `intergrax.capability_catalog.governance` | `CapabilityGovernanceEvaluator` |
| Product listing | `intergrax.marketplace.listing` | Wraps canonical `CapabilityCatalogEntry` |
| Publisher metadata | `intergrax.contracts.marketplace.publisher` | Display-only |
| Commercial metadata | `intergrax.contracts.marketplace.commercial` | Display-only; must not affect rank/govern |
| Provenance | `intergrax.contracts.capability_catalog.provenance` | On catalog entries |
| Availability projection | Discovery evidence + listing views | Not install/enable authority |
| Marketplace visibility | Listing + source identity | Stage 7 private sources |
| Recommendation host | **Gap** — no dedicated recommendation SPI (ME-RB1-006) |
| Lifecycle handoff contracts | **Gap** — domain ports only today (ME-RB1-005) |
| Usage attribution handoff | `intergrax.contracts.capability_metering` | Events ≠ billing |

**Join surface:** `MarketplaceCatalogService` joins Stage-3 discovery candidates with marketplace product metadata keyed by canonical identity.

---

## 4. Agent vertical

**Target model:** Agent Marketplace = **Agent vertical over Capability Marketplace**, not a separate marketplace engine.

| Semantics | Classification |
| --------- | -------------- |
| Federated catalog row / discovery identity | **COMMON** (`CapabilityCatalogEntry`, agent adapter) |
| Listing, publisher presentation, commercial display | **COMMON** (marketplace metadata on shared listing) |
| Categories / marketplace browse UX | **COMMON** product (future UI) |
| Trust labels for **execution** | **AGENT DISTRIBUTION** (qualification, attestation) |
| Package/version resolution, install, bind, activate | **AGENT DISTRIBUTION** (AC-3 / AC-4) |
| Dynamic acquisition orchestration | **AGENT DISTRIBUTION** (`dynamic_acquisition.py`) — separate from catalog core |

Adapter: `intergrax.capability_catalog.adapters.agent` maps `CatalogSourceProvider` → `CapabilityCatalogSource` (read-only).

Public product concept: [`AGENT_MARKETPLACE.md`](../overview/AGENT_MARKETPLACE.md) (vertical concept document).

---

## 5. Tool vertical

**Target model:** Tool Marketplace = **Tool vertical over Capability Marketplace**.

| Semantics | Classification |
| --------- | -------------- |
| Tool identity in federated catalog | **COMMON** + tool adapter (`tools.registry.catalog` read) |
| Listing / publisher / commercial | **COMMON** |
| Tool profile, registry, runtime, invocation | **TOOL DOMAIN** / **EXECUTION** |
| Permissions, enablement, installation | **TOOL DOMAIN** — not marketplace |

Adapter: `intergrax.capability_catalog.adapters.tool` (+ private enterprise source Stage 7).

---

## 6. Skill vertical

**Target model:** Skill Marketplace = **Skill vertical over Capability Marketplace**.

| Semantics | Classification |
| --------- | -------------- |
| Skill identity, manifest digest in catalog | **COMMON** + skill adapter |
| Composition metadata (required tools/agents) | **SKILL DOMAIN** contracts on entries; listing is **COMMON** |
| SkillResolver, composition, runtime | **SKILL DOMAIN** — no `execute_skill` in marketplace |
| Listing must not imply direct execution | **Normative** — discovery only |

Adapter: `intergrax.capability_catalog.adapters.skill` (+ private skill source).

---

## 7. Human consumers

Supported today via:

- `MarketplaceCatalogService.list_listings` / `get_listing` with optional `query_text` (substring filter on listing fields).
- Federated catalog snapshots for inspect/browse composition in Tier-3 wiring.

**Gap:** No dedicated human-facing API contract bundle (ME-RB1-007) — behavior exists as library surface only.

---

## 8. Machine consumers

Supported today via:

- `CapabilityDiscoveryQuery`, ranking, governance pipelines on catalog snapshots.
- `WorkStageCapabilityNeed` + `work_stage_discovery` for staged need → governed candidates (Autonomous Work consumes; catalog core does not import AW).

**Gap:** No stable `CapabilityNeed` envelope for Virtual Workers at marketplace boundary (ME-RB1-008). AW types are consumer-specific, not marketplace public API.

---

## 9. Contract architecture

Platform operates on **contracts**, not hardcoded consumers → implementations.

| Layer | Packages |
| ----- | -------- |
| Discovery / identity / query / governance / ranking | `intergrax.contracts.capability_catalog` |
| Product metadata | `intergrax.contracts.marketplace` |
| Usage attribution | `intergrax.contracts.capability_metering` |
| Default implementations | `intergrax.capability_catalog`, `intergrax.marketplace`, `intergrax.capability_metering` |

**Finding:** `CapabilityCatalogSource` Protocol lives in implementation package (`intergrax.capability_catalog.source`), not contracts — promote in ME-RB2.

---

## 10. Plugin architecture

Replaceable variation points (target state):

| Variation | Contract today | Verdict |
| --------- | -------------- | ------- |
| Catalog source | `CapabilityCatalogSource` (impl package) | PARTIAL |
| Ranker | `CapabilityRanker` | ENTERPRISE_READY |
| Governance evaluator | contracts + adapter evaluators | ENTERPRISE_READY |
| Listing projection | functions in `marketplace.projection` | PARTIAL |
| Marketplace metadata backend | `MarketplaceCapabilityCatalogSource` class | PARTIAL |
| Search / text filter | inline in `MarketplaceCatalogService` | PARTIAL (documented as product filter) |
| Recommendation | none | MISSING |
| Availability evidence | typed evidence contracts | PARTIAL |
| Lifecycle handoff | domain-specific (AC-4, tool/skill ports) | MISSING common envelope |

Do **not** add empty Protocols without semantics (ME-RB2 scope).

---

## 11. Lifecycle handoff

Marketplace/catalog discovery **ends** at governed, ranked **candidates** and optional **selection intent** — not lifecycle mutation.

**Target flow (conceptual, ME-RB4):**

```text
MarketplaceCapabilitySelection (future)
        ↓
CapabilityLifecycleHandoffRequest (future)
        ↓
vertical adapter (Agent / Tool / Skill)
        ↓
domain authority (Agent Distribution, Tool domain, Skill domain)
```

**ME-RB1 decision:** Prefer **Option B** — generic typed envelope + domain-specific payload ports — because install/activate/trust semantics differ per vertical. Do **not** implement until ME-RB4; no `Union[Any]` / `dict[str, Any]` handoff bags.

Existing separate paths: AC-4 `dynamic_acquisition` (Agent Distribution), tool/skill enablement via domain registries, Execution Engine public contracts.

---

## 12. Governance boundary

- **Governance narrowing** during discovery: Capability Catalog `CapabilityGovernanceEvaluator` + vertical adapter evaluators.
- **Governance engine of record:** platform Governance contracts — marketplace does not adjudicate enterprise policy alone.
- Marketplace commercial metadata **must not** influence governance or ranking (enforced by separation of imports in architecture gates).

---

## 13. Decision boundary

```text
Search   → find candidates (discovery query + optional listing text filter)
Ranking  → order candidates (CapabilityRanker)
Selection → consumer choice / orchestration policy (not marketplace core)
Decision System → semantic decision when required (separate subsystem)
Governance → permit/deny candidate visibility/eligibility
```

Marketplace **must not** embed a mini Decision System.

---

## 14. Execution boundary

```text
Marketplace / Catalog
    ↓ handoff intent (future typed)
Execution public contracts
    ↓
Execution Engine
    ↓ private orchestration
Nexus
```

No catalog/marketplace imports of execution runtime or Nexus (P0 gates).

---

## 15. Nexus boundary

**P0:** Marketplace core and Capability Catalog core **MUST NOT** import `intergrax.nexus` or `intergrax.runtime.nexus` or reference Nexus as lifecycle authority. Enforced by AST gates (`test_marketplace_architecture_gates`, program boundary tests).

Product docs that name Nexus as public runtime must defer to Execution Engine public boundary (see TOOL-MARKETPLACE-RB0 remediations).

---

## 16. Observability boundary

Discovery/handoff traceability is a **future** ME-10 concern. V1 emits usage events via metering contracts at execution/domain boundaries — not inside marketplace listing code.

---

## 17. Metering boundary

**Capability Metering** owns usage event shape and consumer ports. Marketplace may attach **display-only** commercial metadata; settlement and billing remain external.

---

## 18. Commercial boundary

Publisher and commercial metadata are **presentation** contracts only. No checkout, purchase, or grant APIs in marketplace package (forbidden API names gated by tests).

---

## 19. Forbidden flows

1. Marketplace → pip / subprocess / tool runtime registry mutation.  
2. Marketplace → agent install/activate/register.  
3. Catalog core → domain runtime registries (adapters read catalog surfaces only).  
4. Ranking/governance modules → marketplace product contracts (authority separation gate).  
5. Capability Marketplace Engine → Nexus as discovery or lifecycle dependency.

---

## 20. Future Virtual Worker usage

The engine must remain reusable for typed **capability need** requests (required capability, tenant, constraints, risk envelope) **without** importing Autonomous Work implementation. Today `WorkStageCapabilityNeed` proves staged discovery; a marketplace-neutral `CapabilityNeed` contract is **ME-RB12** scope.

---

## 21. E2E target scenarios (roadmap)

| ID | Scenario |
| -- | -------- |
| ME-13 | Marketplace → Agent Distribution → Execution |
| ME-14 | Marketplace → Tool domain → Execution |
| ME-15 | Marketplace → Skill domain → Composition |
| ME-16 | Mixed Agent + Tool + Skill acquisition |
| ME-17 | Virtual Worker machine consumer |
| ME-18 | Dynamic Organization resource composition |

V1 code proves federated read, governance, marketplace join, and metering substrate — not full E2E product flows.

---

## IntegrationMarketplaceCatalog (classification)

**Answer A:** `IntegrationMarketplaceCatalog` is a **separate bounded-context catalog** (integrations registry + trust score projection). It is **not** a fourth Capability Marketplace vertical. Naming is historical/product. **Do not migrate** in ME-RB1; optional future bridge is a design proposal only.

---

## Architecture gates (ME-RB1)

Enforced in tests:

- `tests/unit/marketplace/test_marketplace_architecture_gates.py`
- `tests/unit/marketplace/test_common_marketplace_engine_rb1_gates.py`
- `tests/unit/capability_catalog/test_architecture_gates.py`
- `tests/unit/architecture/test_capability_catalog_v1_program_boundaries.py`

Program packages must not import `intergrax.runtime`, applications, or Nexus; marketplace must not import Agent Distribution implementation.

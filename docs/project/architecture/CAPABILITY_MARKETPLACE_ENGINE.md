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
- Typed handoff **intent** to domain lifecycle authorities (`intergrax.contracts.marketplace` lifecycle handoff contracts, ME-RB4).

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
| Lifecycle handoff contracts | `intergrax.contracts.marketplace` + `intergrax.marketplace.handoff` (ME-RB4) |
| Discovery → selection → handoff traceability | `intergrax.contracts.marketplace.handoff_traceability` + `intergrax.marketplace.handoff_traceability` (ME-10) |
| Usage attribution handoff | `intergrax.contracts.capability_metering` | Events ≠ billing |

**Join surface:** `MarketplaceCatalogService` joins Stage-3 discovery candidates with marketplace product metadata keyed by canonical identity.

---

## 4. Agent vertical

**Target model (ME-RB3 aligned):** Agent Marketplace = **Agent vertical over Capability Marketplace**, not a separate marketplace engine.

```text
Agent Distribution catalog (CatalogSourceProvider)
        ↓
AgentCatalogCapabilitySource (read-only adapter)
        ↓
CapabilityCatalogSource → FederatedCapabilityCatalog → MarketplaceCatalogService
```

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

**Target model (ME-RB3 aligned):** Tool Marketplace = **Tool vertical over Capability Marketplace**.

```text
Tool registry/catalog read (iter_bundles)
        ↓
ToolBundleCatalogSource
        ↓
CapabilityCatalogSource → common engine
```

| Semantics | Classification |
| --------- | -------------- |
| Tool identity in federated catalog | **COMMON** + tool adapter (`tools.registry.catalog` read) |
| Listing / publisher / commercial | **COMMON** |
| Tool profile, registry, runtime, invocation | **TOOL DOMAIN** / **EXECUTION** |
| Permissions, enablement, installation | **TOOL DOMAIN** — not marketplace |

Adapter: `intergrax.capability_catalog.adapters.tool` (+ private enterprise source Stage 7).

---

## 6. Skill vertical

**Target model (ME-RB3 aligned):** Skill Marketplace = **Skill vertical over Capability Marketplace**.

```text
Skill registry/catalog read (iter_bundles + catalog manifests)
        ↓
SkillBundleCatalogSource
        ↓
CapabilityCatalogSource → common engine
```

| Semantics | Classification |
| --------- | -------------- |
| Skill identity, manifest digest in catalog | **COMMON** + skill adapter |
| Composition metadata (required tools/agents) | **SKILL DOMAIN** contracts on entries; listing is **COMMON** |
| SkillResolver, composition, runtime | **SKILL DOMAIN** — no `execute_skill` in marketplace |
| Listing must not imply direct execution | **Normative** — discovery only |

Adapter: `intergrax.capability_catalog.adapters.skill` (+ private skill source).

---

## 6.1 ME-RB3 — Domain vertical alignment (closed)

| Vertical | Domain read surface | Adapter | Common contract | Status |
| -------- | ------------------- | ------- | --------------- | ------ |
| Agent | `CatalogSourceProvider` / `AgentCatalogEntry` | `AgentCatalogCapabilitySource` | `CapabilityCatalogSource` | **aligned** |
| Tool | `tools.registry.catalog` (`iter_bundles`) | `ToolBundleCatalogSource` | `CapabilityCatalogSource` | **aligned** |
| Skill | `skills.registry.catalog` + catalog manifests | `SkillBundleCatalogSource` | `CapabilityCatalogSource` | **aligned** |

Normative stack (all verticals):

```text
domain read surface
        ↓
vertical adapter (capability_catalog.adapters.*)
        ↓
CapabilityCatalogSource
        ↓
FederatedCapabilityCatalog
        ↓
Capability Marketplace Engine (discovery + MarketplaceCatalogService)
```

Proof tests: `tests/unit/marketplace/test_me_rb3_domain_vertical_alignment.py`, adapter tests under `tests/unit/capability_catalog/adapters/`, ME-RB2 plugin proof for custom `CapabilityCatalogSource` without core changes.

**External/custom vertical provider proof:** third-party `CapabilityCatalogSource` + matching `MarketplaceMetadataSource` (no default adapter subclasses) federate with Agent/Tool/Skill builtins and surface as `MarketplaceCapabilityListing` via `MarketplaceCatalogService` — **PASSED** (`test_custom_vertical_provider_appears_in_common_marketplace`).

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
| Catalog source port | `CapabilityCatalogSource` in `intergrax.contracts.capability_catalog` |
| Product metadata | `intergrax.contracts.marketplace` |
| Listing projection SPI | `MarketplaceListingProjection` in `intergrax.contracts.marketplace` |
| Marketplace metadata SPI | `MarketplaceMetadataSource` in `intergrax.contracts.marketplace` |
| Usage attribution | `intergrax.contracts.capability_metering` |
| Default implementations | `intergrax.capability_catalog`, `intergrax.marketplace`, `intergrax.capability_metering` |

**ME-RB2 (closed):** `CapabilityCatalogSource`, `MarketplaceListingProjection`, and `MarketplaceMetadataSource` are public replaceable ports; default implementations remain in implementation packages only.

---

## 10. Plugin architecture

Replaceable variation points (post ME-RB2):

| Variation | Contract today | Verdict |
| --------- | -------------- | ------- |
| Catalog source | `CapabilityCatalogSource` (`intergrax.contracts.capability_catalog`) | ENTERPRISE_READY |
| Ranker | `CapabilityRanker` | ENTERPRISE_READY |
| Governance evaluator | contracts + adapter evaluators | ENTERPRISE_READY |
| Listing projection | `MarketplaceListingProjection` + `DefaultMarketplaceListingProjection` | ENTERPRISE_READY |
| Marketplace metadata backend | `MarketplaceMetadataSource` + `InMemoryMarketplaceMetadataSource` | ENTERPRISE_READY |
| Search | `CapabilitySearchStrategy` + `DefaultMarketplaceListingTextSearchStrategy` | ENTERPRISE_READY (ME-5) |
| Recommendation | `CapabilityRecommendationStrategy` + `DefaultTopRankedCapabilityRecommendationStrategy` | ENTERPRISE_READY (ME-5) |
| Availability evidence | typed evidence contracts | PARTIAL |
| Lifecycle handoff | `MarketplaceLifecycleHandoffRequest` + domain-owned handoff ports | ENTERPRISE_READY (ME-RB4-C1) |

Do **not** add empty Protocols without semantics (ME-RB2 scope).

**ME-RB1 gaps:** ME-RB1-002, ME-RB1-003, ME-RB1-004 — **closed in ME-RB2**.

---

## 11. Lifecycle handoff (ME-RB4)

Marketplace/catalog discovery **ends** at governed, ranked **candidates** and **selection**; lifecycle mutation remains domain-owned.

**Canonical flow:**

```text
DISCOVERY
  ↓
RANKING
  ↓
GOVERNANCE NARROWING
  ↓
SELECTION (MarketplaceCapabilitySelection)
  ↓
LIFECYCLE HANDOFF (MarketplaceLifecycleHandoffRequest)
  ↓
MarketplaceLifecycleHandoffService → MarketplaceLifecycleHandoffHandler
  ↓
vertical handoff adapter (Agent / Tool / Skill)
  ↓
DOMAIN AUTHORITY (Agent Distribution, Tool domain, Skill domain)
```

**Invariant:** `HANDOFF_ACCEPTED` ≠ installed ≠ active ≠ routable ≠ executed. Handoff outcome describes delegation to domain authority only.

**Contracts:** generic envelope + typed vertical payloads (`AgentLifecycleHandoffPayload`, `ToolLifecycleHandoffPayload`, `SkillLifecycleHandoffPayload`); plugin `MarketplaceLifecycleHandoffHandler`; explicit `LifecycleHandoffResolver` mapping — no global registry, no `dict[str, Any]` bags.

**Agent path:** `AgentMarketplaceLifecycleHandoffHandler` → `AgentMarketplaceLifecycleHandoffPort` (`intergrax.contracts.agent_distribution`) with optional `AgentDistributionAcquisitionBridge` → `DynamicAgentAcquisitionPort`. Typed `DynamicAgentAcquisitionError` maps to `DOMAIN_UNAVAILABLE`; unexpected errors propagate.

**Tool path:** `ToolMarketplaceLifecycleHandoffHandler` → `ToolMarketplaceLifecycleHandoffPort` (`intergrax.contracts.tools`). `ToolLifecycleHandoffUnavailableError` → `DOMAIN_UNAVAILABLE`.

**Skill path:** `SkillMarketplaceLifecycleHandoffHandler` → `SkillMarketplaceLifecycleHandoffPort` (`intergrax.contracts.skills`). `SkillLifecycleHandoffUnavailableError` → `DOMAIN_UNAVAILABLE`.

**Neutral ack:** `DomainLifecycleHandoffAck` / `DomainLifecycleHandoffDisposition` live in `intergrax.contracts.lifecycle_handoff` (handoff ack ≠ lifecycle state). Marketplace service maps only `MarketplaceLifecycleHandlerError` to `HANDLER_FAILED`; programming defects propagate.

Proof: `tests/unit/marketplace/test_me_rb4_lifecycle_handoff.py`, `tests/unit/marketplace/test_me_rb4_handoff_architecture_gates.py`.

---

## 12. Governance boundary

- **Governance narrowing** during discovery: Capability Catalog `CapabilityGovernanceEvaluator` + vertical adapter evaluators.
- **Governance engine of record:** platform Governance contracts — marketplace does not adjudicate enterprise policy alone.
- Marketplace commercial metadata **must not** influence governance or ranking (enforced by separation of imports in architecture gates).

**Ownership (ME-6):** Marketplace **consumes** governed results (`GovernedCapabilityCandidate`). Capability Catalog governance **composes** decision evidence from independent inputs. Trust and availability **authorities remain domain-owned** — catalog governance consumes read-only projections (`CapabilityAgentGovernanceEvidence`, `CapabilityToolGovernanceEvidence`, `CapabilitySkillGovernanceEvidence`, Stage-3 `AvailabilityDisposition`), never Agent Distribution verification, tool runtime state, or skill runtime activation.

**Invariant:** `ranked != admissible`; `trusted != available != policy allowed`. Trust and availability are evidence dimensions — not automatic allow shortcuts.

**Evaluator failure semantics (ME-6-C1):** Fail-closed applies to known contractual evaluator unavailability (`CapabilityGovernanceEvaluatorUnavailableError` → `BLOCKED` + `EVALUATOR_FAILURE` in `STRICT`). Unexpected programming defects remain visible and propagate. Malformed evaluator output remains `CapabilityGovernanceError` (integrity), not `EVALUATOR_FAILURE`.

---

## 13. Decision boundary

```text
Discovery facts
    ↓
Search (CapabilitySearchStrategy)
    ↓
Ranking (CapabilityRanker)
    ↓
Governance narrowing
    ├── policy evidence (tool projection evaluator)
    ├── trust evidence (agent projection evaluator)
    └── availability evidence (baseline availability-preserving evaluator)
    ↓
Allowed / Blocked partition (GovernedCapabilityCandidate | BlockedCapabilityCandidate)
    ↓
Recommendation (CapabilityRecommendationStrategy)
    ↓
Selection → consumer choice / orchestration policy (not marketplace core)
Decision System → semantic decision when required (separate subsystem)
```

`MarketplaceDiscoveryService` orchestrates search → rank via contracts only (no embedded algorithms). Governance narrowing remains a separate Capability Catalog boundary; recommendation consumes only governance-admissible `GovernedCapabilityCandidate` input via `MarketplaceRecommendationService` (never raw ranked candidates).

**Invariant:** Recommendation cannot consume raw discovery or raw ranked candidates.

`MarketplaceCatalogService.list_listings` delegates optional `query_text` to an injected search strategy (default preserves legacy substring semantics).

Marketplace **must not** embed a mini Decision System or a governance engine.

---

## 14. Execution boundary

```text
Marketplace / Catalog
    ↓ CapabilityHandoffEnvelope (ME-10) + lifecycle handoff request (ME-RB4)
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

## 16. Observability and traceability boundary

**ME-10 (implemented):** Discovery → governance → ranking → **explicit selection** → **`CapabilityHandoffEnvelope`** is observational traceability only. It binds:

- `discovery_correlation_id` (marketplace-scoped; not `run_id` / execution IDs)
- governed visible candidate counts (no foreign-private candidate leakage)
- `CapabilityMarketplaceExplicitSelection` with **`CapabilityReleaseIdentity`**
- neutral `CapabilityHandoffConsumerTarget` (agent / tool / skill domain)

Trace evidence uses optional `CapabilityHandoffTraceEvidenceConsumer` (in-memory reference provider). **Not** `RuntimeEvent`, **not** execution lineage, **not** `CapabilityUsageEvent`.

Hard separations:

```text
discovery facts ≠ selection ≠ handoff ≠ execution ≠ usage
handoff ≠ installation ≠ entitlement ≠ billing
```

V1 usage events remain at execution/domain boundaries via Capability Metering — not inside marketplace listing or handoff delivery code.

```text
Catalog → Visibility → Governance → Ranking → Explicit Selection
    → CapabilityHandoffEnvelope → Domain Adapter / Consumer → Execution (downstream)
```

---

## 17. Metering boundary

**Capability Metering** owns usage event shape and consumer ports. Marketplace may attach **display-only** commercial metadata; settlement and billing remain external.

Hard invariants:

```text
usage != pricing
usage != billing
commercial metadata != charge authority
Marketplace != Billing
```

Canonical flow:

```text
Exact capability release (provenance on usage event)
        ↓
authoritative usage producer (domain / execution boundary — not listing views)
        ↓
CapabilityUsageEvent (immutable factual measurement)
        ↓
CapabilityUsageRecorder → CapabilityUsageConsumer (pluggable sink)
        ↓
external metering / billing consumer
```

Marketplace listing/search/recommendation **must not** emit capability usage events (product analytics is a separate concern).

---

## 18. Commercial boundary

Publisher and commercial metadata are **presentation** contracts only (`MarketplaceCommercialMetadata`: labels, display price text, opaque `pricing_reference` — not charge authority). No checkout, purchase, or grant APIs in marketplace package (forbidden API names gated by tests). No `calculate_price`, invoicing, or settlement in marketplace core.

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

## ME-7 — Publisher / version / provenance (closed)

Canonical identity layers (do not collapse):

```text
Logical capability identity     → CapabilityLogicalIdentity.logical_id (+ kind)
Publisher identity (claim)      → CapabilityProvenance.publisher (not source_id)
Version / release label         → CapabilityProvenance.version_label (opaque; no auto-normalize)
Source identity                 → CapabilitySourceIdentity (catalog/discovery origin)
Integrity reference             → CapabilityProvenance.content_digest (opaque; compare exact)
Trust verdict                   → governance evidence (ME-6; not provenance)
```

Stage-3 discovery key (`CapabilityIdentityKey`) is **source-qualified logical identity** — it intentionally excludes version and publisher. Exact released artifact addressing uses `CapabilityReleaseIdentity` (parallel contract) built from `CapabilityCatalogEntry` provenance facts.

```text
Domain/source authority
      ↓
publisher + version + provenance (CapabilityProvenance on CapabilityCatalogEntry)
      ↓
FederatedCapabilityCatalog (merge fail-closed on discovery-key conflicts)
      ↓
Marketplace listing projection (preserve; never rewrite canonical facts)
      ↓
search / rank / governance / recommendation (validators preserve catalog_entry)
      ↓
selection / lifecycle handoff (MarketplaceCapabilitySelection.capability)

facts preserved unchanged end-to-end
```

**Release multiplicity (Model A — one canonical release per snapshot row):**

```text
DISCOVERY IDENTITY (Stage-3 / federation merge key)
  kind + source_id + source_kind + logical_id
        ↓
  exactly one canonical CapabilityCatalogEntry per federated snapshot row
        ↓
RELEASE FACTS (audit / handoff — not merge key)
  publisher + version_label + content_digest + package_reference
        ↓
CapabilityReleaseIdentity (parallel exact-release contract)
```

Capability Catalog stores **one canonical discoverable release** per source-qualified logical identity in a snapshot. It is **not** a multi-version release history registry, artifact registry, or package repository.

- Same discovery identity with differing catalog facts → `CapabilityCatalogIdentityConflict` (fail-closed; no silent last/first wins).
- Publisher identity does not partition discovery identity. Conflicting publisher claims for the same discovery identity fail closed.
- Version identity does not partition discovery identity. Conflicting release facts for the same discovery identity fail closed.
- `CapabilityReleaseIdentity` provides exact release audit identity, not discovery multiplicity.
- Multiple releases may appear as distinct catalog rows **only** when discovery identity differs (e.g. different `source_id`, or intentionally distinct `logical_id` rows — not semver coexistence under one logical id).
- A newer snapshot/read cycle may replace the surfaced release (v1 → v2) without both coexisting in one snapshot.

Proofs: `tests/unit/marketplace/test_me7_publisher_version_provenance.py`.

**Gap (out of scope):** historical / simultaneous multi-version release registry under one source-qualified logical identity → future design if product requires Model B.

---

## ME-8 — Commercial & metering boundary (closed)

| Concern | Contract / port | Marketplace role |
| ------- | ----------------- | ------------------ |
| Usage fact | `CapabilityUsageEvent` | None (no emission on discovery) |
| Metering sink | `CapabilityUsageConsumer` / `CapabilityUsageRecorder` | Handoff only via external adapters |
| Commercial display | `MarketplaceCommercialMetadata` | Presentation on listings |
| Pricing / invoice / settlement | External billing domain | **Forbidden** in marketplace core |

`CapabilityUsageEvent` carries source-qualified identity, frozen provenance (release facts), tenant, optional run/task correlation, and integer `quantity` — never monetary authority fields.

Proofs: `tests/unit/marketplace/test_me8_commercial_metering_boundary.py`, `tests/unit/contracts/capability_metering/test_capability_metering_contract_import_gates.py`, Stage-13 metering tests.

**Gap register (out of scope):** full billing consumer SPI and financial ledger integration remain outside Capability Marketplace Engine.

---

## ME-9 — Multi-Tenant / Private Marketplace (ME-9-C1 correction)

Marketplace visibility is **explicit**, **typed**, and **fail-closed**. It is not IAM, not organization directory, not execution entitlement, not governance, and not commercial classification.

**Invariants:**

- **Tenant scope ≠ organization scope** — no inference between them.
- **Visible ≠ authorized** to install, activate, or execute.
- **Visibility narrowing ≠ governance admissibility** — independent layers.
- **Private Marketplace** = same Marketplace Engine + authorized `MarketplaceQueryContext` + visibility narrowing (not a second engine).

```text
Federated catalog (single federated truth)
      ↓
Marketplace query + MarketplaceQueryContext (caller-authorized tenant_id / organization_id)
      ↓
Marketplace visibility narrowing
      ↓ hard scope isolation (non-disableable)
      ↓ optional MarketplaceVisibilityPolicyExtension (restrict only; never widen)
Search (CapabilitySearchStrategy / listing text search)
      ↓
Ranking (CapabilityRanker — only visible candidates)
      ↓
Governance narrowing (CapabilityGovernanceEvaluator)
      ↓
Recommendation (CapabilityRecommendationStrategy)
      ↓
Selection / lifecycle handoff
```

| Scope | Semantics |
| ----- | --------- |
| `PUBLIC` | Discoverable by any marketplace caller (still subject to governance) |
| `TENANT_PRIVATE` | Discoverable only when `MarketplaceQueryContext.tenant_id` matches listing `tenant_id` |
| `ORGANIZATION_PRIVATE` | Discoverable only when `MarketplaceQueryContext.organization_id` matches listing `organization_id` |

Listings without `visibility` metadata default to **PUBLIC** (backward compatible). Missing tenant context excludes `TENANT_PRIVATE`; missing organization context excludes `ORGANIZATION_PRIVATE` — **PUBLIC only** when both scope ids are absent.

Platform tenant identity for discovery scope reuse: `CapabilityDiscoveryScope.tenant_id` (catalog) remains separate from `MarketplaceQueryContext` (marketplace product visibility).

Proofs: `tests/unit/marketplace/test_me9_multi_tenant_private_marketplace.py`, `tests/unit/contracts/marketplace/test_marketplace_visibility_contracts.py`.

---

## Architecture gates (ME-RB1)

Enforced in tests:

- `tests/unit/marketplace/test_marketplace_architecture_gates.py`
- `tests/unit/marketplace/test_common_marketplace_engine_rb1_gates.py`
- `tests/unit/capability_catalog/test_architecture_gates.py`
- `tests/unit/marketplace/test_me_rb3_domain_vertical_alignment.py`
- `tests/unit/marketplace/test_me_rb2_plugin_architecture.py`
- `tests/unit/marketplace/test_me7_publisher_version_provenance.py`
- `tests/unit/marketplace/test_me8_commercial_metering_boundary.py`
- `tests/unit/marketplace/test_me9_multi_tenant_private_marketplace.py`
- `tests/unit/contracts/capability_metering/test_capability_metering_contract_import_gates.py`
- `tests/unit/architecture/test_capability_catalog_v1_program_boundaries.py`

Program packages must not import `intergrax.runtime`, applications, or Nexus; marketplace must not import Agent Distribution implementation.

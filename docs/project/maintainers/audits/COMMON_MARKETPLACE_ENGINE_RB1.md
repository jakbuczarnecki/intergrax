# ME-RB1 — Common Marketplace Engine Architecture Audit & Freeze

**Audit ID:** ME-RB1  
**Date:** 2026-09-16  
**Branch:** `development`  
**HEAD:** `35d6fb1017e54438d9b27b90b258427af2b0b933`  
**origin/development:** `35d6fb1017e54438d9b27b90b258427af2b0b933`  
**Worktree:** unrelated local modifications (nexus tests, platform_proofs, token_optimization) — not touched.

**Canonical architecture:** [`CAPABILITY_MARKETPLACE_ENGINE.md`](../../architecture/CAPABILITY_MARKETPLACE_ENGINE.md)

---

## Program verdict

```text
COMMON MARKETPLACE ENGINE READY WITH REMEDIATIONS
```

Ownership and boundaries are sufficiently frozen for one common engine; ME-RB2+ close contract/plugin gaps and ME-RB4 introduces lifecycle handoff without recomposing core packages.

---

## COMMON MARKETPLACE ENGINE GAP REGISTER

| ID | Severity | Area | Current state | Target state | Owner | Affected contract | Reason | Recommended task |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ME-RB1-001 | P1 | Docs / product | `AGENT_MARKETPLACE.md` reads as standalone ecosystem; Nexus in mental model | Agent vertical concept over common engine; EE public boundary | Docs | — | Reconciliation Phase 4 | Point to `CAPABILITY_MARKETPLACE_ENGINE.md` (partial fix in RB1) |
| ME-RB1-002 | P1 | Contracts | `CapabilityCatalogSource` Protocol in impl package | Typed port in `contracts.capability_catalog` | Catalog | `CapabilityCatalogSource` | Enterprise invariant: contracts not implementations | ME-RB2 |
| ME-RB1-003 | P1 | Marketplace SPI | Listing projection is module functions | Pluggable `MarketplaceListingProjection` with default | Marketplace | `contracts.marketplace` | TM-RB0-002 carryover | ME-RB2 |
| ME-RB1-004 | P2 | Marketplace SPI | `MarketplaceCapabilityCatalogSource` concrete only | Optional metadata backend port + default | Marketplace | `contracts.marketplace` | TM-RB0-003 | ME-RB2 |
| ME-RB1-005 | P1 | Handoff | Domain-specific acquisition/enablement only | Typed generic envelope + vertical payload (Option B) | Platform | new (ME-RB4) | Phase 8 — no pseudo dict handoff | ME-RB4 |
| ME-RB1-006 | P2 | Discovery | No recommendation strategy host | `MarketplaceRecommendationStrategy` or catalog-level analogue | Catalog/Marketplace | TBD | Phase 9 pluginability | ME-5 |
| ME-RB1-007 | P2 | API surface | Library methods only for human browse | Stable human-facing read API contracts (optional) | Marketplace | TBD | Phase 12 | ME-RB2 or product API tranche |
| ME-RB1-008 | P2 | Machine consumer | `WorkStageCapabilityNeed` AW-scoped | Marketplace-neutral `CapabilityNeed` for workers/orgs | Catalog/Marketplace | TBD | Phase 13 — no AW import in core | ME-RB12 |
| ME-RB1-009 | P3 | Integration naming | `IntegrationMarketplaceCatalog` separate from capability engine | Document as bounded context A; optional future bridge design | Integrations | — | Phase integration audit | Design proposal only |
| ME-RB1-010 | P2 | Search | Text filter inline in `MarketplaceCatalogService` | Document as product filter or extract search strategy port | Marketplace | — | Phase 11 boundary | ME-5 or doc-only accept |
| ME-RB1-011 | P3 | Historical audits | V1 final audit Nexus rows without EE context | Addendum pointers | Docs | — | TM-RB0-005 | Docs maintenance |

**P0 architecture blockers in audited trees:** none (Nexus/runtime/agent_distribution imports absent from marketplace core; catalog core isolation gates green).

---

## Duplication audit (summary)

| Pair | Verdict |
| ---- | ------- |
| Agent Distribution catalog vs Capability Catalog agent adapter | **JUSTIFIED DOMAIN SPECIALIZATION** — AD owns lifecycle; adapter projects read model |
| Capability Catalog vs Marketplace | **JUSTIFIED LAYERING** — discovery vs product metadata join |
| Tool bundle catalog vs tool adapter | **JUSTIFIED DOMAIN SPECIALIZATION** |
| Skill registry catalog vs skill adapter | **JUSTIFIED DOMAIN SPECIALIZATION** |
| IntegrationMarketplaceCatalog vs Capability Marketplace | **LEGACY OVERLAP (naming only)** — separate bounded context |
| AC-4 dynamic acquisition vs Marketplace | **JUSTIFIED SEPARATION** — acquisition is Agent Distribution, not marketplace engine |

---

## Recommended next task

**ME-RB2 — Contract & Plugin Architecture** — close ME-RB1-002/003/004 without changing discovery/governance semantics.

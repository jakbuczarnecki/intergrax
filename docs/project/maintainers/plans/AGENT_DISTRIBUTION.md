# Agent Distribution and Management — Plan

**Status:** Active (architecture frozen — AGENT-PLATFORM-2)  
**Architecture (1:1):** [`architecture/AGENT_DISTRIBUTION.md`](../../architecture/AGENT_DISTRIBUTION.md)  
**ADR:** [`adr/entries/2026-08-12/ADR-AGENT-004.md`](../../technical/adr/entries/2026-08-12/ADR-AGENT-004.md)  
**Evidence:** [`audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md`](../audit/AGENT_PLATFORM_COMPOSITION_AND_DISTRIBUTION_GAP_AUDIT.md)  
**Last updated:** 2026-08-12

---

## Goal

Implement the Tier-0 Agent Distribution domain so operators can discover, install, bind, configure, enable, upgrade, rollback, and uninstall agents **without** hot-loading Python, **without** LKW-local stores, and **without** breaking deterministic runtime graphs or Nexus capability routing.

## Architecture delivery (AGENT-PLATFORM-2) — Done

| Item | Status |
|------|--------|
| Canonical `AGENT_DISTRIBUTION.md` | done |
| Deterministic `MaterializedRuntimeLock` model | done (architecture) |
| `RuntimeRevision` + activation semantics | done (architecture) |
| Effective roster merge specification | done (architecture) |
| Cross-link from agent execution hub | done |
| Plan pair (this file) | done |

## Implementation waves (AP-3+)

| Wave | Deliverable | Depends on |
|------|-------------|------------|
| AP-3 | Tier-0 contracts: identity, catalog IF, installation/binding records | AP-2 arch |
| AP-4 | Store interfaces + transactional domain services | AP-3 |
| AP-5 | `AgentPackageTrust` coordinator | AP-3, plugin evidence patterns |
| AP-6 | Effective roster merge + `CandidateDependencySpecification` builder | AP-4 |
| AP-7 | `MaterializedRuntimeLock` producer + graph simulation gates | AP-6, runtime graph util |
| AP-8 | Materialization adapters (OCI, venv bundle) | AP-7 |
| AP-9 | `RuntimeRevision` activation + rollback orchestration | AP-8 |
| AP-10 | `build_application_registry` extension + snapshot fields | AP-9 |
| AP-11 | Generic Tier-3 harness admin API routes | AP-4..AP-9 |
| AP-12 | LKW consumer proof wiring | AP-11 |

## Non-goals (program)

- Marketplace billing, reviews, publisher portal
- Second Nexus or registry
- Runtime hot-load
- LKW-specific installer or persistence
- Mandating Docker for all topologies

## Verification intent (future)

- Digest-pinned install → lock → graph → activate → registry → routable capability (LKW proof)
- Rollback restores prior `RuntimeRevision` and lock digest
- Catalog outage does not affect active revision reproducibility
- Concurrent install/upgrade serialization on installation slot

## AGENT-PLATFORM-3 gate

**Done** (2026-08-12) — Tier-0 contracts and store ports landed under `intergrax/agent_distribution/`.

| Deliverable | Status |
|-------------|--------|
| `AgentPackageIdentity` / catalog contracts | done |
| `CatalogSourceProvider` port | done |
| Trust/provenance evidence surface | done |
| Installation / binding contracts | done |
| Effective roster projection models | done |
| Dependency + `MaterializedRuntimeLock` contracts | done |
| `RuntimeRevision` + materialization I/O contracts | done |
| Store ports (installation, binding, lock, revision, artifact metadata) | done |
| Focused unit tests + tier-boundary check | done |

**Next:** AP-4 may begin (store interfaces + transactional domain services).

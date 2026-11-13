# Scenario Factory — architecture (multi-layer feature)

**Status:** **Proposed** — documentation only; no product code. Implementation (SFX-P5+) is gated by `ARCH-FREEZE` = CLOSED and `GATE-SFX-ENTRY`.  
**Pair:** [plan/SCENARIO_FACTORY.md](../plan/SCENARIO_FACTORY.md)  
**Stage:** `SCENARIO-FACTORY-X` (SFX) — Enterprise Scenario Factory · roadmap `PLANNED / MANDATORY / POST-ARCH-FREEZE PRIORITY` ([roadmap](../../maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3, §3.0.7b)  
**SSOT:** [`SFX-MASTER-PLAN.md`](../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) (R1, operator-approved 2026-11-12) — this page is a summary; the SSOT wins on any difference.  
**Dossier:** [`SFX_INDEX.md`](../../maintainers/qualification/scenario_factory_x/SFX_INDEX.md)

## Purpose

Turn a business problem into a discoverable, configurable, executable, testable and demonstrable scenario, built **only** from existing platform mechanisms: no second engine, governance authority, ToolRuntime, registry or audit mechanism. Scenarios produced by SFX stay proposal-only or sandbox (LAB). SFX is product tooling; it consumes SEF-X (engineering engine) and READINESS-X (Delivery Kit) and replaces neither SCENARIO-GATE nor PROD-Q (OD-SFX-01).

## Layering — Option B

| Layer | Location | Role |
|---|---|---|
| Tier-2 | `agents/scenario_factory` | Single semantic implementation: service layer, contracts + JSON Schemas + canonicalizer, discovery/spec agents (STAGING), resolver, blueprint/profile composer, integration setup, `EngineeringHandoffPort`, scenario qualification and security gates, reporter/ROI, catalog sidecar, persistence adapters onto existing owners |
| Tier-3 | `applications/scenario_studio_application` | Doors only: Studio API v1 (FastAPI on `intergrax/fastapi_core`), `sfx` CLI, MCP surface with an explicit allowlist, and the React + TypeScript + Vite SPA under `web/` (OD-SFX-02) |
| Dev tooling | `agent-env/` (rule, `sfx-*` skills, persona briefs) + generated `.claude/` / `.cursor/` adapters | IDE front door: offline CLI, optional MCP live mode (OD-SFX-03); v0 is proposal-only (SFX-P4-A, OD-SFX-16) |

Rejected options: A (new core domain in `intergrax/`) and C (external product). D (IDE-only) is a subset of B. Boundaries stay `intergrax/ ↛ agents/ ↛ applications/`.

## Component view

```mermaid
flowchart TB
  subgraph FD[Front doors — adapters only]
    ST[Studio SPA React+TS]
    IDE[Claude Code / Cursor skills+agents]
  end
  subgraph T3[Tier-3 applications/scenario_studio_application]
    API[Studio API v1]
    CLI[sfx CLI]
    MCP[MCP surface explicit allowlist]
  end
  subgraph T2[Tier-2 agents/scenario_factory]
    SVC[Service layer = single semantic implementation]
    CON[Contracts + JSON Schemas + canonicalizer]
    AG[Discovery / Spec agents — STAGING, NexusLoop]
    RES[Resolver] --- BP[Blueprint/Profile composer] --- INT[Integration setup]
    HO[EngineeringHandoffPort<br/>DETERMINISTIC_SCAFFOLD · AGENTIC_ENGINEERING]
    QUAL[Scenario qualification + security gates]
    REP[Reporter / ROI] --- CATS[Catalog sidecar]
    PERS[Persistence adapters → existing owners]
  end
  subgraph P[Frozen platform — reuse only]
    EXE[execute_scenario_task → HostTaskExecutionPort → NexusLoop → PolicyEngine → ToolRuntime]
    CW[Collaborative Work: WorkItem · WorkArtifact(Version) · DecisionBinding · ContentStore]
    SES[SessionManager / SessionStorage]
    LR[Long-running checkpoints / SchedulerLedger / DLQ]
    EV[Evidence plane · usage · observability]
    CATP[FederatedCapabilityCatalog + UCA]
    SCAF[intergrax.scaffold + init_scenario_implementation]
  end
  SEF[SEF-X EEA — after GATE-SEF-CONSUME]
  ADM[Scenario programme admission owner (OD-SFX-13) + PROD-Q]
  ST-->API; IDE-->CLI; IDE-.->MCP
  API & CLI & MCP --> SVC
  SVC --> CON & AG & RES & BP & INT & HO & QUAL & REP & CATS
  PERS --> CW & SES & LR
  AG --> EXE
  RES --> CATP
  HO --> SCAF
  HO -.-> SEF
  REP --> EV
  QUAL -->|admission dossier only| ADM
```

## Platform mechanisms — reuse only

SFX owns product composition only. Every semantic authority stays with its existing owner (full list SFX-CAP-01..37 in SSOT §5):

- **Execution:** `execute_scenario_task` → `HostTaskExecutionPort` → NexusLoop → PolicyEngine → ToolRuntime (LAB / PRODUCTION_ATTACHED profiles).
- **Durable state:** Collaborative Work (`WorkItem`, `WorkArtifact` + `WorkArtifactVersion`, `ArtifactContentStore`, `CollaborativeDecisionBindingRepository`), `SessionManager` / `SessionStorage`, long-running checkpoints / `SchedulerLedger` / DLQ, evidence plane (`EvidencePersistencePort`). No new store, no new state engine (SSOT §7.5).
- **Resolution and integrations:** `FederatedCapabilityCatalog` + UCA, `AgentRegistry`, integration registry and profiles, secrets store / credential broker (references only).
- **Build:** `intergrax.scaffold` and `scripts/proof/init_scenario_implementation.py`; conformance via `scenario_architecture_conformance`.
- **Hosting and IDE:** `fastapi_core` `create_app`, MCP composition (`compose_application_mcp_server`), `agent-env` + `sync_agent_env.py`.
- **Not changed:** `ApplicationManifest`, `ApplicationPackage`, `ApplicationGraphSpec`. The catalog is a Tier-2 **sidecar** (`sfx.scenario_catalog_entry.v1`) persisted as a `WorkArtifactVersion`, modelled on `AgentCatalogEntry`: metadata only, not a registry (SSOT §7.6).

## Engineering modes

One `EngineeringHandoffPort`, two modes (SSOT §3):

- `DETERMINISTIC_SCAFFOLD` — template fill from an approved blueprint with the existing scaffold tools; no LLM; same blueprint digest gives the same tree digest.
- `AGENTIC_ENGINEERING` — business logic, tests and evidence by the SEF-X EEA; available only after `GATE-SEF-CONSUME`; a human submits every intent (HO-2); merge stays operator-only (HO-6). SFX closure requires this mode verified end to end on Accepted and certified SEF-X mechanisms.

## Parity contract (Studio ↔ CLI ↔ MCP)

| ID | Rule |
|---|---|
| PC-1 | One schema set generated from the Tier-2 pydantic contracts; Studio client and MCP tool schemas come from the same source |
| PC-2 | All doors call the same Tier-2 service functions; identical invalid input gives an identical `sfx.error` code in every door |
| PC-3 | One authorization path (harness principal → tenant reconciliation → PolicyEngine); the offline CLI is limited to L0/L1 on local files |
| PC-4 | Identical artifact identity, lifecycle transitions and side effects for the same operation |
| PC-5 | Canonical form = RFC 8785 JSON Canonicalization; `canonical_digest = sha256(canonical form)`; LLM stages compared by replay |
| PC-6 | API v1 = MCP tool set v1 = CLI v1, 1:1 operation map, one version constant |

## Authority levels

L0 read-only · L1 proposal-only · L2 sandbox-executable (LAB, fixture adapters, budgets, egress deny) · L3 production. **SFX never grants L3** and never holds L3 credentials. Generated code must not mutate environment profile, policy mode, governance or certification records, `AgentContract.lifecycle_state` = PRODUCTION, exposure policies or registries (SSOT §13).

## Certification and admission separation

- The SFX lifecycle ends at `SANDBOX_CERTIFIED` → `ADMISSION_DOSSIER_READY` (SSOT §7.9). Invariant SFX-INV-01: generated code is never a certified scenario.
- **Product certification** (SFX stage, IIAC, independent session) certifies the factory only, never a generated scenario.
- **Scenario admission happens outside SFX** and requires SCENARIO-GATE CLOSED, an admission decision by the owner assigned in OD-SFX-13, PROD-Q live qualification per integration, and a STRICT+PRODUCT host with revision-bound authority. SFX exports an admission dossier as input; it creates no admission authority (SSOT §14).

## ADRs (planned)

All **Proposed**, to be published in [`T-IGX-037-C`](../../../../governance/tickets/T-IGX-037-C.md) (SFX-P2-01). Acceptance is OD-SFX-06 and a `GATE-SFX-ENTRY` condition.

| ADR | Topic | Status |
|---|---|---|
| ADR-SFX-001 | Placement and ownership (Option B); ECAP §16 applicability (C-ECAP-01) | Planned — Proposed |
| ADR-SFX-002 | Contracts, canonicalization, workspace location, persistence mapping | Planned — Proposed |
| ADR-SFX-003 | Studio stack, auth, hosting | Planned — Proposed |
| ADR-SFX-004 | IDE toolkit + CLI + MCP parity | Planned — Proposed |
| ADR-SFX-005 | Catalog sidecar | Planned — Proposed |
| ADR-SFX-006 | Authority levels, engineering modes, admission separation | Planned — Proposed |

## Related domain architecture

Coordinates, does not override: [`COLLABORATIVE_WORK.md`](../../architecture/COLLABORATIVE_WORK.md), [`GOVERNED_EXECUTION.md`](../../architecture/GOVERNED_EXECUTION.md), [`CAPABILITY_CATALOG_AND_DISCOVERY.md`](../../architecture/CAPABILITY_CATALOG_AND_DISCOVERY.md), [`APPLICATION_HOSTING.md`](../../architecture/APPLICATION_HOSTING.md), [`AGENT_DISTRIBUTION.md`](../../architecture/AGENT_DISTRIBUTION.md), [`INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md). Architecture principles: [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md).

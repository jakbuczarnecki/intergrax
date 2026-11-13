# SCENARIO-FACTORY-X — campaign index (SSOT hub)

**Stage ID:** `SCENARIO-FACTORY-X` (SFX) — Enterprise Scenario Factory  
**Roadmap status:** `PLANNED / MANDATORY / POST-ARCH-FREEZE PRIORITY`  
**Master plan:** [`SFX-MASTER-PLAN.md`](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) — Revision R1, operator-approved 2026-11-12; approval scope W0–W2  
**Base pin:** `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` (`origin/development`)  
**Sequencing:** OD-SFX-15 — pre-freeze SFX-P0..P4 open; **STOP before SFX-P5**; target `PRE_IMPLEMENTATION_READY`  
**Tickets:** [`T-IGX-037-A`](../../../../../governance/tickets/T-IGX-037-A.md) · [`T-IGX-037-B`](../../../../../governance/tickets/T-IGX-037-B.md) · [`T-IGX-037-C`](../../../../../governance/tickets/T-IGX-037-C.md) · [`T-IGX-037-D`](../../../../../governance/tickets/T-IGX-037-D.md) · [`T-IGX-037-E`](../../../../../governance/tickets/T-IGX-037-E.md)

## Authority map (one SSOT per concern)

| Concern | SSOT artifact |
| --- | --- |
| Programme sequence and stage status | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3, §3.0.7b, §5 |
| Campaign scope, architecture, gates (binding) | [`SFX-MASTER-PLAN.md`](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) |
| Campaign WBS and units (with tickets) | [`SFX_WBS.md`](./SFX_WBS.md) |
| Gaps SFX-GAP-* and conflicts register | [`SFX_GAP_REGISTER.md`](./SFX_GAP_REGISTER.md) |
| Requirements REQ-SFX-001..054 | [`SFX_REQUIREMENTS_TRACEABILITY.json`](./SFX_REQUIREMENTS_TRACEABILITY.json) |
| SHA ladder | [`SFX_SHA_LADDER.md`](./SFX_SHA_LADDER.md) |
| Operator decisions | [`operator_decisions/`](./operator_decisions/) |
| Capability pair (hub pointers) | [`architecture/SCENARIO_FACTORY.md`](../../../capabilities/architecture/SCENARIO_FACTORY.md) · [`plan/SCENARIO_FACTORY.md`](../../../capabilities/plan/SCENARIO_FACTORY.md) |
| Research, inventory, design, UX, IDE walkthrough (W1–W2) | `docs/project/maintainers/qualification/scenario_factory_x/` (_created by T-IGX-037-B..E_) |
| ADR-SFX-001..006 | `docs/project/technical/adr/entries/<date>/ADR-SFX-00N.md` (_T-IGX-037-C; Proposed_) |
| SEF-X dependency (GATE-SEF-CONSUME) | [`ASF-SEF-MASTER-PLAN.md`](../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md) |

## Operator decision records

Records are written under [`operator_decisions/`](./operator_decisions/). The implementer does not sign; each record holds its own signature status (`PENDING SIGNATURE` until signed by the operator or a commissioned delegate).

| Record | Subject |
| --- | --- |
| [`OD-SFX-01.md`](./operator_decisions/OD-SFX-01.md) | Placement and stage registration (first product priority after ARCH-FREEZE; SFX consumes SEF-X and READINESS-X) |
| [`OD-SFX-02.md`](./operator_decisions/OD-SFX-02.md) | Studio stack: React + TypeScript + Vite SPA in the Tier-3 host, FastAPI backend on `fastapi_core` |
| [`OD-SFX-03.md`](./operator_decisions/OD-SFX-03.md) | IDE model: one factory core; CLI offline + agent-env skills/rules/agents; optional MCP live mode |
| [`OD-SFX-04.md`](./operator_decisions/OD-SFX-04.md) | Pre-freeze scope: docs, roadmap, research, design, IDE toolkit v0 in agent-env only |
| [`OD-SFX-15.md`](./operator_decisions/OD-SFX-15.md) | Pre-freeze sequencing: SFX-P0..P4 now; STOP before SFX-P5; target `PRE_IMPLEMENTATION_READY` |
| [`OD-SFX-16.md`](./operator_decisions/OD-SFX-16.md) | C-OD-01 resolution (adapters ship with agent-env); delegated ticket approvals; sub-session independence |

Scheduled decisions OD-SFX-05..14 have no record yet (master plan §16).

**Delegation records:**

- [`DELEGATION-SFX-PRE-FREEZE.md`](./operator_decisions/DELEGATION-SFX-PRE-FREEZE.md): a dedicated delegate session may write the `Approved:` lines of T-IGX-037-A..E only. It signed them in commit `250d3aeeb`. This record does not delegate:
  - OD signatures;
  - ADR acceptance;
  - IIAC activation;
  - FRZ promotion;
  - merges;
  - GATE-SFX-ENTRY;
  - any SFX-P5+ ticket.
- [`DELEGATION-SFX-OD-SIGNING.md`](./operator_decisions/DELEGATION-SFX-OD-SIGNING.md): a separate dedicated signing session may sign OD-SFX-01..04, 15 and 16 only. It signed them in commit `4758c4051`.

## Phase status

| Phase | Wave | Scope | Ticket | Status |
| --- | --- | --- | --- | --- |
| P0 | W0 | Roadmap registration A1–A10, SSOT plan, capability pair, dossier skeleton, OD records | T-IGX-037-A | **In progress** (`plan_approved`) |
| P1 | W1 | Competitive evidence matrix, IDE/MCP facts, capability inventory re-pin | T-IGX-037-B | Planned (`plan_approved`) |
| P2 | W1 | ADR-SFX-001..006 (Proposed), contract schema drafts, canonicalization, metrics, gap analysis | T-IGX-037-C | Planned (`plan_approved`) |
| P3 | W2 | UX design, prototype, usability protocol + round 0, GEN set seal | T-IGX-037-D | Planned (`plan_approved`) |
| P4 | W2 / W3 | P4-A01/A02 proposal-only IDE toolkit v0 (W2); P4-B01/B02 generator + role registry (W3) | T-IGX-037-E (A01/A02); TBD (post-freeze) for B01/B02 | A: Planned (`plan_approved`); B: **Blocked** (GATE-SFX-ENTRY) |
| GATE-SFX-ENTRY | — | ARCH-FREEZE CLOSED; ADR-SFX-001..006 Accepted; OD-SFX-05/09/14 signed; inventory re-pinned; GEN set sealed | — | **Not met** |
| P5 | W3 | Tier-2 package, contracts, canonicalizer, persistence adapters, Tier-3 host, service + CLI v0 | TBD (post-freeze) | **Blocked** (ARCH-FREEZE, GATE-SFX-ENTRY) |
| P6 | W4 | Discovery agent, spec author + validator, session resume | TBD (post-freeze) | **Blocked** |
| P7 | W4 | Resolver, blueprint/profile composer, integration setup, semantic diff | TBD (post-freeze) | **Blocked** |
| P8 | W4 | Engineering handoff: scaffold adapter, job lifecycle, agentic adapter | TBD (post-freeze) | **Blocked** (P8-03 also GATE-SEF-CONSUME) |
| P9 | W5 | Test planner, self-escalation, supply-chain, isolation and adversarial gates | TBD (post-freeze) | **Blocked** |
| P10 | W5 | Studio API, MCP surface, parity suite, web, SPA hosting, IDE toolkit v1 | TBD (post-freeze) | **Blocked** (P10-04 also OD-SFX-05) |
| P11 | W6 | Run reporter, ROI model, catalog sidecar, admission dossier export | TBD (post-freeze) | **Blocked** |
| P12 | W7 | REF-1..3 both modes, GEN set reveal, usability rounds 1–3, performance | TBD (post-freeze) | **Blocked** (also GATE-SEF-CONSUME) |
| P13 | W8 | Evidence + SHA ladder, ENT-AUDIT CHANGED-SINCE, IIAC product certification, operator closure | TBD (post-freeze) | **Blocked** (GATE-SFX-FULL) |

## Terminal state (target — not claimed)

Pre-freeze: `ROADMAP REGISTERED → RESEARCH CITED → ADRs PROPOSED → UX + GEN SEAL → IDE TOOLKIT v0 → PRE_IMPLEMENTATION_READY`.

Stage: `GATE-SFX-ENTRY → W3..W7 (REBIND per wave) → GATE-SFX-FULL → INDEPENDENT AUDIT → IIAC PRODUCT CERTIFICATION → SCENARIO-FACTORY-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`.

SFX closure certifies the factory only. It admits no generated scenario to production; admission happens outside SFX (OD-SFX-13 owner, SCENARIO-GATE, PROD-Q).

# Scenario Factory — delivery plan (multi-layer feature)

**Architecture pair:** [architecture/SCENARIO_FACTORY.md](../architecture/SCENARIO_FACTORY.md)  
**SSOT:** [`SFX-MASTER-PLAN.md`](../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) (R1, operator-approved 2026-11-12; approval scope W0–W2)  
**Campaign dossier (authoritative WBS, gaps, SHA ladder, traceability):** [`SFX_INDEX.md`](../../maintainers/qualification/scenario_factory_x/SFX_INDEX.md)  
**Satellites (on demand):** `satellites/scenario_factory/` — research, inventory, design drafts, UX, IDE walkthrough (created by T-IGX-037-B..E)

Do not duplicate the WBS here — edit the dossier files only.

## Current state

- **Stage:** `SCENARIO-FACTORY-X` — `PLANNED / MANDATORY / POST-ARCH-FREEZE PRIORITY`; production delta = 0.
- **Pre-freeze campaign SFX-P0..P4 open** (OD-SFX-15), running in parallel with MCP-SERVER-X, PDC-X and ARCH-FREEZE. Target: `PRE_IMPLEMENTATION_READY`.
- **STOP before SFX-P5.** W3+ needs `ARCH-FREEZE` = CLOSED, `GATE-SFX-ENTRY` and per-wave ticket approvals.
- Tickets T-IGX-037-A..E are `plan_approved` (delegate, [`DELEGATION-SFX-PRE-FREEZE`](../../maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md)).
- ADR-SFX-001..006: not yet published (planned as Proposed in T-IGX-037-C).

## Waves

| Wave | Phases / units | Scope | Gate to start | Ticket | Status |
|---|---|---|---|---|---|
| W0 | SFX-P0-01..03 | Roadmap registration A1–A10, SSOT plan, capability pair, dossier skeleton, OD records | — | [T-IGX-037-A](../../../../governance/tickets/T-IGX-037-A.md) | In progress |
| W1 | SFX-P1-01..03 | Competitive evidence matrix, IDE/MCP facts, capability inventory re-pin | W0 | [T-IGX-037-B](../../../../governance/tickets/T-IGX-037-B.md) | Planned (plan_approved) |
| W1 | SFX-P2-01..02 | ADR-SFX-001..006 Proposed, contract schema drafts, canonicalization and metrics specs, gap analysis | P1 | [T-IGX-037-C](../../../../governance/tickets/T-IGX-037-C.md) | Planned (plan_approved) |
| W2 | SFX-P3-01..02 | UX design, prototype Artifact, usability protocol + round 0, GEN set seal | P2-01 / P2-02 | [T-IGX-037-D](../../../../governance/tickets/T-IGX-037-D.md) | Planned (plan_approved) |
| W2 | SFX-P4-A01..A02 | Proposal-only IDE toolkit v0 in `agent-env/` + regenerated adapters | P2-02 | [T-IGX-037-E](../../../../governance/tickets/T-IGX-037-E.md) | Planned (plan_approved) |
| W3 | SFX-P4-B01..B02, SFX-P5-01..07 | Agent generation in `sync_agent_env.py`, role_registry entries; Tier-2 package, contracts, canonicalizer, persistence adapters, Tier-3 host, service + CLI v0 | GATE-SFX-ENTRY | TBD (post-freeze) | Blocked |
| W4 | SFX-P6-01..03, P7-01..04, P8-01..03 | Discovery, specification, session resume; resolver, blueprint/profile, integration setup, diff; engineering handoff (scaffold, jobs, agentic) | W3 (P8-03 also GATE-SEF-CONSUME) | TBD (post-freeze) | Blocked |
| W5 | SFX-P9-01..05, P10-01..08 | Test planner, self-escalation / supply-chain / isolation / adversarial gates; Studio API, MCP, parity suite, web, SPA hosting, IDE toolkit v1 | W4 (P10-04 also OD-SFX-05) | TBD (post-freeze) | Blocked |
| W6 | SFX-P11-01..04 | Run reporter, ROI model, catalog sidecar, admission dossier export | W5 | TBD (post-freeze) | Blocked |
| W7 | SFX-P12-01..06 | REF-1..3 in both modes, GEN set reveal and run, usability rounds 1–3, performance | W6 + GATE-SEF-CONSUME | TBD (post-freeze) | Blocked |
| W8 | SFX-P13-01..04 | Evidence + SHA ladder, ENT-AUDIT CHANGED-SINCE, IIAC product certification, operator closure OD-SFX-11 | W7 + GATE-SFX-FULL | TBD (post-freeze) | Blocked |

Every post-freeze wave ends with **REBIND-Wn** (ECAP A1–A8 delta audit, D1–D6 classification, IIAC ACT-N11 audited re-bind, independent reviewer session).

## Gates

| Gate | Opens | Conditions (SSOT) |
|---|---|---|
| **GATE-SFX-ENTRY** | W3+ | ARCH-FREEZE CLOSED; ADR-SFX-001..006 Accepted (OD-SFX-06); OD-SFX-05/09/14 signed; inventory re-pinned at `FREEZE_CERTIFICATION_SHA`; GEN set sealed (§10) |
| **GATE-SEF-CONSUME** | SFX-P8-03; agentic runs in W7 | ADR-SEF-001 Accepted; OD-SEF-04 resolved; CON-N-01..05 and CON-E-01/02/04/05 implemented; SEF GATE-SEF-FULL passed; SEF scoped B6 (T-P12-003) on consumed mechanisms; OD-SFX-10 signed; SFX registered as SEF intake consumer (§3.3) |
| **GATE-SFX-FULL** | SFX closure (W8) | REF-1..3 and the GEN set verified end to end in `AGENTIC_ENGINEERING` mode on Accepted and certified SEF-X mechanisms; HO-1..7 green; network restriction on agentic engineering (C-SEF-02) (§3.1, §11) |

Critical path: ARCH-FREEZE → GATE-SFX-ENTRY → W3 → W4 → W5 → W6 → max(W6, GATE-SEF-CONSUME) → W7 → W8. SEF-X and SCENARIO-READINESS-X do not depend on SFX; SFX is not a predecessor of SCENARIO-GATE.

## Ticket map

| Ticket | Units | Worktree / branch |
|---|---|---|
| [T-IGX-037-A](../../../../governance/tickets/T-IGX-037-A.md) | SFX-P0-01..03 | `sfx-p0` / `sfx-p0-registration` |
| [T-IGX-037-B](../../../../governance/tickets/T-IGX-037-B.md) | SFX-P1-01..03 | `sfx-w1-research` (stacked on `sfx-p0-registration`) |
| [T-IGX-037-C](../../../../governance/tickets/T-IGX-037-C.md) | SFX-P2-01..02 | `sfx-w1-arch` (stacked) |
| [T-IGX-037-D](../../../../governance/tickets/T-IGX-037-D.md) | SFX-P3-01..02 | `sfx-w2-ux` (stacked) |
| [T-IGX-037-E](../../../../governance/tickets/T-IGX-037-E.md) | SFX-P4-A01..A02 | `sfx-w2-ide` (stacked) |
| TBD (post-freeze) | SFX-P4-B01..SFX-P13-04 | per wave, after GATE-SFX-ENTRY |

## Domain plan dependencies (rows added when implementation begins)

SFX adds no rows to domain plans before W3. Post-freeze waves are expected to cite or add rows in: [`COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) (artifact, decision-binding and work-item reuse; CW requalification), [`APPLICATION_HOSTING.md`](../../maintainers/plans/APPLICATION_HOSTING.md) (Tier-3 Studio host, SPA mount), [`CAPABILITY_CATALOG_AND_DISCOVERY.md`](../../maintainers/plans/CAPABILITY_CATALOG_AND_DISCOVERY.md) (resolver), [`AGENT_DISTRIBUTION.md`](../../maintainers/plans/AGENT_DISTRIBUTION.md) (catalog sidecar pattern), [`INTEGRATIONS.md`](../../maintainers/plans/INTEGRATIONS.md) (integration setup), and the SEF-X track ([`ASF-SEF-MASTER-PLAN.md`](../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md)) for GATE-SEF-CONSUME.

## Open decisions

OD-SFX-05..14 are scheduled decisions (SSOT §16); OD-SFX-13 gates scenario admission only, not SFX closure. Conflicts C-SEF-01..04, C-RDY-01, C-AF-01, C-IIAC-01 and C-ECAP-01 are tracked in [`SFX_GAP_REGISTER.md`](../../maintainers/qualification/scenario_factory_x/SFX_GAP_REGISTER.md); C-OD-01 is resolved by OD-SFX-16.

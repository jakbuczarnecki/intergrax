# Autonomous Organization Scenario Portfolio

**Document type:** Maintainer-level frozen scenario portfolio record  
**Portfolio:** Autonomous Organization Scenario Portfolio **v1**  
**Selection status:** **10 / 10 FROZEN — CLOSED FOR V1 SELECTION**  
**Initiative SSOT:** [`AUTONOMOUS_ORGANIZATION_INITIATIVE.md`](AUTONOMOUS_ORGANIZATION_INITIATIVE.md)

---

## Authority and boundaries

This file is the **Canonical SSOT** for **selection-frozen Autonomous Organization Scenario Portfolio v1**: membership, IDs **VO-S01–VO-S10**, stable slugs, and frozen problem identity.

| Role | Document |
|------|----------|
| **Frozen VO problem portfolio (this file)** | Selection-frozen organizational E2E problems for design, VO overlay, and proof work |
| **Enterprise Frozen-30** | [`ENTERPRISE_E2E_SCENARIO_CATALOG.md`](ENTERPRISE_E2E_SCENARIO_CATALOG.md) — separate substrate portfolio |
| **Current per-package lifecycle** | `platform_proofs/scenarios/<slug>/SCENARIO_SPEC.md` YAML frontmatter |
| **Implementation priority** | [`AUTONOMOUS_ORGANIZATION_ROADMAP.md`](AUTONOMOUS_ORGANIZATION_ROADMAP.md) — **not** lifecycle |

**This portfolio is not authority for:** current scenario-package lifecycle, implementation status, executable status, verified status, or proof acceptance.

Adding, removing, or replacing a scenario after v1 freeze requires explicit **portfolio governance** — not casual table edits.

---

## What FROZEN means here

| FROZEN **does** mean | FROZEN **does not** mean |
|----------------------|---------------------------|
| Portfolio membership frozen | Implemented |
| ID (VO-Sxx) frozen | Executable |
| Stable slug frozen | Verified |
| Fundamental problem identity frozen | Proof accepted |
| | Production ready |
| | Architecture frozen |

**Scenario package lifecycle** remains the source of truth for implementation progress.

---

## Identity ≠ implementation priority

**VO-S01–VO-S10** are stable **identity** labels, not default build order.

Canonical **implementation priority** (program record only):

1. VO-S01 · 2. VO-S02 · 3. VO-S03 · 4. VO-S10 · 5. VO-S04 · 6. VO-S05 · 7. VO-S06 · 8. VO-S08 · 9. VO-S09 · 10. VO-S07

Recorded in [`AUTONOMOUS_ORGANIZATION_ROADMAP.md`](AUTONOMOUS_ORGANIZATION_ROADMAP.md).

---

## Frozen catalog (10)

| ID | Scenario | Stable slug |
|----|----------|-------------|
| VO-S01 | Mission-Driven Strategic Market Entry | `mission_driven_strategic_market_entry` |
| VO-S02 | Temporary Capability Venture Lifecycle | `temporary_capability_venture_lifecycle` |
| VO-S03 | Strategic Portfolio Reallocation Under Runway Pressure | `strategic_portfolio_reallocation` |
| VO-S04 | Distribution Bottleneck & Organizational Self-Evaluation | `organizational_distribution_bottleneck` |
| VO-S05 | Organizational Scaling, Complexity & Redundancy Consolidation | `organizational_complexity_consolidation` |
| VO-S06 | Live-Commitment Organizational Reorganization | `live_commitment_organizational_reorganization` |
| VO-S07 | Autonomous Opportunity Discovery | `autonomous_opportunity_discovery` |
| VO-S08 | Critical Capability Loss & Organizational Reconstitution | `critical_capability_loss_reconstitution` |
| VO-S09 | Strategic Mandate Change With Existing Commitments | `strategic_mandate_change` |
| VO-S10 | Delegated Goal Drift & Local Optimization | `delegated_goal_drift` |

---

## VO-S01 — Mission-Driven Strategic Market Entry

**Real problem:** Leadership assigns entry into a new market under partial information, regulatory variance, and finite budget — without handing engineers a finished org design.

**Fundamental purpose:** Prove the organization can **form** from mission, constraints, and resources and acquire first capabilities responsibly.

**Unique pressure:** Greenfield composition — no inherited teams; every unit must **earn** its existence from need.

**Exclusions:** Not a single-agent market research demo; not predefined “Sales / Legal / Ops” roles baked into config.

**North Star relation:** Minimal viable **self-formation** under constitution — foundation for all later reorganization scenarios.

---

## VO-S02 — Temporary Capability Venture Lifecycle

**Real problem:** A time-boxed initiative (pilot, joint venture, surge program) must spin up, run, and **sunset** without orphaning commitments or ghost resources.

**Fundamental purpose:** Prove **create → operate → retire** organizational structures tied to capability lifecycle.

**Unique pressure:** Explicit end date and teardown — tests whether the org can dismantle itself cleanly.

**Exclusions:** Not permanent headcount planning; not a static project template with fixed RACI YAML.

**North Star relation:** Organization as **state** — structures appear and disappear as capabilities demand.

---

## VO-S03 — Strategic Portfolio Reallocation Under Runway Pressure

**Real problem:** Runway or capital constraints force reprioritization across concurrent initiatives with unequal sunk cost and contractual exposure.

**Fundamental purpose:** Prove **resource economics** and portfolio-level tradeoffs under governance.

**Unique pressure:** Competing goals with irreversible partial spend — not a single-project cut.

**Exclusions:** Not generic “cost optimization” without organizational units and allocations at stake.

**North Star relation:** Connects mission-level goals to **who keeps resources** when the org must shrink or pivot.

---

## VO-S04 — Distribution Bottleneck & Organizational Self-Evaluation

**Real problem:** Product outcomes, retention, activation, and customer value look strong; engineering delivery works — yet pipeline, distribution, and customer acquisition underperform, acquisition cost may be high, and resources are visibly skewed toward engineering and product. Global business results stay weak while no one tells the system to “create a Growth Team” or “reorganize sales.” The org must infer from **business evidence** that weak outcomes may stem from **organizational structure** or **misallocation of capabilities and resources**.

**Fundamental purpose:** Prove **organizational self-evaluation** — business evidence → organizational hypothesis → structural and resource evaluation (and targeted adjustment under governance).

**Unique pressure:** Major parts of the organization can each look locally healthy while **global** business performance remains poor.

**Exclusions:** Not work-queue or inter-unit workflow optimization; not APM or generic process-bottleneck detection; not a prescribed answer such as a named Growth Team.

**North Star relation:** Closed loop from **business evidence** → **organizational hypothesis** → structural or resource change proposal under governance.

---

## VO-S05 — Organizational Scaling, Complexity & Redundancy Consolidation

**Real problem:** Success duplicates teams, handoffs, and overlapping responsibilities until coordination cost dominates.

**Fundamental purpose:** Prove the org can **reduce complexity** — merge, deduplicate, simplify — without breaking delivery.

**Unique pressure:** Deliberate **consolidation** under load, opposite of greenfield S01.

**Exclusions:** Not headcount reduction without responsibility re-mapping; not manual reorg spreadsheet import.

**North Star relation:** Self-reorganization for **efficiency**, not only crisis.

---

## VO-S06 — Live-Commitment Organizational Reorganization

**Real problem:** Mid-flight customer, regulatory, or contractual commitments must survive a major internal restructure.

**Fundamental purpose:** Prove **reorganization under active commitments** with continuity and evidence.

**Unique pressure:** Cannot pause the world — migration of responsibility while work continues.

**Exclusions:** Not greenfield; not “new team” without commitment migration semantics.

**North Star relation:** Core **commitment management + reorganization** intersection.

---

## VO-S07 — Autonomous Opportunity Discovery

**Real problem:** Strategically relevant opportunities appear outside planned backlogs — the org must notice, qualify, and optionally pursue without a human ticket per insight.

**Fundamental purpose:** Prove **open-ended discovery** bounded by constitution and resource limits.

**Unique pressure:** No predefined task list — highest cognitive / epistemic demand in v1 portfolio.

**Exclusions:** Not unbounded autonomous spending; not alert noise without organizational decision path.

**North Star relation:** Approaches North Star **mission-only input** — deferred in implementation priority until substrate scenarios qualify.

---

## VO-S08 — Critical Capability Loss & Organizational Reconstitution

**Real problem:** Loss of a key provider, skill, system, or legal permission removes a capability the org depended on.

**Fundamental purpose:** Prove **reconstitution** — substitute, rebuild, or replan — under uncertainty and evidence requirements.

**Unique pressure:** Sudden **capability hole** with ongoing obligations.

**Exclusions:** Not single-run failover only; must implicate organizational redesign of responsibilities.

**North Star relation:** Resilience of **organizational capability graph**, not single executor swap.

---

## VO-S09 — Strategic Mandate Change With Existing Commitments

**Real problem:** Legitimate human authority shifts the **strategic mandate** (e.g. from maximize expansion/growth to preserve cash, reach profitability, protect strategic customers) while the organization continues under the **same constitution** and authority boundaries. Legacy goals, initiatives, capabilities, and legal or contractual commitments remain until reconciled.

**Fundamental purpose:** Prove **mandate migration** — assess which existing goals stay valid or become invalid, which capabilities remain needed, which initiatives to end, how to change resource allocation, whether reorganization is required, and which commitments must still be fulfilled — without unconstitutional authority expansion (AO-P5).

**Unique pressure:** The mandate answers *what strategic outcome to pursue now*; the constitution answers *non-self-modifiable boundaries* — mandate changes at the top while goals, structure, and commitments **lag** the new direction.

**Exclusions:** Not a change to constitution or authority limits; not voluntary pivot only without legacy commitments; must respect AO-P5 (no self-expansion of authority).

**North Star relation:** Tests **goal system** realignment to a new **strategic mandate** with constitution and authority boundaries unchanged.

---

## VO-S10 — Delegated Goal Drift & Local Optimization

**Real problem:** Sub-units optimize local KPIs that diverge from global mission (metric gaming, silo success, misaligned incentives).

**Fundamental purpose:** Prove **alignment detection and correction** across delegated authority.

**Unique pressure:** Everything “works locally” while the org fails globally — subtle governance challenge.

**Exclusions:** Not a single bad agent run; must involve **organizational** measurement and realignment.

**North Star relation:** Bridges **economics + goal system** before meta-org scenarios (S04–S05).

---

## Outside frozen ten (referenced, not v1 portfolio members)

### VO-IQ-01 — Strategic Transformation Under Market Shock

**Role:** Later **integrated qualification scenario** — validates that capabilities exercised across VO-S01–S10 **work together** under compound shock. Not selection-frozen in the ten-scenario v1 table.

### VO-NS-01 — Self-Organizing Virtual Organization Challenge

**Role:** **North Star** integrative challenge. Input emphasizes mission, objectives, constraints, resources, constitution, and human authority boundaries. Developer must **not** predefine future teams, roles, capability graph, initiative portfolio, or organizational topology. Documented in [`AUTONOMOUS_ORGANIZATION_ROADMAP.md`](AUTONOMOUS_ORGANIZATION_ROADMAP.md).

---

## Portfolio selection philosophy (aligned with enterprise catalog)

A frozen VO scenario **must**:

1. Represent a real organizational or enterprise-scale problem.
2. Exercise **need → capability → responsibility → execution** — not a fixed org chart demo.
3. Be falsifiable — able to show platform limits honestly.
4. Prefer platform reuse; treat gaps as **stop + architecture**, not local hacks.
5. Remain a normal Scenario Proof — see Initiative SSOT scenario governance.

**INTERGRAX FIT** is resolved during scenario design and implementation preparation — not at portfolio freeze time.

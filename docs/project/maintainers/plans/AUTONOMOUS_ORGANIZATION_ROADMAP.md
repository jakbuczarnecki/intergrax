# Autonomous Organization Roadmap

**Document type:** Program outcome roadmap (no calendar dates)  
**Initiative SSOT:** [`AUTONOMOUS_ORGANIZATION_INITIATIVE.md`](AUTONOMOUS_ORGANIZATION_INITIATIVE.md)  
**Frozen scenarios:** [`AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md`](AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md)

Roadmap entries are **outcome gates**, not release promises. Progress is measured by **qualified Scenario Proofs** and resolved **platform gaps**, not by documentation alone.

---

## Current program position (VO-DOC-0)

| Item | State |
|------|--------|
| Initiative foundation docs | Established |
| VO portfolio v1 | 10 / 10 selection-frozen |
| VO scenario packages | **Not required** by portfolio freeze |
| Organizational runtime | **Not implemented** |
| VO-S01 implementation | **Not started** (await VO-GATE-0 / operator go) |

---

## Canonical implementation priority

Identity order **VO-S01–VO-S10** ≠ build order. Implementation priority:

| Priority | ID |
|----------|-----|
| 1 | VO-S01 |
| 2 | VO-S02 |
| 3 | VO-S03 |
| 4 | VO-S10 |
| 5 | VO-S04 |
| 6 | VO-S05 |
| 7 | VO-S06 |
| 8 | VO-S08 |
| 9 | VO-S09 |
| 10 | VO-S07 |

This table is the **only** canonical implementation priority record for the program. Per-package lifecycle remains in `SCENARIO_SPEC.md`.

---

## Epoch I — Formation

**Scenarios:** VO-S01, VO-S02

**Outcome gate:** The organization can **arise from mission** (plus constraints and resources) and can **create and retire** capabilities / structures without a predefined org chart.

**Capability emphasis:** Mission → goals; capability gap discovery; acquisition/composition; venture lifecycle teardown (see Capability Model clusters 1, 4, 5, 6, 10, 11).

---

## Epoch II — Economics & Alignment

**Scenarios:** VO-S03, VO-S10

**Outcome gate:** The organization can manage **scarce resources** across a portfolio and detect **delegated goal drift** so local optimization still serves global mission.

**Capability emphasis:** Resource economics, alignment, epistemic world model updates (clusters 7, 14, 9, 2).

---

## Epoch III — Meta-Organization

**Scenarios:** VO-S04, VO-S05

**Outcome gate:** The organization can **evaluate itself** (bottlenecks, fitness) and **reduce complexity** through consolidation without losing governed delivery.

**Capability emphasis:** Self-evaluation, self-reorganization for efficiency (clusters 9, 10, 6).

---

## Epoch IV — Transformation & Resilience

**Scenarios:** VO-S06, VO-S08, VO-S09

**Outcome gate:** The organization can **restructure under live commitments**, **reconstitute** after critical capability loss, and **realign** when strategic mandate changes while honoring constitution and existing obligations.

**Capability emphasis:** Commitment migration, identity continuity, mandate realignment (clusters 11, 12, 13, 5, 10).

---

## Epoch V — Open-Ended Cognition

**Scenarios:** VO-S07

**Outcome gate:** The organization can discover **material opportunities or problems** without a human-supplied task for each insight — within constitution and resource bounds.

**Capability emphasis:** Discovery, experimentation, mission-linked goal formation (clusters 3, 4, 1).

**Note:** Last in implementation priority — depends on qualified substrate from earlier epochs.

---

## Qualification

**Scenario:** VO-IQ-01 — *Strategic Transformation Under Market Shock*

**Outcome gate:** Integrated proof that organizational capabilities exercised across **VO-S01–S10** compose under compound external shock — not a replacement for individual scenario proofs.

**Portfolio status:** Referenced program milestone; **not** a member of frozen ten-scenario v1 selection table.

---

## North Star

**Scenario:** VO-NS-01 — *Self-Organizing Virtual Organization Challenge*

**Outcome gate:** End-to-end challenge where input is primarily **mission, objectives, constraints, resources, constitution, and human authority boundaries** — without developer-predefined teams, roles, capability graph, initiative portfolio, or organizational topology.

**Portfolio status:** North Star integrative test; **not** frozen portfolio v1 membership.

---

## Relationship to Frozen-30

Epoch work **assumes** enterprise execution substrate scenarios and platform mechanisms mature through **Frozen-30** and normal platform evolution. VO epochs do **not** renumber or replace Frozen-30.

Parallel track reference: [`ENTERPRISE_E2E_SCENARIO_CATALOG.md`](ENTERPRISE_E2E_SCENARIO_CATALOG.md).

---

## Downstream documentation stages

| Stage | Purpose |
|-------|---------|
| VO-DOC-1 | Public README and documentation map links |
| VO-DOC-2 | VO gates integrated into existing scenario scaffolding process |
| VO-DOC-3 | Dedicated program status and gap tracking SSOT |
| VO-GATE-0 | Documentation audit before VO-S01 activation |

---

## Success criteria (program-level, not per scenario)

An epoch is **not** complete when design docs exist. It completes when associated scenarios reach **accepted proof outcomes** under normal Proof Library discipline — adversarial cases, evidence, and honest falsification — without VO-specific runtimes or bypass paths.

# Autonomous Organization

**Status:** Strategic initiative / North Star program — **not** a shipped product, platform domain, or runtime  
**Program SSOT (maintainers):** [`AUTONOMOUS_ORGANIZATION_INITIATIVE.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_INITIATIVE.md)

---

## North Star

Intergrax pursues a long-term direction:

> **Governed Self-Organizing Virtual Organization**

A governed, self-organizing virtual organization receives **mission, goals, constraints, resources, and constitution** — not a predefined org chart or fixed workflow — and can discover needed capabilities, assign responsibilities and authority, compose execution resources, evaluate organizational fitness, and change internal structure over time while preserving commitments, memory, governance, and auditability.

**Canonical definition:**

> **Self-Organizing Virtual Organization** is a persistent, governed system that receives mission, goals, constraints and resources rather than a predefined organizational structure or workflow, and can autonomously discover the capabilities it needs, assign responsibilities and authority, compose execution resources, evaluate its own organizational fitness, and change its internal structure over time while preserving commitments, memory, governance and auditability.

This initiative is **not** a feature, a single platform domain, a product SKU, a new runtime, a new execution engine, an “AI CEO plus predefined workers” pattern, or a parallel scenario system. It is advanced through **real enterprise Scenario Proofs** on the existing Intergrax platform.

---

## What problem it addresses

Enterprise operations need more than task-level agents. They need systems that can **reorganize** when missions, markets, constraints, or capabilities change — without abandoning governance, evidence, or commitments.

Autonomous Organization names that **strategic program**: how Intergrax could eventually support organizations that **form and transform themselves** under human constitution and platform authority — reusing Virtual Workers and existing execution substrate rather than inventing a side stack.

---

## Platform principle (non-negotiable)

**Platform operates on contracts, not implementations.**

Any mechanism that emerges from this program must be pluggable, governable, replaceable, and reachable through **explicit platform contracts**. The platform must not depend on vendor-specific core logic, hardcoded strategies, or private execution flows.

---

## Relationship to Virtual Workforce and Autonomous Work

```text
Virtual Worker
  = persistent holder of one business responsibility

Virtual Workforce
  = collection / fleet of Virtual Workers

Autonomous Organization
  = system that can determine which capabilities, responsibilities,
    units, workers, teams, and allocations should exist
    and how they should change
```

| Topic | Boundary |
|-------|----------|
| Virtual Workers | Autonomous Organization **may use** Virtual Workers; it does **not** replace [`Virtual Workforce`](VIRTUAL_WORKFORCE.md) or [`Autonomous Work`](../architecture/AUTONOMOUS_WORK.md) ownership of worker lifecycle |
| Autonomous Work | Composes existing platform mechanisms under worker semantics; Autonomous Organization sits **above** composition of responsibilities — **does not** supersede Autonomous Work domain ownership |
| Organizational vs worker goals | **Future architecture boundary** — whether organizational goals and Autonomous Work goals share one contract is **not decided** in this initiative foundation; to be tested through scenarios |

Canonical technical depth: [`AUTONOMOUS_WORK.md`](../architecture/AUTONOMOUS_WORK.md) · product framing: [`VIRTUAL_WORKFORCE.md`](VIRTUAL_WORKFORCE.md)

---

## Relationship to Cognitive Layer

Autonomous Organization **uses cognitive capabilities** (beliefs, hypotheses, prediction, discovery, learning, goal formation) as **compositional** support — not as owner of the whole organization.

Cognitive Layer is **candidate architecture / supporting capability direction**, not a claim of full implementation. Cognition may **propose** goals, plans, and organizational changes; **Governance, Authorization, and Execution Engine** remain owners of legal execution and control (see program invariants AO-P4 in Initiative SSOT).

---

## Relationship to Frozen-30 (Enterprise E2E portfolio)

| Portfolio | Role |
|-----------|------|
| **[Enterprise E2E Scenario Catalog v1 (Frozen-30)](../maintainers/plans/ENTERPRISE_E2E_SCENARIO_CATALOG.md)** | **Enterprise safety + execution substrate** — governance, authority, uncertainty, evidence, recovery, resilience, memory integrity, auditability |
| **Autonomous Organization Scenario Portfolio v1** | **Goal → Capability → Organization → Learning → Reorganization** — see [`AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md) |

Frozen-30 is **not modified** by this initiative. There is **no second shared catalog** merging the two portfolios.

---

## How the program advances

Progress is **scenario-first**, on normal Intergrax Scenario Proof rails:

- frozen problem identity in the VO portfolio;
- design and quality gates per [`PLATFORM_PROOF_AUTHORING_GUIDE.md`](../../../platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md);
- platform gap stops — reusable contract-first capabilities before resume;
- adversarial proof and evidence — not architecture demos.

**Scenario-first principle:** each scenario exists to solve a **real enterprise E2E problem**, not to showcase a pre-drawn architecture or existing component.

Implementation order and epoch gates: [`AUTONOMOUS_ORGANIZATION_ROADMAP.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_ROADMAP.md)  
Capability direction and gaps: [`AUTONOMOUS_ORGANIZATION_CAPABILITY_MODEL.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_CAPABILITY_MODEL.md)

---

## Maturity wording

| Claim | Allowed? |
|-------|----------|
| Strategic initiative with frozen scenario portfolio v1 | Yes |
| Runtime / OrganizationEngine / production organizational autonomy | **No** — not implemented |
| VO scenarios executable or verified | Only when per-package `SCENARIO_SPEC.md` lifecycle says so |

North Star challenge **VO-NS-01** remains a **future** integrative test; it is not part of the frozen ten-scenario v1 portfolio.

---

## Maintainer entry points

| Document | Purpose |
|----------|---------|
| [`AUTONOMOUS_ORGANIZATION_INITIATIVE.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_INITIATIVE.md) | Program governance SSOT and router |
| [`AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md) | Frozen VO scenario identities |
| [`AUTONOMOUS_ORGANIZATION_CAPABILITY_MODEL.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_CAPABILITY_MODEL.md) | Capability domains and gap clusters |
| [`AUTONOMOUS_ORGANIZATION_ROADMAP.md`](../maintainers/plans/AUTONOMOUS_ORGANIZATION_ROADMAP.md) | Outcome epochs (no release calendar) |
| [`PROOF_LIBRARY.md`](../proofs/PROOF_LIBRARY.md) · [`platform_proofs/README.md`](../../../platform_proofs/README.md) | Proof Library gateway |

Public integration into root README and documentation maps is planned as **VO-DOC-1** — not part of this foundation drop.

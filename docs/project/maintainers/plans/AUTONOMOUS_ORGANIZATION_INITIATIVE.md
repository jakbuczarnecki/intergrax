# Autonomous Organization Initiative — Program SSOT

**Document type:** Maintainer program governance and router  
**Initiative:** Integrax **Autonomous Organization** (Governed Self-Organizing Virtual Organization)  
**Task foundation:** VO-DOC-0  
**Current program position:** Foundation documentation established; **no VO scenario packages required by this SSOT**; runtime organizational mechanisms **not implemented**

---

## Authority

This file is the **Single Source of Truth** for:

- program scope and non-goals;
- initiative invariants (AO-P1 … AO-P8);
- scenario development governance and VO Qualification Overlay;
- links to portfolio, capability model, roadmap, and platform proof rails.

It **does not** duplicate Frozen-30, Proof Library protocol, Autonomous Work architecture, or Authoring Guide normative text. It **routes** to them.

| Artifact | Role |
|----------|------|
| **This file** | Program governance SSOT |
| [`AUTONOMOUS_ORGANIZATION.md`](../../overview/AUTONOMOUS_ORGANIZATION.md) | Public initiative overview |
| [`AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md`](AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md) | Frozen VO scenario identities (v1) |
| [`AUTONOMOUS_ORGANIZATION_CAPABILITY_MODEL.md`](AUTONOMOUS_ORGANIZATION_CAPABILITY_MODEL.md) | Capability domains, gap clusters, dependency spine |
| [`AUTONOMOUS_ORGANIZATION_ROADMAP.md`](AUTONOMOUS_ORGANIZATION_ROADMAP.md) | Epoch outcome gates and implementation priority |
| [`ENTERPRISE_E2E_SCENARIO_CATALOG.md`](ENTERPRISE_E2E_SCENARIO_CATALOG.md) | **Frozen-30** — enterprise safety + execution substrate (separate portfolio) |
| [`PROOF_LIBRARY.md`](../../proofs/PROOF_LIBRARY.md) · [`platform_proofs/README.md`](../../../../platform_proofs/README.md) | Proof Library gateway |
| [`PLATFORM_PROOF_AUTHORING_GUIDE.md`](../../../../platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md) | Scenario lifecycle, quality gate, platform gap discipline |
| [`AUTONOMOUS_WORK.md`](../../architecture/AUTONOMOUS_WORK.md) | Virtual Worker domain ownership |
| [`VIRTUAL_WORKFORCE.md`](../../overview/VIRTUAL_WORKFORCE.md) | Product-facing Virtual Workforce framing |

**Per-scenario current lifecycle** remains authoritative in `platform_proofs/scenarios/<slug>/SCENARIO_SPEC.md` frontmatter — never this initiative SSOT.

---

## Program definition

**Autonomous Organization** is a **Strategic Initiative / North Star Program** for a governed self-organizing virtual organization — advanced only through bounded Scenario Proofs and platform evolution, without a VO-specific runtime or parallel proof framework.

North Star integrative challenge: **VO-NS-01** (documented in Roadmap; not in frozen ten). Integrated qualification: **VO-IQ-01** (later).

---

## Initiative invariants

### AO-P1 — Organization Is State, Not Configuration

Organizational structure is **mutable runtime state**, not a static upfront configuration file that defines teams and roles for all time.

### AO-P2 — Capability Before Role

Required flow:

```text
need → capability → responsibility → role / unit / executor
```

Forbidden framing: “developer creates SalesAgent” as the primary design motion.

### AO-P3 — Agent Is an Executor, Not Organizational Identity

Agents execute work. They are **not** the organization and not durable organizational identity.

### AO-P4 — Cognition Proposes, Core Governs and Executes

Cognition may propose goals, hypotheses, plans, and organizational changes. **Governance, Authorization, and Execution Engine** retain ownership of legal execution and control.

### AO-P5 — Organization May Change Itself, But Not Its Constitution

The organization may change teams, units, responsibilities, goals, resource allocation, and executors. It **must not** unilaterally expand authority, constitutional boundaries, or human-controlled constraints.

### AO-P6 — Organization Identity Survives Executor Replacement

Replacing agents, models, workers, or providers must **not** automatically erase organizational identity, commitments, or continuity obligations.

### AO-P7 — Platform Mechanisms Are Reused, Never Reimplemented in Scenarios

If a platform owner exists, the scenario **must** use it. If capability is missing:

```text
scenario → STOP → platform gap → architecture decision
→ reusable platform capability → qualification → scenario resumes
```

### AO-P8 — Platform Contracts Before Implementations

Generic platform mechanisms must be available through **public contracts** at the owning layer. Higher layers must not depend on concrete implementations of lower layers.

**Hard rule:** **Platform operates on contracts, not implementations.** No vendor-specific core logic; no private side flows; no layer boundary violations.

If this program requires a **new platform contract or boundary move**, stop implementation work and record:

```text
ARCHITECTURAL DECISION REQUIRED
```

---

## SCENARIO-FIRST PRINCIPLE

Every VO scenario exists to solve a **real enterprise E2E problem** with material consequences — not to demonstrate a feature, prove a pre-drawn architecture, or showcase an existing component.

A scenario that reveals a missing generic platform capability is a **discovery opportunity**. It is **not** a reason to weaken the problem or build a local substitute.

---

## Platform gap rule

When a scenario exposes a missing **generic** platform capability:

**Do not:** build a local substitute in the scenario; bypass the platform; hardcode the missing mechanism; soften the business problem.

**Do:**

```text
detect gap
↓
stop scenario implementation at that boundary
↓
separate architecture analysis
↓
decide correct platform owner
↓
design contract-first reusable mechanism
↓
implement / qualify separately
↓
resume scenario
```

Platform gap design is a **separate process** — not an ad-hoc patch inside scenario implementation.

---

## Reuse over invention

```text
scenario need
↓
search existing platform owner
↓
reuse if correct
↓
if incomplete: architecture decision → generalize platform owner if justified
```

Not: `scenario need → new local implementation`.

---

## Zero bypass

VO scenarios must **exercise** Intergrax value. They must not:

- call vendors outside platform contracts;
- create private execution or governance flows;
- bypass Decision System where semantically required;
- bypass Execution Engine;
- stand up local persistence, diagnostics, or memory for generic platform concerns when a platform owner exists or should exist.

---

## Legacy rule (implementation-era guidance)

If future VO work encounters legacy mechanisms with **no real user**, **no production dependency**, and **no active contract requirement**, **removal** is preferred over gratuitous backward compatibility. This foundation task removes nothing.

---

## Scenario development governance

Each VO scenario is a **normal Intergrax Scenario Proof**. Forbidden: VO-specific runtime, VO-specific scenario engine, parallel proof framework, parallel lifecycle.

**Canonical path:**

```text
FROZEN PROBLEM
↓
activate scenario
↓
create_scenario_proof.py
↓
REAL PROBLEM
↓
SOLUTION ARCHITECTURE
↓
SCENARIO QUALITY GATE
↓
VO QUALIFICATION OVERLAY
↓
ACCEPTED FOR IMPLEMENTATION
↓
INTERGRAX FIT
↓
GAP DECISION
↓
if platform gap: STOP scenario
↓
separate architecture process
↓
reusable platform capability
↓
qualification
↓
resume scenario
↓
init_scenario_implementation.py
↓
normal platform-native implementation
↓
adversarial proof
↓
evidence
↓
verdict
```

Normative lifecycle detail: [`PLATFORM_PROOF_AUTHORING_GUIDE.md`](../../../../platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md).

---

## VO Qualification Overlay

Additional qualification for Autonomous Organization scenarios **on top of** existing Scenario Quality Gate and Application requirements. **Does not** change scaffolds or generators in VO-DOC-0.

| Test | Fail condition |
|------|----------------|
| **Organization Reduction Test** | Developer predefines most future org structure |
| **Capability Emergence Test** | Capabilities listed upfront without emerging from the problem |
| **Responsibility Emergence Test** | Responsibilities not traceable to capability need |
| **Organizational State Test** | Topology treated as static configuration, not runtime state |
| **Core Authority Test** | Real actions not routed through Governance, Authorization, Execution Engine, and normal platform contracts |
| **Continuity Test** | Reorganization breaks commitments, identity, memory, evidence, or active work correctness |
| **Application Survival Test** | Standard Scenario requirement — mandatory |
| **Application Observability Test** | Standard Scenario requirement — mandatory |

---

## Frozen-30 vs VO portfolio

| | Frozen-30 | VO portfolio v1 |
|---|-----------|-----------------|
| **Focus** | Enterprise safety + execution substrate | Goal → capability → organization → learning → reorganization |
| **SSOT** | [`ENTERPRISE_E2E_SCENARIO_CATALOG.md`](ENTERPRISE_E2E_SCENARIO_CATALOG.md) | [`AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md`](AUTONOMOUS_ORGANIZATION_SCENARIO_PORTFOLIO.md) |
| **Interaction** | Complementary; **do not merge catalogs or renumber Frozen-30** |

---

## Future architecture boundary (explicitly open)

**Organizational goal vs Autonomous Work `WorkerGoal` contract unification** — **not resolved** in VO-DOC-0. Mark for scenario-driven architecture review.

---

## Related program stages (outside this file)

| Stage | Intent |
|-------|--------|
| VO-DOC-1 | Public README / documentation map integration |
| VO-DOC-2 | Formal VO gates in existing scenario process |
| VO-DOC-3 | Program status and gap tracking SSOT |
| VO-GATE-0 | Documentation audit before VO-S01 |

No `AUTONOMOUS_ORGANIZATION_STATUS.md` in VO-DOC-0 — roadmap and this SSOT carry **current program position** until VO-DOC-3.

# Autonomous Organization Capability Model

**Document type:** Architecture direction — capability domains and gap clusters (not code)  
**Initiative SSOT:** [`AUTONOMOUS_ORGANIZATION_INITIATIVE.md`](AUTONOMOUS_ORGANIZATION_INITIATIVE.md)  
**Status:** Planned capability model for scenario-driven discovery — **not implemented as runtime**

> **Naming discipline:** Domains, critical capabilities, and gap clusters below are **not** class names, module names, or service identifiers.

---

## Purpose

This model describes **what a self-organizing virtual organization must be able to do** over time, so VO scenarios can expose **platform gaps** without inventing parallel stacks. It complements Frozen-30 (execution substrate) with **organizational composition and change**.

Atomic capability inventories may grow in later program stages; v1 keeps **domains**, **critical capabilities**, **fourteen gap clusters**, and the **dependency spine**.

---

## Dependency spine

Organizational flow (center):

```text
MISSION
↓
GOALS
↓
WORLD MODEL
↓
NEEDS
↓
REQUIRED CAPABILITIES
↓
CAPABILITY GAP
↓
ACQUIRE / COMPOSE
↓
RESPONSIBILITY
↓
AUTHORITY
↓
RESOURCE ALLOCATION
↓
EXECUTION
↓
OUTCOME
↓
LEARNING
↓
SELF-EVALUATION
↓
REORGANIZATION
↓
COMMITMENT MIGRATION
↓
IDENTITY / CONTINUITY
↺
```

**Surround (continuous constraints):** governance, constitution, evidence, risk, memory.

Execution at the bottom of the spine **delegates to existing Intergrax substrate** (Governance, Authorization, Execution Engine, evidence, recovery) — Autonomous Organization does not replace Frozen-30 concerns.

---

## Fourteen canonical gap clusters

These clusters group expected **platform and architecture discovery** work — **not** product module boundaries:

1. Mission → Goal Formation  
2. Persistent Organizational World Model  
3. Autonomous Discovery & Experimentation  
4. Capability Requirement & Gap Discovery  
5. Capability Acquisition & Composition  
6. Responsibility & Organizational Design  
7. Strategic Resource Allocation  
8. Organizational Memory & Institutional Learning  
9. Organizational Self-Evaluation  
10. Self-Reorganization  
11. Commitment Management & Migration  
12. Organizational Identity & Continuity  
13. Organizational Governance Integration  
14. Capability / Organization Economics & Stabilization  

Scenarios should map discovered gaps to **one or more clusters** when stopping for architecture — without renaming clusters after code packages.

---

## Domains and critical capabilities

### 1. Mission & Constitution

**Critical capabilities:** ingest and persist mission; represent human constitution and non-negotiable constraints; separate amendable policy from constitutional bounds (AO-P5).

**Gap cluster touchpoints:** 1, 13.

---

### 2. Goal System

**Critical capabilities:** decompose mission into governed goals; delegate sub-goals with alignment constraints; detect drift vs mission (VO-S10).

**Gap cluster touchpoints:** 1, 14.

**Open boundary:** unification with Autonomous Work `WorkerGoal` — **future architecture review**.

---

### 3. Environment & World Model

**Critical capabilities:** maintain organizational situational model (markets, regulators, partners, dependencies); update beliefs on evidence; scope model to decision needs.

**Gap cluster touchpoints:** 2, 3.

---

### 4. Beliefs, Uncertainty & Epistemics

**Critical capabilities:** represent confidence and unknowns; route high-uncertainty decisions through governance; integrate with enterprise uncertainty substrate where applicable.

**Gap cluster touchpoints:** 2, 3, 13.

---

### 5. Discovery & Experimentation

**Critical capabilities:** propose hypotheses and bounded experiments; measure outcomes; stop or scale based on evidence — not ad-hoc agent loops.

**Gap cluster touchpoints:** 3, 4.

---

### 6. Capability Management

**Critical capabilities:** express required capabilities from needs; detect internal vs external acquisition; compose capabilities from platform contracts; retire obsolete capabilities.

**Gap cluster touchpoints:** 4, 5, 14.

---

### 7. Organizational Design

**Critical capabilities:** form units from responsibility bundles; adjust topology as state (AO-P1); avoid predefined role catalogs as source of truth.

**Gap cluster touchpoints:** 6, 10.

---

### 8. Responsibility & Authority

**Critical capabilities:** assign responsibility after capability identification (AO-P2); map authority to platform Authorization; revoke and transfer without orphan actions.

**Gap cluster touchpoints:** 6, 13.

---

### 9. Resources & Economics

**Critical capabilities:** allocate budget, attention, and executor capacity; trade off portfolio initiatives under constraints (VO-S03); detect local optimization harm (VO-S10).

**Gap cluster touchpoints:** 7, 14.

---

### 10. Planning & Execution Composition

**Critical capabilities:** compose work from goals through governed execution; bind Virtual Workers and agents as **executors** (AO-P3); no private execution stacks.

**Gap cluster touchpoints:** 5, 13 — **reuse** Execution Engine and Autonomous Work where correct.

---

### 11. Coordination & Conflict Resolution

**Critical capabilities:** detect cross-unit conflicts; escalate under policy; resolve without bypassing Decision System when required.

**Gap cluster touchpoints:** 6, 13.

---

### 12. Organizational Memory & Knowledge

**Critical capabilities:** institutional memory beyond single worker horizon; attributable history for reorganization; retrieve relevant context for org-level decisions.

**Gap cluster touchpoints:** 8, 12 — **reuse** platform memory contracts where generic.

---

### 13. Outcomes, Learning & Adaptation

**Critical capabilities:** capture outcomes at org level; update policies and models; feed self-evaluation — distinct from single-run telemetry.

**Gap cluster touchpoints:** 8, 9.

---

### 14. Organizational Self-Evaluation

**Critical capabilities:** assess fitness, bottlenecks, redundancy (VO-S04, S05); trigger reorganization proposals under governance.

**Gap cluster touchpoints:** 9, 10.

---

### 15. Self-Reorganization & Lifecycle

**Critical capabilities:** execute structural change; spin up/down ventures (VO-S02); consolidate complexity; preserve continuity tests (Initiative overlay).

**Gap cluster touchpoints:** 10, 11.

---

### 16. Governance, Safety, Continuity & Evidence

**Critical capabilities:** integrate org decisions with platform Governance and evidence; no parallel “org governance”; crisis and recovery alignment with Frozen-30 themes.

**Gap cluster touchpoints:** 13 — **primarily reuse**, extend only via contract-first gaps.

---

### 17. Commitment Management

**Critical capabilities:** represent external and internal commitments; migrate ownership across reorg (VO-S06, S09); forbid silent abandonment.

**Gap cluster touchpoints:** 11.

---

### 18. Organizational Identity

**Critical capabilities:** stable org identity across executor replacement (AO-P6); continuity narratives auditable to humans and regulators.

**Gap cluster touchpoints:** 12.

---

## Scenario mapping hint (non-normative)

| Scenario | Primary gap cluster emphasis |
|----------|------------------------------|
| VO-S01 | 1, 4, 5, 6 |
| VO-S02 | 5, 10, 11 |
| VO-S03 | 7, 14, 1 |
| VO-S04 | 9, 10 |
| VO-S05 | 9, 10, 6 |
| VO-S06 | 11, 10, 12 |
| VO-S07 | 3, 4, 1 |
| VO-S08 | 5, 10, 12 |
| VO-S09 | 1, 11, 13 |
| VO-S10 | 14, 2, 9 |

---

## Platform contract-first reminder

Every generic mechanism discovered through this model must land as a **reusable platform contract** (AO-P8). Scenario-local implementations of generic concerns are **disqualifying** unless explicitly bounded to non-generic, problem-specific surface area.

If a gap implies a **new execution authority** or core boundary move → **ARCHITECTURAL DECISION REQUIRED** — do not implement inside a scenario session by default.

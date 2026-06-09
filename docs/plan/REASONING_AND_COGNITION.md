# Reasoning and Cognition — Implementation Plan

**Architecture (1:1):** [`architecture/REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

(Global)

1. **Contract** — Pydantic / Protocol public API
2. **Trace** — cognition transitions emit `TraceEvent` / `RuntimeEvent` (`ops:planning`, `DECISION_EMITTED`)
3. **Test** — unit + integration, deterministic, no network
4. **Documentation** — update this plan + architecture pair when contracts change
5. **No regression** — `pytest -m gate` green; Echo through NexusLoop
6. **Reuse Tier-0** — extend existing planner modules; no parallel LLM/log/trace stacks
7. **Separation** — reasoning/planning docs stay in this pair; orchestration scheduling stays in `ORCHESTRATION` / `NEXUS_EXECUTION_FLOW`
8. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts

---

## Phase COG-DOC — Domain pair establishment (Band 2al)

**Status:** **Done** (2026-06-08) — architecture + plan pair created; hub + audit routing updated  
**Prerequisites:** Phase FLOW **Done** (FLOW-1, FLOW-11, FLOW-12 cognition closeouts) · FAUDIT-32 §7 baseline L2  
**Goal:** Establish **18th domain pair** as canonical source of truth for Reasoning and Cognition Layer (RCL) — consolidate scattered §7 audit content without runtime refactor  
**Priority ladder:** **Band 2al** (§4.0 PLATFORM_FOUNDATION) — **closed** on doc merge  
**Execution order:** [§6.2al](#62al-phase-cog-doc-execution-order-band-2al--closed) · queue: [§6.1al](#61al-harness-implementation-queue--reasoning-and-cognition-domain-pair-closed)

**Delivery rule:** COG-DOC.* = docs-only PRs; no code unless doc audit finds contract drift → route to COG-DEPTH.*

**ADR:** **No ADR needed** — documentation boundary split only; runtime contracts unchanged. Rationale: RCL codifies existing modules (`TaskPlanner`, `EngineBackedNexusPlanner`, `DecisionRecord`) under one domain pair; no new Tier-0 mechanism.

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| COG-DOC.1 | **`architecture/REASONING_AND_COGNITION.md`** — full RCL canon (planes, contracts, code map, gaps) | **Done** | **Critical** | `docs/architecture/` | Hub links; audit §7 points here |
| COG-DOC.2 | **`plan/REASONING_AND_COGNITION.md`** — this file; Phase COG-DEPTH register | **Done** | **Critical** | `docs/plan/` | 1:1 pair check green |
| COG-DOC.3 | **Hub update** — 18 domain pairs; audit routing §7 → RCL pair | **Done** | High | `intergrax_runtime_architecture.md` | `check_docs_domain_pairs.py` OK |
| COG-DOC.4 | **Cross-ref sync** — `ORCHESTRATION`, `NEXUS_EXECUTION_FLOW` §7–§8, §24; `AGENTS.md`; `INTEGRAX_HARNESS_AUDIT_MAP` §7 | **Done** | High | `docs/*` | No orphan §7 content |
| COG-DOC.5 | **Gate script** — `python scripts/check_docs_domain_pairs.py` | **Done** | Medium | CI scripts | 18 pairs reported |
| COG-DOC.6 | **Routing modes §9.4** — MULTI_AGENT vs pipeline graph vs engine planner | **Done** | High | `architecture/REASONING_AND_COGNITION.md` §9.4–§9.5 | Cross-ref TIER3 §23, ORCH §55 |

---

## Phase COG-DEPTH — Reasoning layer maturity uplift (Band 2am — planned)

**Status:** **Planned** — **0/22 Done** · canonical register: [COG-DEPTH — Master deliverables register](#cog-depth--master-deliverables-register-all-22-tasks)  
**Prerequisites:** Phase COG-DOC **Done** · Phase FLOW **Done** · default queue = §6.1 gate maintenance until Band 2am prioritized  
**Goal:** Raise FAUDIT-32 §7 from **L2 → L3+** — unified planner stack, Prompt Registry on planners, Nexus `DecisionRecord`, reasoning failure taxonomy, optional `ReasoningProfile`  
**Priority ladder:** **Band 2am** (§4.0) — **not active**; requires explicit operator reprioritization off §6.1 maintenance  
**Traceability:** [Appendix A](#appendix-a--reasoning-and-cognition-traceability-phase-cog-depth)

**Delivery rule:** One **COG-* ID per PR** → update master table + architecture gap register §21 → `pytest -m gate` green.

**Principle:** **evolve, not rewrite** · reuse `EnginePlannerOrchestrator` · no second planner OS · Tier-1 domain-agnostic · LLM planners fail safe to `TaskPlanner`.

**Out of scope:** K.1/K.2 business agents · autonomous prompt mutation without Prompt Registry · replacing `TaskClassifier` with product-specific rules in Tier-1 · deep RL planning policies (see AHI ADR-ADAPT-001).

### COG-DEPTH — Maturity targets

| Area | Current (post COG-DOC) | Target | Primary IDs |
|------|------------------------|--------|-------------|
| LLM Nexus planner | L2 bridged | L3 | COG-1.*, COG-2.1 |
| Engine planner unification | L2 dual stack | L3 | COG-1.* |
| Prompt Registry on planners | L2 partial | L3 | COG-2.* |
| Nexus planning DecisionRecord | L1 gap | L3 | COG-4.* |
| Reasoning failure taxonomy | L1 gap | L3 | COG-6.* |
| LLM classifier | L0 | L2 optional | COG-3.* |
| Model routing for reasoning | L2 partial | L3 | COG-5.* |
| Planning observability SLOs | L2 | L3 | COG-OBS.* |

**Success gate:** P0 + P1 **Done**; FAUDIT §7 **L3+** on re-run; zero ad-hoc planner prompts in Nexus hot path.

```text
Wave COG0 — Canon + ADR check (5 tasks) — COG-DOC Done
Wave COG1 — Planner unification P0 (5 tasks)
Wave COG2 — Prompt Registry on planners P0 (4 tasks)
Wave COG3 — Classifier extensions P1 (3 tasks)
Wave COG4 — DecisionRecord at planning boundary P1 (2 tasks)
Wave COG5 — ReasoningProfile + model routing P1 (3 tasks)
Wave COG6 — Failure taxonomy + trace P1 (3 tasks)
Wave COG7 — Observability SLOs P2 (2 tasks)
Total COG-DEPTH: 22 (excluding COG-DOC)
```

---

### 6.2al Phase COG-DOC execution order (Band 2al — closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | COG-DOC.1 | Architecture canon | Critical |
| 2 | COG-DOC.2 | Plan register | Critical |
| 3 | COG-DOC.3 | Hub 18-pair index | High |
| 4 | COG-DOC.4 | Cross-ref sync | High |
| 5 | COG-DOC.5 | Domain pair gate script | Medium |

### 6.1al Harness implementation queue — Reasoning and Cognition domain pair (closed)

**Status:** **Closed** (2026-06-08)  
**Band:** 2al  
**Outcome:** 18th domain pair live; §7 audit routes to `REASONING_AND_COGNITION`.

---

### 6.2am Phase COG-DEPTH execution order (Band 2am — planned)

Work **one COG ID per PR** when Band 2am is activated.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| COG1 | COG-1.1–COG-1.5 | 5 | **P0** — Engine planner ↔ Nexus unification |
| COG2 | COG-2.1–COG-2.4 | 4 | **P0** — Prompt Registry planner prompts |
| COG3 | COG-3.1–COG-3.3 | 3 | **P1** — Classifier extensions (**COG-3.1 Partial** via ORCH-CONFIG.1) |
| COG4 | COG-4.1–COG-4.2 | 2 | **P1** — Planning-phase DecisionRecord |
| COG5 | COG-5.1–COG-5.3 | 3 | **P1** — ReasoningProfile + routing |
| COG6 | COG-6.1–COG-6.3 | 3 | **P1** — Failure taxonomy |
| COG7 | COG-OBS.1–COG-OBS.2 | 2 | **P2** — Planning SLO metrics |
| **Total** | | **22** | |

---

## COG-DEPTH — Master deliverables register (all 22 tasks)

### Wave COG1 — Planner unification (P0)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-1.1 | **`EngineBackedNexusPlanner` → `EnginePlannerOrchestrator` adapter** — shared parse/validate path for `NexusPlan` | Planned | **Critical** | `orchestration_wiring.py`, `nexus_llm_plan_builder.py` | Unit: same task → equivalent plan shape vs current bridge |
| COG-1.2 | **Unified planner diagnostics** — single `planner_build_debug` surface on Nexus planning trace | Planned | High | `engine_planner_diagnostics.py` | `PLAN_CREATED` payload includes planner source |
| COG-1.3 | **Plan validation gate** — reject LLM plans with cycles/unknown agents before graph build | Planned | High | `planning_runner.py` | Invalid plan → fallback or FAILED with `COG-PLAN-VALID` |
| COG-1.4 | **`allow_dynamic_replan` wire** — document + test engine replan boundary vs Nexus plan immutability | Planned | Medium | `plan_loop_controller.py`, ADR note | Integration test replan does not mutate committed NexusPlan |
| COG-1.5 | **Gate test** — `planner_kind=engine` regression suite with mock LLM | Planned | High | `tests/unit/runtime/nexus/planning/` | `-m gate` green |

### Wave COG2 — Prompt Registry on planners (P0)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-2.1 | **`nexus.task_planner.v1` prompt id** — replace inline string in `nexus_llm_plan_builder.py` | Planned | **Critical** | `prompts/`, `nexus_llm_plan_builder.py` | Golden catalog entry; gate script |
| COG-2.2 | **Tool planner prompt ids** — ensure `ToolPlanningConfig` uses registry in all reference hosts | Planned | High | `tool_planning_config.py` | Lab host smoke |
| COG-2.3 | **Engine planner `PlannerPromptConfig` registry binding** | Planned | High | `engine_plan_models.py` | Forced-plan replay uses registry |
| COG-2.4 | **Author guide Appendix** — planner prompt authoring for Tier-3 | Planned | Medium | `guides/AGENT_CREATION_GUIDE.md` | TOC entry + cross-ref |

### Wave COG3 — Classifier extensions (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-3.1 | **`classifier_kind=rules`** — `RulesTaskClassifier`, `IntentRoute`, orchestration tokens (ORCH-CONFIG.1) | **Done** | Medium | `orchestration_wiring.py`, `intent_routing.py` | `test_intent_routing.py`, `test_orchestration_cfg_simulation.py` |
| COG-3.2 | **Optional LLM classifier** — capability + message → classification with fallback to deterministic | **Done** | Medium | `llm_task_classifier.py` | `test_llm_task_classifier.py`; fallback on parse fail |
| COG-3.3 | **Classification trace enrichment** — confidence + rationale fields on hook payload | **Done** | Low | `task_contract.py`, `task_metadata_bridge.py` | `test_intent_routing.py`, `test_llm_task_classifier.py` |

### Wave COG4 — Planning-phase DecisionRecord (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-4.1 | **`DecisionRecord` on PLAN_CREATED** — planner choice, classification, fallback flag | Planned | High | `planning_runner.py`, `decision_record.py` | `DECISION_EMITTED` at planning phase |
| COG-4.2 | **Gate test FAUDIT-COG-1 extension** — planning + UAEP paths | Planned | High | `tests/integration/` | Both phases covered |

### Wave COG5 — ReasoningProfile and model routing (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-5.1 | **`ReasoningProfile` on `ApplicationEnvironmentProfile`** — planner LLM id, parse retries, prompt ids | Planned | High | `environment_profile.py` | Profile round-trip test |
| COG-5.2 | **Wire ReasoningProfile → orchestration wiring** — optional separate adapter for planners | Planned | High | `orchestration_wiring.py` | `planner_kind=engine` uses profile LLM |
| COG-5.3 | **Policy hook for planner model selection** — FAUDIT-LLM.1 partial close | Planned | Medium | `policy_engine.py` | Deny over-budget planner model |

### Wave COG6 — Reasoning failure taxonomy (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-6.1 | **`ReasoningFailureKind` enum** — §17 taxonomy as code | Planned | High | `intergrax/contracts/` | Used in trace payloads |
| COG-6.2 | **Emit failure kind on planner fallback and policy block** | Planned | High | `planning_runner.py`, `nexus_llm_plan_builder.py` | Integration test asserts kind |
| COG-6.3 | **Ops dashboard hints** — `ops:planning` failure counters | Planned | Medium | observability bridge | Metric names documented in OBS plan |

### Wave COG7 — Planning observability (P2)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-OBS.1 | **Planner latency + fallback rate metrics** | Planned | Medium | metrics export | SLO doc in OBSERVABILITY plan |
| COG-OBS.2 | **`scripts/check_reasoning_gates.py`** — optional CI: no inline planner prompts | Planned | Low | `scripts/` | Fails if ad-hoc prompt detected in hot path |

---

## Appendix A — Reasoning and Cognition traceability (Phase COG-DEPTH)

| Architecture § | Topic | Task IDs |
|----------------|--------|----------|
| §5 Three planes | Plane boundaries | COG-DOC.* |
| §9 Classification | Classifier extensions | COG-3.* · ORCH-CONFIG.1 (rules **Partial**) |
| §9.4 Routing modes | Authoring canon (docs) | COG-DOC.6 **Done** |
| §10 Nexus planning | Planner unification | COG-1.* |
| §10.4 LLM planner | Prompt Registry | COG-2.1 |
| §12 Engine planner | Orchestrator bridge | COG-1.1 |
| §14 DecisionRecord | Planning phase emit | COG-4.* |
| §15 Prompt compilation | Registry on all planners | COG-2.* |
| §16 Model selection | ReasoningProfile | COG-5.* |
| §17 Failure taxonomy | ReasoningFailureKind | COG-6.* |
| §18 Observability | SLO metrics | COG-OBS.* |
| §21 Gap register | Maturity uplift | All COG-DEPTH |

### Historical closeout traceability (pre-RCL domain)

These items implemented under FLOW/ORCH phases — **Done**; canon now owned by RCL:

| Legacy ID | Deliverable | RCL architecture § |
|-----------|-------------|-------------------|
| FLOW-1 | `EngineBackedNexusPlanner` | §10.4 |
| FLOW-11 | Pre-plan policy hooks | §10.5 |
| FLOW-12 | `DecisionRecord` UAEP gate | §14 |
| FLOW-17 | `multi_agent_order` | §10.3 |
| ORCH-1 | Planner strategies explicit | §10 |
| ORCH-2 | Declarative `graph_spec` | §11 |
| FAUDIT-COG.1 | DecisionRecord contract | §14 |

---

## Appendix B — FAUDIT-32 §7 scorecard (baseline)

| Audit question | Pre-RCL | Post COG-DOC | Post COG-DEPTH target |
|----------------|---------|--------------|----------------------|
| Structured plan contract? | Yes (`NexusPlan`) | Yes — canon §10 | Maintain |
| DecisionRecord per step? | UAEP only | Documented §14 | Planning + UAEP |
| Reasoning separated from execution? | Yes (UAEP) | Canon §4, §8 | Maintain |
| Planning strategies explicit? | Yes | Canon §10, Appendix B | Maintain |
| Prompt compilation layered? | Partial | Cross-ref §15 | COG-2.* Done |
| Reasoning failures classified? | No | Taxonomy §17 doc | COG-6.* code |
| **Layer score** | **L2** | **L2** (plan accurate) | **L3+** |

---

## Appendix C — Operator reading order

1. [`architecture/REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md) — RCL canon
2. This plan — COG-DEPTH register when implementing
3. [`architecture/NEXUS_EXECUTION_FLOW.md`](../architecture/NEXUS_EXECUTION_FLOW.md) — end-to-end flow only
4. [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix I §I.4 — host planner configuration

---

*End of Reasoning and Cognition Implementation Plan.*

# Reasoning and Cognition — Implementation Plan

**Architecture (1:1):** [`architecture/REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5, §16 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — incremental after IDEAL-L3 W2 closeout; closed COG-LC (2026-06-17)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-7.1 | §7 Cognition | Ship `ReasoningProfile` contract + environment wire | P1 | **Done** (COG-5.1 · COG-PROD.1) |
| AUDIT-IDEAL-7.2 | §7 Cognition | Complete `allow_dynamic_replan` runtime path | P1 | **Done** |
| AUDIT-IDEAL-7.3 | §7 Cognition | Reasoning failure taxonomy on all planner kinds | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

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

## Phase COG-DEPTH — Reasoning layer maturity uplift (Band 2as — closed)

**Status:** **Done** (2026-06-09) — **22/22 Done** · canonical register: [COG-DEPTH — Master deliverables register](#cog-depth--master-deliverables-register-all-22-tasks)  
**Prerequisites:** Phase COG-DOC **Done** · Phase FLOW **Done**  
**Goal:** Raise FAUDIT-32 §7 from **L2 → L3+** — unified planner stack, Prompt Registry on planners, Nexus `DecisionRecord`, reasoning failure taxonomy, optional `ReasoningProfile`  
**Priority ladder:** **Band 2as** (§4.0) — **closed**; default queue = §6.1 maintenance  
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

### 6.2as Phase COG-DEPTH execution order (Band 2as — closed)

**Status:** **Done** (2026-06-09) · **22/22 Done** · canonical register: [COG-DEPTH — Master deliverables register](#cog-depth--master-deliverables-register-all-22-tasks).

Work **one COG ID per PR** — phase **closed**; historical order below.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| COG1 | COG-1.1–COG-1.5 | 5 | **P0** — Engine planner ↔ Nexus unification |
| COG2 | COG-2.1–COG-2.4 | 4 | **P0** — Prompt Registry planner prompts |
| COG3 | COG-3.1–COG-3.3 | 3 | **P1** — Classifier extensions (**Done** — ORCH-CONFIG.1 + COG-3.*) |
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
| COG-1.1 | **`EngineBackedNexusPlanner` → `EnginePlannerOrchestrator` adapter** — shared parse/validate path for `NexusPlan` | **Done** | **Critical** | `nexus_plan_bridge.py`, `orchestration_wiring.py` | `test_nexus_plan_bridge.py` |
| COG-1.2 | **Unified planner diagnostics** — single `planner_build_debug` surface on Nexus planning trace | **Done** | High | `nexus_plan_bridge.py` | `PLAN_CREATED` payload includes `planner_source` |
| COG-1.3 | **Plan validation gate** — reject LLM plans with cycles/unknown agents before graph build | **Done** | High | `plan_validator.py` · `planning_runner.py` | Unknown agent/dep → FAILED before graph build |
| COG-1.4 | **`allow_dynamic_replan` wire** — document + test engine replan boundary vs Nexus plan immutability | **Done** | Medium | `interrupts/handler.py`, ADR-FLOW-003 | `test_audit_ideal_depth_gate.py` |
| COG-1.5 | **Gate test** — `planner_kind=engine` regression suite with mock LLM | **Done** | High | `test_nexus_plan_bridge.py`, `test_engine_planner_orchestration_gate.py` | `-m gate` green |

### Wave COG2 — Prompt Registry on planners (P0)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-2.1 | **`nexus.task_planner.v1` prompt id** — replace inline string in `nexus_llm_plan_builder.py` | **Done** | **Critical** | `prompts/nexus_task_planner/`, `nexus_planner_prompts.py` | `check_reasoning_gates.py` |
| COG-2.2 | **Tool planner prompt ids** — ensure `ToolPlanningConfig` uses registry in all reference hosts | **Done** | High | `reasoning_wiring.py`, `tool_planning_config.py` | `test_catalog_runtime_bridge.py` |
| COG-2.3 | **Engine planner `PlannerPromptConfig` registry binding** | **Done** | High | `reasoning_wiring.py` | `resolve_engine_planner_prompt_config()` |
| COG-2.4 | **Author guide Appendix** — planner prompt authoring for Tier-3 | **Done** | Medium | `guides/AGENT_CREATION_GUIDE.md` | Appendix COG-2.4 |

### Wave COG3 — Classifier extensions (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-3.1 | **`classifier_kind=rules`** — `RulesTaskClassifier`, `IntentRoute`, orchestration tokens (ORCH-CONFIG.1) | **Done** | Medium | `orchestration_wiring.py`, `intent_routing.py` | `test_intent_routing.py`, `test_orchestration_cfg_simulation.py` |
| COG-3.2 | **Optional LLM classifier** — capability + message → classification with fallback to deterministic | **Done** | Medium | `llm_task_classifier.py` | `test_llm_task_classifier.py`; fallback on parse fail |
| COG-3.3 | **Classification trace enrichment** — confidence + rationale fields on hook payload | **Done** | Low | `task_contract.py`, `task_metadata_bridge.py` | `test_intent_routing.py`, `test_llm_task_classifier.py` |

### Wave COG4 — Planning-phase DecisionRecord (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-4.1 | **`DecisionRecord` on PLAN_CREATED** — planner choice, classification, fallback flag | **Done** | High | `planning_runner.py`, `decision_record.py` | `DECISION_EMITTED` at planning phase |
| COG-4.2 | **Gate test FAUDIT-COG-1 extension** — planning + UAEP paths | **Done** | High | `test_planning_decision_record_gate.py` | Planning phase covered |

### Wave COG5 — ReasoningProfile and model routing (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-5.1 | **`ReasoningProfile` on `ApplicationEnvironmentProfile`** — planner LLM id, parse retries, prompt ids | **Done** | High | `contracts/reasoning_profile.py`, `environment_profile.py` | Profile on lab defaults |
| COG-5.2 | **Wire ReasoningProfile → orchestration wiring** — optional separate adapter for planners | **Done** | High | `orchestration_wiring.py`, `nexus_factory.py` | `planner_prompt_id` + `planner_llm_profile_id` policy context |
| COG-5.3 | **Policy hook for planner model selection** — FAUDIT-LLM.1 partial close | **Done** | Medium | `runtime_policy_engine.py`, `ReasoningProfile.denied_planner_model_ids` | Planning-phase deny gate |

### Wave COG6 — Reasoning failure taxonomy (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-6.1 | **`ReasoningFailureKind` enum** — §17 taxonomy as code | **Done** | High | `contracts/reasoning_failure.py` | Used in `nexus_plan_bridge` debug |
| COG-6.2 | **Emit failure kind on planner fallback and policy block** | **Done** | High | `nexus_llm_plan_builder.py`, `planning_runner.py` | `failure_kind` in metadata + `DECISION_EMITTED` |
| COG-6.3 | **Ops dashboard hints** — `ops:planning` failure counters | **Done** | Medium | `planning_metrics.py` | `ops_planning_failure_*_total` export |

### Wave COG7 — Planning observability (P2)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-OBS.1 | **Planner latency + fallback rate metrics** | **Done** | Medium | `planning_metrics.py` | Export hooks on planning path |
| COG-OBS.2 | **`scripts/check_reasoning_gates.py`** — optional CI: no inline planner prompts | **Done** | Low | `scripts/check_reasoning_gates.py` | Gate script green |

---

## Appendix A — Reasoning and Cognition traceability (Phase COG-DEPTH)

| Architecture § | Topic | Task IDs |
|----------------|--------|----------|
| §5 Three planes | Plane boundaries | COG-DOC.* |
| §9 Classification | Classifier extensions | COG-3.* · ORCH-CONFIG.1 **Done** |
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
| **Layer score** | **L2** | **L2** (plan accurate) | **L3** (COG-DEPTH **Done**) |

---

## Appendix C — Operator reading order

1. [`architecture/REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md) — RCL canon
2. This plan — COG-DEPTH register when implementing
3. [`architecture/NEXUS_EXECUTION_FLOW.md`](../architecture/NEXUS_EXECUTION_FLOW.md) — end-to-end flow only
4. [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix I §I.4 — host planner configuration

---

### COG-DEPTH — Paydown log

| Date | COG ID | Summary |
|------|--------|---------|
| 2026-06-09 | COG-1.*–COG-OBS.* | Phase COG-DEPTH **22/22 Done**; reference host engine planner presets |

---

## Phase COG-PROD — Production reasoning plane hardening (Band 2au)

**Status:** **Done** (2026-06-12) — doc↔code drift closed; production wiring complete  
**Prerequisites:** Phase COG-DEPTH **Done**  
**Goal:** L3+ reasoning plane with honest production wiring — planner LLM separation, parse retries, full prompt template binding, planning `DecisionRecord` enrichment, missing wiring helpers  
**Priority ladder:** Band 2au (maintenance queue after COG-DEPTH)

**Delivery rule:** One **COG-PROD.\*** ID per PR → update architecture §21 + this register → gate green.

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-PROD.1 | **`resolve_planner_llm_adapter()`** — producer/planner separation (mirror critic) | **Done** | **Critical** | `reasoning_wiring.py`, `orchestration_wiring.py`, `nexus_factory.py` | Separate adapter when `planner_llm_profile` set |
| COG-PROD.2 | **`planner_parse_retries` wire** + `nexus_task_planner` `user_template` | **Done** | High | `nexus_plan_bridge.py`, `nexus_planner_prompts.py`, `prompts/nexus_task_planner/` | Retries exercised in unit test |
| COG-PROD.3 | **Planning `DecisionRecord` enrichment** + `resolve_engine_planner_prompt_config()` | **Done** | High | `planning_runner.py`, `reasoning_wiring.py` | Gate test asserts classification + policy fields |
| COG-PROD.4 | **Doc reconciliation** — architecture stale gaps removed | **Done** | High | `architecture/REASONING_AND_COGNITION.md` | No §2/§14 contradictions vs §21 |
| COG-PROD.5 | **`check_reasoning_gates.py` uplift** — registry-only planner prompts | **Done** | Medium | `scripts/check_reasoning_gates.py` | CI script green |

### COG-PROD — Sprint execution order

| Sprint | IDs | Files (primary) |
|--------|-----|-----------------|
| S1 Doc | COG-PROD.4 | `docs/architecture/REASONING_AND_COGNITION.md`, `docs/plan/REASONING_AND_COGNITION.md` |
| S2 Planner LLM | COG-PROD.1 | `reasoning_wiring.py`, `orchestration_wiring.py`, `nexus_factory.py`, `contracts/reasoning_profile.py` |
| S3 Parse + prompt | COG-PROD.2 | `nexus_plan_bridge.py`, `nexus_planner_prompts.py`, `prompts/nexus_task_planner/1.yaml` |
| S4 Decision + engine prompt | COG-PROD.3, COG-PROD.5 | `planning_runner.py`, `reasoning_wiring.py`, `scripts/check_reasoning_gates.py`, tests |

---

## Phase COG-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — Full Harness LC sprint closeout after COG-PROD  
**Prerequisites:** Phase COG-PROD **Done**  
**Goal:** Close P1/P2 gaps from Layer Completion audit — doc reconciliation, Plane 2 engine prompt wire, planning latency metrics, classifier failure emission, CI bundle, LLM classifier registry prompt  
**ADR:** **No ADR needed** — extends existing `ReasoningProfile`, `RuntimeConfig`, and planning metrics contracts

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-LC-S1 | **Doc reconciliation** — remove stale `EnginePlan`/`EnginePlannerOrchestrator` active refs; fix Appendix A; AUDIT-IDEAL header | **Done** | High | `docs/architecture/REASONING_AND_COGNITION.md`, `docs/plan/REASONING_AND_COGNITION.md` | No doc↔code contradictions in §21 |
| COG-LC-S2 | **Engine prompt wire (Plane 2)** — `RuntimeConfig.engine_planner_prompt_id` + task/request metadata | **Done** | **Critical** | `catalog_runtime_bridge.py`, `reasoning_wiring.py`, `graph_executor.py` | `test_catalog_runtime_bridge.py`, `test_reasoning_wiring.py` |
| COG-LC-S3 | **Planning latency metrics** — `record_planner_latency` in `planning_runner` | **Done** | High | `planning_runner.py`, `planning_metrics.py` | `test_planning_metrics.py` |
| COG-LC-S4 | **Classifier failure emission** — `CLASSIFIER_*` in runtime trace metadata | **Done** | High | `llm_task_classifier.py`, `planning_runner.py` | `test_llm_task_classifier.py` |
| COG-LC-S5 | **CI bundle** — `check_reasoning_gates.py` in AGENTS.md + `check_audit_ideal_gates.py` | **Done** | Medium | `scripts/` | gate script green |
| COG-LC-S6 | **LLM classifier registry prompt** — `nexus_task_classifier` prompt asset | **Done** | Medium | `prompts/nexus_task_classifier/`, `nexus_classifier_prompts.py` | `check_reasoning_gates.py` |

### 6.1av Harness implementation queue — Reasoning audit maintenance (planned)

**Source:** Layer 5 audit (2026-06-18) — `REASONING_AND_COGNITION` layer 7 · [`guides/audit/results/2026-06-18/REASONING_AND_COGNITION.md`](guides/audit/results/2026-06-18/REASONING_AND_COGNITION.md)  
**Priority ladder:** **Band 1** (§6.1) — doc/gate hygiene; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **COG-MAINT-01** | Code/Docs | P2 | **Done** | Align §17 failure taxonomy with `ReasoningFailureKind` — canonical mapping table in architecture §17 | Trace payloads match §17 codes |
| 2 | **COG-MAINT-02** | CI | P2 | **Done** | Extend `check_reasoning_gates.py` for SYS-INV-22 plane-separation import boundaries | Gate fails on forbidden cross-plane imports in hot paths |
| 3 | **COG-MAINT-03** | Test | P3 | **Done** | Acceptance test: `allow_dynamic_replan` replan after policy interrupt on reference host | E2E replan boundary proven |

**Suggested PR order:** COG-MAINT-01 → COG-MAINT-02 → COG-MAINT-03.

**Explicitly excluded:** L4 adaptive planner selection (AHI scope) — observe-only default in canon §21.

---

*End of Reasoning and Cognition Implementation Plan.*

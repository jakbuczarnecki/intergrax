# Nexus Execution Flow — Implementation Plan

**Architecture (1:1):** [`architecture/NEXUS_EXECUTION_FLOW.md`](../../architecture/NEXUS_EXECUTION_FLOW.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (NEXUS_EXECUTION_FLOW plan).

- **Implement / audit default:** §6.1 FLOW maintenance · open P0/P1 rows · Phase AUDIT-IDEAL gap table. Historical flow registers — [`plan/satellites`](plan/satellites) satellite on demand
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/NEXUS_EXECUTION_FLOW.md`](../../architecture/NEXUS_EXECUTION_FLOW.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/NEXUS_EXECUTION_FLOW.md`](../../technical/guides/audit_slices/NEXUS_EXECUTION_FLOW.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/NEXUS_EXECUTION_FLOW_appendices.md`](plan/NEXUS_EXECUTION_FLOW_appendices.md) | appendices |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §5, §6 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-8.1 | §8 Runtime | Long-running workflow resume E2E on product hosts | P2 | **Done** |
| AUDIT-IDEAL-8.2 | §8 Runtime | Checkpoint introspection API for ops (beyond lab) | P2 | **Done** |
| AUDIT-IDEAL-10.1 | §10 Subagents | Evaluator-loop standard node in product graph specs | P2 | **Done** |
| AUDIT-IDEAL-10.2 | §10 Subagents | Budget delegation enforcement on all delegation paths | P2 | **Done** |
| AUDIT-IDEAL-6.6 | §6 LLM (shared) | ACP `StepLLMRouter` backed by `LLMAdapter` | P1 | **Done** — [M-LLM-X.5](plan/LLM_ADAPTERS.md) · LC-3 |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

### 6.1aj Harness implementation queue — Nexus execution depth (closed)

**Purpose:** Single ordered list for **Phase FLOW** (Band 2aj). **Closed 2026-06-09** — **18/18 harness Done** (FLOW-8 harness ORCH-CONFIG.5); product host **Deferred** §6.3. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **FLOW-2** | Code | **Done** | ADR-FLOW-001 — `DELEGATES_TO` → child node | Delegation integration tests |
| 2 | **FLOW-14** | Code | **Done** | `SubtaskContract` in delegation expansion | Scopes on child `DelegationSpec` |
| 3 | **FLOW-3** | Code | **Done** | `max_delegation_depth` enforcement | Depth limit test |
| 4 | **FLOW-15** | Code | **Done** | Subagent budget envelope | Child budget exceeded → fail |
| 5 | **FLOW-6** | Code | **Done** | Graph cycle detection | Cyclic graph fails fast |
| 6 | **FLOW-1** | Code | **Done** | Real `EngineBackedNexusPlanner` | LLM plan parse tests |
| 7 | **FLOW-4** | Code | **Done** | Run-level retry profile field | Graph retry integration |
| 8 | **FLOW-13** | Code | **Done** | `max_inflight_nodes` profile + wire | Backpressure event test |
| 9 | **FLOW-7** | Code | **Done** | `MergePolicy` / composer profile | Multi-agent merge tests |
| 10 | **FLOW-9** | Code | **Done** | Multi-agent eval hooks | Registry observation |
| 11 | **FLOW-11** | Code | **Done** | Pre-plan policy hooks | Planning boundary tests |
| 12 | **FLOW-5** | Code | **Done** | `AgentGraph.on_error` wire | Integration test |
| 13 | **FLOW-10** | Code/Docs | **Done** | Reserved lifecycle ADR ([ADR-FLOW-002](adr/entries/2026-06-07/ADR-FLOW-002.md)) | Lifecycle doc |
| 14 | **FLOW-12** | Code | **Done** | `DecisionRecord` regression gate | Gate test per step |
| 15 | **FLOW-16** | Docs | **Done** | `MODIFY_PLAN` ADR (ADR-FLOW-003) | ADR accepted |
| 16 | **FLOW-17** | Code | **Done** | `MULTI_AGENT` ordering policy | Stable order gate test |
| 17 | **FLOW-DOC.*** | Docs | **Done** | Flow reference + Appendix N paydown | Zero open FLOW-GAP |
| 18 | **FLOW-DOC.2** | Docs | **Done** | §3.1 interaction scenario table | Cross-ref TIER3 §23, ORCH §55 |
| — | **FLOW-8** | Harness + Product | **Partial** | Harness: `test_orchestration_cfg_simulation.py` · Product §42.43: **§6.3** |

**Suggested PR order:** See [Phase FLOW — Suggested PR order](.#flow--suggested-pr-order).

**Explicitly excluded:** K.1, K.2 (unless FLOW-8 activated), nested harness per child.

### 6.1av Harness implementation queue — Nexus execution flow audit maintenance (closed)

**Source:** Layer 4 audit (2026-06-18) — `NEXUS_EXECUTION_FLOW` layers 8–10 · [`../audit_results/2026-06-18/NEXUS_EXECUTION_FLOW.md`](../../../audit_results/2026-06-18/NEXUS_EXECUTION_FLOW.md)
**Priority ladder:** **Band 1** (§6.1) — incremental after gate maintenance; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **FLOW-MAINT-01** | Code | P2 | **Done** | Wire `ResiliencePolicy.allow_partial_result` into `graph_runner` lifecycle transitions | When `False`, non-all-completed multi-node graph → `FAILED` not `PARTIALLY_COMPLETED`; unit/integration test |
| 2 | **FLOW-MAINT-02** | Docs | P3 | **Done** | Production-ready checklist in architecture §1.4 (strict profile + W-OPS SLO + reference host presets) | Operator runbook cross-ref; no new mechanisms |
| 3 | **FLOW-MAINT-03** | Test/CI | P3 | **Done** | Windows acceptance teardown guard for `signals.db` lock flake | `tests/acceptance/agent_os` stable on Windows CI |
| 4 | **FLOW-MAINT-04** | Test | P3 | **Done** | Bootstrap fail-fast test when engine planner path lacks `llm_adapter` | `test_orchestration_wiring.py::test_engine_planner_requires_llm_adapter` |

**Suggested PR order:** none — §6.1av queue closed (2026-06-18).

**Explicitly excluded:** UC-6 production research agents; FLOW-8 / FLOW-GAP-20 product hosts — [§6.3](PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

### 6.1aw Harness implementation queue — Nexus execution flow audit maintenance (2026-06-19)

**Source:** Interactive layer audit (2026-06-19) — `NEXUS_EXECUTION_FLOW` layers 8–10 · [`../audit_results/2026-06-19/NEXUS_EXECUTION_FLOW.md`](../../../audit_results/2026-06-19/NEXUS_EXECUTION_FLOW.md) · prior: [`../audit_results/2026-06-18/NEXUS_EXECUTION_FLOW.md`](../../../audit_results/2026-06-18/NEXUS_EXECUTION_FLOW.md)
**Priority ladder:** **Band 1** (§6.1) — test depth + doc sync + audit artifact; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **FLOW-MAINT-05** | Test | P3 | **Done** | Lifecycle regression: multi-node partial graph + `allow_partial_result=False` → `TaskState.FAILED`; `True` → `PARTIALLY_COMPLETED` | `test_graph_runner_resilience.py`; gate green |
| 2 | **FLOW-MAINT-DOC-01** | Docs | P3 | **Done** | Close §6.1av header; sync architecture §1.4 partial-results test row with FLOW-MAINT-05 | Canon matches test evidence |
| 3 | **FLOW-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/2026-06-19` | `NEXUS_EXECUTION_FLOW.md` + `progress.json`; L3 verdict layers 8–10 |

**Suggested PR order:** none — §6.1aw queue closed (2026-06-19).

**Explicitly excluded:** UC-6 production research agents; FLOW-8 / FLOW-GAP-20 product hosts — [§6.3](PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

### 6.1ax Harness implementation queue — Nexus scenario production status (closed)

**Source:** Maturity taxonomy rollout — [`guides/MATURITY_TAXONOMY.md`](../../technical/guides/MATURITY_TAXONOMY.md) · architecture §12.2
**Priority ladder:** **Band 1** (§6.1) — docs only; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **P2-ARCH-04** | Docs | P2 | **Done** | Add Nexus scenario production status matrix (S1–S8) with four-axis A/I/P/E mapping | Architecture §12.2; cross-refs MATURITY_TAXONOMY + SYSTEM_INVARIANTS; legacy §12.1 labels preserved |

**Suggested PR order:** none — §6.1ax queue closed (2026-06-20).

### 6.1ak Harness implementation queue — Critic & Verification Layer (closed)

**Purpose:** Single ordered list for **Phase CRIT-V** (Band 2ak). **Closed 2026-06-08** — CRIT-V-0…7 + **CRIT-V-FOLLOWUP** closeout **Done**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **CRIT-V-0.*** | Docs | **Done** | Architecture RFC + ADR + canon §55 + README | Cross-links resolve |
| 2 | **CRIT-V-1.*** | Code | **Done** | `CriticProfile` + CVL contracts | Unit tests |
| 3 | **CRIT-V-2.*** | Code | **Done** | `eval.judge` + `eval.trajectory` tools | Tool gate tests |
| 4 | **CRIT-V-3.1–3.3** | Code | **Done** | `CriticOrchestrator` + L0/L1 gateways | `test_critic_orchestrator.py` |
| 5 | **CRIT-V-3.4–3.5** | Code | **Done** | Graph partial + final hooks | Integration tests |
| 5 | **CRIT-V-4.*** | Code | **Done** | `EvaluatorLoopExecutor` | Loop budget tests |
| 6 | **CRIT-V-5.*** | Code | **Done** | Semantic `NexusEvalRunner` | Eval integration test |
| 7 | **CRIT-V-6.*** | Code/Docs | **Done** | Tier-3 wiring + Appendix W | CI assembly script |
| 8 | **CRIT-V-7.*** | Code/Docs | **Done** | FAUDIT-EVAL.1 + flow reference sync | Closeout gate green |
| 9 | **CRIT-V-FOLLOWUP** | Code | **Done** | L1 client, L2 HITL, UAEP hook, policy bridge | `test_critic_closeout.py`, gate green |

**Suggested PR order:** See [§6.2ak](.#62ak-phase-crit-v-execution-order-band-2ak--closed).

**Explicitly excluded:** FLOW-8 product app; domain rubric packs in Tier-0; mandatory universal LLM-judge.

---

### 6.2aj Phase FLOW execution order (Band 2aj — closed 2026-06-07)

**Status:** **Done** · register: [Phase FLOW](plan/ORCHESTRATION.md) · queue: [§6.1aj](.#61aj-harness-implementation-queue--nexus-execution-depth-closed)

Work **one FLOW ID per PR**; after each step update FLOW master table + §6.1aj + Appendix N; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | FLOW-2 | Delegation graph expansion (ADR-FLOW-001) | **Critical** | — |
| 2 | FLOW-14 | `SubtaskContract` on expanded child node | High | FLOW-2 |
| 3 | FLOW-3 | `max_delegation_depth` enforcement | High | FLOW-2 |
| 4 | FLOW-15 | Subagent budget envelope | Medium | FLOW-14 |
| 5 | FLOW-6 | Strict graph cycle detection | High | — |
| 6 | FLOW-1 | LLM-backed Nexus planner | High | — (parallel with 5–8 after step 1) |
| 7 | FLOW-4 | Run-level retry profile | Medium | FLOW-2 |
| 8 | FLOW-13 | `max_inflight_nodes` profile wire | Medium | — |
| 9 | FLOW-7 | Merge policy / composer profile | Medium | — |
| 10 | FLOW-9 | Multi-agent evaluation hooks | Medium | FLOW-7 optional |
| 11 | FLOW-11 | Pre-plan policy hooks | Medium | — |
| 12 | FLOW-5 | `AgentGraph.on_error` wire | Low | FLOW-4 optional |
| 13 | FLOW-10 | Reserved lifecycle states ADR | Low | — |
| 14 | FLOW-12 | `DecisionRecord` regression gate | Medium | — |
| 15 | FLOW-16 | `MODIFY_PLAN` ADR (ADR-FLOW-003) | Low | — |
| 16 | FLOW-17 | `MULTI_AGENT` ordering policy | Low | — |
---

## Cross-domain registers (canonical elsewhere)

| Need | Source |
|------|--------|
| CRIT-V | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |
| ORCH | [`ORCHESTRATION.md`](ORCHESTRATION.md) |
| Platform §6 | [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) |

---
## Phase FLOW — Nexus execution depth

**Status:** **Done** (2026-06-07) — **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3 §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **maintenance only** — §6.1 gate (Band 3 §6.3 frozen)  
**Execution order:** [§6.2aj](.#62aj-phase-flow-execution-order-band-2aj--active) · queue: [§6.1aj](.#61aj-harness-implementation-queue--nexus-execution-depth-closed)
**Traceability:** **Appendix N (FLOW)** — [`§Appendix N`](.#appendix-n--nexus-execution-flow-traceability-phase-flow)

**Delivery rule:** One **FLOW-*** ID per PR → update master table + §6.1aj + Appendix N paydown → `pytest -m gate` + §6.1 scripts green.

**Maturity target (phase complete):**

| AUDIT_MAP § | Baseline (FAUDIT-32) | Target after FLOW |
|-------------|----------------------|-------------------|
| §5 Policy (pre-plan hooks) | L2 partial | **L3** (FLOW-11) |
| §7 Reasoning / planning | L2 | **L3** (FLOW-1, FLOW-12) |
| §8 Execution runtime | L3 | **L3** (FLOW-10, maintain) |
| §9 Orchestration / graph | L3 partial | **L3+** (FLOW-4–7, FLOW-6, FLOW-13, FLOW-16) |
| §10 Subagents | L2 | **L3** (FLOW-2, FLOW-3, FLOW-14, FLOW-15) |
| §25 Evaluation | L2 | **L3** (FLOW-9) |

**Explicitly out of scope:** Nested full harness per child; new graph node **types** (Tier-1 canon change); K.1/K.2 business agents (FLOW-8 → §6.3 unless reprioritized).

### FLOW — Master register

| ID | Wave | Gap | Deliverable | Status | Priority | Module / test | Acceptance |
|----|------|-----|-------------|--------|----------|---------------|------------|
| FLOW-DOC.1 | FLOW0 | — | **Flow reference sync** — paydown §23 gaps in `architecture/NEXUS_EXECUTION_FLOW.md` after each FLOW PR | **Done** | Low | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | No stale `FLOW-GAP` rows for Done IDs |
| FLOW-2 | FLOW1 | FLOW-GAP-02 | **ADR-FLOW-001 implementation** — expand `DELEGATES_TO` to child `PlanStep` + `ExecutionNode`; `DelegationSpec` on **child**; `GraphExecutor` routes `child_agent_id` | **Done** | **Critical** | `graph_spec_to_plan.py`, `graph_builder.py`, `graph_executor.py` | `test_graph_spec_to_plan.py` + integration delegation path; canon §42.14.3 note updated |
| FLOW-3 | FLOW1 | FLOW-GAP-03 | **`max_delegation_depth` enforcement** — count expanded delegation chain in `GraphExecutor`; fail with trace | **Done** | High | `graph_executor.py`, `environment_profile.py` | Unit test depth exceeded |
| FLOW-1 | FLOW2 | FLOW-GAP-01 | **Real `EngineBackedNexusPlanner`** — bridge `engine_planner_orchestrator` → `NexusTaskPlannerProtocol`; typed `NexusPlan` from LLM parse | **Done** | High | `orchestration_wiring.py`, `planning/engine_planner_*.py` | `test_orchestration_wiring.py` + planner integration tests |
| FLOW-6 | FLOW2 | FLOW-GAP-06 | **Strict cycle detection** — `ExecutionGraph.batches()` raises on cycle; no unsafe fallback | **Done** | High | `execution_graph.py` | Unit test cyclic graph → error |
| FLOW-4 | FLOW3 | FLOW-GAP-04 | **Opt-in run-level retry** — `OrchestrationProfile.max_run_retries`; wire `RetryCoordinator` in `NexusGraphRunner` | **Done** | Medium | `environment_profile.py`, `graph_runner.py`, `nexus_factory.py` | Integration test graph retry once |
| FLOW-7 | FLOW3 | FLOW-GAP-07 | **`MergePolicy` / `FinalResponseComposerProfile`** — deterministic + structured merge; optional LLM merge hook (policy-gated) | **Done** | Medium | `final_response_composer.py`, `environment_profile.py` | Multi-agent merge unit tests |
| FLOW-9 | FLOW3 | FLOW-GAP-11 | **Evaluation hooks on multi-agent fan-in** — post-graph eval observation; evaluator-node cookbook; registry write on multi-node runs | **Done** | Medium | `nexus_loop.py`, `evaluation_wiring.py`, docs §18 | `EvaluationProfile` observation recorded; guide §18 |
| FLOW-11 | FLOW3 | FLOW-GAP-09 | **Pre-plan / pre-LLM policy extension points** — document + wire hooks at planning boundary | **Done** | Medium | `planning_runner.py`, `policy_engine.py` | Hook tests + Appendix H cross-ref |
| FLOW-5 | FLOW4 | FLOW-GAP-05 | **`AgentGraph.on_error(retry)`** — wire to `RetryPolicy` / graph executor | **Done** | Low | `graph_builder.py`, `orchestration_wiring.py` | Integration test declared retry |
| FLOW-10 | FLOW4 | FLOW-GAP-08 | **Reserved lifecycle states** — ADR: implement `WAITING_FOR_RESOURCES`/`EXPIRED` **or** trim enum + canon sync | **Done** | Low | `task_lifecycle.py`, `adr/entries/2026-06-07/ADR-FLOW-002.md` | [ADR-FLOW-002](adr/entries/2026-06-07/ADR-FLOW-002.md) accepted; reserved v1 semantics |
| FLOW-12 | FLOW4 | §24 / FAUDIT-COG | **`DecisionRecord` regression gate** — verify FAUDIT-COG.1 emit on every UAEP decision path; gate test; sync flow §24 | **Done** | Medium | `uaep.py`, `tests/integration/agents` | `DECISION_EMITTED` + `decision_record` on each step decision |
| FLOW-13 | FLOW4 | FLOW-GAP-12 | **`max_inflight_nodes` profile + wire** — field on `OrchestrationProfile`; `resolve_max_inflight_nodes()`; `nexus_factory` → `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `orchestration_wiring.py`, `nexus_factory.py` | `GRAPH_BACKPRESSURE` event when cap hit; profile round-trip test |
| FLOW-14 | FLOW4 | FLOW-GAP-13 | **`SubtaskContract` in delegation expansion** — `graph_spec_to_plan` / ADR-FLOW-001 child node uses `SubtaskContract.to_delegation_spec()` (`objective`, `permission_scopes`, `inherit_tool_policy=False`) | **Done** | Medium | `graph_spec_to_plan.py`, `subtask_contract.py` | Unit test scopes + objective on child `DelegationSpec` |
| FLOW-15 | FLOW4 | FLOW-GAP-14 | **Subagent budget envelope** — optional `budget_envelope` on `SubtaskContract` / `DelegationSpec`; enforce in child `GraphExecutor` run via existing budget bridge | **Done** | Medium | `subtask_contract.py`, `delegation.py`, `graph_executor.py` | Child run exceeds envelope → fail with trace |
| FLOW-16 | FLOW4 | FLOW-GAP-15 | **`MODIFY_PLAN` ADR** — [ADR-FLOW-003](adr/entries/2026-06-07/ADR-FLOW-003.md): document reserved semantics (policy-gated replan hook) **or** trim `AgentDecision` enum | **Done** | Low | `adr/entries/2026-06-07/ADR-FLOW-003.md`, `interrupts/handler.py` | ADR accepted; `MODIFY_PLAN_NOT_SUPPORTED` when no handoff |
| FLOW-17 | FLOW4 | FLOW-GAP-16 | **`MULTI_AGENT` ordering policy** — `OrchestrationProfile.multi_agent_order` (`registry` \| `priority` \| `stable_alpha`); deterministic step order in `TaskPlanner` | **Done** | Low | `environment_profile.py`, `task_planner.py` | Gate test: two agents same capability → stable declared order |
| FLOW-8 | FLOW5 | FLOW-GAP-10 | **Harness CFG simulation** (ORCH-CONFIG.5) + optional Tier-3 §42.43 product host | **Partial** | Harness + Product | `tests/integration/runtime/test_orchestration_cfg_simulation.py` · product §6.3 gate |
| FLOW-DOC.2 | FLOW5 | — | **Phase closeout** — Appendix N (FLOW), flow reference §23 paydown (all gaps), maturity dashboard §0.5 | **Done** | Low | `docs/*` | All non-deferred FLOW rows **Done**; zero open `FLOW-GAP` in §23 |

### FLOW — Suggested PR order

```text
FLOW-2 → FLOW-14 → FLOW-3 → FLOW-15 → FLOW-6 → FLOW-1 → FLOW-4 → FLOW-13 → FLOW-7 → FLOW-9 → FLOW-11 → FLOW-5 → FLOW-10 → FLOW-12 → FLOW-16 → FLOW-17 → FLOW-DOC.*
```

**Parallel OK after FLOW-2:** FLOW-1, FLOW-6, FLOW-13 (disjoint modules). **FLOW-14** same PR as FLOW-2 or immediately after.

**FLOW-8:** Schedule only after explicit product decision ([§6.3](.#63-end-of-plan--deferred-product-work-only)).

### FLOW — Paydown log

| Date | FLOW ID | Summary |
|------|---------|---------|
| 2026-06-07 | — | Phase FLOW scheduled from `architecture/NEXUS_EXECUTION_FLOW.md` §25; queue §6.1aj; Appendix N (FLOW) |
| 2026-06-07 | — | Audit gap closeout: FLOW-13–17 + FLOW-GAP-12–16 added; FLOW-12 narrowed to regression gate; **0/18** |
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Phase FLOW implementation complete: delegation expansion, graph hardening, profile wiring, ADR-FLOW-002/003; gate **906 passed**; **18/18 harness** (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3) |

**Phase FLOW complete when:** FLOW-1–7, FLOW-9, FLOW-11–17, FLOW-DOC.* **Done**; FLOW-8 **Deferred** or Done per product; §6.1aj closed; **zero open `FLOW-GAP-*`** in flow reference §23; AUDIT_MAP §5/§7/§9/§10/§25 at target maturity; gate green.

---

## Phase FLOW-CTL — Interrupt, cancel, and resume hardening (Band 2av — planned)

**Status:** **Done** (2026-06-09) — architecture canon §28; FLOW-CTL.1–5 implemented on harness paths.

**Goal:** Formalize **interrupt anywhere** and **resume from checkpoint** guarantees — cooperative cancel in long UAEP loops, unified operator API, trace coverage.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| FLOW-CTL-DOC.1 | Canon §28 + UAEP cross-ref | **Done** | High | ORCHE §58 index |
| FLOW-CTL.1 | **Cooperative cancel gate** — `RuntimeExecutionContext.should_cancel()` + active task registry | **Done** | Medium | `active_task_registry.py`, UAEP |
| FLOW-CTL.2 | **Unified cancel API** — `POST /v1/tasks/{id}/cancel` on lab harness host | **Done** | High | `harness_task_routes.py` |
| FLOW-CTL.3 | **Mid-step interrupt budget** — `max_interrupts_per_run` on governance options | **Done** | Low | `ExecutionInterruptHandler` |
| FLOW-CTL.4 | **Resume API parity** — `POST /v1/tasks/{id}/resume` with checkpoint store | **Done** | Medium | `harness_task_routes.py` |
| FLOW-CTL.5 | **Trace completeness** — new governance events in phase coverage | **Done** | Medium | `phase_coverage.py` + B07 gate |
| FLOW-CTL.6 | **Product host parity** — task control routes on scaffold-opt-in hosts | **Done** | High | H-APP-WIRING.1 **Done**; closes FLOW-GAP-17 |

**Prerequisites:** Phase FLOW **Done**; REL checkpoint store **Done**.

**Cross-plan:** FLOW-CTL.2 ↔ REL-ADV autonomy downgrade; FLOW-CTL.4 ↔ ORCH-6 async posture; FLOW-CTL.6 ↔ H-APP-WIRING.

**Audit note (2026-06-09, synced):** FLOW-CTL **Done**; FLOW-GAP-17–19 **Closed** on reference hosts (H-APP-WIRING); FLOW-GAP-20 **Deferred** §6.3.

**Explicitly excluded:** Distributed transaction rollback across external systems (use idempotency + compensation patterns in REL §34).

---

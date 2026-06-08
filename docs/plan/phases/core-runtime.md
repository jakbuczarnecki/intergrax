# Implementation Phases — Core Runtime

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase V-REM — Phase V Runtime Remediation (audit closeout)

**Source:** Plan/code audit (2026-06-05) — reconcile Phase V **Done** claims vs runtime evidence; aligned with [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) layers 5, 19, 21, 23, 25, 26.  
**Status:** **Done** (2026-06-05) — **10/10 Done**.  
**Prerequisites:** Phase V contracts **Done**; Phase H-APP **Done** (Tier-3 `ApplicationSecurityProfile` hooks exist).  
**Goal:** Close every **Partial** Phase V row and **A.4** EvalRunner gap — move from governance/evidence-only to **runtime-enforced** behavior. **Achieved 2026-06-05.**  
**Priority ladder:** **Band 2i** (§4.0) — closed.  
**Execution order:** [§6.2v](#62v-phase-v-rem-execution-order-band-2i--closed-2026-06-05).  
**Traceability:** [Appendix J](#appendix-j--phase-v-remediation-traceability-audit-gap--v-rem-id).

**Explicitly out of scope:** K.1/K.2, new product Tier-3 apps, full 32-layer re-audit (use [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) separately).

**Delivery rule:** One `V-REM.*` ID per PR → update master table + Appendix J + paydown log → `pytest -m gate` + relevant architecture scripts green.

### V-REM — Traceability (audit gap → task ID)

| Parent ID | Gap summary | V-REM ID |
|-----------|-------------|----------|
| V-CG.2–V-CG.4 | Incorrect system edge mapping agents→application breaks lineage/impact/CI | V-REM-CG.1, V-REM-CG.2 |
| V-ALG.3 | No runtime cutoff for retired/deprecated agents | V-REM-ALG.1 |
| V-ALG.4 | No production-eligible-only filter at selection | V-REM-ALG.2 |
| V-PE.1 | PromptMeta missing owner/risk; no YAML prompt assets | V-REM-PE.1, V-REM-PE.2 |
| V-SEC.2 | Tool injection defense not wired in execution path | V-REM-SEC.1 |
| V-SEC.3 | Retrieval poisoning defense not enforced per tenant/app | V-REM-SEC.2 |
| V-SEC.4 | Tenant isolation + audit trail hooks missing in main path | V-REM-SEC.3 |
| A.4 / A.4.1 | NexusEvalRunner missing integration tests + gate | V-REM-A.1 |

### V-REM — Master deliverables register

| ID | Stream | Deliverable | Status | Priority | Closes | Acceptance |
|----|--------|-------------|--------|----------|--------|------------|
| V-REM.0.1 | Governance | **Appendix J** — audit gap → V-REM ID matrix (100% mapped) | **Done** | Critical | — | Every Partial row has V-REM ID |
| V-REM.0.2 | Governance | Sync Phase V header, §0.5, §4.0 Band 2i, Appendix H, §6.1z | **Done** | High | — | No Phase V domain row marked **Done** while child Partial |
| V-REM-CG.1 | V-CG | **Fix system edge mapping** — per-application agents→application edges from manifest/roster (not global cross-product) | **Done** | **Critical** | V-CG.2–4 | Unit: lab/legal/poc graphs have correct agent-application edges |
| V-REM-CG.2 | V-CG | **Re-run graph guard** — lineage, impact, compatibility on corrected mapping; update `phase_v_capability_graph_guard.py` fixtures | **Done** | High | V-CG.2–4 | CI guard green; impact blast radius matches expected for sample change |
| V-REM-ALG.1 | V-ALG | **Runtime lifecycle filter** — AgentRegistry / NexusLoop reject or reroute retired/deprecated agents | **Done** | High | V-ALG.3 | Unit tests: deprecated agent not selected for new runs |
| V-REM-ALG.2 | V-ALG | **Production-eligible gate** — discovery/selection requires owner + certification metadata for production mode | **Done** | High | V-ALG.4 | Test: agent without owner blocked in strict/production profile |
| V-REM-PE.1 | V-PE | **Extend PromptMeta / YamlPromptRegistry** — add `owner`, `risk_tier`, version governance fields + validation | **Done** | High | V-PE.1 | Schema round-trip + registry validation tests |
| V-REM-PE.2 | V-PE | **Seed YAML prompt assets catalog** — minimal harness reference prompts under versioned assets path | **Done** | Medium | V-PE.1 | E2E governance validation passes on `harness_capability_summary` |
| V-REM-SEC.1 | V-SEC | **Wire tool injection defense** — `ApplicationSecurityProfile` → ToolRuntime / pre-tool hook on main execution path | **Done** | High | V-SEC.2 | Unit: dangerous payload denied via middleware |
| V-REM-SEC.2 | V-SEC | **Wire retrieval poisoning defense** — per-tenant/app middleware on RAG retrieval path | **Done** | High | V-SEC.3 | Unit: quarantine/trust score filters retrieval chunks |
| V-REM-SEC.3 | V-SEC | **Wire tenant isolation + audit trail** — enforcement + security audit events in UnifiedTaskRunner/NexusLoop | **Done** | High | V-SEC.4 | Unit: tenant boundary violation blocked at intake |
| V-REM-A.1 | Phase A | **NexusEvalRunner integration tests + gate** — NexusLoop→UnifiedTaskRunner→EvalRunner path | **Done** | Medium | A.4, A.4.1 | Gate tests in `tests/integration/eval/test_nexus_eval_runner.py` |

```text
Wave V-REM-0 (governance):  V-REM.0.1 -> V-REM.0.2  — Done (plan sync)
Wave V-REM-1 (graph):       V-REM-CG.1 -> V-REM-CG.2  — Done (2026-06-05)
Wave V-REM-2 (lifecycle):   V-REM-ALG.1 -> V-REM-ALG.2  — Done (2026-06-05)
Wave V-REM-3 (prompt):      V-REM-PE.1 -> V-REM-PE.2  — Done (2026-06-05)
Wave V-REM-4 (security):    V-REM-SEC.1 -> V-REM-SEC.2 -> V-REM-SEC.3  — Done (2026-06-05)
Wave V-REM-5 (eval):        V-REM-A.1  — Done (2026-06-05)
```

**Phase V-REM complete when:** All rows **Done**; parent V-CG.2–4, V-ALG.3–4, V-PE.1, V-SEC.2–4, A.4 marked **Done**; Appendix H rows updated; §6.1z queue closed. **Status: complete (2026-06-05).**

### V-REM — Paydown log

| Date | V-REM ID | Summary |
|------|----------|---------|
| 2026-06-05 | V-REM.0.1, V-REM.0.2 | Audit → plan: Phase V-REM register, Appendix J, §6.1z queue, status sync |
| 2026-06-05 | V-REM-CG.1–A.1 | Runtime remediation: capability graph edges, lifecycle routing, V-SEC wiring, prompt governance, NexusEvalRunner gate |

---

## Phase ORCH — Orchestration control plane closeout

**Status:** **Done** (2026-06-05) — **6/6** deliverables Done (ORCH-DOC.* + ORCH-1–4); gate **581 passed**  
**Prerequisites:** R-Delegate **Done**, Q+-N.* runners **Done**, H-APP.3.1–3.2 **Done**, V-MA.* **Done**  
**Goal:** Close orchestration audit residuals (AUDIT_MAP §7–§10) — wire declared Tier-3 profile fields to runtime; bridge declarative graph spec to execution plan; cap graph batch concurrency.  
**Priority ladder:** **Band 2j** (§4.0) — **default implementation queue** after §6.1 gate on each PR.  
**Execution order:** [§6.2bb](#62bb-phase-orch-execution-order-band-2j--active) · queue: [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-active)  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane)

**Delivery rule:** One **ORCH-*** ID per PR → update master table + §6.1b + paydown log below → `pytest -m gate` + §6.1 scripts green.

**Audit verdict (baseline — preserve as acceptance context):**

| Area | Maturity (L0–L4) | Residual before ORCH | Close via |
|------|------------------|----------------------|-----------|
| Nexus stack (§8) | **L3–L4** | — | ORCH-DOC.* (documented) |
| Planning strategies (§7) | **L3–L4** | — | ORCH-1 **Done** |
| Declarative graph (§9) | **L3–L4** | — | ORCH-2 **Done** |
| Graph concurrency (§9) | **L3** | — | ORCH-3 **Done** |
| Subagent delegation (§10) | **L3–L4** | — | R-Delegate (Done) |

### ORCH — Master register

| ID | Wave | Deliverable | Status | Priority | Module / test | Acceptance |
|----|------|-------------|--------|----------|---------------|------------|
| ORCH-DOC.1 | ORCH0 | **Appendix I** — orchestration control plane map (§I.1–I.10) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| ORCH-DOC.2 | ORCH0 | **Cross-ref sync** — plan, README, strategy, AUDIT_MAP §7–§10, audit prompt ref #6, canon §42.43 | **Done** | Medium | `docs/*` | Links resolve |
| ORCH-1 | ORCH1 | **Wire `planner_kind` / `classifier_kind`** — registry maps kinds → `TaskPlanner` / `ClassifyingTaskClassifier`; `build_nexus_loop_from_environment` passes resolved instances to `NexusLoop` | **Done** | **Critical** | `orchestration_wiring.py`, `nexus_factory.py` | `test_orchestration_wiring.py` |
| ORCH-2 | ORCH2 | **`ApplicationGraphSpec` → `NexusPlan` seed** — `graph_spec_to_plan.py` + `GraphSpecSeedingPlanner` when task has no plan id | **Done** | **High** | `graph_spec_to_plan.py`, `PlanStep.delegation` | `test_graph_spec_to_plan.py`, `test_lab_graph_spec.py` |
| ORCH-3 | ORCH3 | **`max_parallel_nodes` on `OrchestrationProfile`** — cap concurrent nodes per graph batch in `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `graph_executor.py` | `test_graph_executor_parallel_cap.py` |
| ORCH-4 | ORCH4 | **Docs closeout** — Appendix I + plan sync | **Done** | Low | `docs/*` | No “planned wiring” residuals |

**Supported `planner_kind` values (ORCH-1 contract):**

| Kind | Implementation | Notes |
|------|----------------|-------|
| `null` / `default` | `TaskPlanner()` | Current harness default |
| `engine` | `EnginePlanner` adapter implementing plan contract | Requires `RuntimeConfig` on build context — lab/legal hosts only in v1 |
| Unknown kind | — | **Fail fast** at Nexus bootstrap with typed error (no silent fallback) |

**Supported `classifier_kind` values (ORCH-1 contract):**

| Kind | Implementation |
|------|----------------|
| `null` / `default` | `ClassifyingTaskClassifier(registry)` |

**Explicitly out of scope:** Nested full harness per child (use R-Delegate); new graph node types (Tier-1 canon change); product-specific orchestration in `agents/`.

### ORCH — Paydown log

| Date | ORCH ID | Summary |
|------|---------|---------|
| 2026-06-05 | ORCH-DOC.1, ORCH-DOC.2 | Governance + orchestration audit docs; Appendix H/I; AUDIT_MAP cross-refs |
| 2026-06-05 | ORCH-1, ORCH-2, ORCH-3 | Orchestration wiring, graph spec plan seed, parallel cap; gate **581** |
| 2026-06-05 | ORCH-4 | Plan + author guide closeout |

**Phase ORCH complete when:** ORCH-1–4 **Done**; §6.1b queue closed; Appendix I has no “planned wiring” gaps; gate **581** green. **Status: complete (2026-06-05).**

---

## Phase FLOW — Nexus execution depth

**Status:** **Done** (2026-06-07) — **17/18** deliverables Done (**FLOW-8 Deferred** §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **recommended next harness band** after §6.1 gate (before §6.3 product)  
**Execution order:** [§6.2aj](#62aj-phase-flow-execution-order-band-2aj--active) · queue: [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed)  
**Traceability:** **Appendix N (FLOW)** — [`§Appendix N`](#appendix-n--nexus-execution-flow-traceability-phase-flow)

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
| FLOW-DOC.1 | FLOW0 | — | **Flow reference sync** — paydown §23 gaps in `architecture/NEXUS_EXECUTION_FLOW.md` after each FLOW PR | **Done** | Low | `docs/architecture/NEXUS_EXECUTION_FLOW.md` | No stale `FLOW-GAP` rows for Done IDs |
| FLOW-2 | FLOW1 | FLOW-GAP-02 | **ADR-FLOW-001 implementation** — expand `DELEGATES_TO` to child `PlanStep` + `ExecutionNode`; `DelegationSpec` on **child**; `GraphExecutor` routes `child_agent_id` | **Done** | **Critical** | `graph_spec_to_plan.py`, `graph_builder.py`, `graph_executor.py` | `test_graph_spec_to_plan.py` + integration delegation path; canon §42.14.3 note updated |
| FLOW-3 | FLOW1 | FLOW-GAP-03 | **`max_delegation_depth` enforcement** — count expanded delegation chain in `GraphExecutor`; fail with trace | **Done** | High | `graph_executor.py`, `environment_profile.py` | Unit test depth exceeded |
| FLOW-1 | FLOW2 | FLOW-GAP-01 | **Real `EngineBackedNexusPlanner`** — bridge `engine_planner_orchestrator` → `NexusTaskPlannerProtocol`; typed `NexusPlan` from LLM parse | **Done** | High | `orchestration_wiring.py`, `planning/engine_planner_*.py` | `test_orchestration_wiring.py` + planner integration tests |
| FLOW-6 | FLOW2 | FLOW-GAP-06 | **Strict cycle detection** — `ExecutionGraph.batches()` raises on cycle; no unsafe fallback | **Done** | High | `execution_graph.py` | Unit test cyclic graph → error |
| FLOW-4 | FLOW3 | FLOW-GAP-04 | **Opt-in run-level retry** — `OrchestrationProfile.max_run_retries`; wire `RetryCoordinator` in `NexusGraphRunner` | **Done** | Medium | `environment_profile.py`, `graph_runner.py`, `nexus_factory.py` | Integration test graph retry once |
| FLOW-7 | FLOW3 | FLOW-GAP-07 | **`MergePolicy` / `FinalResponseComposerProfile`** — deterministic + structured merge; optional LLM merge hook (policy-gated) | **Done** | Medium | `final_response_composer.py`, `environment_profile.py` | Multi-agent merge unit tests |
| FLOW-9 | FLOW3 | FLOW-GAP-11 | **Evaluation hooks on multi-agent fan-in** — post-graph eval observation; evaluator-node cookbook; registry write on multi-node runs | **Done** | Medium | `nexus_loop.py`, `evaluation_wiring.py`, docs §18 | `EvaluationProfile` observation recorded; guide §18 |
| FLOW-11 | FLOW3 | FLOW-GAP-09 | **Pre-plan / pre-LLM policy extension points** — document + wire hooks at planning boundary | **Done** | Medium | `planning_runner.py`, `policy_engine.py` | Hook tests + Appendix H cross-ref |
| FLOW-5 | FLOW4 | FLOW-GAP-05 | **`AgentGraph.on_error(retry)`** — wire to `RetryPolicy` / graph executor | **Done** | Low | `graph_builder.py`, `orchestration_wiring.py` | Integration test declared retry |
| FLOW-10 | FLOW4 | FLOW-GAP-08 | **Reserved lifecycle states** — ADR: implement `WAITING_FOR_RESOURCES`/`EXPIRED` **or** trim enum + canon sync | **Done** | Low | `task_lifecycle.py`, `adr/ADR-FLOW-002.md` | [ADR-FLOW-002](adr/ADR-FLOW-002.md) accepted; reserved v1 semantics |
| FLOW-12 | FLOW4 | §24 / FAUDIT-COG | **`DecisionRecord` regression gate** — verify FAUDIT-COG.1 emit on every UAEP decision path; gate test; sync flow §24 | **Done** | Medium | `uaep.py`, `tests/integration/agents/` | `DECISION_EMITTED` + `decision_record` on each step decision |
| FLOW-13 | FLOW4 | FLOW-GAP-12 | **`max_inflight_nodes` profile + wire** — field on `OrchestrationProfile`; `resolve_max_inflight_nodes()`; `nexus_factory` → `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `orchestration_wiring.py`, `nexus_factory.py` | `GRAPH_BACKPRESSURE` event when cap hit; profile round-trip test |
| FLOW-14 | FLOW4 | FLOW-GAP-13 | **`SubtaskContract` in delegation expansion** — `graph_spec_to_plan` / ADR-FLOW-001 child node uses `SubtaskContract.to_delegation_spec()` (`objective`, `permission_scopes`, `inherit_tool_policy=False`) | **Done** | Medium | `graph_spec_to_plan.py`, `subtask_contract.py` | Unit test scopes + objective on child `DelegationSpec` |
| FLOW-15 | FLOW4 | FLOW-GAP-14 | **Subagent budget envelope** — optional `budget_envelope` on `SubtaskContract` / `DelegationSpec`; enforce in child `GraphExecutor` run via existing budget bridge | **Done** | Medium | `subtask_contract.py`, `delegation.py`, `graph_executor.py` | Child run exceeds envelope → fail with trace |
| FLOW-16 | FLOW4 | FLOW-GAP-15 | **`MODIFY_PLAN` ADR** — [ADR-FLOW-003](adr/ADR-FLOW-003.md): document reserved semantics (policy-gated replan hook) **or** trim `AgentDecision` enum | **Done** | Low | `adr/ADR-FLOW-003.md`, `interrupts/handler.py` | ADR accepted; `MODIFY_PLAN_NOT_SUPPORTED` when no handoff |
| FLOW-17 | FLOW4 | FLOW-GAP-16 | **`MULTI_AGENT` ordering policy** — `OrchestrationProfile.multi_agent_order` (`registry` \| `priority` \| `stable_alpha`); deterministic step order in `TaskPlanner` | **Done** | Low | `environment_profile.py`, `task_planner.py` | Gate test: two agents same capability → stable declared order |
| FLOW-8 | FLOW5 | FLOW-GAP-10 | **§42.43 reference Tier-3 app** — 3+ agent `graph_spec` demo (PM/UX/Legal pattern) | **Deferred** | Product | `applications/` new host or lab extension | Acceptance multi-agent + HITL path · **§6.3 gate** |
| FLOW-DOC.2 | FLOW5 | — | **Phase closeout** — Appendix N (FLOW), flow reference §23 paydown (all gaps), maturity dashboard §0.5 | **Done** | Low | `docs/*` | All non-deferred FLOW rows **Done**; zero open `FLOW-GAP` in §23 |

### FLOW — Suggested PR order

```text
FLOW-2 → FLOW-14 → FLOW-3 → FLOW-15 → FLOW-6 → FLOW-1 → FLOW-4 → FLOW-13 → FLOW-7 → FLOW-9 → FLOW-11 → FLOW-5 → FLOW-10 → FLOW-12 → FLOW-16 → FLOW-17 → FLOW-DOC.*
```

**Parallel OK after FLOW-2:** FLOW-1, FLOW-6, FLOW-13 (disjoint modules). **FLOW-14** same PR as FLOW-2 or immediately after.

**FLOW-8:** Schedule only after explicit product decision ([§6.3](#63-end-of-plan--deferred-product-work-only)).

### FLOW — Paydown log

| Date | FLOW ID | Summary |
|------|---------|---------|
| 2026-06-07 | — | Phase FLOW scheduled from `architecture/NEXUS_EXECUTION_FLOW.md` §25; queue §6.1aj; Appendix N (FLOW) |
| 2026-06-07 | — | Audit gap closeout: FLOW-13–17 + FLOW-GAP-12–16 added; FLOW-12 narrowed to regression gate; **0/18** |
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Phase FLOW implementation complete: delegation expansion, graph hardening, profile wiring, ADR-FLOW-002/003; gate **906 passed**; **17/18** (**FLOW-8 Deferred**) |

**Phase FLOW complete when:** FLOW-1–7, FLOW-9, FLOW-11–17, FLOW-DOC.* **Done**; FLOW-8 **Deferred** or Done per product; §6.1aj closed; **zero open `FLOW-GAP-*`** in flow reference §23; AUDIT_MAP §5/§7/§9/§10/§25 at target maturity; gate green.

---

## Phase LEG — Legacy tool plan boolean closeout

**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (LEG-1–2); gate **612 passed**

**Audit basis:** Phase O.5a residual; `check_legacy_tool_plan_booleans.py`; Appendix J §J.6.

**Priority ladder:** **Band 2o** (§4.0) — closed; default queue = **§6.1** maintenance.

### LEG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| LEG-1 | LEG1 | **`tool_invocation_plan_from_capability_payload`** — gateway maps booleans → `tool_ids` without `from_legacy` | **Done** | `tool_runtime.py`, `tool_gateway.py` | `test_capability_payload_tool_plan.py` |
| LEG-2 | LEG2 | **Engine planner `tool_ids`** — parser populates `EnginePlan.tool_ids`; schema optional `tool_ids` | **Done** | `engine_planner_parse.py`, `engine_planner_messages.py` | `test_engine_plan_json_parser.py` |
| LEG-3 | LEG3 | **`plan_from_like` canonical path** — `from_tool_ids` only; `tool_gateway` removed from audit grandfather | **Done** | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | audit script green |

**Residual:** `ToolInvocationPlan.from_legacy()` retained in `tool_runtime.py` for explicit deprecation tests only; `EnginePlan.use_rag`/`use_websearch` remain on LLM schema for backward-compatible planner output.

---

## Phase CLEAN — Legacy module closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CLEAN-1–4)

**Audit basis:** Phase U-Leg residual; `scripts/check_legacy_modules_removed.py`; prior `check_tools_agent_*` audits merged.

**Priority ladder:** closeout between Band 2p and 2q; default queue = **Band 2q** [Phase AS](#phase-as--agent-assembly-control-plane-closeout).

### CLEAN — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CLEAN-1 | CLEAN1 | **Remove `legacy/chat_router.py`** — YAML assets tested without runtime module | **Done** | `tests/unit/chat_agent/` | prompt YAML tests green |
| CLEAN-2 | CLEAN2 | **Remove `tools/tools_agent.py`** — `CatalogToolPlanner` + `ToolPlanningService` canonical | **Done** | `catalog_tool_planner.py`, `tool_planning_service.py` | `test_catalog_tool_planner.py` |
| CLEAN-3 | CLEAN3 | **Unified CI audit** — `check_legacy_modules_removed.py` replaces `check_tools_agent_*` | **Done** | `scripts/`, `.github/workflows/unit-tests.yml` | audit script green in CI |
| CLEAN-4 | CLEAN4 | **Docs sync** — plan, HARNESS_ENVIRONMENT, AGENT_CREATION_GUIDE, README, TOOLS | **Done** | `docs/*` | no stale `ToolsAgent` production paths |

**Retained (not CLEAN scope):** `ToolInvocationPlan.from_legacy()` + deprecation tests; `EnginePlan.use_rag`/`use_websearch` on LLM schema; `intergrax/legacy/rag_answers/` archive with import guard; diagnostic type names (`CoreLLMUsedToolsAgentAnswerDiagV1`).

---

## Phase AS — Agent assembly control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (AS-DOC.1 + AS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §18; ideal model §17 in [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](IDEAL_HARNESS_AI_ARCHITECTURE.md); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix N**.

**Priority ladder:** **Band 2q** (§4.0) — closed; default queue = **§6.1** maintenance.

### AS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| AS-DOC.1 | AS0 | **Appendix N** — agent assembly control plane (contract, capabilities, skills, lifecycle) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| AS-1 | AS1 | **`agent_assembly_resolver`** — contract metadata validation at register time | **Done** | `runtime/registry/agent_assembly_resolver.py`, `agent_registry.py` | `test_agent_assembly_resolver.py` |
| AS-2 | AS2 | **Lifecycle metadata enforcement** — `production_eligible` owner/runbook requirements | **Done** | `agent_assembly_resolver.py`, `agent_routing_policy.py` | resolver + routing tests |
| AS-3 | AS3 | **`skill_ids` → `allowed_tools` resolution audit** — CI script + docs cross-ref | **Done** | `scripts/check_agent_skill_resolution.py`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), Legal domain steps, product-only contract variants — [§6.3a](#63a-business-backlog-register-consolidated).

---


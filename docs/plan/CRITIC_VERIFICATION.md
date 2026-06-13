# Critic Verification — Implementation Plan

**Architecture (1:1):** [`architecture/CRITIC_VERIFICATION.md`](../architecture/CRITIC_VERIFICATION.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §18 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-25.1 | §25 Evaluation | Shadow eval path automation (DEBT-25-01) | P1 | Planned |
| AUDIT-IDEAL-25.2 | §25 Evaluation | Human review sample queue (beyond CLI) | P2 | **Done** |
| AUDIT-IDEAL-25.3 | §25 Evaluation | Context/RAG eval blocking product release CI | P1 | Planned |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

(Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---



**Status:** **Done** (2026-06-05) — runtime governance via V-REM, H-APP, DX-5.8; documentation via GOV-DOC.*  
**Prerequisites:** Phase V-REM **Done**, H-APP.2.4–2.8 **Done**, DX-5.8 **Done**  
**Goal:** Close governance/policy/observability audit (AUDIT_MAP §5, §21) with a single authoring map and traceability — **no** new OS features.  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane)

**Delivery rule:** GOV-DOC.* = docs-only PRs; no code unless regression found → route to **REG-*** under §6.1.

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| GOV-DOC.1 | **Appendix H** — control plane map (profiles, bundles, hooks, EP groups, mandatory vs optional observability) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + §H.1–H.8 present |
| GOV-DOC.2 | **Cross-ref sync** — plan Documentation model, README, `guides/HARNESS_ENVIRONMENT.md`, [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11.5, AUDIT_MAP §5/§21, audit prompt ref #5 | **Done** | Medium | `docs/*` | Links resolve; no orphan audit layer |
| GOV-DOC.3 | **`guides/EXTENSION_AUTHOR_GUIDE.md` §10** — `intergrax.policy_rules` author surface | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` | DX-5.8 traceability |
| GOV-PROD.1 | Unified product observability dashboard (beyond lab debug APIs) | **Deferred** | — | — | **§6.3** product decision; optional `observability_backend` remains harness path |

**Explicitly out of scope:** K.1/K.2 policy; product-specific legal/org policy fragments beyond lab reference YAML.

---



**Status:** **Done** (2026-06-06) — 32-layer audit (`scope: C`) + **23/23 FAUDIT remediation** implemented → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed)  
**Source:** [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8  
**Traceability:** **Appendix M** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done. Layer maturity uplift (L2→L3 depth) remains incremental maintenance — see Appendix M.

**Gate evidence (verify step):** `uv run pytest -m gate -q` → **901 passed**; `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py` → **OK**.

### FAUDIT-32 — Layer scorecard (summary)

| # | Layer | Score | Crit | High | Plan accurate? |
|---|-------|-------|------|------|----------------|
| 1 | Strategic Harness Model | L3 | 0 | 0 | Yes |
| 2 | Tier Model and Dependency Boundaries | L2 | **1** | 1 | **Partial** |
| 3 | Interface and Task Intake | L2 | 0 | 2 | Partial |
| 4 | Identity, Trust and Tenancy | L2 | 0 | 2 | Partial |
| 5 | Policy and Governance | L3 | 0 | 2 | Partial |
| 6 | LLM and Model Adapter Layer | L3 | 0 | 1 | Yes |
| 7 | Reasoning, Planning and Cognition | L2 | 0 | 1 | Partial |
| 8 | Execution Runtime and Agent OS | L3 | 0 | 0 | Yes |
| 9 | Orchestration, Scheduler and Execution Graph | L3 | 0 | 1 | Partial |
| 10 | Subagents and Multi-Agent Coordination | L2 | 0 | 2 | Partial |
| 11 | Tool Layer | L3 | 0 | 1 | Yes |
| 12 | Skill Layer | L3 | 0 | 0 | Yes |
| 13 | Integration Layer | L3 | 0 | 0 | Yes |
| 14 | RAG and Retrieval Layer | L3 | 0 | 0 | Yes |
| 15 | Memory Layer | L2 | 0 | 2 | Partial |
| 16 | Context Engineering Layer | L3 | 0 | 0 | Yes |
| 17 | Prompt Engineering and Prompt Registry | L2 | 0 | 1 | **No** |
| 18 | Agent Assembly and Agent Contracts | L2 | 0 | 1 | Yes |
| 19 | Registry Architecture | L2 | 0 | 2 | **No** |
| 20 | Capability Graph Architecture | L2 | 0 | 2 | **No** |
| 21 | Observability and Telemetry | L2 | 0 | 2 | **No** |
| 22 | Error Handling and Reliability | L2 | 0 | 1 | **No** |
| 23 | Security and Data Governance | L2 | 0 | 2 | **No** |
| 24 | Cost and Resource Governance | L2 | 0 | 1 | **No** |
| 25 | Evaluation and Benchmarking | L2 | 0 | 1 | **No** |
| 26 | Testing, CI and Architecture Gates | L3 | 0 | 0 | Yes |
| 27 | Developer Experience, Scaffold and Lab | L3 | 0 | 1 | Yes |
| 28 | Product Environment and Tier-3 Applications | L3 | 0 | 2 | Partial |
| 29 | Modality, Vision, Audio and Dedicated ML | L3 | 0 | 1 | Yes |
| 30 | Operational Excellence and SLOs | L3 | 0 | 2 | Partial |
| 31 | Agent Lifecycle Governance | L2 | 0 | 2 | Partial |
| 32 | Architecture Governance and Documentation Loop | L3 | 0 | 1 | Yes |

**Plan accuracy note:** Rows marked **No** or **Partial** mean the phase closeout register claims **Done** for **wiring/bridge** work, but FAUDIT found **High** gaps vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` / `INTEGRAX_HARNESS_AUDIT_MAP.md` §8 — tracked as **FAUDIT.\*** residuals, not reopening closed closeout phases.

### FAUDIT-32 — Remediation register (implementation queue → §6.1ah)

| ID | Layer | Gap | Severity | Module / acceptance |
|----|-------|-----|----------|-------------------|
| FAUDIT-TIER.1 | §2 | Tier-0 imports `applications/*` in `capability_graph_applications.py` | **Critical** | Move manifest catalog to Tier-3 injection or static metadata; zero `from applications` under `intergrax/` |
| FAUDIT-TIER.2 | §2 | No CI gate for `intergrax/` → `applications/` imports | High | `scripts/check_intergrax_no_applications_imports.py` in §6.1 |
| FAUDIT-INTAKE.1 | §3 | No canonical `TaskEnvelope`; `Task` + `RuntimeRequest` split | High | Typed envelope alias or consolidation; plan W-OPS.6 naming sync |
| FAUDIT-INTAKE.2 | §3 | Worker≡HTTP intake parity test matrix incomplete | High | Acceptance test: CLI/worker/HTTP same `Task` shape |
| FAUDIT-ID.1 | §4 | No user/service/agent identity distinction | High | Identity contracts + propagation to delegation |
| FAUDIT-ID.2 | §4 | `DelegationSpec` lacks permission scope audit | High | Scope field + trace on child runs |
| FAUDIT-POL.1 | §5 | No pre-LLM / pre-output policy hooks in runtime | High | PolicyEngine extension points documented + wired |
| FAUDIT-LLM.1 | §6 | No policy-driven model routing / fallback chain | High | Router module or AdaptiveProfile integration |
| FAUDIT-COG.1 | §7 | No universal `DecisionRecord` per UAEP step | High | Typed decision artifact + trace event |
| FAUDIT-ORCH.1 | §9 | No graph backpressure beyond parallel cap | High | Queue depth / shed policy in `GraphExecutor` |
| FAUDIT-SUB.1 | §10 | No formal `SubtaskContract`; `inherit_tool_policy=True` default | High | Contract type + safer delegation defaults |
| FAUDIT-MEM.1 | §15 | Entity graph memory absent; STM retention partial | High | Route to MEM-9.* / new MEM row if scoped |
| FAUDIT-PE.1 | §17 | No golden prompt content regression in CI | High | Golden YAML fixtures + gate test |
| FAUDIT-REG.1 | §19 | `HarnessRegistrySnapshot` omits agents + eval registry | High | Extend snapshot + assembly resolver |
| FAUDIT-CG.1 | §20 | Capability graph seed skips `prompt:*` nodes | High | `_seed_node_ids()` parity |
| FAUDIT-CG.2 | §20 | Blast-radius not enforced at release | High | `phase_v_capability_graph_guard.py` impact check |
| FAUDIT-OBS.1 | §21 | `RuntimeEventType` missing `LLM_CALL` / `POLICY_DECISION` | High | Canon event catalog + bridge from trace |
| FAUDIT-REL.1 | §22 | Shallow error taxonomy (`VALIDATION_ERROR` only + 2) | High | Expand classifier per AUDIT_MAP §22 |
| FAUDIT-SEC.1 | §23 | No `DataClassification` model | High | Security profile + enforcement hooks |
| FAUDIT-COST.1 | §24 | Per-tenant cost attribution not mandatory in NexusLoop | High | Budget gate on main path |
| FAUDIT-EVAL.1 | §25 | `require_baseline_for_release` not CI-enforced | High | `phase_v_closeout_gate.py` eval baseline check |
| FAUDIT-ALG.1 | §31 | Lifecycle states ≠ AUDIT_MAP catalog; weak agent adoption | High | Align or document ADR; scaffold defaults |
| FAUDIT-OPS.1 | §30 | `release_cycles.json` not in repo; W-OPS.5 artifact policy unclear | High | Document committed vs generated artifact |

**Delivery rule:** One **FAUDIT.\*** ID per PR → update §6.1ah + Appendix M paydown log → gate green.

**Explicitly out of scope (audit-and-fix):** source code or test changes during this audit pass; K.1/K.2; new product Tier-3 apps.

---



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



**Status:** **Done** (2026-06-09) — **18/18 harness Done** (FLOW-8 product host **Deferred** §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **closed**; default queue = §6.1 maintenance  
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
| FLOW-10 | FLOW4 | FLOW-GAP-08 | **Reserved lifecycle states** — ADR: implement `WAITING_FOR_RESOURCES`/`EXPIRED` **or** trim enum + canon sync | **Done** | Low | `task_lifecycle.py`, `adr/entries/2026-06-07/ADR-FLOW-002.md` | [ADR-FLOW-002](adr/entries/2026-06-07/ADR-FLOW-002.md) accepted; reserved v1 semantics |
| FLOW-12 | FLOW4 | §24 / FAUDIT-COG | **`DecisionRecord` regression gate** — verify FAUDIT-COG.1 emit on every UAEP decision path; gate test; sync flow §24 | **Done** | Medium | `uaep.py`, `tests/integration/agents/` | `DECISION_EMITTED` + `decision_record` on each step decision |
| FLOW-13 | FLOW4 | FLOW-GAP-12 | **`max_inflight_nodes` profile + wire** — field on `OrchestrationProfile`; `resolve_max_inflight_nodes()`; `nexus_factory` → `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `orchestration_wiring.py`, `nexus_factory.py` | `GRAPH_BACKPRESSURE` event when cap hit; profile round-trip test |
| FLOW-14 | FLOW4 | FLOW-GAP-13 | **`SubtaskContract` in delegation expansion** — `graph_spec_to_plan` / ADR-FLOW-001 child node uses `SubtaskContract.to_delegation_spec()` (`objective`, `permission_scopes`, `inherit_tool_policy=False`) | **Done** | Medium | `graph_spec_to_plan.py`, `subtask_contract.py` | Unit test scopes + objective on child `DelegationSpec` |
| FLOW-15 | FLOW4 | FLOW-GAP-14 | **Subagent budget envelope** — optional `budget_envelope` on `SubtaskContract` / `DelegationSpec`; enforce in child `GraphExecutor` run via existing budget bridge | **Done** | Medium | `subtask_contract.py`, `delegation.py`, `graph_executor.py` | Child run exceeds envelope → fail with trace |
| FLOW-16 | FLOW4 | FLOW-GAP-15 | **`MODIFY_PLAN` ADR** — [ADR-FLOW-003](adr/entries/2026-06-07/ADR-FLOW-003.md): document reserved semantics (policy-gated replan hook) **or** trim `AgentDecision` enum | **Done** | Low | `adr/entries/2026-06-07/ADR-FLOW-003.md`, `interrupts/handler.py` | ADR accepted; `MODIFY_PLAN_NOT_SUPPORTED` when no handoff |
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
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Phase FLOW implementation complete: delegation expansion, graph hardening, profile wiring, ADR-FLOW-002/003; gate **906 passed**; **18/18 harness** (FLOW-8 product **Deferred** §6.3) |

**Phase FLOW complete when:** FLOW-1–7, FLOW-9, FLOW-11–17, FLOW-DOC.* **Done**; FLOW-8 **Deferred** or Done per product; §6.1aj closed; **zero open `FLOW-GAP-*`** in flow reference §23; AUDIT_MAP §5/§7/§9/§10/§25 at target maturity; gate green.

---



**Status:** **Done** (2026-06-02) — **5/5** deliverables Done (TS-DOC.* + TS-1–3); gate **589 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §11–§12; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix J**.

**Priority ladder:** **Band 2k** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bc](#62bc-phase-ts-execution-order-band-2k--closed) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

**Delivery rule:** One **TS-*** ID per PR → update master table + §6.1c + paydown log below → `pytest -m gate` + §6.1 scripts green.

### TS — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TS-DOC.1 | TS0 | **Appendix J** — tools & skills control plane map (§J.1–J.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| TS-DOC.2 | TS0 | **Cross-ref sync** — plan, README, AUDIT_MAP §11–§12, audit prompt ref #7 | **Done** | Medium | `docs/*` | Links resolve |
| TS-1 | TS1 | **`catalog_runtime_bridge.py`** — `tool_profile` / `skill_profile` on `RuntimeConfig` via `materialize_runtime_config` | **Done** | **Critical** | `catalog_runtime_bridge.py`, `runtime_config_bridge.py`, `config.py` | `test_catalog_runtime_bridge.py` |
| TS-2 | TS2 | **Harness host LLM wiring** — `resolve_llm_adapter(env)` → `build_nexus_loop_from_environment` | **Done** | High | `harness_host_runtime.py` | `test_harness_host_runtime_llm.py` |
| TS-3 | TS3 | **`SkillResolverProtocol`** — typed contract for skill composition resolution | **Done** | Medium | `skills/resolver.py`, `contract_resolution.py` | existing skill resolver tests green |

**Residual (not TS scope — track separately):** legacy `use_rag`/`use_websearch` booleans in `engine_planner` / `tool_gateway` (deprecation warnings; `check_legacy_tool_plan_booleans.py`).

### TS — Paydown log

| Date | TS ID | Summary |
|------|-------|---------|
| 2026-06-02 | TS-DOC.1, TS-DOC.2 | Appendix J + cross-refs; AUDIT_MAP §11–§12 authoring map |
| 2026-06-02 | TS-1, TS-2, TS-3 | Catalog runtime bridge, harness LLM wiring, SkillResolverProtocol; gate **589** |

**Phase TS complete when:** TS-1–3 + TS-DOC.* **Done**; §6.1c queue closed; Appendix J has no “planned wiring” gaps; gate **589** green. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (INT-DOC.* + INT-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §13; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix K**.

**Priority ladder:** **Band 2l** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bd](#62bd-phase-int-execution-order-band-2l--closed) · queue: [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed)

### INT — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| INT-DOC.1 | INT0 | **Appendix K** — integration control plane (§K.1–K.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| INT-DOC.2 | INT0 | **Cross-ref sync** — plan, README, AUDIT_MAP §13, audit prompt ref #8 | **Done** | Medium | `docs/*` | Links resolve |
| INT-1 | INT1 | **`integration_runtime_bridge.py`** — explicit `integration_profile` on `RuntimeConfig` | **Done** | **Critical** | `integration_runtime_bridge.py`, `runtime_config_bridge.py` | `test_integration_runtime_bridge.py` |
| INT-2 | INT2 | **`integration_health_wiring.py`** — bootstrap health probes on `wire_application_environment` | **Done** | High | `integration_health_wiring.py`, `environment_wiring.py` | `test_integration_health_wiring.py` |

### INT — Paydown log

| Date | INT ID | Summary |
|------|--------|---------|
| 2026-06-02 | INT-DOC.1, INT-DOC.2 | Appendix K + cross-refs; AUDIT_MAP §13 |
| 2026-06-02 | INT-1, INT-2 | Integration runtime bridge + health wiring |

**Phase INT complete when:** INT-1–2 + INT-DOC.* **Done**; §6.1d queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (RAG-DOC.* + RAG-1); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §14; author map: **Appendix K** §K.5.

**Priority ladder:** **Band 2m** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2be](#62be-phase-rag-execution-order-band-2m--closed) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

### RAG — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| RAG-DOC.1 | RAG0 | **Appendix K** §K.5 + AUDIT_MAP §14 cross-ref | **Done** | High | `docs/*` | RAG bridge documented |
| RAG-1 | RAG1 | **`rag_runtime_bridge.py`** + RAG stack on `wire_application_environment` | **Done** | **Critical** | `rag_runtime_bridge.py`, `environment_wiring.py`, `runtime_config_bridge.py` | `test_rag_runtime_bridge.py` |

### RAG — Paydown log

| Date | RAG ID | Summary |
|------|--------|---------|
| 2026-06-02 | RAG-DOC.1 | Appendix K §K.5 + plan sync |
| 2026-06-02 | RAG-1 | RAG runtime bridge + environment wire; gate **600** |

**Phase RAG complete when:** RAG-1 + RAG-DOC.* **Done**; §6.1e queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CTX-DOC.* + CTX-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §16; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix L**.

**Priority ladder:** **Band 2n** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bf](#62bf-phase-ctx-execution-order-band-2n--closed) · queue: [§6.1f](#61f-harness-implementation-queue--context-engineering-closeout-closed)

### CTX — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| CTX-DOC.1 | CTX0 | **Appendix L** — context engineering control plane (§L.1–L.6) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CTX-DOC.2 | CTX0 | **Cross-ref sync** — plan, README, AUDIT_MAP §16, audit prompt ref #9 | **Done** | Medium | `docs/*` | Links resolve |
| CTX-1 | CTX1 | **`context_runtime_bridge.py`** — dedicated context profile → `RuntimeConfig` | **Done** | **Critical** | `context_runtime_bridge.py`, `runtime_config_bridge.py` | `test_context_runtime_bridge.py` |
| CTX-2 | CTX2 | **`context_wiring.py`** — `ContextManager` + task options from environment; `nexus_factory` wire | **Done** | High | `context_wiring.py`, `nexus_factory.py`, `harness_host_runtime.py` | `test_context_wiring.py` |

### CTX — Paydown log

| Date | CTX ID | Summary |
|------|--------|---------|
| 2026-06-02 | CTX-DOC.1, CTX-DOC.2 | Appendix L + cross-refs; AUDIT_MAP §16 |
| 2026-06-02 | CTX-1, CTX-2 | Context runtime bridge + Nexus ContextManager wiring; gate **608** |

**Phase CTX complete when:** CTX-1–2 + CTX-DOC.* **Done**; §6.1f queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (LEG-1–2); gate **612 passed**

**Audit basis:** Phase O.5a residual; `check_legacy_tool_plan_booleans.py`; Appendix J §J.6.

**Priority ladder:** **Band 2o** (§4.0) — closed; default queue = **§6.1** maintenance.

### LEG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| LEG-1 | LEG1 | **`tool_invocation_plan_from_capability_payload`** — gateway maps booleans → `tool_ids` without `from_legacy` | **Done** | `tool_runtime.py`, `tool_gateway.py` | `test_capability_payload_tool_plan.py` |
| LEG-2 | LEG2 | **Engine planner `tool_ids`** — parser populates `EnginePlan.tool_ids`; schema optional `tool_ids` | **Done** | `engine_planner_parse.py`, `nexus_llm_plan_builder.py` | `test_engine_plan_json_parser.py` |
| LEG-3 | LEG3 | **`plan_from_like` canonical path** — `from_tool_ids` only; `tool_gateway` removed from audit grandfather | **Done** | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | audit script green |

**Residual:** `ToolInvocationPlan.from_legacy()` retained in `tool_runtime.py` for explicit deprecation tests only; `EnginePlan.use_rag`/`use_websearch` remain on LLM schema for backward-compatible planner output.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (PE-DOC.* + PE-1–3); gate **623 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §17; V-REM-PE.1/PE.2 governance schema (**Done**); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix M**.

**Priority ladder:** **Band 2p** (§4.0) — closed; default queue = **§6.1** maintenance.

### PE — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| PE-1 | PE1 | **`PromptProfile`** + `prompt_runtime_bridge` — `catalog_path` → `RuntimeConfig.prompt_catalog_path` | **Done** | `environment_profile.py`, `prompt_runtime_bridge.py`, `config.py` | `test_prompt_runtime_bridge.py` |
| PE-2 | PE2 | **`prompt_wiring`** — `resolve_prompt_registry()`, `PromptRegistryProtocol` | **Done** | `prompt_wiring.py`, `prompt_registry_protocol.py` | `test_prompt_wiring.py` |
| PE-3 | PE3 | **Environment wire** — `materialize_runtime_config`, `build_runtime_context_from_environment`, `ApplicationBuildContext.prompt_registry` | **Done** | `runtime_config_bridge.py`, `environment_wiring.py`, `runtime_context.py` | wiring tests + gate |
| PE-4 | PE4 | **Nexus injection** — `prompt_registry_resolver`; `tools_step`, `tool_planning_prompts`, `engine_plan_models`, `nexus_llm_plan_builder` use `RuntimeContext.prompt_registry` | **Done** | `prompt_registry_resolver.py`, nexus/tools + nexus_llm_plan_builder | `test_tools_step_prompt_registry.py` |
| PE-DOC.1 | PE0 | **Appendix M** — prompt registry control plane (§M.1–M.6) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |

**Residual:** none on Tier-3 host build path. Legacy YAML prompt assets (`chat_router*`, `tools_agent_*`) remain as catalog files only.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CLEAN-1–4)

**Audit basis:** Phase U-Leg residual; `scripts/check_legacy_modules_removed.py`; prior `check_tools_agent_*` audits merged.

**Priority ladder:** closeout between Band 2p and 2q; default queue = **Band 2q** [Phase AS](plan/ORCHESTRATION.md).

### CLEAN — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CLEAN-1 | CLEAN1 | **Remove `legacy/chat_router.py`** — YAML assets tested without runtime module | **Done** | `tests/unit/chat_agent/` | prompt YAML tests green |
| CLEAN-2 | CLEAN2 | **Remove `tools/tools_agent.py`** — `CatalogToolPlanner` + `ToolPlanningService` canonical | **Done** | `catalog_tool_planner.py`, `tool_planning_service.py` | `test_catalog_tool_planner.py` |
| CLEAN-3 | CLEAN3 | **Unified CI audit** — `check_legacy_modules_removed.py` replaces `check_tools_agent_*` | **Done** | `scripts/`, `.github/workflows/unit-tests.yml` | audit script green in CI |
| CLEAN-4 | CLEAN4 | **Docs sync** — plan, HARNESS_ENVIRONMENT, AGENT_CREATION_GUIDE, README, TOOLS | **Done** | `docs/*` | no stale `ToolsAgent` production paths |

**Retained (not CLEAN scope):** `ToolInvocationPlan.from_legacy()` + deprecation tests; `EnginePlan.use_rag`/`use_websearch` on LLM schema; `intergrax/legacy/rag_answers/` archive with import guard; diagnostic type names (`CoreLLMUsedToolsAgentAnswerDiagV1`).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (AS-DOC.1 + AS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §18; ideal model §17 in [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix N**.

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



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REG-DOC.1 + REG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §19; capability graph V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix O**.

**Priority ladder:** **Band 2r** (§4.0) — closed; default queue = **§6.1** maintenance.

### REG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REG-DOC.1 | REG0 | **Appendix O** — registry architecture control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REG-1 | REG1 | **`HarnessRegistrySnapshot`** + `registry_wiring` + `RegistrySnapshotProtocol` | **Done** | `registry_snapshot.py`, `registry_wiring.py` | `test_registry_wiring.py` |
| REG-2 | REG2 | **`registry_assembly_resolver`** — profile ↔ registry conformance at wire time | **Done** | `registry_assembly_resolver.py`, `environment_wiring.py` | `test_registry_wiring.py` |
| REG-3 | REG3 | **Host registry resolution CI** — `check_harness_registry_resolution.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), marketplace UI, Band 3 product hosts — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CG-DOC.1 + CG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §20; Phase V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix P**.

**Priority ladder:** **Band 2s** (§4.0) — closed; default queue = **§6.1** maintenance.

### CG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CG-DOC.1 | CG0 | **Appendix P** — capability graph control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CG-1 | CG1 | **`capability_graph_wiring`** — environment subgraph from catalog + registry snapshot | **Done** | `capability_graph_wiring.py`, `capability_graph_protocol.py` | `test_capability_graph_wiring.py` |
| CG-2 | CG2 | **`capability_graph_assembly_resolver`** — wire-time catalog node validation | **Done** | `capability_graph_assembly_resolver.py`, `environment_wiring.py` | `test_capability_graph_wiring.py` |
| CG-3 | CG3 | **Host capability graph CI** — `check_harness_capability_graph_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only graph nodes — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (OBS-DOC.1 + OBS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §21; complements GOV-AUDIT Appendix H; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix Q**.

**Priority ladder:** **Band 2t** (§4.0) — closed; default queue = **§6.1** maintenance.

### OBS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| OBS-DOC.1 | OBS0 | **Appendix Q** — observability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| OBS-1 | OBS1 | **`observability_runtime_bridge`** + **`observability_wiring`** | **Done** | `observability_runtime_bridge.py`, `observability_wiring.py`, `runtime_config_bridge.py` | `test_harness_observability_wiring.py` |
| OBS-2 | OBS2 | **`observability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `observability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| OBS-3 | OBS3 | **Host observability CI** — `check_harness_observability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only observability dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **8/8** deliverables · OBS-BUS-0–7 **Done**

**Purpose:** Implement the full **Harness Observability Spine (HOS)** — one bus for Harness, applications, and agents; typed extension; causal trees; complete catalog emission; L4 audit §21.

**Architecture:** [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **ADR:** [ADR-OBS-001](adr/entries/2026-06-08/ADR-OBS-001.md)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §21 · complements Phase OBS (wiring closeout) · supersedes residual “live bus emit for all LLM paths” row when OBS-BUS-2 ships.

**Priority ladder:** **Band 2al** (§4.0) — runs **after** Phase CRIT-V (Band 2ak) or in parallel §6.1 maintenance slices; **one OBS-BUS ID per PR**.

**Depends on:** Phase OBS (wiring) **Done** · OBS-DEPTH.1/2 **Done** · FAUDIT-OBS.1 **Done**

### OBS-BUS — Master register

| ID | Area | Deliverable | Status | Modules / artifacts | Acceptance |
|----|------|-------------|--------|---------------------|------------|
| OBS-BUS-0 | OBS0 | **Architecture canon** — `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README links | **Done** | `docs/architecture/OBSERVABILITY.md`, `docs/adr/entries/2026-06-08/ADR-OBS-001.md` | Doc review; links from §33 |
| OBS-BUS-1 | OBS1 | **`RuntimeEventPayload` registry** — typed canonical payloads per `RuntimeEventType` (§42.23.1 families) | **Done** | `intergrax/runtime/events/payload_registry.py`, `payloads/`, `schema_guard.py`, `trace_bridge.py`, `context_skill_recording.py` | Gate: `test_runtime_event_payload_registry.py` |
| OBS-BUS-2 | OBS2 | **`ObservabilityEmitter` + `TraceScope`** — single emit API; `parent_event_id` causal tree | **Done** | `intergrax/runtime/observability/emitter.py`, `trace_scope.py`, `runtime_state.py` | `RuntimeState.trace_event` delegates; `test_observability_emitter.py` |
| OBS-BUS-3 | OBS3 | **Emission coverage** — `AGENT_SELECTED`, `STEP_FAILED`, graph typed payloads, critic `evaluator_loop` bridge | **Done** | `agent_router.py`, `graph_trace_callbacks.py`, `task_trace.py`, `trace_bridge.py`, `graph_node_diag.py` | `check_observability_emission_coverage.py` |
| OBS-BUS-4 | OBS4 | **Extension SDK** — agent/app `DiagnosticPayload` scaffold, namespace rules, `PayloadSchemaRegistry` | **Done** | `extension_sdk.py`, `tracing_templates.py`, `new_agent.py`, `new_application.py` | `check_payload_schema_registry.py` |
| OBS-BUS-5 | OBS5 | **Persistence conformance** — Cassandra/ES adapters implement same protocols; profile docs | **Done** | `document_backed_runtime_event_store.py`, `persistence_conformance.py`, profile wiring | `check_observability_persistence_conformance.py` |
| OBS-BUS-6 | OBS6 | **Export sinks** — OTLP dual-write from unified journal; parser trace link | **Done** | `journal_export.py`, `export_bridge.py`, `task_events.py`, `platform_wiring.py` | `TASK_COMPLETED` carries `journal_ref`; export plugin dual-writes OTLP JSON + parser trace |
| OBS-BUS-7 | OBS7 | **CI gates** — emission coverage + schema registry + L4 §21 evidence | **Done** | `scripts/check_observability_gates.py`, emission/schema/persistence audits, CI workflow | Gate suite green; audit map §21 → **L4** |

### OBS-BUS — Execution order (recommended)

```text
OBS-BUS-0 (docs) → OBS-BUS-1 (typed payloads)
  → OBS-BUS-2 (emitter + TraceScope)
  → OBS-BUS-3 (coverage gaps)
  → OBS-BUS-4 (extension SDK)
  → OBS-BUS-5 (persistence)
  → OBS-BUS-6 (sinks)
  → OBS-BUS-7 (gates / L4 closeout)
```

**DoD:** All OBS-BUS rows **Done**; `build_unified_run_journal` reproduces full Nexus+AgentEngine path without reading source; every `RuntimeEventType` in §42.1.2 has ≥1 production emitter; `parent_event_id` populated for tool/LLM/delegation; extension scaffold documented; gate green.

**Explicitly excluded:** product-specific dashboards (§6.3a); replacing external APM as mandatory deployment.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REL-DOC.1 + REL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §22; H-APP `ReliabilityProfile` **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix R**.

**Priority ladder:** **Band 2u** (§4.0) — closed; default queue = **§6.1** maintenance.

### REL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REL-DOC.1 | REL0 | **Appendix R** — reliability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REL-1 | REL1 | **`reliability_runtime_bridge`** + **`reliability_wiring`** | **Done** | `reliability_runtime_bridge.py`, `reliability_wiring.py`, `runtime_config_bridge.py` | `test_harness_reliability_wiring.py` |
| REL-2 | REL2 | **`reliability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `reliability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| REL-3 | REL3 | **Host reliability CI** — `check_harness_reliability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only retry/fallback policies — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (SEC-DOC.1 + SEC-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §23; V-SEC / V-REM-SEC **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix S**.

**Priority ladder:** **Band 2v** (§4.0) — closed; default queue = **§6.1** maintenance.

### SEC — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| SEC-DOC.1 | SEC0 | **Appendix S** — security control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| SEC-1 | SEC1 | **`security_runtime_bridge`** + **`security_wiring`** | **Done** | `security_runtime_bridge.py`, `security_wiring.py`, `runtime_config_bridge.py` | `test_harness_security_wiring.py` |
| SEC-2 | SEC2 | **`security_assembly_resolver`** — profile ↔ middleware conformance | **Done** | `security_assembly_resolver.py`, `harness_host_runtime.py`, `nexus_factory.py` | assembly validation tests |
| SEC-3 | SEC3 | **Host security CI** — `check_harness_security_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only security dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (COST-DOC.1 + COST-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §24; V-COST **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix T**.

**Priority ladder:** **Band 2w** (§4.0) — closed; default queue = **§6.1** maintenance.

### COST — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| COST-DOC.1 | COST0 | **Appendix T** — cost governance control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| COST-1 | COST1 | **`CostProfile`** + **`cost_runtime_bridge`** + **`cost_wiring`** | **Done** | `environment_profile.py`, `cost_runtime_bridge.py`, `cost_wiring.py`, `policy_wiring.py` | `test_harness_cost_wiring.py` |
| COST-2 | COST2 | **`cost_assembly_resolver`** — profile ↔ budget conformance | **Done** | `cost_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py` | assembly validation tests |
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (EVAL-DOC.1 + EVAL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25; V-EVAL **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix U**.

**Priority ladder:** **Band 2x** (§4.0) — closed; default queue = **§6.1** maintenance.

### EVAL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| EVAL-DOC.1 | EVAL0 | **Appendix U** — evaluation control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| EVAL-1 | EVAL1 | **`EvaluationProfile`** + **`evaluation_runtime_bridge`** + **`evaluation_wiring`** | **Done** | `environment_profile.py`, `evaluation_runtime_bridge.py`, `evaluation_wiring.py`, `policy_wiring.py` | `test_harness_evaluation_wiring.py` |
| EVAL-2 | EVAL2 | **`evaluation_assembly_resolver`** — profile ↔ registry conformance | **Done** | `evaluation_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py`, `runtime.py` | assembly validation tests |
| EVAL-3 | EVAL3 | **Host evaluation CI** — `check_harness_evaluation_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product quality dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **24/24** deliverables Done (CRIT-V-0 through CRIT-V-7)  
**Prerequisites:** Phase EVAL **Done** (registry wiring), Phase FLOW **Done** (graph hooks), Phase M-LLM-R **Done** (typed LLM envelope)  
**Goal:** Deliver production-grade PEV **Verify** infrastructure — L0/L1/L2 critic stack with tier-separated competencies; uplift Evaluation audit layer L2→L3.  
**Priority ladder:** **Band 2ak** (§4.0) — **Done** (2026-06-08). Default queue reverts to §6.1 gate maintenance.  
**Architecture:** [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · canon [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · [ADR-CRITIC-001](adr/entries/2026-06-07/ADR-CRITIC-001.md)  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25 (Evaluation), §7 (Reasoning), §10 (Multi-agent); closes **FAUDIT-EVAL.1** residual  
**Execution order:** [§6.2ak](#62ak-phase-crit-v-execution-order-band-2ak--closed) · queue: [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed)

**Delivery rule:** One **CRIT-V-*** ID per PR → update master table + §6.1ak + gate green.

### CRIT-V — Master register

| ID | Wave | Deliverable | Status | Modules / docs | Acceptance |
|----|------|-------------|--------|----------------|------------|
| CRIT-V-0.1 | 0 | **Architecture RFC** — CVL full spec | **Done** | `architecture/CRITIC_VERIFICATION.md` | Linked from canon §55, README |
| CRIT-V-0.2 | 0 | **ADR-CRITIC-001** — tier-separated PEV verify | **Done** | `docs/adr/entries/2026-06-07/ADR-CRITIC-001.md` | Status Accepted; adr index |
| CRIT-V-0.3 | 0 | **Canon §55** addendum | **Done** | `intergrax_runtime_architecture.md` §55 | Cross-links resolve |
| CRIT-V-0.4 | 0 | **README** sections (root + docs) | **Done** | `README.md`, `docs/README.md` | Navigation table |
| CRIT-V-1.1 | 1 | **`CriticProfile`** on `ApplicationEnvironmentProfile` | **Done** | `contracts/environment_profile.py`, `critic_runtime_bridge.py`, `RuntimeConfig` | Unit: `test_harness_critic_wiring.py` |
| CRIT-V-1.2 | 1 | **CVL contracts** — `CriticRequest`, `CriticVerdict`, `LayerVerdict`, `RubricSpec` | **Done** | `runtime/critic/contracts.py` | Unit: `test_critic_contracts.py` |
| CRIT-V-1.3 | 1 | **`EvaluatorLoopSpec`** — max iterations, revise routing | **Done** | `runtime/critic/evaluator_loop_spec.py` | Unit: `test_evaluator_loop_spec.py` |
| CRIT-V-2.1 | 2 | **`eval.judge` tool** — semantic scoring via separate LLM profile | **Done** | `tools/providers/eval/judge.py`, bundle | `test_eval_critic_tools.py` |
| CRIT-V-2.2 | 2 | **`eval.trajectory` tool** — process scoring from replay slice | **Done** | `tools/providers/eval/trajectory.py` | Uses `trace_reader` |
| CRIT-V-2.3 | 2 | **Registry hook** — judge/trajectory → `OnlineEvaluationObservation` | **Done** | `service.py` `_append_critic_observation` | Observation appended when registry bound |
| CRIT-V-3.1 | 3 | **`CriticOrchestrator`** — L0→L1→L2 pipeline | **Done** | `runtime/critic/critic_orchestrator.py` | Unit: short-circuit, layer order |
| CRIT-V-3.2 | 3 | **`L0Gateway`** — wraps `NexusValidationEngine` + schema | **Done** | `runtime/critic/l0_gateway.py` | Reuses existing validators |
| CRIT-V-3.3 | 3 | **`L1Gateway`** — invokes eval tools via `CriticEvalToolClient` | **Done** | `runtime/critic/l1_gateway.py` | No direct LLM in Tier-1 |
| CRIT-V-3.4 | 3 | **Graph partial hook** — `GraphExecutor` → `verify_partial` | **Done** | `graph_executor.py`, `critic_wiring.py` | Integration test: L0 fail → retry |
| CRIT-V-3.5 | 3 | **Graph final hook** — `GraphRunner` → `verify_final` | **Done** | `graph_runner.py` | Terminal state respects verdict |
| CRIT-V-3.6 | 3 | **Critic trace events** — `critic.*` trace steps | **Done** | `runtime/critic/trace.py`, `trace_bridge.py` | Visible in lab trace API |
| CRIT-V-4.1 | 4 | **`EvaluatorLoopExecutor`** — critique→revise routing | **Done** | `runtime/critic/evaluator_loop_executor.py` | Unit: budget exhaustion → FAIL/HITL |
| CRIT-V-4.2 | 4 | **Graph integration** — `EVALUATOR_LOOP` pattern wired | **Done** | `graph_executor.py`, `evaluator_loop_metadata.py` | Acceptance: 2-iteration loop |
| CRIT-V-5.1 | 5 | **`NexusEvalRunner` semantic mode** — optional L1 via `eval.judge` | **Done** | `eval/nexus_eval_runner.py` | Integration: non-exact pass |
| CRIT-V-5.2 | 5 | **`EvalCase` rubric field** — rubric_ref + semantic_threshold | **Done** | `eval/eval_case.py` | Backward compatible |
| CRIT-V-6.1 | 6 | **`wire_application_critic()`** — Tier-3 wiring | **Done** | `applications/_shared/critic_wiring.py` | Mirror EVAL pattern |
| CRIT-V-6.2 | 6 | **`critic_assembly_resolver`** — wire-time validation | **Done** | `critic_assembly_resolver.py`, `check_harness_critic_wiring.py` | CI script |
| CRIT-V-6.3 | 6 | **Policy bundle** — `critic_governance` fragment | **Done** | `policy_wiring.py` | Merged at host build |
| CRIT-V-6.4 | 6 | **Appendix W** — critic control plane author map | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CRIT-V-7.1 | 7 | **FAUDIT-EVAL.1** — `require_baseline_for_release` CI gate | **Done** | `phase_v_closeout_gate.py`, `check_harness_critic_wiring.py` | Closeout gate green |
| CRIT-V-7.2 | 7 | **Flow reference §18 sync** — CVL hook table | **Done** | `architecture/NEXUS_EXECUTION_FLOW.md` | Hooks documented |
| CRIT-V-7.3 | 7 | **Lab harness demo** — L0+L1 on sample agent (not FLOW-8) | **Done** | `test_harness_critic_wiring.py`, lab host | Trace shows critic steps |
| CRIT-V-F.1 | F | **`ToolRegistryCriticEvalClient`** — L1 bridge to Tier-0 eval tools | **Done** | `runtime/critic/tool_registry_client.py`, `critic_tool_wiring.py` | `test_critic_closeout.py` |
| CRIT-V-F.2 | F | **`critic_llm_profile`** — separate judge LLM adapter | **Done** | `critic_llm_resolver.py`, `environment_profile.py` | Assembly + wiring tests |
| CRIT-V-F.3 | F | **L2 `L2Gateway`** + HITL escalation path | **Done** | `l2_gateway.py`, `critic_orchestrator.py` | `test_critic_l2_gateway.py` |
| CRIT-V-F.4 | F | **UAEP step hook** | **Done** | `uaep.py`, `validate_uaep_step_with_critic_detail` | UAEP critic path |
| CRIT-V-F.5 | F | **`CriticPolicyBridge`** + policy engine | **Done** | `policy_bridge.py`, `runtime_policy_engine.py` | `test_critic_closeout.py` |
| CRIT-V-F.6 | F | **Assembly gate** — require L1 client when semantic/trajectory enabled | **Done** | `critic_assembly_resolver.py` | `test_critic_assembly_resolver.py` |

**Explicitly excluded:** FLOW-8 §42.43 product reference app ([§6.3](#63-end-of-plan--deferred-product-work-only)); domain rubric packs in Tier-0; mandatory universal LLM-judge on all runs.

**Phase CRIT-V complete when:** CRIT-V-1 through CRIT-V-7 **Done**; Evaluation audit layer ≥ **L3**; gate green; FAUDIT-EVAL.1 closed.

---

# Audit Result: Critic & Verification Layer (CVL)

**Audit date:** 2026-06-13  
**Method:** Layer Completion Mode vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` §18 · `INTEGRAX_HARNESS_AUDIT_MAP.md` §25 · code `runtime/critic/` · `tools/providers/eval/` · Tier-3 critic wiring  
**Verdict:** **CRIT-V-0…7 + FOLLOWUP Done** — domain **CRITIC_VERIFICATION** closed at **L3+** (Architecturally Mature).

---

## Audit §CVL-1 — Scope

What was audited:

- PEV Verify infrastructure: L0/L1/L2 stack, `CriticOrchestrator`, graph hooks, evaluator-loop, offline semantic runner.
- Tier-0 tools: `eval.judge`, `eval.trajectory`, registry observation bridge.
- Tier-3 wiring: `CriticProfile`, `wire_application_critic()`, assembly resolver, policy `critic_governance`.
- Cross-domain: NEXUS_EXECUTION_FLOW §18 hooks, FAUDIT-EVAL.1 baseline gate, ACP reflection CVL gateway.

Out of scope: FLOW-8 product reference app (§6.3), domain rubric packs, L4 adaptive thresholds (AHIA).

---

## Audit §CVL-2 — Pre-CRIT-V gaps (historical) — all closed

| GAP-ID | Description | Closed by | Evidence |
|--------|-------------|-----------|----------|
| GAP-CVL-01 | No universal `eval.judge` primitive | CRIT-V-2.1 | `tools/providers/eval/judge.py` |
| GAP-CVL-02 | No trajectory evaluation contract | CRIT-V-2.2 | `tools/providers/eval/trajectory.py` |
| GAP-CVL-03 | Evaluator-loop catalog only | CRIT-V-4.1/4.2 | `evaluator_loop_executor.py` + `graph_executor.py` |
| GAP-CVL-04 | NexusEvalRunner exact-match only | CRIT-V-5.1/5.2 | `nexus_eval_runner.py` + `test_nexus_eval_runner_semantic.py` |
| GAP-CVL-05 | L0→L1→L2 stack not explicit | CRIT-V-1/3 | `contracts.py`, `critic_orchestrator.py` |
| GAP-CVL-06 | Evaluation layer L2 depth | CRIT-V-0…7 | 24/24 register Done; gate green |
| GAP-CVL-07 | L1 client not wired from Tier-3 tools | CRIT-V-F.1/F.2 | `critic_tool_wiring.py`, `tool_registry_client.py` |
| GAP-CVL-08 | No UAEP step critic hook | CRIT-V-F.4 | `validate_uaep_step_with_critic_detail` |
| GAP-CVL-09 | No policy bridge for verdict actions | CRIT-V-F.5 | `policy_bridge.py` |
| GAP-CVL-10 | Architecture §2 stale gap list | **CVL-LC-1** (2026-06-13) | This audit + architecture sync |

**Coverage:** 10 gaps — **all closed**. No open P0/P1 items.

---

## Audit §CVL-3 — Maturity scores (2026-06-13, iteration II)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Architecture Completeness | **96%** | Trajectory dual-mode documented |
| Production Readiness | **92%** | Bootstrap idempotency + combined gate stability |
| Documentation Consistency | **97%** | §7.3 trajectory clarification (CVL-LC-4) |
| Implementation Consistency | **95%** | Graph critic tests pass in combined sessions |

**State:** **Architecturally Mature** — no P0/P1 blockers.

---

## Audit §CVL-4 — Backlog (P2–P4, non-blocking)

| ID | Priority | Item | Notes |
|----|----------|------|-------|
| CVL-BACKLOG-01 | P2 | LLM trajectory judge in runtime path | **Documented** — `eval.trajectory_judge` skill; `eval.trajectory` stays heuristic (CVL-LC-4) |
| CVL-BACKLOG-02 | P2 | Test isolation for critic graph suite | **Done** — `register_default_tools` idempotent override (CVL-LC-3) |
| CVL-BACKLOG-03 | P2 | `NexusEvalRunner.from_nexus_loop` auto-wire semantic client | **Done** — CVL-LC-2 |
| CVL-BACKLOG-04 | P3 | Duplicate CRIT-V master register removed | **CVL-LC-1** doc cleanup |
| CVL-BACKLOG-05 | P4 | L4 adaptive critic thresholds in CI | AHIA / `VerificationLoop` extension |
| CVL-BACKLOG-06 | P4 | FLOW-8 product reference host with critic demo | §6.3 deferred |

---

## Sprint CVL-LC-1 — Documentation sync (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | Architecture §2 historical gaps + status; plan audit register; audit prompt regeneration |
| **Goal** | Honest L3+ layer status — no false “open gap” list at doc open |
| **DoD** | Architecture/plan/audit prompt aligned; closes GAP-CVL-10, CVL-BACKLOG-04 |
| **Files** | `docs/architecture/CRITIC_VERIFICATION.md`, `docs/plan/CRITIC_VERIFICATION.md`, `scripts/generate_domain_audit_prompts.py`, `docs/guides/audit/CRITIC_VERIFICATION.md` |

## Sprint CVL-LC-2 — NexusEvalRunner semantic wiring (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | `from_nexus_loop` extracts L1 client from critic hooks; fail-closed when semantic enabled without client |
| **Goal** | Offline harness eval uses wired critic path without manual client injection |
| **DoD** | Unit test for auto-wire + fail-closed; closes CVL-BACKLOG-03 |
| **Files** | `intergrax/eval/nexus_eval_runner.py`, `intergrax/runtime/critic/l1_gateway.py`, `intergrax/runtime/nexus/nexus_loop.py`, `tests/unit/eval/test_nexus_eval_runner_semantic.py` |

## Sprint CVL-LC-3 — Tool catalog bootstrap idempotency (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | `register_default_tools()` overrides already-registered bundles instead of raising |
| **Goal** | `RuntimeContext.build()` safe after partial catalog registration in test sessions |
| **DoD** | Critic graph integration tests pass in combined CVL+eval gate runs; closes CVL-BACKLOG-02 |
| **Files** | `intergrax/tools/registry/bootstrap.py`, `intergrax/tools/registry/catalog.py`, `tests/unit/tools/registry/test_bootstrap_idempotent.py` |

## Sprint CVL-LC-4 — Trajectory eval dual-mode docs (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | Architecture §7.3 + plan backlog clarify heuristic `eval.trajectory` vs `eval.trajectory_judge` skill |
| **Goal** | Honest L1 trajectory contract — no false expectation of LLM rubric on `eval.trajectory` tool |
| **DoD** | Architecture/plan aligned; closes CVL-BACKLOG-01 (documented) |
| **Files** | `docs/architecture/CRITIC_VERIFICATION.md`, `docs/plan/CRITIC_VERIFICATION.md` |

---

## 6. What to implement next

**Default answer (infrastructure):** **[§6.1](#61-harness-platform-maintenance-default--band-1)** gate green on every PR — CRIT-V and OBS-BUS platform closeouts **Done**.

**Maintenance-only mode:** If CRIT-V paused by explicit decision, revert to §6.1 gate-only maintenance.

**Not default:** K.1, K.2, Legal UAEP domain steps, new product Tier-3 apps — **[§6.3](#63-end-of-plan--deferred-product-work-only)** · **[§6.3a](#63a-business-backlog-register-consolidated)** · **[§4.0a](#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Governance audit (2026-06-05) → GOV-AUDIT **Done**; orchestration audit (2026-06-05) → Phase ORCH + §6.1b; tools/skills audit (2026-06-02) → Phase TS + §6.1c; integration/RAG audit (2026-06-02) → Phase INT + RAG + §6.1d/§6.1e; context engineering audit (2026-06-02) → Phase CTX + §6.1f; prior V-REM/MEM/DX/AA closeouts in [§6.1z](#61z-harness-implementation-queue-consolidated) / [§6.1aa](#61aa-harness-implementation-queue-memory-platform).

### 6.1i Harness implementation queue — prompt registry closeout (closed)

**Purpose:** Single ordered list for **Phase PE** (Band 2p). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **PE-DOC.1** | Docs | **Done** | Appendix M + cross-refs | Author map complete |
| 2 | **PE-1** | Code | **Done** | `prompt_runtime_bridge` + `PromptProfile` | `test_prompt_runtime_bridge.py` |
| 3 | **PE-2** | Code | **Done** | `prompt_wiring` + `PromptRegistryProtocol` | `test_prompt_wiring.py` |
| 4 | **PE-3** | Code | **Done** | environment + runtime context wire | gate green |
| 5 | **PE-4** | Code | **Done** | Nexus prompt registry injection | `test_tools_step_prompt_registry.py` |

### 6.1j Harness implementation queue — legacy module closeout (closed)

**Purpose:** Single ordered list for **Phase CLEAN** (post-2p closeout). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CLEAN-1** | Code | **Done** | Remove `chat_router.py`; YAML-only tests | `test_chat_agent_prompts_yaml.py` |
| 2 | **CLEAN-2** | Code | **Done** | Remove `tools_agent.py`; planner tests | `test_catalog_tool_planner.py` |
| 3 | **CLEAN-3** | CI | **Done** | `check_legacy_modules_removed.py` in CI | workflow green |
| 4 | **CLEAN-4** | Docs | **Done** | Plan + harness docs sync | no stale production refs |

**Suggested PR order (complete):** CLEAN-1 → CLEAN-2 → CLEAN-3 → CLEAN-4.

### 6.1k Harness implementation queue — agent assembly closeout (closed)

**Purpose:** Single ordered list for **Phase AS** (Band 2q). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **AS-DOC.1** | Docs | **Done** | Appendix N + cross-refs | Author map complete |
| 2 | **AS-1** | Code | **Done** | `agent_assembly_resolver` | `test_agent_assembly_resolver.py` |
| 3 | **AS-2** | Code | **Done** | Lifecycle metadata enforcement | resolver + routing tests |
| 4 | **AS-3** | CI | **Done** | `check_agent_skill_resolution.py` | CI green |

**Suggested PR order (complete):** AS-DOC.1 → AS-1 → AS-2 → AS-3.

**Explicitly excluded:** K.1, K.2, new product agents, domain-only contract packs — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1l Harness implementation queue — registry architecture closeout (closed)

**Purpose:** Single ordered list for **Phase REG** (Band 2r). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REG-DOC.1** | Docs | **Done** | Appendix O + cross-refs | Author map complete |
| 2 | **REG-1** | Code | **Done** | `HarnessRegistrySnapshot` + `registry_wiring` | `test_registry_wiring.py` |
| 3 | **REG-2** | Code | **Done** | `registry_assembly_resolver` wire | `test_registry_wiring.py` |
| 4 | **REG-3** | CI | **Done** | `check_harness_registry_resolution.py` | CI green |

**Suggested PR order (complete):** REG-DOC.1 → REG-1 → REG-2 → REG-3.

### 6.1m Harness implementation queue — capability graph closeout (closed)

**Purpose:** Single ordered list for **Phase CG** (Band 2s). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CG-DOC.1** | Docs | **Done** | Appendix P + cross-refs | Author map complete |
| 2 | **CG-1** | Code | **Done** | `capability_graph_wiring` | `test_capability_graph_wiring.py` |
| 3 | **CG-2** | Code | **Done** | `capability_graph_assembly_resolver` | wire-time validation tests |
| 4 | **CG-3** | CI | **Done** | `check_harness_capability_graph_wiring.py` | CI green |

**Suggested PR order (complete):** CG-DOC.1 → CG-1 → CG-2 → CG-3.

### 6.1n Harness implementation queue — observability closeout (closed)

**Purpose:** Single ordered list for **Phase OBS** (Band 2t). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **OBS-DOC.1** | Docs | **Done** | Appendix Q + cross-refs | Author map complete |
| 2 | **OBS-1** | Code | **Done** | `observability_runtime_bridge` + `observability_wiring` | `test_harness_observability_wiring.py` |
| 3 | **OBS-2** | Code | **Done** | `observability_assembly_resolver` | wire-time validation tests |
| 4 | **OBS-3** | CI | **Done** | `check_harness_observability_wiring.py` | CI green |

**Suggested PR order (complete):** OBS-DOC.1 → OBS-1 → OBS-2 → OBS-3.

### 6.1o Harness implementation queue — reliability closeout (closed)

**Purpose:** Single ordered list for **Phase REL** (Band 2u). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REL-DOC.1** | Docs | **Done** | Appendix R + cross-refs | Author map complete |
| 2 | **REL-1** | Code | **Done** | `reliability_runtime_bridge` + `reliability_wiring` | `test_harness_reliability_wiring.py` |
| 3 | **REL-2** | Code | **Done** | `reliability_assembly_resolver` | wire-time validation tests |
| 4 | **REL-3** | CI | **Done** | `check_harness_reliability_wiring.py` | CI green |

**Suggested PR order (complete):** REL-DOC.1 → REL-1 → REL-2 → REL-3.

### 6.1q Harness implementation queue — security closeout (closed)

**Purpose:** Single ordered list for **Phase SEC** (Band 2v). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **SEC-DOC.1** | Docs | **Done** | Appendix S + cross-refs | Author map complete |
| 2 | **SEC-1** | Code | **Done** | `security_runtime_bridge` + `security_wiring` | `test_harness_security_wiring.py` |
| 3 | **SEC-2** | Code | **Done** | `security_assembly_resolver` | wire-time validation tests |
| 4 | **SEC-3** | CI | **Done** | `check_harness_security_wiring.py` | CI green |

**Suggested PR order (complete):** SEC-DOC.1 → SEC-1 → SEC-2 → SEC-3.

### 6.1r Harness implementation queue — cost governance closeout (closed)

**Purpose:** Single ordered list for **Phase COST** (Band 2w). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **COST-DOC.1** | Docs | **Done** | Appendix T + cross-refs | Author map complete |
| 2 | **COST-1** | Code | **Done** | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | `test_harness_cost_wiring.py` |
| 3 | **COST-2** | Code | **Done** | `cost_assembly_resolver` | wire-time validation tests |
| 4 | **COST-3** | CI | **Done** | `check_harness_cost_wiring.py` | CI green |

**Suggested PR order (complete):** COST-DOC.1 → COST-1 → COST-2 → COST-3.

### 6.1s Harness implementation queue — evaluation closeout (closed)

**Purpose:** Single ordered list for **Phase EVAL** (Band 2x). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **EVAL-DOC.1** | Docs | **Done** | Appendix U + cross-refs | Author map complete |
| 2 | **EVAL-1** | Code | **Done** | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | `test_harness_evaluation_wiring.py` |
| 3 | **EVAL-2** | Code | **Done** | `evaluation_assembly_resolver` | wire-time validation tests |
| 4 | **EVAL-3** | CI | **Done** | `check_harness_evaluation_wiring.py` | CI green |

**Suggested PR order (complete):** EVAL-DOC.1 → EVAL-1 → EVAL-2 → EVAL-3.

### 6.1f Harness implementation queue — context engineering closeout (closed)

**Purpose:** Single ordered list for **Phase CTX** (Band 2n). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CTX-DOC.1–2** | Docs | **Done** | Appendix L + cross-refs | Author map complete |
| 2 | **CTX-1** | Code | **Done** | `context_runtime_bridge` | `test_context_runtime_bridge.py` |
| 3 | **CTX-2** | Code | **Done** | `context_wiring` + `nexus_factory` | `test_context_wiring.py` |

### 6.1e Harness implementation queue — RAG closeout (closed)

**Purpose:** Single ordered list for **Phase RAG** (Band 2m). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **RAG-DOC.1** | Docs | **Done** | Appendix K §K.5 + AUDIT_MAP §14 | Author map complete |
| 2 | **RAG-1** | Code | **Done** | `rag_runtime_bridge` + environment wire | `test_rag_runtime_bridge.py` |

### 6.1d Harness implementation queue — integration closeout (closed)

**Purpose:** Single ordered list for **Phase INT** (Band 2l). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **INT-DOC.1–2** | Docs | **Done** | Appendix K + cross-refs | Author map complete |
| 2 | **INT-1** | Code | **Done** | `integration_runtime_bridge` | `test_integration_runtime_bridge.py` |
| 3 | **INT-2** | Code | **Done** | `integration_health_wiring` | `test_integration_health_wiring.py` |

### 6.1c Harness implementation queue — tools/skills closeout (closed)

**Purpose:** Single ordered list for **Phase TS** (Band 2k). **Closed 2026-06-02** — all TS rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **TS-DOC.1–2** | Docs | **Done** | Appendix J + cross-refs | Author map complete |
| 2 | **TS-1** | Code | **Done** | `catalog_runtime_bridge` + `RuntimeConfig.skill_profile` | `test_catalog_runtime_bridge.py` |
| 3 | **TS-2** | Code | **Done** | Harness host `resolve_llm_adapter` wiring | `test_harness_host_runtime_llm.py` |
| 4 | **TS-3** | Code | **Done** | `SkillResolverProtocol` | skill resolver tests green |

**Suggested PR order (complete):** TS-1 → TS-2 → TS-3 → TS-DOC.*.

**Explicitly excluded:** K.1, K.2, new product tools/skills, business agent packs — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1aa Harness implementation queue — memory platform (closed)

**Purpose:** Phase MEM execution queue — **closed 2026-06-02** (48/48 Done). Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **MEM-1.1–MEM-1.4** | Code | **Done** | H-APP `MemoryProfile` + `ContextProfile.budget` + SQLite session → `RuntimeConfig` | MEM-1.5 gate test green |
| 2 | **MEM-2.1–MEM-2.3** | Code | **Done** | `SQLiteUserProfileStore` + bundle wiring + unit tests | LTM survives restart on sqlite profile |
| 3 | **MEM-1.6** | Docs/status | **Done** | H-APP.4.3 → **Done** | Bridge complete |
| 4 | **MEM-4.1–MEM-4.3** | Test | **Done** | Session + LTM + full-stack memory gates | acceptance/integration green |
| 5 | **MEM-5.1–MEM-5.2** | Test/Docs | **Done** | `engine_history_layer` tests + compression docs | unit + guide |
| 6 | **MEM-3.1–MEM-3.3** | Code | **Done** | Memory store plugin EP + reference fixture | bootstrap + gate |
| 7 | **MEM-0.3–MEM-DOC.*** | Docs | **Done** | Author cookbooks + Appendix G sync | guide updated |
| 8 | **MEM-6.*–MEM-7.*** | Code | **Done** | Retention enforcement + memory hooks | P2 after P0/P1 |
| 9 | **MEM-8.*–MEM-9.*** | RFC | **Done (RFC)** | Product memory layer + entity graph design | §6.3 gate for implementation |

**Suggested PR order:** See [Phase MEM — Suggested PR order](#mem--paydown-log).

**Explicitly excluded:** K.1, K.2, Mem0 SaaS product, entity graph ship (RFC only), business agent memory.

### 6.1aj Harness implementation queue — Nexus execution depth (closed)

**Purpose:** Single ordered list for **Phase FLOW** (Band 2aj). **Closed 2026-06-09** — **18/18 harness Done**; product host **Deferred** §6.3. Ongoing: **§6.1** maintenance only.

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
| — | **FLOW-8** | Product | **Deferred** | §42.43 reference app | **§6.3** gate only |

**Suggested PR order:** See [Phase FLOW — Suggested PR order](#flow--suggested-pr-order).

**Explicitly excluded:** K.1, K.2 (unless FLOW-8 activated), nested harness per child.

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

**Suggested PR order:** See [§6.2ak](#62ak-phase-crit-v-execution-order-band-2ak--closed).

**Explicitly excluded:** FLOW-8 product app; domain rubric packs in Tier-0; mandatory universal LLM-judge.

### 6.1al Harness implementation queue — Unified Observability Spine (closed)

**Purpose:** Single ordered list for **Phase OBS-BUS** (Band 2al). **Closed 2026-06-08** — all OBS-BUS rows **Done**; audit map §21 → **L4**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **OBS-BUS-0** | Docs | **Done** | `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README | Links resolve |
| 2 | **OBS-BUS-1** | Code | **Done** | `RuntimeEventPayload` registry | Payload registry gate |
| 3 | **OBS-BUS-2** | Code | **Done** | `ObservabilityEmitter` + `TraceScope` | Causal tree tests |
| 4 | **OBS-BUS-3** | Code | **Done** | Emission coverage gaps | `check_observability_emission_coverage.py` |
| 5 | **OBS-BUS-4** | Code/Docs | **Done** | Extension SDK + scaffold | Agent tracing template |
| 6 | **OBS-BUS-5** | Code | **Done** | Persistence conformance | Integration tests |
| 7 | **OBS-BUS-6** | Code | **Done** | OTLP/journal dual-write | `test_journal_export.py`, `test_export_bridge.py` |
| 8 | **OBS-BUS-7** | CI | **Done** | L4 §21 gates | `check_observability_gates.py` in CI; audit map §21 → L4 |

**Suggested PR order:** See [Phase OBS-BUS — Execution order](#obs-bus--execution-order-recommended).

**Explicitly excluded:** Product dashboards (§6.3a); vendor-only APM as sole store.

### 6.1am Harness implementation queue — Memory intelligence depth (closed)

**Purpose:** Single ordered list for **Phase MEM-DEPTH** (Band 2am). **Closed 2026-06-08** — **26/26 Done**. Canonical: [plan/MEMORY.md](plan/MEMORY.md).

**Suggested PR order:** See [§6.2ab](plan/MEMORY.md#62ab-phase-mem-depth-execution-order-band-2am--closed).

**Explicitly excluded:** K.1/K.2, Mem0 SaaS, Redis session default — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1b Harness implementation queue — orchestration closeout (closed)

**Purpose:** Single ordered list for **Phase ORCH** (Band 2j). **Closed 2026-06-05** — all ORCH rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **ORCH-DOC.1–2** | Docs | **Done** | Appendix I + cross-refs | Author map complete |
| 2 | **ORCH-1** | Code | **Done** | `planner_kind` / `classifier_kind` wiring | `test_orchestration_wiring.py` |
| 3 | **ORCH-2** | Code | **Done** | `ApplicationGraphSpec` → `NexusPlan` | `test_graph_spec_to_plan.py` |
| 4 | **ORCH-3** | Code | **Done** | `max_parallel_nodes` cap | `test_graph_executor_parallel_cap.py` |
| 5 | **ORCH-4** | Docs | **Done** | Closeout sync | Plan + Appendix I updated |

**Suggested PR order (complete):** ORCH-1 → ORCH-2 → ORCH-3 → ORCH-4.

**Explicitly excluded:** K.1, K.2, new graph node types, nested harness per child — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1g Harness implementation queue — governance audit (closed)

**Purpose:** Phase GOV-AUDIT documentation closeout — **closed 2026-06-05**.

| Order | ID | Status | Deliverable |
|-------|-----|--------|-------------|
| 1 | GOV-DOC.1 | **Done** | Appendix H control plane |
| 2 | GOV-DOC.2 | **Done** | Cross-ref sync |
| 3 | GOV-DOC.3 | **Done** | EXTENSION_AUTHOR §10 |
| — | GOV-PROD.1 | **Deferred** | Product dashboard → §6.3 |

### 6.1z Harness implementation queue (consolidated — closed 2026-06-05)

**Purpose:** Single ordered list of **infrastructure** work. Excludes Band 3 / [§6.3a](#63a-business-backlog-register-consolidated). **Closed 2026-06-05** — Phase V-REM complete. Prior DX/AA/MEM/W-OPS/H-APP rows remain **Done**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green; scripts in [§6.1](#61-harness-platform-maintenance-default--band-1) |
| 1 | **V-REM-CG.1** | Code | **Done** | Fix per-application capability graph system edges | V-CG.2–4 closed |
| 2 | **V-REM-CG.2** | Test/CI | **Done** | Re-validate lineage/impact/compatibility on corrected graph | `phase_v_capability_graph_guard.py` green |
| 3 | **V-REM-ALG.1** | Code | **Done** | Runtime filter retired/deprecated agents | Unit tests green |
| 4 | **V-REM-ALG.2** | Code | **Done** | Production-eligible + owner gate at selection | Strict harness test green |
| 5 | **V-REM-PE.1** | Code | **Done** | PromptMeta owner/risk schema | Registry validation tests |
| 6 | **V-REM-PE.2** | Assets | **Done** | YAML prompt assets catalog seed | E2E governance validation |
| 7 | **V-REM-SEC.1** | Code | **Done** | Tool injection defense on execution path | Middleware unit tests |
| 8 | **V-REM-SEC.2** | Code | **Done** | Retrieval poisoning middleware per tenant/app | RagStep filter unit tests |
| 9 | **V-REM-SEC.3** | Code | **Done** | Tenant isolation + audit trail in main path | Intake middleware unit tests |
| 10 | **V-REM-A.1** | Test | **Done** | NexusEvalRunner integration + gate | A.4 → **Done** |
| — | **REG-*** | Regression | As needed | Fix gate/CI failures only | No feature scope |

**Closed (no implementation — do not reopen without regression):**

| ID | Resolution |
|----|------------|
| DX-0.3–DX-8.2 (except DX-5.7) | **Done** — 2026-06-02 DX residual closeout |
| AA-LABAG.1, AA-SIG.2, AA-LABAPP.6 | **Done** |
| AA-LABAG.2 | **Won't fix** — mocks remain in `agents/lab/` until leadership requests move |
| W-OPS.1–15, H-APP.0–6.3, P-Ext, Q–V contracts, MEM 48/48 | **Done** |
| V-REM.0.1, V-REM.0.2 | **Done** — 2026-06-05 plan sync |
| V-REM-CG.1–A.1 | **Done** — 2026-06-05 runtime remediation |

**Explicitly excluded from this queue (business — implement only after §6.3 decision):** K.1, K.2, K.6, B.15, S-Ops.4, A.5, AA-LEG.2.2+, AA-LEGAPP.6–8, AA-RES.4–5, AA-RESAPP.6, AA-ORG.3–4, new Tier-3 product apps, domain skills — full list: [§6.3a](#63a-business-backlog-register-consolidated).

**Suggested PR order:** V-REM-CG.1 → V-REM-CG.2 → V-REM-ALG.1 → V-REM-ALG.2 → V-REM-SEC.1 → V-REM-SEC.2 → V-REM-SEC.3 → V-REM-PE.1 → V-REM-PE.2 → V-REM-A.1. Regressions → **REG-*** under §6.1.

**Explicitly excluded:** K.1, K.2, new product eval modes requiring business datasets — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1t Harness implementation queue — Adaptive Harness Intelligence (closed)

**Purpose:** Single ordered list for **Phase W-ADAPT** (Band 2y). **Closed 2026-06-02** — **70/70 Done** (Wave W-ADAPT-0 through Wave W-ADAPT-7 **Done**). Maintenance-only; see [§6.1](#61-harness-platform-maintenance-default--band-1).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **W-ADAPT-0.2–0.5** | Docs/Code | **Done** | ADR-ADAPT-001 + `intergrax/runtime/adaptive/` scaffold | Import + gate stub |
| 2 | **W-ADAPT-1.1–1.12** | Code | **Done** | Observe (L4-O): signals + utility + report | `phase_w_adapt_report.py` |
| 3 | **W-ADAPT-2.1–2.12** | Code | **Done** | Recommend (L4-R): engines + proposals (no apply) | Proposals in report |
| 4 | **W-ADAPT-3.1–3.7** | Code | **Done** | Shadow (L4-S): ProfileVersionStore + executor.shadow | Integration test green |
| 5 | **W-ADAPT-4.1–4.10** | Code | **Done** | Apply (L4-A): canary, apply, rollback, events | Policy learning HITL enforced |
| 6 | **W-ADAPT-5.1–5.12** | Code/Docs | **Done** | Verify (L4-V): VerificationLoop + runtime L4 closeout | `--enforce-l4-runtime` |
| 7 | **W-ADAPT-6.1–6.5** | Code | **Done** | ProcessPatternMiner + daily scheduler | pattern report |
| 8 | **W-ADAPT-7.1–7.7** | Code/Docs | **Done** | Tier-3 AdaptiveProfile + Appendix V + acceptance | E2E observe→recommend |

**Suggested PR order:** See [Phase W-ADAPT — Suggested PR order](#w-adapt--suggested-pr-order).

**Explicitly excluded:** K.1, K.2, deep RL, foundation model training, autonomous prompt edits — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1v Harness implementation queue — LLM completion response envelope (closed)

**Purpose:** Single ordered list for **Phase M-LLM-R** (Band 2z). **Closed 2026-06-06** — **39/39 Done**. Runs **in parallel** with W-ADAPT waves 5–7 (Tier-0 LLM contract; independent of L4 runtime loop).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **M-LLM-R.0.2–0.3** | Docs | **Done** | ADR-LLM-001 + canon §5.2.2 addendum | Linked from plan |
| 2 | **M-LLM-R.1.1–1.8** | Code | **Done** | Contract types + builders + public exports | Import smoke; no dict returns |
| 3 | **M-LLM-R.2.1–2.6** | Code | **Done** | `LLMAdapter` ABC typed signatures | ABC compiles; stubs updated |
| 4 | **M-LLM-R.3.1–3.7** | Code | **Done** | All provider adapters return envelope | Conformance per provider family |
| 5 | **M-LLM-R.4.1–4.6** | Code | **Done** | Nexus runtime consumers | `test_context_preflight + ACP agent tests` + tool planner |
| 6 | **M-LLM-R.5.1–5.3** | Code | **Done** | RAG + websearch + legacy | RAG unit tests green |
| 7 | **M-LLM-R.6.1–6.4** | Code | **Done** | Agents + scaffold + CI lint | `check_llm_adapter_typed_returns.py` + `check_agents_llm_adapter_response.py` |
| 8 | **M-LLM-R.7.1–7.5** | Code | **Done** | Usage alignment + replay/trace bridge | `test_replay_engine` + diagnostics |
| 9 | **M-LLM-R.8.1–8.4** | Docs/CI | **Done** | Docs + conformance + closeout | M-LLM.14 Done; Appendix L complete |

**Suggested PR order:** See [Phase M-LLM-R — Suggested PR order](#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06).

**Explicitly excluded:** K.1, K.2, product HTTP API DTOs, provider SDK rewrites — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1w Harness implementation queue — Integration expansion (M.6 P4 closed)

**Purpose:** Ordered backlog for **Phase M.6 P4** (Band 2aa). **Status:** **Done** (2026-06-02) — **28/28 Done** · catalog **127**.  
**Register:** [M.6 P4 — Master register](#m6-p4--master-register-28-slugs) · **Execution order:** [§6.2ae](#62ae-phase-m6-p4-execution-order--done)  
**Policy:** One slug per PR; runs **in parallel** with §6.1 maintenance — pull only when harness ops/adaptive/INT health needs the slug.

| Order | Wave | IDs | Slugs | Priority | Status |
|-------|------|-----|-------|----------|--------|
| 0 | CAT | M-P4-CAT.1, M-P4-CAT.2 | *(categories)* | **P0** | **Done** (beta) |
| 1 | H-INT-1 | M-P4.1–M-P4.4 | `pgvector`, `duckdb`, `influxdb`, `timescaledb` | P0/P1 | **Done** |
| 2 | H-INT-2 | M-P4.5–M-P4.7 | `grafana`, `loki`, `tempo` | **P0** | **Done** |
| 3 | H-INT-3 | M-P4.8–M-P4.11 | `aws_secrets_manager`, `azure_key_vault`, `gcp_secret_manager`, `doppler` | P0/P1 | **Done** |
| 4 | H-INT-4 | M-P4.12–M-P4.16 | `unleash`, `launchdarkly`, `github_actions`, `redpanda`, `cloudflare_r2` | P0/P1 | **Done** |
| 5 | H-INT-5 | M-P4.17–M-P4.28 | `memgraph`, `falkordb`, `incident_io`, `kubernetes`, `servicenow`, `bitbucket`, `asana`, `sendgrid`, `mailgun`, `mlflow`, `huggingface_hub`, `ollama` | P1/P2 | **Done** |

**Per-slug checklist (M.4):** contract → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → `architecture/INTEGRATIONS.md` → canon §7.1.3 row → gate green → paydown log row.

**Explicitly excluded:** CRM, payments, blockchain, duplicate vector SaaS, LLM vendor APIs — see [M.6 P4 register](#m6-p4--harness-platform-expansion-planned).

### 6.1x Harness implementation queue — Integration depth (M.6 P5 done)

**Purpose:** Closeout record for **Phase M.6 P5** (Band 2ab). **Status:** **Done** (2026-06-02) — **33/34**.  
**Register:** [M.6 P5 — Master register](#m6-p5--master-register-34-slugs) · **Execution order:** [§6.2af](#62af-phase-m6-p5-execution-order-band-2ab--planned)  
**Policy:** One slug per PR (or one harden wave ≤4 slugs); runs **in parallel** with §6.1 maintenance — pull when W-OPS / W-ADAPT / EVAL / prod stack needs the slug.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P5-CAT.1–3 | `ci_cd` extend, `security_scanner`, category mapping | **P0** | **Done** (CAT.2 deferred: `trivy`) |
| 1 | H-INT-6 | M-P5.1–M-P5.10 | Ops/metrics/CI/local cloud: prometheus, clickhouse, vault, pagerduty, github, gitlab_ci, circleci, azure_pipelines, mailpit, localstack | **P0** | **Done** |
| 2 | H-INT-7 | M-P5.11–M-P5.20 | Eval/async/artifacts: langfuse, phoenix, braintrust, mlflow, influxdb, timescaledb, temporal, redpanda, minio, s3 | **P0/P1** | **Done** |
| 3 | H-INT-8 | M-P5.21–M-P5.28 | Data plane lab: neo4j, mongodb, elasticsearch, nats, chroma, weaviate, launchdarkly, signoz | **P1/P2** | **Done** |
| 4 | H-INT-9 | M-P5.29–M-P5.34 | P2 reserve: codecov, trivy, grafana_oncall, opentelemetry_collector, snowflake, supabase | **P2** | **Done** |
| 5 | PRE | M-P5-PRE.1 | Tier-3 presets: `harness_metrics_stack`, `harness_eval_stack`, `harness_async_stack`, `harness_ci_stack` | **P0** | **Done** |

**Explicitly excluded:** Band 3 product agents; see [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334).

### 6.1y Harness implementation queue — Integration expansion (M.6 P6 Done)

**Purpose:** Ordered backlog for **Phase M.6 P6** (Band 2ac). **Status:** **Done** (2026-06-02) — **32/32**.  
**Register:** [M.6 P6 — Master register](#m6-p6--master-register-32-slugs) · **Execution order:** [§6.2ag](#62ag-phase-m6-p6-execution-order-band-2ac--done)  
**Policy:** One slug per PR (or one CAT wave before first slug in a new category); runs **in parallel** with §6.1 maintenance — pull when security/sandbox/identity/GitOps/speech harness gaps block ops.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P6-CAT.1–9 | New categories: `security_scanner`, `sandbox_host`, `identity_provider`, `speech_provider`, `workflow_orchestrator`, `vision_serving`, `ml_inference_host`, `billing_meter`, `crm` | **P0** | **Done** |
| 1 | H-INT-10 | M-P6.1–M-P6.4 | Security + secrets: `trivy`, `snyk`, `semgrep`, `infisical` | **P0** | **Done** |
| 2 | H-INT-11 | M-P6.5–M-P6.7 | Cloud sandbox: `e2b`, `modal`, `daytona` | **P0/P1** | **Done** |
| 3 | H-INT-12 | M-P6.8–M-P6.10 | Identity: `auth0`, `keycloak`, `workos` | **P0/P1** | **Done** |
| 4 | H-INT-13 | M-P6.11–M-P6.13 | GitOps CI: `argocd`, `buildkite`, `jenkins` | **P0/P1** | **Done** |
| 5 | H-INT-14 | M-P6.14–M-P6.15 | Speech catalog: `elevenlabs`, `deepgram` | **P0** | **Done** |
| 6 | H-INT-15 | M-P6.16–M-P6.19 | Enterprise ops: `newrelic`, `splunk`, `zendesk`, `statsig` | **P1** | **Done** |
| 7 | H-INT-16 | M-P6.20–M-P6.24 | Data/workflow: `prefect`, `airflow`, `typesense`, `neon`, `pulsar` | **P1** | **Done** |
| 8 | H-INT-17 | M-P6.25–M-P6.32 | Reserve: `algolia`, `confluent`, `backblaze_b2`, `triton`, `replicate`, `stripe`, `salesforce`, `hubspot` | **P2** | **Done** |
| 9 | PRE | M-P6-PRE.1 | Tier-3 presets: `harness_security_stack`, `harness_sandbox_stack`, `harness_identity_stack`, `harness_gitops_stack` | **P0** | **Done** |
| 10 | WIRE | M-P6-WIRE.1–7 | Tool surface + sandbox/speech/identity bridges + promote gate + infra `p6` | **P0** | **Done** |

**Per-slug checklist:** see [M.6 P6 register](#m6-p6--harness-integration-expansion-planned).

**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS`; four Tier-3 presets; gate green.

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done)**. **Next product work** = [§6.3](#63-end-of-plan--deferred-product-work-only) (product prioritization only).

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/check_harness_no_getattr.py
  python scripts/check_legacy_modules_removed.py
  python scripts/check_agent_skill_resolution.py
  python scripts/check_harness_registry_resolution.py
  python scripts/check_harness_capability_graph_wiring.py
  python scripts/check_legacy_tool_plan_booleans.py
  python scripts/check_trace_bridge_event_catalog.py
  python scripts/check_plugin_catalog.py
  python scripts/check_llm_adapter_typed_returns.py
  python scripts/check_agents_llm_adapter_response.py
  uv run python scripts/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/check_scaffold_harness_alignment.py
  python scripts/check_agents_no_tier3_imports.py
  python scripts/check_intergrax_no_applications_imports.py
  uv run python scripts/check_harness_prompt_golden_catalog.py
  uv run python scripts/check_agents_lifecycle_metadata.py
  uv run intergrax doctor --ci
  uv run python scripts/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
  uv run python scripts/phase_v_capability_graph_guard.py --enforce
```

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3.

**Maintenance depth (2026-06-07):** **OBS-DEPTH.1 Done** — unified run journal. **T10-DEPTH.1 Done** — broker task index + PagerDuty acknowledge adapter. **T-EXPAND T11 Done** — 160 tools. **LEG-DEPTH.1–3 + O.5 depth Done** — planner schema uses `tool_ids`; legacy booleans accepted with deprecation trace; `from_legacy()` gated by `check_legacy_tool_plan_booleans.py`. **OBS-DEPTH.2 Done** — `check_trace_bridge_event_catalog.py` + gate test. **OBS live emit Done** — `RuntimeState.trace_event` → `runtime_event_bus`. **Celery purge_completed Done** — optional KV task index. **notify.dispatch_due Done** — Tier-0 dispatcher tool. **T-EXPAND T12 Done** — 170 tools (health slot probes + notify dispatcher). **T-EXPAND T13 Done** — 172 tools (`eval.judge`, `eval.trajectory` / CRIT-V). **L2→L3 §21 Done** — `test_observability_layer_depth_gate.py` regression gate.

### 6.1ah Harness implementation queue — FAUDIT-32 remediation (closed)

**Status:** **Done** (2026-06-06) — **23/23 Done**  
**Source:** [Phase FAUDIT-32](#phase-faudit-32--full-architecture-audit-closeout) · **Appendix M**  
**Priority ladder:** **Band 2ad** (§4.0) — runs **after** FAUDIT-TIER.1 on every harness PR that touches `intergrax/runtime/architecture/`

**Execution order (recommended):**

```text
Wave P0 (architecture integrity):
  FAUDIT-TIER.1 → FAUDIT-TIER.2

Wave P1 (identity + intake + observability):
  FAUDIT-INTAKE.1 → FAUDIT-ID.1 → FAUDIT-OBS.1 → FAUDIT-EVAL.1

Wave P2 (control-plane depth):
  FAUDIT-PE.1 → FAUDIT-REG.1 → FAUDIT-CG.1 → FAUDIT-CG.2
  → FAUDIT-SEC.1 → FAUDIT-REL.1 → FAUDIT-COST.1

Wave P3 (orchestration + cognition + memory):
  FAUDIT-ORCH.1 → FAUDIT-SUB.1 → FAUDIT-COG.1 → FAUDIT-LLM.1
  → FAUDIT-POL.1 → FAUDIT-MEM.1 → FAUDIT-ALG.1 → FAUDIT-OPS.1
  → FAUDIT-INTAKE.2 → FAUDIT-ID.2
```

| ID | Status | Priority | Blocks |
|----|--------|----------|--------|
| FAUDIT-TIER.1 | **Done** | **Critical** | `intergrax/applications/reference/harness_manifest_catalog.py` |
| FAUDIT-TIER.2 | **Done** | High | `scripts/check_intergrax_no_applications_imports.py` |
| FAUDIT-INTAKE.1 | **Done** | High | `intergrax/contracts/task_envelope.py` |
| FAUDIT-INTAKE.2 | **Done** | Medium | `tests/unit/runtime/architecture/test_faudit_remediation.py` |
| FAUDIT-ID.1 | **Done** | High | `intergrax/contracts/actor_identity.py` |
| FAUDIT-ID.2 | **Done** | Medium | `DelegationSpec.permission_scopes` |
| FAUDIT-POL.1 | **Done** | High | `PolicyEngine.evaluate_pre_llm/pre_output` |
| FAUDIT-LLM.1 | **Done** | High | `intergrax/llm_adapters/registry/model_router.py` |
| FAUDIT-COG.1 | **Done** | High | `intergrax/contracts/decision_record.py` + UAEP emit |
| FAUDIT-ORCH.1 | **Done** | Medium | `GraphExecutor` inflight backpressure |
| FAUDIT-SUB.1 | **Done** | High | `SubtaskContract` + safer defaults |
| FAUDIT-MEM.1 | **Done** | High | `retention_enforcement.py` + `PolicyScopedMemoryView` STM purge |
| FAUDIT-PE.1 | **Done** | High | `prompt_golden_catalog.py` + `tests/fixtures/prompt_golden/` + CI script |
| FAUDIT-ALG.1 | **Done** | High | lifecycle states + reference agent `owner_team` adoption + CI script |
| FAUDIT-REG.1 | **Done** | High | `HarnessRegistrySnapshot` agent/eval fields |
| FAUDIT-CG.1 | **Done** | High | prompt seeds in `capability_graph_wiring.py` |
| FAUDIT-CG.2 | **Done** | Medium | `phase_v_capability_graph_guard.py` impact log |
| FAUDIT-OBS.1 | **Done** | High | `RuntimeEventType.LLM_CALL/POLICY_DECISION` |
| FAUDIT-REL.1 | **Done** | High | expanded `RuntimeErrorCode` + classifier |
| FAUDIT-SEC.1 | **Done** | High | `intergrax/contracts/data_classification.py` |
| FAUDIT-COST.1 | **Done** | High | `run_budget` wired in `nexus_factory` |
| FAUDIT-EVAL.1 | **Done** | High | `phase_v_closeout_gate.py` eval baseline |
| FAUDIT-OPS.1 | **Done** | Medium | `build/architecture_hardening/release_cycles.json` |

**DoD (§6.1ah queue closure):** All **Planned** rows **Done**; Appendix M scorecard shows **0 Critical**, **≤5 High** (documented deferrals only); tier gate green.

### 6.1ai Harness implementation queue — FAUDIT-32 follow-up (closed)

**Status:** **Done** (2026-06-06) — post-remediation depth for PE/ALG/MEM adoption  
**Priority ladder:** **Band 2ad** (§4.0) — runs after §6.1ah closure

| ID | Status | Deliverable |
|----|--------|-------------|
| FAUDIT-PE.1+ | **Done** | Real `prompts/` golden hashes in `tests/fixtures/prompt_golden/expectations.json`; `scripts/check_harness_prompt_golden_catalog.py`; gate test |
| FAUDIT-ALG.1+ | **Done** | `lifecycle_state` + `owner_team` on reference Tier-2 agents; `scripts/check_agents_lifecycle_metadata.py` |
| FAUDIT-MEM.1+ | **Done** | `should_forget_stm_record` wired in `PolicyScopedMemoryView.read` |

**Explicitly deferred (Band 3 / product):** MEM-9 entity graph memory implementation (RFC only); K.1/K.2 business agents.

### 6.2bo Phase EVAL execution order (Band 2x — closed 2026-06-02)

**Status:** **Done** · register: [Phase EVAL](#phase-eval--evaluation-control-plane-closeout) · queue: [§6.1s](#61s-harness-implementation-queue--evaluation-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | EVAL-DOC.1 | Appendix U + plan sync | High |
| 2 | EVAL-1 | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | Critical |
| 3 | EVAL-2 | `evaluation_assembly_resolver` | High |
| 4 | EVAL-3 | `check_harness_evaluation_wiring.py` | Medium |

### 6.2bn Phase COST execution order (Band 2w — closed 2026-06-02)

**Status:** **Done** · register: [Phase COST](#phase-cost--cost-governance-control-plane-closeout) · queue: [§6.1r](#61r-harness-implementation-queue--cost-governance-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | COST-DOC.1 | Appendix T + plan sync | High |
| 2 | COST-1 | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | Critical |
| 3 | COST-2 | `cost_assembly_resolver` | High |
| 4 | COST-3 | `check_harness_cost_wiring.py` | Medium |

### 6.2bm Phase SEC execution order (Band 2v — closed 2026-06-02)

**Status:** **Done** · register: [Phase SEC](#phase-sec--security-control-plane-closeout) · queue: [§6.1q](#61q-harness-implementation-queue--security-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | SEC-DOC.1 | Appendix S + plan sync | High |
| 2 | SEC-1 | `security_runtime_bridge` + `security_wiring` | Critical |
| 3 | SEC-2 | `security_assembly_resolver` | High |
| 4 | SEC-3 | `check_harness_security_wiring.py` | Medium |

### 6.2bl Phase REL execution order (Band 2u — closed 2026-06-02)

**Status:** **Done** · register: [Phase REL](#phase-rel--reliability-control-plane-closeout) · queue: [§6.1o](#61o-harness-implementation-queue--reliability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REL-DOC.1 | Appendix R + plan sync | High |
| 2 | REL-1 | `reliability_runtime_bridge` + `reliability_wiring` | Critical |
| 3 | REL-2 | `reliability_assembly_resolver` | High |
| 4 | REL-3 | `check_harness_reliability_wiring.py` | Medium |

### 6.2bk Phase OBS execution order (Band 2t — closed 2026-06-02)

**Status:** **Done** · register: [Phase OBS](#phase-obs--observability-control-plane-closeout) · queue: [§6.1n](#61n-harness-implementation-queue--observability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | OBS-DOC.1 | Appendix Q + plan sync | High |
| 2 | OBS-1 | `observability_runtime_bridge` + `observability_wiring` | Critical |
| 3 | OBS-2 | `observability_assembly_resolver` | High |
| 4 | OBS-3 | `check_harness_observability_wiring.py` | Medium |

### 6.2bj Phase CG execution order (Band 2s — closed 2026-06-02)

**Status:** **Done** · register: [Phase CG](#phase-cg--capability-graph-control-plane-closeout) · queue: [§6.1m](#61m-harness-implementation-queue--capability-graph-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CG-DOC.1 | Appendix P + plan sync | High |
| 2 | CG-1 | `capability_graph_wiring` | Critical |
| 3 | CG-2 | `capability_graph_assembly_resolver` | High |
| 4 | CG-3 | `check_harness_capability_graph_wiring.py` | Medium |

### 6.2bi Phase REG execution order (Band 2r — closed 2026-06-02)

**Status:** **Done** · register: [Phase REG](#phase-reg--registry-architecture-control-plane-closeout) · queue: [§6.1l](#61l-harness-implementation-queue--registry-architecture-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REG-DOC.1 | Appendix O + plan sync | High |
| 2 | REG-1 | `HarnessRegistrySnapshot` + `registry_wiring` | Critical |
| 3 | REG-2 | `registry_assembly_resolver` | High |
| 4 | REG-3 | `check_harness_registry_resolution.py` | Medium |

### 6.2bg Phase AS execution order (Band 2q — closed 2026-06-02)

**Status:** **Done** · register: [Phase AS](#phase-as--agent-assembly-control-plane-closeout) · queue: [§6.1k](#61k-harness-implementation-queue--agent-assembly-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | AS-DOC.1 | Appendix N + plan sync | High |
| 2 | AS-1 | `agent_assembly_resolver` | Critical |
| 3 | AS-2 | Lifecycle state on `AgentContract` | High |
| 4 | AS-3 | `skill_ids` resolution audit script | Medium |

### 6.2bh Phase CLEAN execution order (closed 2026-06-02)

**Status:** **Done** · register: [Phase CLEAN](#phase-clean--legacy-module-closeout) · queue: [§6.1j](#61j-harness-implementation-queue--legacy-module-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CLEAN-1 | Remove `chat_router.py` | Critical |
| 2 | CLEAN-2 | Remove `tools_agent.py` | Critical |
| 3 | CLEAN-3 | `check_legacy_modules_removed.py` in CI | High |
| 4 | CLEAN-4 | Docs sync | Low |

### 6.2bf Phase CTX execution order (Band 2n — closed 2026-06-02)

**Status:** **Done** · register: [Phase CTX](#phase-ctx--context-engineering-control-plane-closeout) · queue: [§6.1f](#61f-harness-implementation-queue--context-engineering-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CTX-1 | `context_runtime_bridge` | Critical |
| 2 | CTX-2 | `context_wiring` + Nexus factory wire | High |
| 3 | CTX-DOC.1–2 | Appendix L + plan sync | Low |

### 6.2be Phase RAG execution order (Band 2m — closed 2026-06-02)

**Status:** **Done** · register: [Phase RAG](#phase-rag--rag-retrieval-control-plane-closeout) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | RAG-1 | `rag_runtime_bridge` + environment wire | Critical |
| 2 | RAG-DOC.1 | Appendix K §K.5 + plan sync | Low |

### 6.2bd Phase INT execution order (Band 2l — closed 2026-06-02)

**Status:** **Done** · register: [Phase INT](#phase-int--integration-control-plane-closeout) · queue: [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | INT-1 | `integration_runtime_bridge` | Critical |
| 2 | INT-2 | `integration_health_wiring` | High |
| 3 | INT-DOC.1–2 | Appendix K + plan sync | Low |

### 6.2bc Phase TS execution order (Band 2k — closed 2026-06-02)

**Status:** **Done** · register: [Phase TS](#phase-ts--tools--skills-control-plane-closeout) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

Work **one TS ID per PR**; after each step update the TS master table + §6.1c + paydown log; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | TS-1 | `catalog_runtime_bridge` + `materialize_runtime_config` | Critical | TS-DOC.* (parallel OK) |
| 2 | TS-2 | Harness host LLM adapter wiring | High | — |
| 3 | TS-3 | `SkillResolverProtocol` | Medium | — |
| 4 | TS-DOC.1–2 | Appendix J + plan sync | Low | TS-1–3 |

### 6.2aj Phase FLOW execution order (Band 2aj — closed 2026-06-07)

**Status:** **Done** · register: [Phase FLOW](#phase-flow--nexus-execution-depth) · queue: [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed)

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
| 17 | FLOW-DOC.* | Docs closeout | Low | FLOW-1–17 (except deferred FLOW-8) |

### 6.2ak Phase CRIT-V execution order (Band 2ak — closed)

**Status:** **Done** (2026-06-08) · register: [Phase CRIT-V](#phase-crit-v--critic--verification-layer) · queue: [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed)

Work **one CRIT-V ID per PR**; after each step update CRIT-V master table + §6.1ak; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | CRIT-V-0.* | Architecture + ADR + canon + README | High | — |
| 2 | CRIT-V-1.1 | `CriticProfile` on environment profile | **Critical** | CRIT-V-0 |
| 3 | CRIT-V-1.2 | CVL contracts (`CriticRequest`, `CriticVerdict`, …) | **Critical** | CRIT-V-1.1 |
| 4 | CRIT-V-1.3 | `EvaluatorLoopSpec` | High | CRIT-V-1.2 |
| 5 | CRIT-V-2.1 | `eval.judge` tool | **Critical** | CRIT-V-1.2, M-LLM-R |
| 6 | CRIT-V-2.2 | `eval.trajectory` tool | High | CRIT-V-2.1 |
| 7 | CRIT-V-2.3 | Registry observation hook for judge/trajectory | Medium | CRIT-V-2.1 |
| 8 | CRIT-V-3.1 | `CriticOrchestrator` | **Critical** | CRIT-V-2.1 |
| 9 | CRIT-V-3.2–3.3 | L0/L1 gateways | High | CRIT-V-3.1 |
| 10 | CRIT-V-3.4–3.5 | Graph partial + final hooks | High | CRIT-V-3.1 |
| 11 | CRIT-V-3.6 | Critic trace events | Medium | CRIT-V-3.4 |
| 12 | CRIT-V-4.1–4.2 | Evaluator-loop executor + graph wire | High | CRIT-V-3.4 |
| 13 | CRIT-V-5.1–5.2 | Semantic offline eval runner | Medium | CRIT-V-2.1 |
| 14 | CRIT-V-6.1–6.3 | Tier-3 critic wiring + policy + CI | High | CRIT-V-3.1 |
| 15 | CRIT-V-6.4 | Appendix W author map | Medium | CRIT-V-6.1 |
| 16 | CRIT-V-7.1 | FAUDIT-EVAL.1 baseline CI gate | High | CRIT-V-6.3 |
| 17 | CRIT-V-7.2–7.3 | Flow reference sync + lab demo | Medium | CRIT-V-3.6 |

### 6.2bb Phase ORCH execution order (Band 2j — closed 2026-06-05)

**Status:** **Done** · register: [Phase ORCH](#phase-orch--orchestration-control-plane-closeout) · queue: [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed)

Work **one ORCH ID per PR**; after each step update the ORCH master table + §6.1b + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | Depends on |
|-------|-----|-------------|----------|------------|
| 1 | ORCH-1 | Planner/classifier kind registry + `nexus_factory` wiring | **Critical** | ORCH-DOC.* |
| 2 | ORCH-2 | `graph_spec_to_plan` + planning runner integration | High | ORCH-1 (shared factory path) |
| 3 | ORCH-3 | `max_parallel_nodes` on `OrchestrationProfile` + `GraphExecutor` | Medium | — (parallel OK after ORCH-1) |
| 4 | ORCH-4 | Docs closeout — Appendix I + plan §0.5 | Low | ORCH-1–3 |

### 6.2v Phase V-REM execution order (Band 2i — closed 2026-06-05)

**Status:** **Done** · register: [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout) · queue: [§6.1z](#61z-harness-implementation-queue-consolidated) (closed)

Work **one V-REM ID per PR**; after each step update the V-REM master table + Appendix J + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | Closes |
|-------|-----|-------------|----------|--------|
| 1 | V-REM-CG.1 | Fix per-application capability graph system edge mapping | **Critical** | V-CG.2–4 |
| 2 | V-REM-CG.2 | Re-validate lineage/impact/compatibility on corrected graph | High | V-CG.2–4 |
| 3 | V-REM-ALG.1 | Runtime filter for retired/deprecated agents | High | V-ALG.3 |
| 4 | V-REM-ALG.2 | Production-eligible + owner gate at agent selection | High | V-ALG.4 |
| 5 | V-REM-SEC.1 | Tool injection defense on main execution path | High | V-SEC.2 |
| 6 | V-REM-SEC.2 | Retrieval poisoning middleware per tenant/app | High | V-SEC.3 |
| 7 | V-REM-SEC.3 | Tenant isolation + audit trail in UnifiedTaskRunner/NexusLoop | High | V-SEC.4 |
| 8 | V-REM-PE.1 | PromptMeta owner/risk schema + validation | High | V-PE.1 |
| 9 | V-REM-PE.2 | YAML prompt assets catalog seed | Medium | V-PE.1 |
| 10 | V-REM-A.1 | NexusEvalRunner integration tests + gate | Medium | A.4, A.4.1 |

**Phase V-REM closeout:** **Done** (2026-06-05). Verified via `phase_v_closeout_gate.py --enforce --enforce-l4`.

### 6.2w Phase W-OPS execution order (Band 2d — complete 2026-06-06)

**Status:** **Done** · register: [Phase W-OPS](#phase-w-ops--operational-harness-maturity-ideal-l3-ops)

Work **one W-OPS ID per PR**; after each step update the W-OPS table + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | IDEAL gap |
|-------|-----|-------------|----------|-----------|
| 1 | W-OPS.1 | Side-effect tool idempotency keys + dedup | **Critical** | Reliability §8.3 |
| 2 | W-OPS.2 | Integration circuit breaker (`_shared`) | **Critical** | Reliability §8.2 |
| 3 | W-OPS.3 | Long-running / checkpoint / retry gate tests | High | Reliability §8.3 |
| 4 | W-OPS.6 | `tenant_id` on TaskEnvelope → trace/events | High | Identity §3.2 |
| 5 | W-OPS.7 | Mandatory harness API key (staging profile) | High | Identity §3.2 |
| 6 | W-OPS.4 | SLO catalog + incident budget + runbooks | **Critical** | Observability §11 |
| 7 | W-OPS.5 | L3-ops evidence (2 release cycles) | **Critical** | §12.3 vs V-V6 CI |
| 8 | W-OPS.8 | `harness.*` platform skill packs | Medium | Capability §3.6 |
| 9 | W-OPS.9 | `requires_skills` shipped demo | Medium | Registries §19 |
| 10 | W-OPS.10 | Harness lab stable stack health (catalog slugs) | Medium | Capability §3.6 |
| 11 | W-OPS.11 | Online/shadow evaluation registry writes | Medium | Evaluation §18 |
| 12 | W-OPS.12 | W-ML Celery Tier-3 scale-out (optional) | Low | Modality §3.5.1 |
| 13 | W-OPS.13 | ToolsAgent removal roadmap | Low | Cognition hygiene |
| 14 | W-OPS.14 | Typed wiring (no `load_callable`) | Low | DX §22 |
| 15 | W-OPS.15 | Architecture metrics threshold enforcement | Low | §21.6 |

**Wave P0 (orders 1–7)** must be **Done** before declaring **operational IDEAL L3**. **Wave P1/P2** run in parallel with P0 when owners differ.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new product applications, Problem Radar wave 2+.

### 6.2x Phase H-APP execution order (Band 2e — complete 2026-06-03)

**Status:** **Done** · canonical register: [Phase H-APP — Master deliverables register](#h-app--master-deliverables-register-all-43-tasks) · audit narrative: [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7.

Work **one H-APP ID per PR**; after each step update the H-APP master table + paydown log; keep §6.1 scripts green.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| H0 | H-APP.0.1–H-APP.0.5 | 5 | Terminology, CI guards, `poc_template` getattr fix, manifest conformance |
| H1 | H-APP.1.1–H-APP.1.8 | 8 | `ApplicationEnvironmentProfile`, unified wiring, runtime bridge, LLM resolver |
| H2 | H-APP.2.1–H-APP.2.8 | 8 | Identity, policy DSL, execution modes, V-SEC per application |
| H3 | H-APP.3.1–H-APP.3.6 | 6 | Orchestration profile, graph spec, Nexus factory, shadow/sandbox |
| H4 | H-APP.4.1–H-APP.4.8 | 8 | Context, memory, reliability, observability profiles |
| H5 | H-APP.5.1–H-APP.5.5 | 5 | Migrate lab/legal/research/poc/docker_verify + scaffold |
| H6 | H-APP.6.1–H-APP.6.3 | 3 | Operational L3 sign-off (release cycles + CI + audit §4) |
| **Total** | | **43** | |

**Suggested PR order (same as Phase H-APP paydown):** H-APP.0.3 → H-APP.1.1–H-APP.1.4 → H-APP.1.5–H-APP.1.8 → H-APP.3.4–H-APP.3.5 → H-APP.2.1–H-APP.2.8 → H-APP.4.1–H-APP.4.8 → H-APP.3.1–H-APP.3.3 → H-APP.5.1–H-APP.5.5 → H-APP.0.1–H-APP.0.5 → H-APP.6.1–H-APP.6.3.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new **product** Tier-3 apps, Problem Radar wave 2+, marketplace UI, catalog hot-reload.

### 6.2y Phase DX execution order (Band 2f — mostly done)

**Status:** **Done** (2026-06-02) · **47/47 Done** · canonical register: [Phase DX — Master deliverables register](#dx--master-deliverables-register-all-47-tasks).

Work **one DX ID per PR**; after each step update the DX master table + paydown log; keep §6.1 scripts green. **Start with DX1 (scaffold/H-APP alignment)** before DX2 facades — otherwise new authors copy broken `factory.py` patterns.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| DX0 | DX-0.1–DX-0.4 | 4 | LangGraph mapping, responsibility matrix, progressive disclosure |
| DX1 | DX-1.1–DX-1.6 | 6 | **P0** — scaffold + poc/legal/research factories on H-APP path only |
| DX2 | DX-2.1–DX-2.6 | 6 | `HarnessApplication`, `AgentGraph`, `IntergraxAgent` + `@step` |
| DX3 | DX-3.1–DX-3.6 | 6 | `--minimal` stack, `intergrax run`, `doctor`, TTFRun acceptance |
| DX4 | DX-4.1–DX-4.4 | 4 | Integration presets + picker + gate tests |
| DX5 | DX-5.1–DX-5.8 | 8 | Host hooks, YAML loader, logging, event catalog, policy rule plugins |
| DX6 | DX-6.1–DX-6.5 | 5 | Tier-2 hygiene, external `intergrax init` template |
| DX7 | DX-7.1–DX-7.5 | 5 | JSON Schema + spec versioning + UI feed (Phase 2 prep) |
| DX8 | DX-8.1–DX-8.3 | 3 | `doctor --ci`, DX metrics artifact, scaffold alignment script |
| **Total** | | **47** | |

**Suggested PR order:** DX-1.1 → DX-1.2 → DX-1.3 → DX-1.6 → DX-8.3 → DX-2.1 → DX-2.2 → DX-2.3 → DX-2.5 → DX-3.1 → DX-3.2 → DX-3.5 → DX-3.6 → DX-4.1 → DX-4.4 → DX-1.4–DX-1.5 → DX-2.4 → DX-2.6 → DX-3.3–DX-3.4 → DX-5.1–DX-5.2 → DX-6.1–DX-6.2 → DX-4.2–DX-4.3 → DX-5.3–DX-5.8 → DX-6.3–DX-6.5 → DX-7.1–DX-7.5 → DX-8.1–DX-8.2 → DX-0.1–DX-0.4.

**Success gate for Phase DX full closeout:** All rows **Done** or **Won't fix**; DX-3.5 + DX-8.1 green in CI; DX-3.6 quickstart validated; DX-7.1 schemas under `build/harness_specs/`. **Core path (DX1–DX2, DX3.2–3.3, DX8.3) already meets harness authoring needs.**

**Explicitly out of NOW:** K.1, K.2, visual environment builder UI, new product Tier-3 apps, Problem Radar wave 2+.

### 6.2z Phase AA execution order (Band 2g — mostly done)

**Status:** **Mostly Done** (2026-06-02) · platform **Done** · domain **Deferred** · canonical register: [Phase AA — Master deliverables register](#aa--master-deliverables-register-all-tasks).

Work **one AA ID per PR/session**; after each step update the AA master table + paydown log + conformance matrix; keep §6.1 scripts green. **Legal:** follow **hard reset** policy (AA-LEG.0.1) — no incremental preservation of legacy pipeline code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| AA0 | AA-0.1, AA-0.2, AA-S0.1–AA-S0.6, AA-LG.1, AA-APP.0.1–AA-APP.0.3 | 12 | Scaffold checklist, tier guards, deploy triad standard |
| AA1 | AA-D0.1–AA-D0.7 | 7 | README, guides, TIER3_READINESS, USAGE |
| AA2 | AA-LEG.0.2–AA-LEG.3.1 | 12 | **Legal agent hard reset** |
| AA3 | AA-LEGAPP.1–AA-LEGAPP.8 | 8 | `legal_application` + deploy triad |
| AA4 | AA-ECHO.1–AA-ECHO.5 | 5 | Reference echo agent |
| AA5 | AA-SIG.1–AA-SIG.3 | 3 | Signoff probe |
| AA6 | AA-PR.1–AA-PR.5 | 5 | Problem radar (docs/hygiene; frozen feature) |
| AA7 | AA-ORG.1–AA-ORG.5 | 5 | Organization worker |
| AA8 | AA-RES.1–AA-RES.6 | 6 | Research agents |
| AA9 | AA-LABAG.1–AA-LABAG.2 | 2 | Lab mocks |
| AA10 | AA-LABAPP.1–AA-LABAPP.7 | 7 | Lab application host |
| AA11 | AA-POC.1–AA-POC.5 | 5 | POC template (canonical shell) |
| AA12 | AA-RESAPP.1–AA-RESAPP.6 | 6 | Research application host |
| **Total** | | **83** | |

**Suggested PR order:** AA-S0.2 → AA-S0.5 → AA-APP.0.1 → AA-APP.0.3 → AA-POC.1 → AA-POC.2 → AA-LABAPP.2 → AA-ECHO.2 → AA-LEG.0.3 → AA-LEG.1.1 → AA-LEG.1.2 → AA-LEG.1.3 → AA-LEG.2.1 → AA-LEG.2.2 → … → AA-LEGAPP.1–AA-LEGAPP.6 → AA-D0.1 → AA-D0.3–AA-D0.5 → AA-RESAPP.* → AA-LABAPP.1 → AA-APP.0.2 → remaining ARCHITECTURE.md rows.

**Per-application deploy triad gate (AA-APP.0.2):** for each of `lab_application`, `legal_application`, `local_workspace_application`, `poc_template_application`, `research_application` assert:

1. `docker/Dockerfile` + `docker-compose.yml` + `build-docker.sh` / `.bat`
2. `BUILD_AND_DEPLOY.md` present and matches scaffold generator output (or documented drift)
3. `ARCHITECTURE.md` § **Dependencies** lists required `pyproject.toml` extras (e.g. `harness-author`, provider-specific `llm-*`, `dev-ci` for tests)

**Doc pair gate (AA-D0.6):** for each listed Tier-2 agent and Tier-3 application assert `ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md` exist and cross-link. Gate: `tests/unit/applications/test_agent_app_doc_pair.py`.

**Success gate for Phase AA platform closeout:** **Met** (2026-06-02) — conformance matrix **OK**; legal tree = scaffold; `lab_application` on `build_harness_host_runtime`; AA-APP.0.2 green; gate **533**. **Full AA register closeout** additionally requires Band 3 domain rows **Done** or explicitly **Deferred** (current policy: **Deferred**).

**Explicitly out of NOW:** K.1/K.2 implementation, Legal **live LLM** E2E (Band 3), new product hosts beyond the four listed, Legal UAEP step port (AA-LEG.2.2+) unless product reprioritizes §6.3.

### 6.2aa Phase MEM execution order (Band 2h — closed)

**Status:** **Done** (2026-06-02) · **48/48 Done** · canonical register: [Phase MEM — Master deliverables register](#mem--master-deliverables-register-all-48-tasks).

Work **one MEM ID per PR**; after each step update the MEM master table + paydown log; keep §6.1 scripts green. **Start with MEM-1.*** before MEM-3/MEM-7 — bridge must exist before plugins/hooks.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEM0 | MEM-0.1–MEM-0.4, MEM-PAR.1, MEM-CHk.1, MEM-PERS.1, MEM-ST.1, MEM-OBS.2 | 9 | Register, audit baseline, parity tables (MEM-OBS.2 Done) |
| MEM1 | MEM-1.1–MEM-1.6, MEM-2.1–MEM-2.3 | 9 | **P0** — H-APP bridge + SQLite user LTM |
| MEM2 | MEM-3.*, MEM-4.*, MEM-5.*, MEM-CTX.1, MEM-DOC.1–6, MEM-GRAPH.1, MEM-TASK.* | 18 | **P1** — gates, plugins, context docs |
| MEM3 | MEM-6.*, MEM-7.*, MEM-OBS.1, MEM-DOC.5, MEM-CTX.2, MEM-PERS.2, MEM-ST.4 | 9 | **P2** — retention, hooks, metrics |
| MEM4 | MEM-8.*, MEM-9.1, MEM-PERS.3 | 4 | **P3** — product RFCs |
| **Total** | | **48** | |

**Success gate:** P0 + P1 **Done**; H-APP.4.3 **Done**; user LTM durable on sqlite lab profile; `MemoryProfile` drives all reference hosts.

**Explicitly out of NOW:** K.1/K.2, Mem0 auto-ingest ship (MEM-8.2), entity graph implementation (MEM-9.1 beyond RFC).

### 6.2ab Phase MEM-DEPTH execution order (Band 2am — closed)

**Status:** **Done** (2026-06-08) · **26/26 Done** · canonical register: [Phase MEM-DEPTH — Master deliverables register](#mem-depth--master-deliverables-register-all-26-tasks).

Work **one MEM-DEPTH ID per PR**; after each step update the MEM-DEPTH master table + paydown log; keep §6.1 scripts green. **Start with MEM-DEPTH-0.3 (ADR) then MEM-DEPTH-1.*** — architecture decision before Context Compiler code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEMD0 | MEM-DEPTH-0.1–0.4 | 4 | Canon doc, plan register, ADR, cross-links |
| MEMD1 | MEM-DEPTH-1.1–1.6 | 6 | **P0** — Context Compiler + never-overflow |
| MEMD2 | MEM-DEPTH-2.1–2.2 | 2 | **P1** — Mongo session persistence parity |
| MEMD3 | MEM-DEPTH-3.1–3.5 | 5 | **P1** — Lifecycle automation |
| MEMD4 | MEM-DEPTH-4.1–4.3 | 3 | **P1** — Explore / discovery |
| MEMD5 | MEM-DEPTH-5.1–5.6 | 6 | **P2/P3** — Entity graph, temporal validity, quality gates |
| **Total** | | **26** | |

**Success gate:** P0 + P1 **Done**; never-overflow acceptance green; Memory Layer **L3+** on FAUDIT re-run.

**Explicitly out of NOW:** K.1/K.2, Mem0 SaaS, entity graph ship without §6.3 decision (MEM-DEPTH-5.1).

### 6.1p Phase P-Ext paydown (Band 2c — optional parallel with §6.1)

**Status:** **Done** (2026-06-02) · closure complete; extend catalogs via Appendix I + author guide.

| Order | ID | Deliverable | Priority |
|-------|-----|-------------|----------|
| 1 | P-Ext.0.5 | Fixture pip package (`tests/fixtures/plugin_packages/`) | P0 |
| 2 | P-Ext.0.6 | EP discovery tests (all three groups) | P0 |
| 3 | P-Ext.1.6 | Integration EP test via fixture | P0 |
| 4 | P-Ext.1.10 | Tier-3 `integration_wiring` → `bootstrap_catalogs()` | P0 |
| 5 | P-Ext.2.9–2.11 | External tool example + unit + EP tests | P0 |
| 6 | P-Ext.3.6–3.8 | External skill example + unit + EP tests | P0 |
| 7 | P-Ext.0.7 | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | P1 |
| 8 | P-Ext.4.3, 4.5, 1.8 | Conflict policy + CI smoke (incl. integration counts) | P1 |
| 9 | P-Ext.1.5, 1.7, 5.5–5.6 | Slug/docs cleanup + author guide matrix | P2 |
| 10 | P-Ext.2.12, 3.9–3.11 | Tool/skill lazy bootstrap, scaffold plugin template, importer docs | P2 |
| 11 | P-Ext.1.3a, 1.4, 1.9, 1.11–1.12 | Typed resolve expansion, health API, integration wiring helper | P3 |
| 12 | P-Ext.5.1, 3.10, 3.12 | Scaffold CLI (all three catalogs) + harness `requires_skills` demo | P3 |

Full task register: [Appendix I](#appendix-i--plugin-catalog-traceability-phase-p-ext).

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (closed).

### 6.2ag Phase M.6 P6 execution order (Band 2ac — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P6](#m6-p6--harness-integration-expansion-planned) · queue: [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done)

```text
Wave H-INT-0 (categories):  M-P6-CAT.1 → M-P6-CAT.2 → M-P6-CAT.3 → M-P6-CAT.4 → M-P6-CAT.5 → M-P6-CAT.6 → M-P6-CAT.7 → M-P6-CAT.8 → M-P6-CAT.9
Wave H-INT-10 (security):   M-P6.1 → M-P6.2 → M-P6.3 → M-P6.4
Wave H-INT-11 (sandbox):    M-P6.5 → M-P6.6 → M-P6.7
Wave H-INT-12 (identity):   M-P6.8 → M-P6.9 → M-P6.10
Wave H-INT-13 (gitops CI):  M-P6.11 → M-P6.12 → M-P6.13
Wave H-INT-14 (speech):     M-P6.14 → M-P6.15
Wave H-INT-15 (enterprise): M-P6.16 → M-P6.17 → M-P6.18 → M-P6.19
Wave H-INT-16 (data/wf):    M-P6.20 → M-P6.21 → M-P6.22 → M-P6.23 → M-P6.24
Wave H-INT-17 (reserve):    M-P6.25 → M-P6.26 → M-P6.27 → M-P6.28 → M-P6.29 → M-P6.30 → M-P6.31 → M-P6.32
Wave PRE (presets):         M-P6-PRE.1  (after H-INT-10 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P5 **Done**; M-P5.FU wiring **Done**; Phase SEC closeout **Done** (V-SEC patterns for `security_scanner`).  
**Parallelism:** H-INT-10 unblocks STABLE promote gate; H-INT-11 unblocks cloud `sandbox.exec`; H-INT-12 unblocks multi-tenant hosts; H-INT-14 unifies speech catalog.  
**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-done)

```text
Wave H-INT-0 (categories):  M-P5-CAT.1 → M-P5-CAT.2 → M-P5-CAT.3
Wave H-INT-6 (ops/CI):      M-P5.1 → M-P5.2 → M-P5.3 → M-P5.4 → M-P5.5 → M-P5.6 → M-P5.7 → M-P5.8 → M-P5.9 → M-P5.10
Wave H-INT-7 (eval/async):  M-P5.11 → M-P5.12 → M-P5.13 → M-P5.14 → M-P5.15 → M-P5.16 → M-P5.17 → M-P5.18 → M-P5.19 → M-P5.20
Wave H-INT-8 (data lab):    M-P5.21 → M-P5.22 → M-P5.23 → M-P5.24 → M-P5.25 → M-P5.26 → M-P5.27 → M-P5.28
Wave H-INT-9 (P2 reserve):  M-P5.29 → M-P5.30 → M-P5.31 → M-P5.32 → M-P5.33 → M-P5.34
Wave PRE (presets):         M-P5-PRE.1  (after H-INT-6 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P4 **Done**; M-P4.FU wiring **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** H-INT-6 unblocks W-OPS metrics + multi-CI; H-INT-7 unblocks EVAL/W-ADAPT; H-INT-8 is lab-only.  
**Closeout target:** catalog **136** slugs; `HARNESS_M6_P5_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.2ae Phase M.6 P4 execution order (Band 2aa — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P4](#m6-p4--harness-platform-expansion-done) · queue: [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed)

```text
Wave H-INT-0 (categories):  M-P4-CAT.1 → M-P4-CAT.2  (before first slug in new category)
Wave H-INT-1 (storage):     M-P4.1 → M-P4.2 → M-P4.3 → M-P4.4
Wave H-INT-2 (obs stack):   M-P4.5 → M-P4.6 → M-P4.7
Wave H-INT-3 (secrets):     M-P4.8 → M-P4.9 → M-P4.10 → M-P4.11
Wave H-INT-4 (control):     M-P4.12 → M-P4.13 → M-P4.14 → M-P4.15 → M-P4.16
Wave H-INT-5 (enterprise):  M-P4.17 → M-P4.18 → M-P4.19 → M-P4.20 → M-P4.21 → M-P4.22 → M-P4.23 → M-P4.24 → M-P4.25 → M-P4.26 → M-P4.27 → M-P4.28
```

**Prerequisites:** Phase M core + M.6 P1/P2/P3 **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** Any wave after H-INT-0 may start when a slug is needed — prefer H-INT-1 → H-INT-2 → H-INT-3 order for W-OPS/adaptive unblock.  
**Closeout:** **Done** — catalog **127** in `layout.py`; `tests/unit/integrations/providers/test_p5_m6_p4_providers.py` (42 tests).

### 6.2ad Phase M-LLM-R execution order (Band 2z — closed 2026-06-06)

**Status:** **Done** · register: [Phase M-LLM-R](#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06) · queue: [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

```text
Wave M-LLM-R-0 (planning):     M-LLM-R.0.2 → 0.3  (0.1 **Done**)
Wave M-LLM-R-1 (contracts):    M-LLM-R.1.1 → 1.8
Wave M-LLM-R-2 (ABC):          M-LLM-R.2.6 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
Wave M-LLM-R-3 (providers):    M-LLM-R.3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7
Wave M-LLM-R-4 (Nexus):        M-LLM-R.4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
Wave M-LLM-R-5 (RAG/web):      M-LLM-R.5.1 → 5.2 → 5.3
Wave M-LLM-R-6 (agents):       M-LLM-R.6.1 → 6.2 → 6.3 → 6.4
Wave M-LLM-R-7 (obs/replay):   M-LLM-R.7.1 → 7.2 → 7.3 → 7.4 → 7.5
Wave M-LLM-R-8 (closeout):     M-LLM-R.8.1 → 8.2 → 8.3 → 8.4
```

**Prerequisites:** Phase M-LLM **Done** (M-LLM.1–13); no dependency on W-ADAPT runtime L4 gate.

**Parallelism:** May run alongside W-ADAPT-5+; coordinate M-LLM-R.7.5 with W-ADAPT signal work if both touch `signal_collector.py`.

**Closeout gate:** `scripts/check_llm_adapter_typed_returns.py` + `scripts/check_agents_llm_adapter_response.py` + full `tests/unit/llm_adapters/` gate green (M-LLM-R.8.3, M-LLM-R.6.4).

### 6.2ac Phase W-ADAPT execution order (Band 2y — closed)

**Status:** **Done** (2026-06-02) · register: [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime) · queue: [§6.1t](#61t-harness-implementation-queue--adaptive-harness-intelligence-closed)

```text
Wave W-ADAPT-0 (planning):        W-ADAPT-0.2 → 0.3 → 0.4 → 0.5  (**Done**)
Wave W-ADAPT-1 (observe L4-O):    W-ADAPT-1.1 → 1.12  (**Done**)
Wave W-ADAPT-2 (recommend L4-R):  W-ADAPT-2.1 → 2.12  (**Done**)
Wave W-ADAPT-3 (shadow L4-S):      W-ADAPT-3.1 → 3.2 → 3.3 → 3.4 → 3.6 → 3.7 → 3.5  (**Done**)
Wave W-ADAPT-4 (apply L4-A):       W-ADAPT-4.1 → 4.10  (**Done**)
Wave W-ADAPT-5 (verify L4-V):      W-ADAPT-5.1 → 5.3 → 5.4 → 5.5 → 5.2 → 5.11 → 5.6 → 5.7 → 5.8 → 5.9 → 5.10 → 5.12  (**Done**)
Wave W-ADAPT-6 (patterns):         W-ADAPT-6.2 → 6.1 → 6.3 → 6.5 → 6.4  (**Done**)
Wave W-ADAPT-7 (Tier-3 + docs):    W-ADAPT-7.1 → 7.2 → 7.3 → 7.4 → 7.5 → 7.6 → 7.7  (**Done**)
```

**Prerequisites:** Phase V + V-REM + W-OPS + EVAL + COST + CG closeouts **Done**.

**Runtime L4 gate:** `uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime` (added in W-ADAPT-5.6).

### 6.2 Harness architecture hardening (Band 2 — Phase V) — Done

**Status:** **Done** (2026-06-05) — Phase V contracts + V-REM runtime enforcement complete. Closeout: `phase_v_closeout_gate.py --enforce --enforce-l4`.

| Item | Status | Notes |
|------|--------|-------|
| V-CG … V-V6 | **Done** | Governance + CI + runtime enforcement |
| V-REM | **Done** | 10/10 closed — §6.1z queue closed |
| W-ML | **Done** | [architecture/MODALITY.md](architecture/MODALITY.md) |
| P-Ext | **Done** | Appendix I |
| M.6 P5 / M.6 P6 / R-Skill expansion | **On demand** | W-OPS.10, W-OPS.8, §6.1x, §6.1y |

**Forbidden in Band 2/2d:** K.1, K.2, product-specific skills, new product application hosts.

### 6.3 End of plan — deferred product work only (Band 3)

**This section is the last band in the implementation plan.** Nothing here is the default “next step” after harness work.

| ID | Deliverable | Status | Gate to start |
|----|-------------|--------|----------------|
| K.1 | Problem Radar prototype | **Deferred** | Explicit product decision + [Appendix A](#appendix-a--agent-operating-system-certification-checklist) |
| K.2 | Vendor Discovery prototype | **Deferred** | Same as K.1 |
| K.6 / B.15 / S-Ops.4 | Legal live LLM E2E | **Deferred** | Product/CI budget decision |
| `agents/legal` UAEP domain steps | Scaffold shell **Done** (Band 2g); step port **Deferred** | **Business** | [§6.3a](#63a-business-backlog-register-consolidated) AA-LEG.2.2+ |
| Tier-3 product apps | New `applications/<product>/` beyond lab + reference hosts | **Deferred** | Product decision; scaffold exists (Phase N **Done**) |
| Domain skills | Product agent skill packs (non-`harness.*`) | **Deferred** | With K.1 or K.2 |
| `agents/problem_radar/` | Wave 1 scaffold frozen | **Deferred** | Do not extend until K.1 reprioritized |

**When Band 3 may start:** Record the decision in this plan (date + chosen K.1 vs K.2), then follow [guides/AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md). Tier-3 scaffold reference (Phase N) applies **only after** that decision — not as ongoing harness work.

**Tier-3 scaffold (for when Band 3 is approved):**

```bash
python -m intergrax.scaffold new-stack <slug> --profile lab --capability <slug>.basic
```

See [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md). Existing hosts (`lab_application`, `legal_application`, `research_application`, `poc_template_application`) are sufficient for **all harness** work. **Product:** [`local_workspace_application`](../applications/local_workspace_application/) — Local Knowledge Workspace (LKW) — first business environment after harness GA; see [ARCHITECTURE.md](../applications/local_workspace_application/ARCHITECTURE.md).

### 6.3a Business backlog register (consolidated)

**Single register for Band 3 and AA domain-deferred rows.** Do not duplicate in harness session summaries.

| ID | Deliverable | Module | Priority | Depends on |
|----|-------------|--------|----------|------------|
| **LKW.0** | Local Knowledge Workspace — scaffold + architecture baseline | `agents/local_{indexer,search,synthesizer}/`, `applications/local_workspace_application/` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **LKW.1** | Wave 1 — ingest + search smoke on explicit paths | `agents/local_*/steps/` | **High** | LKW.0 |
| **LKW.2** | Multi-agent pipeline (`local.workspace.pipeline` graph) | `local_workspace_application/` + Nexus graph | High | LKW.1 |
| **LKW.3** | Tier-0 `filesystem.*` read tools + allowlist policy | `intergrax/tools/providers/filesystem/` | Medium | LKW.1 |
| **LKW.4** | Background ingest queue + incremental index | Tier-0 queue + Tier-3 worker | Medium | LKW.2 |
| **LKW.5** | `LKW_DATA_HOME` + Chroma persistent local index | `local_workspace_application/host/settings.py` | High | LKW.1 |
| **LKW.6** | Local OS daemon (Win/Linux/macOS) + interaction intake on host | `local_workspace_application/` | High | LKW.1 |
| **LKW.6b** | Slack Socket Mode + slash command → Nexus (interaction surface) | Tier-3 + `slack` integration | Medium | LKW.6 |
| **LKW.7** | Background file watcher + incremental index + optional Slack notify | Tier-0 queue + Tier-3 worker | Medium | LKW.3 |
| **LKW.8** | Tray / file-picker UI (localhost HTTP/MCP client) | Product (out of harness) | Low | LKW.6 |
| **DSW.0** | Dispute Simulation Workspace — scaffold + architecture baseline | `agents/dispute_{intake,analyst,strategist,scenario}/`, `applications/dispute_sim_application/` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **DSW.1** | Wave 1 — case intake + RAG ingest + timeline artifact | `agents/dispute_intake/steps/` | **High** | DSW.0 |
| **DSW.2** | Multi-agent pipeline (`dispute.pipeline` graph) | `dispute_sim_application/` + Nexus graph | High | DSW.1 |
| **DSW.3** | Analyst matrix + strategist brief domain steps | `agents/dispute_analyst/`, `agents/dispute_strategist/` | High | DSW.1 |
| **DSW.4** | Scenario variants + correspondence review + HITL | `agents/dispute_scenario/` | High | DSW.3 |
| **DSW.5** | Optional subgraph to `legal.review` for clause drill-down | Nexus graph | Medium | DSW.3 |
| **DSW.6** | Case persistence + retention policy | `dispute_sim_application/host/settings.py` | Medium | DSW.1 |
| **DSW.7** | Polish dispute eval fixtures + regression | `tests/` / agent eval | Medium | DSW.4 |
| **K.1** | Problem Radar prototype (wave 2+) | `agents/problem_radar/` | Product | Explicit reprioritization |
| **K.2** | Vendor Discovery prototype | (greenfield) | Product | K.1 decision or parallel product call |
| **AA-LEG.2.2** | Legal UAEP steps (one step per PR from `SPEC_FROM_LEGACY.md`) | `agents/legal/steps/` | High | Product/legal owner |
| **AA-LEG.2.3** | Remove any parallel legal runtime (Nexus gateway only) | `agents/legal/` | High | AA-LEG.2.2 |
| **AA-LEG.2.4** | Legal agent tests per ported step | `agents/legal/tests/` | High | AA-LEG.2.2 |
| **AA-LEGAPP.6** | `legal_application` host smoke on real steps | `legal_tests/` | High | AA-LEG.2.2 |
| **AA-LEGAPP.8** | Consolidate duplicate legal test trees | `legal_tests/` vs agent tests | Low | AA-LEG.2.4 |
| **AA-RES.4** | Research skill ids on contracts | `agents/research/` | Medium | Product |
| **AA-RES.5** | Research UAEP + graph delegation tests | `agents/research/tests/` | High | Product |
| **AA-RESAPP.6** | Research application smoke + manifest wiring | `research_application_tests/` | High | AA-RES.5 |
| **AA-ORG.3** | Organization worker scaffold-align (`contract`, `steps/`) | `agents/organization_worker/` | Medium | Harness demo |
| **AA-ORG.4** | Lab manifest flag + integration test | `lab_application/manifest.py` | Medium | AA-ORG.3 |
| ~~AA-LABAPP.6~~ | ~~Extra lab host smoke~~ | — | — | **Done** (2026-06-02) — not in business queue |
| **K.6 / B.15 / S-Ops.4** | Legal full E2E with live LLM | CI / acceptance | Low | CI budget approval |
| **Tier-3 product** | New `applications/<product>/` beyond four reference hosts | `applications/` | Product | Phase N scaffold + §6.3 decision |
| **Domain skills** | Non-`harness.*` skill packs for product agents | `intergrax/skills/providers/` | Product | With K.1 or K.2 |
| **A.5** | Full Legal regression (all steps, live model) | Phase A row | Low | K.6 / B.15 |
| **Phase E** | Legal agent refactoring (parallel track) | `agents/legal/` | On demand | Product architecture |

**Not business (infrastructure — closed; see [§6.1z](#61z-harness-implementation-queue-consolidated)):** DX-5.7, AA-LEG.0.2, OPS-L3.1 **Done**; ongoing **§6.1** maintenance only.

### 6.1u Archived — Phase U cadence (complete 2026-06-01)

Security, policy, contracts, typing (U-Sec through U-CI). See Phase U definition of done. Residual U-Leg.* moved to §4.1 — not reopened as a new phase.

### 6.1s Archived — Phase S cadence (complete 2026-06-01)

See Phase S definition of done and Appendix F. Do not reopen S.* unless regression (fix under T.* or U.*).

### 6.1a Archived — Phase Q cadence (complete 2026-06-01)

Phase Q used **one Q.* deliverable per PR** → update Appendix C + paydown log. See Appendix C for Waves 1–9 and gate **417** at close. Do not reopen Q.* unless regression found (residual hardening → Appendix D).

### 6.1b Phase N (complete)

Tier-3 scaffold cadence remains the reference for new applications (`new-stack`); lab defaults include RAG/websearch tools and legal + research skill bundles.

### 6.4 Historical gate milestones (archived)

Phases F–L, J, Q, Q+, R, S, T, U, and §4.1 are **Done**. Gate milestones: **417** (Phase Q), **481** (harness completion, 2026-06-02). Phase tables: §2–§3; paydown: Appendices C–G.

> **Note:** Older phase closers said “next: Phase K (K.1/K.2).” That meant harness prerequisites were met, **not** that product work becomes the default implementation queue. **Current rule:** §4.0 Band 3 / §6.3 only after explicit product prioritization.

### D.2 Debug API (Done)

Standalone laboratory server:

```bash
uv run uvicorn intergrax.debug.app:create_debug_app --factory --host 127.0.0.1 --port 8099
```

Endpoints (mirror CLI):

```text
GET /debug/tasks?tenant=t1&limit=20
GET /debug/tasks/{run_id}?tenant=t1
GET /debug/tasks/{run_id}/trace?tenant=t1&include_runtime=true
```

Mount on an existing app:

```python
from intergrax.debug.router import create_debug_router

app.include_router(create_debug_router(db_path=Path("build/intergrax_trace.db")))
```

Environment: `INTERGRAX_TRACE_DB` (same as CLI).

### D.3 Experiment registry (Done)

SQLite registry at `build/intergrax_experiments.db` (`INTERGRAX_EXPERIMENTS_DB`).

```bash
python -m intergrax.debug experiments register --hypothesis "..." --capability echo.basic
python -m intergrax.debug experiments link-run EXPERIMENT_ID RUN_ID
python -m intergrax.debug experiments decide EXPERIMENT_ID --decision keep
python -m intergrax.debug experiments list --decision pending
```

HTTP: `GET/POST /debug/experiments`, `POST /debug/experiments/{id}/decision`, `POST /debug/experiments/{id}/runs/{run_id}`.

### D.4 Experiment workflow API (Done)

Platform-level `notebooks/` was **removed** (2026-06-12). §35 workflow: `intergrax.experiments.workflow.ExperimentSession`; tests in `tests/unit/experiments/`.

```python
from pathlib import Path
from intergrax.experiments.workflow import ExperimentSession, ensure_repo_root_on_path

ensure_repo_root_on_path()
session = ExperimentSession(trace_db=Path("build/experiments/trace.db"))
```

### D.5 Cost in trace (Done)

`AgentExecutionResult.cost` and `duration_seconds` are derived from LLM usage (`intergrax/contracts/runtime_cost.py`):

- Mapping: `runtime_answer_to_agent_result()` reads `llm_usage_report` or `stats.extra.cost`
- NexusLoop: aggregates multi-agent cost into task metadata (`execution_cost`) and `RunStats.llm_usage` on finalize
- Debug API/CLI: `stats.cost` on run detail; CLI `tasks show` prints cost line

Cost proxy: **1 cost unit = 1 LLM token** (laboratory default, matches EvalRunner).

### F.1 Shadow workspace (Done)

Isolated temporary filesystem for experiments (§20). Enable on a Nexus task:

```python
task = Task(
    tenant_id="t1",
    user_id="u1",
    message="analyze vendor",
    context=TaskContext(capability="research.web_search"),
    metadata={"shadow_workspace": True},  # optional: "shadow_workspace_cleanup": True
)
```

UAEP agents receive `ctx.metadata["shadow_workspace"]` in `run_step`. Result metadata includes `shadow_workspace_id`.

Root directory: `INTERGRAX_SHADOW_ROOT` (default `build/shadow_workspaces/`).

### F.2 Sandbox runtime (Done)

Controlled session for risky tool use (§21). Enable on a Nexus task:

```python
task = Task(..., metadata={"sandbox": True})
```

Agents invoke allowlisted operations through the tool gateway:

```python
await ctx.invoke_tool(ToolRequest(
    tool_name="sandbox.exec",
    agent_id=ctx.agent_id,
    input={"operation": "write_file", "payload": {"path": "out.txt", "content": "..."}},
))
```

Operations: `echo`, `write_file`, `read_file`, `list_files`. Root: `INTERGRAX_SANDBOX_ROOT` (default `build/sandbox_sessions/`).

### F.3 Advanced HITL (Done)

Human responses beyond approve:

```python
# Re-submit paused task with verdict
task = Task(..., task_id=original_task_id, metadata={"human_response": "reject"})
# or "approve" / "escalate"
```

- **reject** → task `FAILED`, decision persisted
- **escalate** → `INTERRUPT_ESCALATED` event, escalation chain in metadata, stays `WAITING_FOR_HUMAN`
- Store: `INTERGRAX_HUMAN_DECISIONS_DB` (default `build/intergrax_human_decisions.db`)

Optional on `NexusLoop`: `human_decision_store=SQLiteHumanDecisionStore(...)`.

### F.4 Long-running tasks (Done)

Enable durable pause/resume on Nexus tasks (§26):

```python
from intergrax.runtime.task import Task, TaskExecutionOptions, TaskLongRunningOptions

task = Task(
    tenant_id="t1",
    user_id="u1",
    message="monitor vendors for 30 days",
    context=TaskContext(capability="hitl.basic"),
    options=TaskExecutionOptions(
        long_running=TaskLongRunningOptions(
            enabled=True,
            notify_channel="slack",  # or "teams" / "log"
        ),
    ),
)
```

On pause (`WAITING_FOR_HUMAN`), NexusLoop persists a checkpoint with `resume_token` in result metadata.

Resume with the same `task_id` and token:

```python
Task(
    ...,
    task_id=original_task_id,
    options=TaskExecutionOptions(
        long_running=TaskLongRunningOptions(enabled=True, resume_token=token),
    ),
    metadata={"human_approved": True, "resume_token": token},
)
```

Optional on `NexusLoop`: `checkpoint_store=SQLiteTaskCheckpointStore(...)`, `notification_adapter=LoggingNotificationAdapter()`.

Env:

- `INTERGRAX_TASK_CHECKPOINTS_DB` (default `build/intergrax_task_checkpoints.db`)
- `INTERGRAX_RUNTIME_EVENTS_DB` (optional; enables SQLite runtime events in NexusLoop / debug API)
- `INTERGRAX_TASK_MEMORY_DB` (optional; TaskMemory SQLite path for lab / debug)
- `INTERGRAX_SLACK_WEBHOOK_URL` / `INTERGRAX_TEAMS_WEBHOOK_URL` (stub adapters; no network unless configured)

### H.6 Organization Worker lab runbook (Done)

Reference flow for §38 — virtual worker via Slack / Teams without orchestration in adapters.

**Agent:** `agents/organization_worker/` — capability `org.vendor_report`.

**Lab app factory:**

```python
from intergrax.lab import create_organization_worker_lab_app

app = create_organization_worker_lab_app()  # pre-wired registry + HITL intake enricher
```

**HTTP (debug API):**

```bash
uv run uvicorn intergrax.lab.organization_worker:create_organization_worker_lab_app --factory --host 127.0.0.1 --port 8099
```

1. **Intake + execute** (Slack-shaped slash command):

```bash
curl -s -X POST "http://127.0.0.1:8099/debug/interactions/intake?execute=true&tenant=T1" \
  -H "Content-Type: application/json" \
  -d '{"command":"/intergrax","text":"org.vendor_report Acme Corp Q1","user_id":"U1","team_id":"T1"}'
```

Response includes `state: waiting_for_human`, `resume_token`, HITL notification on configured channel (`slack` / `teams` / `log`).

2. **Resume after approval:**

```bash
curl -s -X POST "http://127.0.0.1:8099/debug/tasks/{task_id}/human-response?tenant=T1" \
  -H "Content-Type: application/json" \
  -d '{"response":"approve","resume_token":"<token from intake>"}'
```

Teams intake uses the same endpoints with Bot Framework activity JSON (`channelId: msteams`).

**Registry helper:** `build_organization_worker_registry()` in `intergrax.runtime.registry`.

**Tests:** `tests/integration/debug/test_organization_worker_demo.py` (gate).

### D.1 Debug CLI (Done)



```bash

python -m intergrax.debug tasks list --tenant t1 --limit 20

python -m intergrax.debug tasks show RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1 --format json --runtime

python -m intergrax.debug --db path/to/trace.db tasks list

```



Reuse:



- `SQLiteRunTraceStore` / `RunTraceReader` — `intergrax/runtime/nexus/tracing/`

- `trace_bridge` — `intergrax/runtime/events/trace_bridge.py`

- `NexusLoop.event_bus` — in-process runs (not persisted; CLI uses SQLite trace)



---

## Appendix A — Business agents readiness checklist

Gate before Problem Radar / Vendor Discovery. Run:

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
uv run pytest tests/ -m gate -q
```

### Agent creation & registration

| # | Question | Status |
|---|----------|--------|
| 1 | Scaffold in minutes (`intergrax.scaffold new-agent`)? | ✅ |
| 2 | UAEP structure generated (contract, steps, tests)? | ✅ |
| 3 | First run in < 1 hour? | ✅ |
| 4 | Register via `AgentRegistry` only (no Nexus edits)? | ✅ |
| 5 | Capabilities in contract? | ✅ |

### Execution & observability

| # | Question | Status |
|---|----------|--------|
| 6 | Runs through NexusLoop / lab `/v1/lab/run`? | ✅ |
| 7 | UnifiedTaskRunner same path as HTTP? | ✅ |
| 8 | Graph sequential + parallel? | ✅ |
| 9 | Trace via `/debug/tasks/{id}`? | ✅ |
| 10 | Runtime events + checkpoints + progress? | ✅ |

### Recovery, HITL, memory, isolation

| # | Question | Status |
|---|----------|--------|
| 11 | Nexus validates output? | ✅ |
| 12 | Retry / alternate agent on validation failure? | ✅ |
| 13 | HITL pause + resume? | ✅ |
| 14 | Checkpoint recovery? | ✅ |
| 15 | Shared context in graphs? | ✅ |
| 16 | Sandbox + shadow workspace? | ✅ |

### Tooling & composition

| # | Question | Status |
|---|----------|--------|
| 17 | Canonical agent guide exists? | ✅ |
| 18 | Lab application (Tier-3)? | ✅ |
| 19 | Same agent reusable across applications? | ✅ |
| 20 | Applications contain wiring only? | ✅ |

### Go / no-go

| Criterion | Threshold | Current |
|-----------|-----------|---------|
| Checklist | ≥ 90% | **20/20** |
| Acceptance suite | 10/10 green | ✅ |
| Sign-off exercise | 1 new agent, < 1h, zero runtime edits | **Done** (`signoff_probe`) |

**Verdict:** **L1 Agent Operating System certified** (technical). **Phase S** (harness environment GA) is next; **K.1/K.2** wait until S is **Done**.

### Sign-off record

```text
Date:           2026-05-27
Agent exercise: signoff_probe
Capability:     signoff.probe
Time to first run: ~15 min (scaffold + smoke test)
Runtime files modified: none (only agents/signoff_probe/ added)
Smoke test:     agents/signoff_probe/tests — 1 passed
HTTP proof:     lab_application wiring + POST /v1/lab/run
Trace proof:    GET /debug/tasks/{id}, /trace?include_runtime=true, /events
                (test_lab_application_runs_signoff_probe_with_trace)
Acceptance suite: pass (tests/acceptance/agent_os)
Gate suite:     pass (228+ tests)
Trace:          NexusLoop smoke + HTTP debug API (SQLite trace store in lab factory)
Decision:       L1 certified — GO Phase S (harness environment), then Phase K (K.1/K.2)
```

---

## Appendix B — Technical debt backlog

**Purpose:** consolidated backlog for review and **incremental paydown**.  
**Source:** canon §2 map, §0.5 maturity, Phase G–K gaps, lab sign-off findings (2026-05-27).  
**How to use:** pick items by priority; apply §0.6 (Tier-1 only when reusable across agents).  
**Status:** `Open` | `Done` | `Deferred`

### B.0 Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-05-29 | M.6-gcp | `providers/gcp/` — cloud_platform facade; ADC/service account + category slug defaults |
| 2026-05-29 | M.6-azure | `providers/azure/` — cloud_platform facade; token health + category slug defaults |
| 2026-05-29 | M.6-aws | `providers/aws/` — cloud_platform facade; STS health + category slug defaults |
| 2026-05-29 | M.6-cassandra | `providers/cassandra/` + `contracts/document_store.py`; CQL partition-scoped CRUD |
| 2026-05-29 | M.6-ms365_graph | `providers/ms365_graph/` + `contracts/collaboration_suite.py`; Graph mail/calendar/directory |
| 2026-05-30 | M.6-prometheus | `providers/prometheus/` + `contracts/observability_backend.py`; PromQL query API |
| 2026-05-30 | M.6-confluence | `providers/confluence/` + `contracts/wiki_knowledge.py`; REST wiki; single-entry `opens.py` |
| 2026-05-30 | M.6-jira | `providers/jira/` + `contracts/issue_tracker.py`; REST v3; single-entry `opens.py` |
| 2026-05-30 | M.6-mysql | `providers/mysql/` — beta `RelationalStore` (pymysql); single-entry `opens.py` |
| 2026-05-30 | M.6-provider-layout | Providers grouped under `providers/<category>/<slug>/`; `layout.py` slug map; tests mirrored by category |
| 2026-05-30 | M.6-p2-batch | P2/P3 integrations — 22 slugs (`azure_blob`, `gcs`, `dynamodb`, cloud queues, SQL variants, SMTP, OTEL, GitHub/Linear/Azure DevOps, Notion/SharePoint, Google Workspace, Brave/SerpAPI, Playwright); `_shared/p2/`; **324** integration unit tests |
| 2026-05-30 | M.7-agent-guide-integrations | `guides/AGENT_CREATION_GUIDE.md` Appendix E — agents vs Tier-3 wiring |
| 2026-05-30 | N.2.1-unified-wiring | `ApplicationBuildContext`, `builder_key`/`factory_path`, lab+legal on `build_application_registry` |
| 2026-05-30 | N.2-conformance | `build_registry_from_manifest`, `load_agent_from_binding` + unit tests |
| 2026-05-30 | N.1-manifest | `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` + unit tests |
| 2026-05-30 | N.10-new-stack | `scaffold new-stack` — agent + application; `TIER3_READINESS.md` |
| 2026-05-30 | N.9-scaffold-acceptance | `test_scaffold_acceptance.py` — lab/product runtime E2E; fix product `agent_factories.py` indent |
| 2026-05-30 | N.8-agent-guide-4e | `guides/AGENT_CREATION_GUIDE.md` Step 4E — `new-application`, Docker scripts, §7.4.8 links |
| 2026-05-30 | N.4-product-scaffold | `--profile product` → FastAPI Core host, `agent_factories.py`, auth stub env; `new_application_product.py` |
| 2026-05-30 | N.5-docker-build-scripts | `build-docker.sh` / `build-docker.bat` in scaffold + lab/legal/research/poc; `docker_templates.py` |
| 2026-05-30 | N.0-docs | Canon §7.4.8–§7.4.10 + Phase N plan (application environment, manifest, scaffold steps) |
| 2026-05-30 | M.8-lab-profile | `wire_lab_integrations()` + `providers/log/` — lab uses `IntegrationProfile.lab()` |
| 2026-05-30 | M.4-kafka-rabbitmq-adopt | Queueing bootstrap + integration tests use `integrations/providers/{kafka,rabbitmq}/` only |
| 2026-05-30 | M.4-rabbitmq | `providers/rabbitmq/` + runtime `build_rabbitmq_transport()` delegate |
| 2026-05-29 | M.4-lab_json | `providers/lab_json/` + runtime `create_interaction_adapter(LAB)` delegate — **M.4 P0 complete** |
| 2026-05-29 | M.4-webhook | `providers/webhook/` + runtime `create_notification_adapter(WEBHOOK)` delegate |
| 2026-05-29 | M.4-teams-adopt | Runtime notifications/interactions/verifier + long_running delegate to `providers/teams/` |
| 2026-05-29 | M.4-teams | `providers/teams/` — dual category catalog entry |
| 2026-05-29 | M.4-slack-adopt | Runtime notifications/interactions/verifier + long_running delegate to `providers/slack/` |
| 2026-05-29 | M.4-slack | `providers/slack/` — dual category + resolve dispatches by category |
| 2026-05-29 | M.4-bing | `providers/bing/` — SearchProvider adapter over legacy Bing v7 |
| 2026-05-29 | M.4-google_cse | `providers/google_cse/` — SearchProvider adapter over legacy CSE |
| 2026-05-29 | M.4-celery | `providers/celery/` — message bus + worker helpers; no `kv_store` |
| 2026-05-29 | M.4-kafka | `providers/kafka/` + transport delegate; requires `kv_store` |
| 2026-05-29 | M.4-sqlite-adopt | Runtime `open_*` + apps delegate to `integrations/providers/relational_store/sqlite/` |
| 2026-05-29 | M.4-sqlite | `providers/sqlite/` + bundle (10 domain stores); lazy bootstrap + package `__init__` |
| 2026-05-29 | M.4-redis | Complete bundle: `create_redis_integration()` — KV, idempotency, rate limit, semaphore, rerank |
| 2026-05-27 | B.08, B.10 | `wire_nexus_observability` + SQLite defaults in Legal / Research / Lab factories; integration test |
| 2026-05-27 | B.01, B.02 | `RuntimeCheckpoint` full snapshot + UAEP mid-step cursor/resume; acceptance `05b` |
| 2026-05-27 | B.12, B.14 | Production `POST /v1/interactions/intake` on lab; Legal legacy `AgentEngine` removed |
| 2026-05-27 | B.05 | Escalation notification template + scheduler wiring in lab + SAFETY_VIOLATION timeout→escalate |
| 2026-05-27 | B.09, B.17 | Injectable `trace_store` on debug API; gate uses `pytest -m gate` (`testpaths` includes `agents/`) |
| 2026-05-27 | Platform stabilization | All Tier-3 hosts: validating runtime events, plugin bootstrap, resilient delivery (lab/legal/research/poc); shared `_shared/platform_wiring` + `notification_wiring` |
| 2026-05-27 | Infra paydown | SQLite DLQ ledger + debug `/notifications/*`; `ValidatingRuntimeEventPersistence`; Tier-3 plugin bootstrap |
| 2026-05-27 | B.07, B.11, B.13, B.18, B.24 | Schema registry + phase coverage + `RuntimePlugin`; metrics export + `GET /debug/tasks/{id}/metrics`; retry/DLQ delivery; echo + research_mock HTTP trace acceptance; agents vendor import gate test |
| 2026-05-27 | K.3–K.5 | `coerce_replay_policy_engine` + `ExecutionGuard.evaluate_replay`; ChatAgent production import guard; CI gate paths aligned with full gate (**394** tests) |
| 2026-05-27 | B.06, §18 | `BEFORE/AFTER_TOOL_CALL` + agent-selection hooks; product interaction intake on legal/research (**397** gate) |

### B.1 Runtime & §42 convergence

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.01 | **UAEP mid-step checkpoint** — resume inside a long-running step (not only between steps / HITL) | §42.9.3, §26 | **High** | **Done** | Long-running domain agents (Legal, Research) | Tier-1 | `uaep_step_cursor`, `should_resume_uaep_step`, optional `resume_step` (2026-05-27) |
| B.02 | **Full checkpoint snapshot** — plan + graph node states + UAEP index + pending decisions in one durable blob | §42.9.2 | **High** | **Done** | Multi-agent graphs, crash recovery | Tier-1 | `plan_snapshot`, `graph_snapshot`, `pending_decisions` in `RuntimeCheckpoint` (2026-05-27) |
| B.03 | **Policy engine facade** — single `PolicyEngine` for replay, validation, runtime policy | §42.11 | **Medium** | **Done** | Indirect — consistent governance for all agents | Tier-1 | `PolicyEngine` + `coerce_policy_engine`; Nexus/UAEP/interrupt handler (2026-05-27) |
| B.04 | **Dual `AgentDecision` cleanup** — converge tools-agent variant with canonical §42.7 enum | §42.7 | **Medium** | **Done** | Agents emitting decisions must use one contract | Tier-1 | `ToolPlanDecision` in `tools.core.tool_plan_decision`; no `tools_agent` re-export (2026-06-02) |
| B.05 | **Escalation policy production path** — `SAFETY_VIOLATION` / HITL expiry → real escalation (not stub) | §42.38, §42.10 | **Medium** | **Done** | HITL-heavy agents | Tier-1 | `escalation.v1` template, `wire_long_running_scheduler`, lab startup, SAFETY_VIOLATION timeout→escalate (2026-05-27) |
| B.06 | **Hook / middleware parity** — full §42.20 pipeline vs current Nexus-embedded hooks | §42.20, §42.22 | **Low** | **Done** | Extension agents via plugins | Tier-1 | Lifecycle + **tool call** + **agent selection** hooks; decision/interrupt/retry hooks remain optional (2026-05-27) |
| B.07 | **§42 maturity remainder** — schema versioning (§42.29), full `ExecutionPhase` coverage, plugin contracts | §42 | **Medium** | **Done** (baseline) | Platform stability for new agents | Tier-1 | `runtime/schema/registry.py`, `events/phase_coverage.py`, `plugins/contract.py` (2026-05-27) |

### B.2 Observability & debug surface

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.08 | **Application trace store split** — factories used `InMemoryRunTraceStore` while debug API reads SQLite | §33, §42.24 | **High** | **Done** | HTTP `/debug/tasks/*` 503 in product apps | Tier-3 | `wire_nexus_observability` + `open_run_trace_store` (2026-05-27) |
| B.09 | **Debug API trace reader** — only SQLite file path; no injectable in-memory / shared store handle | §19 | **Medium** | **Done** | Lab tests, local dev without file I/O | Tier-1 | `trace_store` on `create_debug_router` / `create_debug_app`; lab passes Nexus store (2026-05-27) |
| B.10 | **NexusLoop runtime events in app factories** — all Tier-3 factories pass runtime events to Nexus | §42.24 | **Medium** | **Done** | Events 503 on `/debug/tasks/{id}/events` | Tier-3 | Legal + Research default SQLite; lab when path passed (2026-05-27) |
| B.11 | **Metrics layer** — event-first, trace-second, **metrics-third** unified export | §42.1, §33 | **Low** | **Done** | Ops visibility, SLOs | Tier-0 | `runtime/metrics/export.py` + `GET /debug/tasks/{run_id}/metrics` (2026-05-27) |

### B.3 Interaction surfaces (§18)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.12 | **Production Slack / Teams webhooks** — inbound intake on product hosts | §18 | **Medium** | **Done** | Organization Worker, HITL from chat | Tier-0 / Tier-3 | `POST /v1/interactions/intake` on lab/legal/research/poc via `wire_interaction_intake_service` (2026-05-27) |
| B.13 | **Outbound delivery hardening** — retries, DLQ, delivery receipts for HITL notifications | §18, §42.10 | **Low** | **Done** | HITL agents in prod | Tier-0 | `RetryingNotificationDelivery` + `SQLiteDeliveryLedger` + debug `/debug/notifications/*` (2026-05-27) |

### B.6 Integration Library (§7.1)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.18 | **Integration catalog package** — `intergrax/integrations/` scaffold | §7.1.1 | **High** | **Done** | All agents needing external systems | Tier-0 | M.1–M.3 + M.5 (2026-05-29) |
| B.19 | **P0 provider wraps** — M.4 catalog slugs | §7.1.3 | **High** | **Done** | Lab + first prod apps | Tier-0 | All P0 slugs wrapped + runtime adoption (2026-05-29) |
| B.20 | **PostgreSQL relational_store** — production DB adapter | §7.1.3 | **Medium** | **Done** (beta) | Multi-tenant applications | Tier-0 | `providers/postgresql/` — domain stores SQLite-first |
| B.21 | **Jira + Confluence providers** — issue/wiki ingestion | §7.1.3 | **Medium** | **Done** (beta) | PM / research agents | Tier-0 | Integrations + catalog tools (Phase O.4, 2026-05-30) |
| B.22 | **MS365 Graph provider** — mail, calendar | §7.1.3 | **Medium** | **Done** (beta) | Org worker, scheduling agents | Tier-0 | `providers/ms365_graph/`; client credentials via `opens.py` |
| B.23 | **Prometheus observability_backend** — PromQL query API | §33, §7.1.3 | **Low** | **Done** (beta) | Ops / SLO | Tier-0 | `providers/prometheus/`; complements B.11 metrics layer design |
| B.28 | **Cassandra document_store** — wide-column adapter for high-volume retention | §7.1.3 P2 | **Medium** | **Done** (beta) | Runtime event archive at scale; ops telemetry | Tier-0 | `providers/cassandra/`; single-entry `opens.py` |
| B.29 | **Elasticsearch observability_backend** — log search / aggregations | §7.1.3 P2 | **Medium** | **Done** (beta) | Ops log triage; optional RAG over logs | Tier-0 | `providers/elasticsearch/`; single-entry `opens.py`; complements B.23 |
| B.30 | **Databricks relational_store** — SQL Warehouse / Unity Catalog SQL | §7.1.3 P2 | **Medium** | **Done** (beta) | Analytics agents, lakehouse reporting | Tier-0 | `providers/databricks/`; single-entry `opens.py`; PAT |
| B.31 | **MongoDB document_store** — flexible JSON persistence | §7.1.3 P2 | **Medium** | **Done** (beta) | Agent memory, unstructured artifacts | Tier-0 | `providers/mongodb/`; PyMongo only in `opens.py`; reuses `DocumentStore` |
| B.32 | **Pinecone vector_store bridge** — catalog entry → `rag/` | §7.1.3 P2 | **Medium** | **Done** (beta) | Production RAG agents | Tier-0 | `providers/pinecone/` thin adapter; SDK only in `opens.py` |
| B.33 | **Qdrant + Chroma vector_store bridges** — same pattern as B.32 | §7.1.3 P2 | **Low** | **Done** (beta) | Self-hosted / dev RAG | Tier-0 | `providers/qdrant/`, `providers/chroma/`; RAG bootstrap via catalog |
| B.34 | **Object storage contract + S3 provider** — blobs for artifacts / sandboxes | §7.1.3 P2 | **Medium** | **Done** (beta) | Large file handoff, exports | Tier-0 | `contracts/object_storage.py`, `providers/s3/`; boto3 only in `opens.py` |
| B.35 | **Notion + SharePoint wiki_knowledge** — internal docs ingestion | §7.1.3 P3 | **Low** | **Done** (beta) | Research / runbook agents | Tier-0 | REST adapters; `_shared/p2/factories.py` |
| B.36 | **GitHub + Linear issue_tracker** — dev workflow sources | §7.1.3 P3 | **Low** | **Done** (beta) | Code-aware agents | Tier-0 | REST; thin provider shells |
| B.37 | **email_smtp notification_channel** — outbound mail without chat | §7.1.3 P3 | **Low** | **Done** (beta) | HITL, scheduled reports | Tier-0 | stdlib SMTP in factory open path |
| B.38 | **OpenTelemetry observability_backend** — trace/metric export | §33, §7.1.3 P3 | **Low** | **Done** (beta) | Unified ops dashboards | Tier-0 | `providers/otel/`; beta noop exporter default |
| B.39 | **Playwright browser_automation** — dynamic web interaction | §7.1.3 P3 | **Low** | **Done** (beta) | Research on JS-heavy sites | Tier-0 | `providers/playwright/`; browser launch in factory |
| B.25 | **AWS cloud_platform facade** — auth + S3/SQS/DynamoDB/ElastiCache defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | AWS-hosted applications | Tier-0 | `providers/aws/`; infrastructure only |
| B.26 | **Azure cloud_platform facade** — MI + Blob/Service Bus/Azure SQL defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | Azure-hosted applications | Tier-0 | `providers/azure/`; infrastructure only |
| B.27 | **GCP cloud_platform facade** — ADC + GCS/Pub/Sub/Cloud SQL defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | GCP-hosted applications | Tier-0 | `providers/gcp/`; infrastructure only |
| B.24 | **Direct vendor SDK in agents** — audit + lint rule | §5.2, §7.1.4 | **Medium** | **Done** | Prevents catalog bypass | Tier-2 | `scripts/check_agents_vendor_imports.py` + gate test `test_vendor_import_guard_b24` (2026-05-27) |

### B.7 Tool Library (§7.1.6)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.40 | **Tool Library scaffold** — catalog, profile, wiring context | §7.1.6 | **High** | **Done** | All agents using external capabilities | Tier-0 | Phase O.2; apps wire tools O.8 (2026-05-30) |
| B.41 | **Context tools** — `rag.retrieve`, `websearch.query` | §7.1.7, §22.1 | **High** | **Done** | RAG / research agents | Tier-0 | Phase O.3 (2026-05-30) |
| B.42 | **Jira catalog tools** — `jira.get_issue`, `jira.search_tasks`, … | §7.1.6 | **Medium** | **Done** | PM / legal workflow agents | Tier-0 | Phase O.4 (2026-05-30) |
| B.43 | **Unified tool model** — deprecate `use_rag` / `use_websearch` flags | §7.1.7, §22.2 | **High** | **Done** | Consistent tool policy + MCP | Tier-1 | Phase O.5 (2026-05-30) |
| B.44 | **Legacy ToolBase migration** | §5.2.2 | **Medium** | **Done** | Single registry | Tier-0 | Phase O.7; `tools_base` deprecated |
| B.45 | **MCP tool export from catalog** | §7.1.6 | **Low** | **Done** | External MCP clients | Tier-3 | Phase O.6 |

### B.4 Legacy & composition

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.14 | **`ChatAgent` / legacy engine removal** — `LEGAL_USE_LEGACY_AGENT_ENGINE` removed | §39, §41 | **Medium** | **Done** | Single execution path for all agents | Tier-1 / Tier-3 | Legal `fastapi_router` requires `UnifiedTaskRunner`; legacy flags removed (2026-05-27) |
| B.15 | **Legal full E2E gate (real LLM)** — deferred acceptance with live model | — | **Low** | **Deferred** | Legal quality assurance | Tier-2 / CI | K.6; separate from Agent OS gate; enable when CI budget approved |
| B.16 | **Lab agent auto-discovery** — manifest-driven roster + scaffold | §7.4 | **Low** | **Done** | Onboarding friction | Tier-3 | Phase N: `ApplicationManifest`, `new-stack` (N.10); explicit `AgentBinding` remains by design (2026-05-30) |
| B.28 | **Per-application `.env.example` missing** — only root `.env.example`; lab/legal vars in README only | §7.4.8 | **Medium** | **Done** | Deployable POC friction | Tier-3 | N.7 backfill + scaffold (2026-05-30) |
| B.29 | **`new-application` scaffold (lab)** — Tier-3 hosts hand-copied from legal/lab | §7.4.8 | **High** | **Done** | Lab + product profiles via CLI; gate acceptance | Tier-3 / platform | N.10 `new-stack` optional |
| B.30 | **No application-level Dockerfile** — only `infra/docker/docling/` | §7.4.8 | **Medium** | **Done** | Per-app `docker/` + build scripts on lab/legal/research/poc | Tier-3 | N.5–N.7 (2026-05-30) |

### B.5 Test & certification hygiene

| ID | Item | Canon | Priority | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------------|------|----------------|
| B.17 | **`agents/` gate collection** — `signoff_probe` test marks `gate` but lives under `agents/` (may not be collected by default `pytest tests/`) | — | **Low** | **Done** | Sign-off smoke not in main gate count | Test infra | `testpaths` includes `agents/`; canonical gate: `uv run pytest -m gate -q` (2026-05-27) |
| B.18 | **HTTP observability acceptance** — trace on echo + multi-agent mock (graph path) | Appendix A #9–10 | **Low** | **Done** | Certification confidence | Test | `test_lab_application_runs_echo_with_trace_observability`, `test_lab_application_runs_research_mock_with_graph_trace` (2026-05-27) |

### B.8 Suggested priority order (for planning)

```text
1. ~~B.08, B.10~~ — observability consistency (Done 2026-05-27)
2. ~~B.01, B.02~~ — checkpoint / full snapshot (Done 2026-05-27)
3. ~~B.03, B.04~~ — governance facade + AgentDecision cleanup (Done 2026-05-27)
4. ~~B.12, B.14~~ — product interaction + legacy removal (Done 2026-05-27)
5. ~~B.05~~ — escalation production path (Done 2026-05-27)
6. ~~B.09, B.17~~ — debug trace injection + gate collection (Done 2026-05-27)
7. ~~B.06~~ — hook parity doc + lifecycle wiring (Done 2026-05-27)
8. ~~B.07, B.11, B.13, B.18, B.24~~ — §42 baseline, metrics export, delivery hardening, HTTP trace acceptance, vendor import guard (Done 2026-05-27)
9. ~~Platform stabilization~~ — all Tier-3 factories aligned (Done 2026-05-27)
10. B.15 — Legal E2E real LLM (**Deferred** — product/CI decision)
11. ~~Phase Q~~ — Harness audit remediation — **Done** (Appendix C)
12. ~~Phase Q+ / Phase R~~ — **Done** (Appendices D, E)
13. ~~Phase S — Harness environment GA~~ — **Done**
14. ~~Phase T — Harness cleanliness~~ — **Done**
15. Phase U — Harness production hardening — **Done**
16. Harness completion backlog (§4.1) — **Done** (2026-06-02)
17. Phase K — K.1/K.2 business agents — **Deferred**
18. Tier-3 product apps / Legal E2E — **Deferred**
```

**Note:** Platform harness (Q–U) is complete. **Harness completion** (legacy + CI) is active. Business agents and product applications are **end of list**.

---

## Appendix C — Harness audit traceability (Phase Q)

**Purpose:** Every finding from the harness implementation audit (2026-06-01) maps to exactly one Phase Q deliverable. Update **Status** when the deliverable is **Done** / **Won't fix** (with reason).

**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### C.1 Nexus, loops, orchestration, errors

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| N-01 | `NexusLoop` monolith ~1200 lines | Q-N.1 | Done (`orchestration/`; ~586 lines) |
| N-02 | Duplicate `_normalize_human_response` | Q-N.2 | Done |
| N-03 | Dual retry (`RetryEngine` vs `max_run_retries`) | Q-N.3 | Done |
| N-04 | `PolicyEngine` \| `RuntimePolicyEngine` union | Q-N.4 | Done |
| N-05 | Hooks NOT_WIRED: decision, interrupt, retry | Q-N.5 | Done |
| N-06 | Hooks PARTIAL: trace persist | Q-N.6 | Done |
| N-07 | `nexus/context/tool_context_helpers.py` misleading name | Q-N.7 | Done |
| N-08 | `RuntimeConfig` monolith | Q-N.8 | Done |
| N-09 | `integration_profile: object` | Q-N.9 | Done |
| N-10 | `production_mode` default in lab | Q-N.10 | Done |
| N-11 | Graph callbacks typed `object` | Q-N.11 | Done |
| N-12 | Duplicate import `InterruptType` | Q-N.12 | Done |
| N-13 | `AgentEngine` static UAEP / event_bus | Q-N.13 | Done |
| N-14 | No unit tests `nexus_loop.py` | Q-N.14 | Done |
| N-15 | Thin `GraphExecutor` unit coverage | Q-N.15 | Done |

### C.2 LLM adapters

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| L-01 | Dead `tracked_llm_call` | Q-L.1 | Done |
| L-02 | Empty `llm_adapters/__init__.py` | Q-L.2 | Done |
| L-03 | `architecture/LLM_ADAPTERS.md` missing provider table | Q-L.3 | Done |
| L-04 | `LLMProfile` docstring `max_retries` wrong | Q-L.4 | Done |
| L-05 | `supports_streaming()` default True | Q-L.5 | Done |
| L-06 | PolicyEngine ignores `llm_cost_evaluation` | Q-L.6 | Done |
| L-07 | Dual usage tracking naming | Q-L.7 | Done |
| L-08 | No structured-output conformance | Q-L.8 | Done |
| L-09 | Bedrock context_window TODO | Q-L.9 | Done |
| L-10 | OpenAI-compat `__dict__.update` fragility | Q-L.10 | Done |
| L-11 | Env vars scattered | Q-L.11 | Done |

### C.3 RAG

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| R-01 | Dead `_build_backend_where` / `_map_hits_to_chunks` | Q-R.1 | Done |
| R-02 | Four parallel retrieval paths | Q-R.2 | Done |
| R-03 | `enable_rag` vs `use_rag` in ContextBuilder | Q-R.3 | Done |
| R-04 | Pipeline `rag_step` always `rag.retrieve` (retired — tool_ids in `on_next_step`) | Q-R.4 | Done |
| R-05 | `top_k` collapses prefetch | Q-R.5 | Done |
| R-06 | `RuntimeConfig` vs `RagProfile` dual config | Q-R.6 | Done |
| R-07 | Unused `RagProfile.extras` | Q-R.7 | Done |
| R-08 | RAG metrics env not in profile | Q-R.8 | Done |
| R-09 | `rag/answers/` parallel stack | Q-R.9 | Done |
| R-10 | `UserProfileManager` bypasses `RetrievalService` | Q-R.10 | Done |
| R-11 | Three “context builder” names | Q-R.11 | Done |
| R-12 | Legacy `use_rag` plan booleans | Q-R.12 | Done |

### C.4 Memory

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| M-01 | No single memory architecture doc | Q-M.1 | Done |
| M-02 | Task memory not visible in scaffold | Q-M.2 | Done |
| M-03 | Silent default when task memory None | Q-M.3 | Done |

### C.5 Observability & metrics

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| O-01 | RAG plugin not in `platform_wiring` | Q-O.1 | Done |
| O-02 | No RAG bridge tests | Q-O.2 | Done |
| O-03 | Parser trace bypasses `ObservabilityBackend` | Q-O.3 | Done |
| O-04 | `metrics/export` substring heuristics | Q-O.4 | Done |
| O-05 | Duplicate import in `metrics/export.py` | Q-O.5 | Done |
| O-06 | `behavioral` never set in export | Q-O.6 | Done |
| O-07 | `/metrics/llm` not on lab host | Q-O.7 | Done |
| O-08 | Observability env scattered | Q-O.8 | Done |
| O-09 | RAG metrics asymmetry vs LLM | Q-O.9 | Done |
| O-10 | `trace_bridge` vs `phase_coverage` drift | Q-O.10 | Done |
| O-11 | Debug router missing type imports | Q-O.11 | Done |
| O-12 | No `trace_bridge` unit tests | Q-O.12 | Done |
| O-13 | Two Prometheus concepts unclear | Q-O.13 | Done |
| O-14 | Runtime events SQLite-first; Cassandra adoption undefined | Q-O.14 | Done |

### C.6 Legacy, style, docs

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| X-01 | Deprecated `ChatAgent` | Q-X.1 | Done |
| X-02 | `task_metadata_bridge` legacy | Q-X.2 | Done |
| X-03 | Copyright / Integrax typo | Q-X.3 | Done |
| X-04 | `tools_base` deprecation | Q-X.4 | Done |
| X-05 | M.6 Future slugs table stale | Q-X.5 | Done |
| D-01 | `docs/README` focus outdated | Q-D.1 | Done |
| D-02 | Canon §52 still “Active” | Q-D.2 | Done |
| D-03 | §0.1 “blocked until L” stale | Q-D.1 (§0.1 fix) | Done |
| D-04 | Guide missing memory/RAG naming | Q-D.4 | Done |
| D-05 | §5.2 process gates not listed for agent authors | Q-D.5 | Done |

### C.7 Tests (cross-cutting)

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| T-01 | NexusLoop unit suite | Q-T.1 / Q-N.14 | Done |
| T-02 | `rag_profile_from_env` tests | Q-T.2 | Done |
| T-03 | `ContextBuilder` tests | Q-T.3 | Done |
| T-04 | `UserProfileManager` tests | Q-T.4 | Done |
| T-05 | Single retrieval per turn test | Q-T.5 | Done |
| T-06 | Platform wiring observability E2E | Q-T.6 | Done |

### C.8 Phase Q paydown log

| Date | Q ID | Summary |
|------|------|---------|
| 2026-06-01 | Q-D.3 | §0.1 strategic objective — Harness GA vs Phase K vs Phase Q |
| 2026-06-01 | Q-O.1,Q-O.2,Q-O.5,Q-O.7 | RAG plugin bootstrap, tests, metrics lint, lab `/metrics/llm` |
| 2026-06-01 | Q-N.2,Q-N.7,Q-N.12 | Duplicate HITL normalize; tool_context_helpers; interrupt import |
| 2026-06-01 | Q-R.1–Q-R.5,Q-R.8 | RAG dead code, single retrieval path, use_rag metadata, prefetch_k |
| 2026-06-01 | Q-L.1,Q-L.2,Q-L.4 | Remove tracked_llm_call; llm_adapters exports; LLMProfile docstring |
| 2026-06-01 | Q-T.2,Q-T.3,Q-T.6 | New unit/integration tests; gate **399 passed** (+2) |
| 2026-06-01 | Q-N.1(partial),Q-N.10,Q-N.13,Q-N.15 | `hitl_runner.py`; lab `harness_production_mode`; AgentEngine `event_bus`; graph checkpoint tests |
| 2026-06-01 | Q-L.9–Q-L.11,Q-O.6,Q-O.11,Q-O.14 | Bedrock windows, OpenAI-compat delegation, LLM env appendix, metrics behavioral, debug types, trace storage §33.1 |
| 2026-06-01 | docs-consolidation | Merged LLM/RAG observability, retry, trace ADR into canon + `architecture/LLM_ADAPTERS.md`; removed satellite `docs/*.md` |
| 2026-06-01 | Q-N.1,Q-X.2,Wave 9 | `graph_runner`, `task_events`, `lifecycle_bridge`; UAEP `execution_options_for_request`; gate **417 passed** |
| 2026-06-01 | Q-X.2(partial),Q-X.4,Q-X.5 | Legacy metadata warnings; `tools_base` timeline; M.6 beta slugs; gate **415 passed** |
| — | — | *(append row per merged PR)* |

**Coverage:** 58 audit rows → 49 unique Q deliverables (some Q IDs satisfy multiple rows). **Target:** 100% **Done** or **Won't fix** — **achieved** (Phase Q complete).

**Appendix B relationship:** Closed by Phase Q where mapped. Residual items tracked in **Appendix D** (Phase Q+).

---

## Appendix D — Post-audit hardening traceability (Phase Q+)

**Source:** Technical debt audit (2026-06-01, after Phase Q Wave 9).  
**Goal:** Cursor-/Claude Code–class harness discipline — typed contracts, single orchestration path, full observability on critical paths.

**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### D.1 Audit verdict → Phase Q+ mapping

| Audit theme | Priority | Q+ IDs | Status |
|-------------|----------|--------|--------|
| Duplicate Tier-0 (`tools_agent`, supervisor, chains, rag/answers, openai/rag) | P0–P2 | Q+-L.1–Q+-L.7 | Done (L.7 Won't fix) |
| `getattr` / duck typing (UAEP, tools, context, plans) | P0 | Q+-T.1–Q+-T.8, Q+.0.3 | Done (zero grandfathered paths) |
| Nexus intake/planning still in `nexus_loop` | P0–P1 | Q+-N.1, Q+-N.2 | Done |
| No `RetryCoordinator` | P1 | Q+-N.3 | Done |
| Observability gaps (metrics heuristics, RAG HTTP, planner errors) | P1 | Q+-O.1–Q+-O.4, Q+-N.5 | Done (O.3 Won't fix) |
| `task_metadata` auto-hydrate | P1 | Q+-M.1, Q+-M.2 | Done |
| Planning monoliths (~680/620 lines) | P2 | Q+-P.1–Q+-P.3 | Done |
| `session_manager` monolith (~596 lines) | P2 | Q+-S.1 | Done |
| LLM SDK getattr quarantine | P3 | Q+-I.1 | Done |
| `harness_production_mode` not wired in lab | P1 | Q+-O.2 | Done |
| Thin `GraphExecutor` handoff/retry tests | P1 | Q+-N.4 | Done |

### D.2 First implementation steps (Wave 1 — start here)

Execute in order; one PR per ID where possible.

| Step | ID | Action | Exit criteria |
|------|-----|--------|---------------|
| **1** | Q+.0.3 | Add `scripts/check_harness_no_getattr.py`; wire to gate (grandfather list for existing hits) | CI enforces on new lines |
| **2** | Q+-T.1 | Introduce `UAEPAgent` Protocol; refactor `supports_uaep` + `UAEPExecutor` | Zero getattr on agent in `uaep.py` |
| **3** | Q+-T.2 | `ToolInvokerProtocol`; fix `catalog_context.py` | Typed registry access |
| **4** | Q+-T.3 | `RuntimeState.trace_event` typed | `tool_access_policy` clean |
| **5** | Q+-T.4 | `can_handle(TaskContext)` on `Agent` | All agents updated |
| **6** | Q+-T.5 | Plan union for `tool_runtime` | No getattr on plan source |

**Then Wave 2:** Q+-L.1 → Q+-L.2 → Q+-L.3 → Q+-M.1 (Legal off ToolsAgent, import gates, opt-in Task hydrate).

### D.3 Phase Q+ paydown log

| Date | Q+ ID | Summary |
|------|-------|---------|
| 2026-06-01 | Q+.0.1,Q+.0.2 | Appendix D + execution order added to plan |
| 2026-06-01 | Q+.0.3,Q+-T.1–T.8,Q+-L.1,Q+-M.1,Q+-N.1,Q+-N.2,Q+-D.* | Wave 1 harness contracts; intake/planning runners; CI getattr/tools_agent gates; docs |
| 2026-06-01 | Q+-L.2–L.3,Q+-N.3,Q+-O.1,Q+-O.2 | Legal `CatalogToolPlanner`; `tool_planner` on RuntimeConfig; RetryCoordinator; typed metrics export; lab harness mode |
| 2026-06-01 | Q+-P.2,Q+-S.1,R-Policy | `step_planner/` package; `session_consolidation.py`; `runtime_config_bridge` wires `ToolScopePolicy` |
| 2026-06-01 | Q+-P.1,Q+-S.1,R-Policy | `engine_planner_*` modules; `session_lifecycle.py`; `tool_policy_resolution` + harness getattr cleanup |
| 2026-06-01 | R-Skill catalog | `research.literature_scan` bundle; `ResearchAgent` skill_ids wiring |
| 2026-06-01 | Q+.0.3 (closeout) | Grandfather list cleared; `parser_trace_flush` uses `TraceEventWithTags` Protocol |
| 2026-06-01 | **Phase Q+** | All Q+-* deliverables **Done** or **Won't fix**; gate **450 passed** |
| 2026-06-01 | Appendix C sync, research skill | C.7 T-* / D-05 aligned; `research.literature_scan` bundle; K.1/K.2 **Ready** |
| 2026-06-01 | Doc sync | §1 alignment table, §6 Phase K cadence, Appendix B.8 renumber, E.1 skill row; README + canon research skill examples |
| — | — | *(append row per merged PR)* |

**Coverage target:** 100% **Done** or **Won't fix** — **met** (2026-06-01).

---

---

## Appendix E — Harness AI alignment traceability (Phase R)

**Source:** Harness AI philosophy audit (2026-06-01) — scaffold, harness+LLM=agent, tool vs skill, context engineering, subagents, policy.  
**Goal:** Step-by-step implementation readiness; every audit theme maps to Phase R deliverables.  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### E.1 Audit theme → Phase R mapping

| Audit theme | Intergrax today | Gap | Phase R IDs | Status |
|-------------|-----------------|-----|-------------|--------|
| Scaffold | `intergrax/scaffold` | No `new-skill` | R-Skill.7, R.0.4 | Done |
| Harness = Nexus + platform + app wiring | Tier-1 + Tier-0 + Tier-3 | Terminology not in glossary | R.0.2 §5.3 | Done |
| LLM separate from agent module | `llm_adapters` | “Runnable instance” undefined | R.0.2 §5.3 | Done |
| Tool = atomic operation | `ToolContract`, `ToolRuntime` | Doc said “tool/skill” | R.0.3, R.0.1 | Done |
| Skill = goal-oriented pack | Was missing (pre-R); **MVP Done** | Registry + importers + first-party packs | R-Skill.1–R-Skill.10 | Done |
| Option 1: skills = tools | — | **Rejected** — breaks LLM/MCP atomic model | R.0.1 ADR | Done |
| Option 2: Skill Library | — | **Adopted** | R-Skill.* | Done |
| Context engineering | §27–28, `MemoryView`, `TaskContextAssemblyOptions` | No central budget API | R-Context.* | Done |
| Subagents | `GraphExecutor`, handoff §42.15 | No isolated child namespace | R-Delegate.* | Done |
| Policy | Multiple engines | No single bundle narrative | R-Policy.* | Done |
| External skill compatibility | — | No importer | R-Skill.8 | Done |

### E.2 Four-layer capability model (canonical)

```text
Integration  →  vendor/backend Protocol (Postgres, Bing, Jira REST)
Tool         →  atomic LLM/MCP operation (rag.retrieve, jira.search_tasks)
Skill        →  composable pack: tool_ids + prompts + policy fragment + metadata
Agent        →  domain module: contract, UAEP steps, skill_ids[], local governance
Harness      →  Nexus + Tier-0 + Tier-3 wiring (orchestration, trace, policy enforcement)
```

### E.3 Phase R paydown log

| Date | R ID | Summary |
|------|------|---------|
| 2026-06-01 | R.0.1,R.0.2,R.0.3,R.0.4 | ADR Option 2; canon §5.3, §7.1.8, §28.1, §42.11.4, §42.14.3; ToolContract docstring; plan Appendix E |
| 2026-06-01 | R-Skill.1–R-Skill.9,R-Context.1,R-Delegate.1,R-Policy.1 | Skill Library MVP, legal pilot, ContextBudget, DelegationSpec, gate **422 passed** |
| 2026-06-01 | R-Skill.10,R-Context.2,R-Delegate.2–4,R-Policy.2 | Event recording, delegation memory, graph integration test, policy bundle wiring |
| 2026-06-01 | **Phase R (MVP)** | All R-* deliverables **Done** or **Won't fix**; gate **450 passed** |
| — | — | *(append row per merged PR)* |

**Coverage target:** 100% **Done** or **Won't fix** — **met** (2026-06-01). Phase S proceeds on this harness baseline.

---

## Appendix F — Harness environment traceability (Phase S)

**Source:** Architecture audit + plan pivot (2026-06-01) — **harness environment before business agents**.  
**Goal:** Track Phase S deliverables.  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### F.1 Theme → Phase S mapping

| Theme | S IDs | Status |
|-------|-------|--------|
| Docs / plan pivot | S.0.1–S.0.4 | **Done** |
| Integration + OTLP | S-Ops.1–S-Ops.3 | **Done** |
| Platform harness skills + lab proof | S-H.1–S-H.5 | **Done** |
| Operator documentation | S-Doc.1–S-Doc.2 | **Done** |
| Business agents (→ Phase K) | K.1, K.2 | **Deferred** (was S-K.*) |
| Legal live LLM E2E | S-Ops.4 / K.6 | **Deferred** |

### F.2 Phase S paydown log

| Date | S ID | Summary |
|------|------|---------|
| 2026-06-01 | S.0.* | Strategy doc; canon; initial Phase S |
| 2026-06-01 | S.0.4 | Pivot: Phase S = harness environment only; K.1/K.2 → Phase K |
| 2026-06-01 | **Phase S** | harness_lab_stack, harness.* skills, OTEL profile, guides/HARNESS_ENVIRONMENT.md, tests |
| — | — | *(append row per merged PR)* |

**Coverage target:** Phase S definition of done met — **yes** (2026-06-01).

---

## Appendix G — Harness production audit traceability (Phase U)

**Source:** Harness-system audit (2026-06-01) — lab/Tier-1/Tier-3 only; **no business agents**.  
**Goal:** Map every finding to exactly one Phase U deliverable. Update **Status** when **Done** / **Won't fix** (with reason).  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### G.1 Security (P0)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| SEC-01 | Lab `POST /v1/lab/run` and `/debug/*` without authentication | U-Sec.1 | Done |
| SEC-02 | MCP enabled by default (`LAB_INCLUDE_MCP=true`) — second open surface | U-Sec.2 | Done |
| SEC-03 | `sandbox.exec` enabled in default lab tool profile | U-Sec.3 | Done |
| SEC-04 | `harness_production_mode()` always `False` — no strict production path | U-Sec.4 | Done |

### G.2 Contracts & policy (P1)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| CON-01 | `Agent` (ABC) vs `UAEPAgent` (Protocol) — no unified inheritance | U-Con.1 | Done |
| CON-02 | `RuntimePolicyBundle` built in lab ctx but not applied to `RuntimeConfig` | U-Pol.1 | Done |
| CON-03 | `PolicyEngine` (NexusLoop) vs `policy_bundle` (RuntimeConfig) — dual systems | U-Pol.2 | Done |
| CON-04 | `ToolPlanningService` imports `ToolsAgentConfig` from Tier-0 `tools_agent` | U-Typ.2 | Done |
| CON-05 | `runtime_state` uses `isinstance(CatalogToolPlanner)` not protocol | U-Typ.3 | Done |
| CON-06 | `create_lab_interaction_adapter()` uses `IntegrationProfile.lab()` not preset | U-Arch.1 | Done |
| CON-07 | Skill `skill_ids` resolved at register — no runtime E2E proof in gate | U-Con.3 | Done |

### G.3 Typing & hygiene (P2)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| TYP-01 | `ToolsAgentConfig` tuple bug (`temperature = None,`) | U-Typ.1 | Done |
| TYP-02 | `RuntimePolicyBundle.budget` / `plan_loop` typed as `Any` | U-Pol.3 | Done |
| TYP-03 | `# type: ignore` on lab integration wiring adapters | U-Arch.2 | Done |
| TYP-04 | `getattr` outside harness audit (tools_agent prune, profile, sandbox) | U-Typ.4 | Done |
| TYP-05 | `hasattr` on harness paths (shared_task_context, engine_plan, platform_wiring) | U-Typ.5 | Done |
| TYP-06 | `ToolPlanDecision` vs `AgentDecision` naming collision risk | U-Leg.3 | Done |

### G.4 Legacy & naming (P3)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| LEG-01 | `tools_agent_answer` and ToolsAgent naming in Tier-1 runtime | U-Arch.3 | Done |
| LEG-02 | `ToolsAgent.run` still full orchestrator — deprecation incomplete | U-Leg.1 | Done |
| LEG-03 | `rag.answers` module remains; tests filtered not removed | U-Leg.2 | Done |
| LEG-04 | Legacy tool plan booleans (`from_legacy`, `uses_legacy_rag_flag_only`) | U-Leg.3 | Done |

### G.5 Documentation & CI (P4)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| DOC-01 | `guides/HARNESS_ENVIRONMENT.md` claims policy bundle wired — lab does not apply bridge | U-Doc.1, U-Pol.1 | Done |
| DOC-02 | Phase K footer still "after Phase S" in harness docs | U-Doc.3 | Done |
| CI-01 | harness-smoke omits Phase T unit tests | U-CI.1 | Done |
| CI-02 | No acceptance test for strict production harness path | U-CI.2 | Done |
| CI-03 | harness-smoke vs gate run on different OS images | U-CI.3 | Done |

### G.6 Phase U paydown log

| Date | U ID | Summary |
|------|------|---------|
| 2026-06-01 | U.0.* | Appendix G + Phase U section added to implementation plan (audit → backlog) |
| 2026-06-02 | §4.1 | Harness completion: U-Leg.1–3, U-Arch.2, U-Typ.4, U-CI.3, harness.skill_registry, research UAEP parity; gate **481** |
| — | — | *(append row per merged PR)* |

**Coverage target:** Phase U + §4.1 harness completion backlog **Done** (2026-06-02). **K.1/K.2 deferred** until product prioritization.

---

## Appendix H — Architecture coverage matrix (Intergrax canon + ideal harness)

**Purpose:** ensure the implementation plan explicitly covers all harness-scope requirements from:

- `intergrax_runtime_architecture.md` (canonical Intergrax runtime architecture)
- `IDEAL_HARNESS_AI_ARCHITECTURE.md` (target/benchmark architecture)

**Rule:** For harness work, this matrix must have **zero `Uncovered` rows**.

### H.1 Coverage status legend

- **Done** — capability implemented and verified by existing phases/tests.
- **Partial closeout** — contracts/governance Done; runtime enforcement gaps scheduled in Phase V-REM.
- **Planned (Phase V-REM)** — explicitly scheduled in Phase V-REM (`V-REM-*` IDs).
- **Deferred (product scope)** — intentionally outside harness-only scope (Band 3 / §6.3).
- **Uncovered** — gap; MUST be added to plan before related implementation proceeds.

### H.2 Harness architecture domains — required coverage

| Domain (harness scope) | Intergrax canon anchor | Ideal harness anchor | Plan coverage | Status |
|------------------------|------------------------|----------------------|---------------|--------|
| Strategic objective + harness-first hierarchy | canon §2, §5.1, §51, §53.1 | ideal §0, §1, §26 | §0, §4.0, Phase V governance | **Done** |
| Tier model and runtime boundaries | canon §5.1, §7.0–§7.4, §42 | ideal §3, §26 | §0.2, §2 map, Phases L/Q+/U, **FAUDIT-TIER.\*** | **Done** — reference manifest catalog in `intergrax/applications/reference/` + CI gate |
| Unified execution runtime (UAEP, lifecycle, interrupts, policy) | canon §42.* | ideal §3.3, §3.4, §5, §8 | §2 map, Phase U, gate suites | **Done** |
| Context engineering core | canon §28.1, §42.35 | ideal §16 | Phase R (Done) + V-CE.* | **Done** |
| Capability graph dependencies + impact analysis | canon §53.2 | ideal §19 + capability graph expectations | V-CG.* | **Done** |
| Agent lifecycle governance (cert/promo/deprec/retire/owner) | canon §15, §53.3 | ideal §17 | V-ALG.* | **Done** |
| Prompt engineering architecture | canon §53.5 | ideal §20 | V-PE.* | **Done** |
| Evaluation and benchmarking operations | canon §53.6 | ideal §18 | V-EVAL.* + A.4 | **Done** |
| Architecture metrics and debt governance | canon §53.7 | ideal §21 + architecture metrics expectations | V-AM.* | **Done** |
| Security/data governance (agent-native threats) | canon §42.37, §53.8 | ideal §23 | Phase U (baseline) + V-SEC.* | **Done** |
| Cost/resource governance | canon §53.9 | ideal §24 | V-COST.* | **Done** |
| Multi-agent coordination pattern catalog | canon §42.43, §53.10 | ideal §6 + §25 | V-MA.* | **Done** |
| Knowledge graph evolution path (Graph-RAG) | canon §53.11 | ideal §3.7.1 + §25 | V-KG.* | **Done** |
| **Adaptive Harness Intelligence (L4 runtime closed loop)** | canon §54 | ideal §25 | **Phase W-ADAPT** · AHIA | **Done** (Band 2y, 70/70) — L4 runtime closed; observe/recommend/apply/verify per AHIA |
| Observability and runtime traceability | canon §33, §42.24 · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) | ideal §11 | Phases OBS + OBS-DEPTH.* + **Phase OBS-BUS** | **L4 Done** — spine, typed payloads, emitter, emission coverage, journal export; gate: `check_observability_gates.py` |
| Registry-driven extensibility (agent/tool/skill/policy/prompt/eval) | canon §7.1.5.1–§7.1.8, §15, §53.2 | ideal §19 | Phase R/U + V-CG/V-PE/V-EVAL + **P-Ext** | **Done** — plugin catalogs production-ready; marketplace UI out of scope |
| Product agents and new product apps | canon §7.4, §52 | ideal §26 | §6.3 only | **Deferred (product scope)** |

### H.3 Completion policy for “architecture-complete harness”

Harness architecture can be considered complete against both architecture documents only when:

1. All harness-scope rows in H.2 are `Done` (no `Partial closeout`, no `Planned`, no `Uncovered`).
2. `Deferred (product scope)` rows remain intentionally isolated to Band 3 (§6.3).
3. Phase V-REM complete and parent V-* Partial rows closed.
4. Phase V KPI thresholds and L3/L4 evidence gates are satisfied.
5. Canon + plan + docs index are synchronized in the same change window.

### H.4 Change control rule

Any future addition to either architecture document that introduces a new harness-scope
domain MUST be reflected in:

- this matrix (Appendix H),
- a concrete Phase V-REM (or successor phase) deliverable ID,
- priority ladder (§4) and “what next” (§6) if it changes execution order.

---

## Appendix I — Plugin catalog traceability (Phase P-Ext)

**Purpose:** Task-level tracker for plugin-native Integration, Tool, and Skill catalogs. **Canonical phase narrative:** [Phase P-Ext](#phase-p-ext--plugin-catalogs-integrations-tools-skills) · paydown: [P-Ext.6](#p-ext6--production-closure-paydown).

**Status:** **Done** (2026-06-02) · **MVP effort:** ~21–32 person-days · **paydown estimate:** ~8–14 person-days.

### I.1 Delivery rule

Same as §6.1: one **P-Ext.\*** ID → PR → update status in this appendix → `pytest -m gate` green. Paydown cadence: [§6.1p](#61p-phase-p-ext-paydown-band-2c--optional-parallel-with-61).

### I.2 Task register

| ID | Layer | Summary | Status | Priority |
|----|-------|---------|--------|----------|
| P-Ext.0.1 | All | `load_plugins()` / entry point discovery | **Done** | P0 |
| P-Ext.0.2 | All | `PluginConflictError`, `PluginLoadError` | **Done** | P0 |
| P-Ext.0.3 | All | `bootstrap_catalogs()` Tier-3 API | **Done** | P0 |
| P-Ext.0.4 | All | `guides/EXTENSION_AUTHOR_GUIDE.md` (EN) | **Done** | P0 |
| P-Ext.0.5 | All | Test fixture pip package | **Done** | P0 |
| P-Ext.0.6 | All | EP discovery tests (3 groups) | **Done** | P0 |
| P-Ext.0.7 | All | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | **Done** | P1 |
| P-Ext.1.1 | Integrations | Entry points `intergrax.integrations` | **Done** | P0 |
| P-Ext.1.2 | Integrations | `bootstrap_core` / optional split | **Done** | P1 |
| P-Ext.1.3 | Integrations | Typed `resolve_*` helpers (top categories) | **Done** | P2 |
| P-Ext.1.3a | Integrations | Expand `resolve_typed` + tests | **Done** | P2 |
| P-Ext.1.4 | Integrations | Health check API (optional) | **Done** | P3 |
| P-Ext.1.5 | Integrations | `IntegrationSlug` cleanup (docs/scripts) | **Done** | P2 |
| P-Ext.1.6 | Integrations | EP test via fixture | **Done** | P0 |
| P-Ext.1.7 | Integrations | Dual-model docs (manifest vs plugin) | **Done** | P2 |
| P-Ext.1.8 | Integrations | CI integration slug count smoke | **Done** | P1 |
| P-Ext.1.9 | Integrations | `test_resolve_typed.py` | **Done** | P3 |
| P-Ext.1.10 | Integrations | Tier-3 `bootstrap_catalogs` in integration_wiring | **Done** | P0 |
| P-Ext.1.11 | Integrations | `_shared/integration_wiring.py` helper | **Done** | P2 |
| P-Ext.1.12 | Integrations | `SqliteIntegrationPlugin` wire or document | **Done** | P3 |
| P-Ext.2.1 | Tools | `ToolPlugin` Protocol | **Done** | P0 |
| P-Ext.2.2 | Tools | `ToolBundleManifest` / bundle metadata | **Done** | P0 |
| P-Ext.2.3 | Tools | `register_tool_plugin()` | **Done** | P0 |
| P-Ext.2.4 | Tools | RAG bundle plugin migration (pilot) | **Done** | P1 |
| P-Ext.2.5 | Tools | Entry points `intergrax.tools` | **Done** | P1 |
| P-Ext.2.6 | Tools | MCP tool export | **Done** | P1 |
| P-Ext.2.7 | Tools | `ToolContract.version` | **Done** | P2 |
| P-Ext.2.8 | Tools | All 13 shipped bundles → `ToolPlugin` | **Done** | P1 |
| P-Ext.2.9 | Tools | `tools/examples/` reference package | **Done** | P0 |
| P-Ext.2.10 | Tools | `test_external_tool_plugin.py` | **Done** | P0 |
| P-Ext.2.11 | Tools | EP tool test via fixture | **Done** | P0 |
| P-Ext.2.12 | Tools | `tool_wiring` lazy `tool_bundle_ids` | **Done** | P2 |
| P-Ext.3.1 | Skills | `SkillPlugin` Protocol | **Done** | P1 |
| P-Ext.3.2 | Skills | `register_skill_plugin()` | **Done** | P1 |
| P-Ext.3.3 | Skills | Entry points `intergrax.skills` | **Done** | P1 |
| P-Ext.3.4 | Skills | harness + research + legal plugin migration | **Done** | P1 |
| P-Ext.3.5 | Skills | `requires_skills` (optional) | **Done** | P3 |
| P-Ext.3.6 | Skills | `skills/examples/` reference package | **Done** | P0 |
| P-Ext.3.7 | Skills | `test_external_skill_plugin.py` | **Done** | P0 |
| P-Ext.3.8 | Skills | EP skill test via fixture | **Done** | P0 |
| P-Ext.3.9 | Skills | `skill_wiring` lazy `skill_bundle_ids` | **Done** | P2 |
| P-Ext.3.10 | Skills | Scaffold `new-skill` → `SkillPlugin` | **Done** | P2 |
| P-Ext.3.11 | Skills | Docs: SkillPlugin vs Cursor importer | **Done** | P2 |
| P-Ext.3.12 | Skills | Shipped `requires_skills` demo (optional) | **Done** | P3 |
| P-Ext.4.1 | Ops | Lazy profile bootstrap | **Done** | P2 |
| P-Ext.4.2 | Ops | `CatalogSnapshot` API | **Done** | P2 |
| P-Ext.4.3 | Ops | Slug conflict policy (bootstrap) | **Done** | P2 |
| P-Ext.4.4 | Ops | `check_plugin_catalog.py` CI | **Done** | P1 |
| P-Ext.4.5 | Ops | CI smoke: tool/skill bundle counts | **Done** | P1 |
| P-Ext.5.1 | Docs | Scaffold `new_*` commands | **Done** | P2 |
| P-Ext.5.2 | Docs | INTEGRATIONS/TOOLS/SKILLS external sections | **Done** | P2 |
| P-Ext.5.3 | Docs | Canon §7.1.5.1 plugin narrative | **Done** | P1 |
| P-Ext.5.4 | Docs | remove `PLUGIN_CATALOG_PLAN.md` | **Done** | P3 |
| P-Ext.5.5 | Docs | Prod path matrix in author guide | **Done** | P2 |
| P-Ext.5.6 | Docs | Lab wiring recipe for external plugins | **Done** | P2 |
| P-Ext.6.1 | Paydown | Fixture pip package (rollup) | **Done** | P0 |
| P-Ext.6.2 | Paydown | External tool + skill examples + tests | **Done** | P0 |
| P-Ext.6.3 | Paydown | EP discovery + lab env | **Done** | P1 |
| P-Ext.6.4 | Paydown | IntegrationSlug cleanup | **Done** | P2 |
| P-Ext.6.5 | Paydown | Scaffold CLI | **Done** | P2 |
| P-Ext.6.6 | Paydown | Integration Tier-3 + typed resolve + health | **Done** | P2 |
| P-Ext.6.7 | Paydown | Conflict policy + CI smoke | **Done** | P1 |
| P-Ext.6.8 | Paydown | Skill Tier-3 + scaffold rollup | **Done** | P2 |
| P-Ext.6.9 | Paydown | Tool Tier-3 lazy wiring rollup | **Done** | P2 |
| P-Ext.6.10 | Paydown | Tier-3 lazy wiring (all catalogs) rollup | **Done** | P2 |

**Paydown summary:** 0 **Planned** · 61 **Done** · 0 **Partial** (Phase P-Ext production closure complete; rollup rows duplicate leaf IDs).

### I.3 Market alignment checklist

| Pattern | Target |
|---------|--------|
| Hexagonal adapters | `IntegrationCategory` + contracts + `IntegrationPlugin` |
| MCP tools | `ToolContract` + `export_mcp_tools` |
| Capability packs | `SkillManifest` + resolver (not LLM-invokable) |
| 12-factor config | env_prefix + `IntegrationProfile.options` |
| Plugin discovery | entry points (hybrid with explicit bootstrap) |
| Tier-3 composition root | `bootstrap_catalogs()` |

### I.4 Paydown log

| Date | P-Ext ID | Summary |
|------|----------|---------|
| 2026-06-02 | — | Phase P-Ext + Appendix I added (migrated from `PLUGIN_CATALOG_PLAN.md`) |
| 2026-06-02 | 0.1–0.4, 1.1–1.2, 2.1–2.8, 3.1–3.5, 4.1–4.2, 4.4, 5.2–5.4 | MVP: protocols, bootstrap, 13 tool + 3 skill plugins, lazy catalog, `custom_memory_kv` test |
| 2026-06-02 | — | Plan updated: **MVP Done** + **P-Ext.6 paydown** backlog (EP fixture, external tool/skill tests, ops/docs) |
| 2026-06-02 | 1.* audit | Integrations audit: 12 core / ~99 full manifest path; `resolve_typed` partial; Tier-3 integration_wiring gap; +P-Ext.1.3a, 1.8–1.12 |
| 2026-06-02 | M.6 P5 closeout | Catalog **135** full (`12` core); timeline 99→127→135; P-Ext integration counts synced |
| 2026-06-02 | 3.* audit | Skills audit: 3/3 `SkillPlugin`, 8 skill_id; Tier-3 `skill_wiring` OK; scaffold legacy; +P-Ext.3.9–3.12, 6.8 |
| 2026-06-02 | 2.* audit | Tools audit section + `tool_wiring` lazy (P-Ext.2.12); P-Ext.4.5 unified counts; +P-Ext.6.9–6.10 |
| 2026-06-02 | P-Ext paydown | Fixture EP package, external examples/tests, Tier-3 wiring, docs, CI smoke (residual: 1.5, 4.3, 5.1, 5.6) |
| 2026-06-02 | P-Ext closure | IntegrationSlug docs cleanup, `warn_override` conflict policy, scaffold CLI, lab wiring recipe |
| 2026-06-02 | P-Ext complete | Phase narrative + §6.1p synced; expanded `check_plugin_catalog.py` smoke suite |
| 2026-06-02 | §6.1 | Gate green **486**: IntegrationBinding test fixes, circular import, catalog re-bootstrap after test clears, scaffold templates |
| 2026-06-02 | TYP-06, U-Typ.4 | `IntegrationProfile` explicit binding accessors; removed `tools_agent.AgentDecision` alias |
| 2026-06-02 | W-OPS.0 | Harness maturity audit → Phase W-OPS + §6.2w in implementation plan |
| 2026-06-05 | V-REM.0.* | Plan audit → Phase V-REM + Appendix J + §6.1z queue (10 open) |
| — | — | *(append row per merged PR)* |

---

## Appendix J — Phase V remediation traceability (audit gap → V-REM ID)

**Purpose:** 100% mapping from **Partial** audit findings (2026-06-05) to concrete remediation IDs. **Canonical phase narrative:** [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout).

**Status:** **12 tasks** · **12 Done** (2026-06-05).

### J.1 Audit gap → remediation matrix

| Audit source | Layer / area | Gap | Severity | Parent plan ID | V-REM ID | Status |
|--------------|--------------|-----|----------|----------------|----------|--------|
| Plan/code audit 2026-06-05 | Capability graph (AUDIT_MAP §19) | System edges agents→application incorrect per host | **Critical** | V-CG.2, V-CG.3, V-CG.4 | V-REM-CG.1, V-REM-CG.2 | **Done** |
| Plan/code audit 2026-06-05 | Agent lifecycle (AUDIT_MAP §31) | Governance contracts exist; no runtime routing cutoff for retired/deprecated | High | V-ALG.3 | V-REM-ALG.1 | **Done** |
| Plan/code audit 2026-06-05 | Agent lifecycle (AUDIT_MAP §31) | Ownership contracts exist; no production-eligible filter at selection | High | V-ALG.4 | V-REM-ALG.2 | **Done** |
| Plan/code audit 2026-06-05 | Prompt registry (AUDIT_MAP §17) | PromptMeta missing owner/risk; no YAML assets for E2E validation | High | V-PE.1 | V-REM-PE.1, V-REM-PE.2 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Tool injection defense not wired on execution path | High | V-SEC.2 | V-REM-SEC.1 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Retrieval poisoning defense not enforced per tenant/app | High | V-SEC.3 | V-REM-SEC.2 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Tenant isolation + audit trail hooks missing in main path | High | V-SEC.4 | V-REM-SEC.3 | **Done** |
| Plan/code audit 2026-06-05 | Evaluation (AUDIT_MAP §25) | NexusEvalRunner exists; missing integration tests + gate | Medium | A.4, A.4.1 | V-REM-A.1 | **Done** |
| Plan sync 2026-06-05 | Plan governance | Appendix J + §6.1z queue + status sync | — | — | V-REM.0.1, V-REM.0.2 | **Done** |

**Coverage target:** 100% **Done** when every **Planned** row is **Done** and parent Partial IDs (V-CG.2–4, V-ALG.3–4, V-PE.1, V-SEC.2–4, A.4) are **Done**.

### J.2 Paydown log

| Date | V-REM ID | Summary |
|------|----------|---------|
| 2026-06-05 | V-REM.0.1, V-REM.0.2 | Appendix J + Phase V-REM section + §6.1z/§6.2v + Appendix H sync |
| 2026-06-05 | V-REM-CG.1–A.1 | Runtime remediation: capability graph, lifecycle routing, V-SEC wiring, prompt governance, EvalRunner gate |
| 2026-06-05 | V-POST.1, V-POST.2 | Phase V closeout gate green; AgentEngine routability guard; NexusLoop tenant-security integration tests |

---

## Appendix K — Adaptive Harness Intelligence traceability (Phase W-ADAPT)

**Purpose:** 100% mapping from [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) (AHIA) to concrete **W-ADAPT.\*** IDs. **Canonical phase narrative:** [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime).

**Status:** **70/70 Done** (Band 2y closed 2026-06-05) — Waves W-ADAPT-0 through W-ADAPT-7 complete.

### K.1 AHIA component → W-ADAPT ID matrix

| AHIA component (§9) | Existing module to reuse | W-ADAPT ID |
|---------------------|--------------------------|------------|
| SignalCollector | `metrics/export.py`, `execution_guard.py`, `online_evaluation_registry.py` | W-ADAPT-1.4–1.11 |
| HarnessOutcomeSignal + utility | — (new) | W-ADAPT-1.1, W-ADAPT-1.8 |
| SignalStore | — (new SQLite) | W-ADAPT-1.3 |
| BanditStateStore | — (new) | W-ADAPT-2.1 |
| RoutingTuningEngine | `rag/routing/query_router.py`, LLM profiles | W-ADAPT-2.2, W-ADAPT-3.7, W-ADAPT-4.10 |
| ExecutionStrategyEngine | `history_evaluator.py`, `nexus_factory.py` | W-ADAPT-2.3, W-ADAPT-4.10 |
| PolicyLearningEngine | `adaptive_governance.py`, `tool_security.py` | W-ADAPT-2.4, W-ADAPT-4.6, W-ADAPT-4.9 |
| EvaluationFeedbackEngine | `evaluation_registry_trends.py` | W-ADAPT-2.5, W-ADAPT-5.3 |
| ProposalBuilder | `adaptive_governance.py` (`AdaptiveLoopProposal`) | W-ADAPT-2.6 |
| AdaptationEngine facade | — (new) | W-ADAPT-2.7 |
| Governance gate | `adaptive_governance.py`, `capability_graph_compatibility.py` | W-ADAPT-2.8–2.9 |
| ProfileVersionStore | — (new; pattern from `agent_promotion.py`) | W-ADAPT-3.1–3.2, W-ADAPT-3.5 |
| AdaptationExecutor | `runtime_governance_bridge.py` (extend) | W-ADAPT-3.3–3.4, W-ADAPT-4.4–4.5, W-ADAPT-4.8 |
| VerificationLoop | `evaluation_registry_trends.py`, `execution_guard.py` | W-ADAPT-5.1–5.5 |
| ProcessPatternMiner | trace persistence | W-ADAPT-6.* |
| AdaptationScheduler | Celery/message bus pattern from W-ML | W-ADAPT-2.12, W-ADAPT-5.12, W-ADAPT-6.5 |
| AdaptiveProfile (Tier-3) | `environment_profile.py` | W-ADAPT-4.1, W-ADAPT-7.1–7.2 |
| Ops reports / CI | `phase_v_governance_report.py` pattern | W-ADAPT-1.12, W-ADAPT-2.11, W-ADAPT-5.6–5.8 |
| Runtime L4 evidence | `maturity_gate_evidence.py` | W-ADAPT-5.7, W-ADAPT-5.11 |
| Author docs | AGENT_CREATION_GUIDE appendices | W-ADAPT-7.3–7.4 |

### K.2 Adaptive loop kind → implementation wave

| `AdaptiveLoopKind` | Engine | Apply wave | Authority default |
|--------------------|--------|------------|-------------------|
| `ROUTING_TUNING` | W-ADAPT-2.2 | W-ADAPT-4.10 | RECOMMEND |
| `EXECUTION_STRATEGY_TUNING` | W-ADAPT-2.3 | W-ADAPT-4.10 | RECOMMEND |
| `POLICY_LEARNING` | W-ADAPT-2.4 | W-ADAPT-4.6, W-ADAPT-4.9 | AUTO_WITH_HUMAN_GATE |
| `EVALUATION_FEEDBACK` | W-ADAPT-2.5 | observe only (W-ADAPT-5.3) | OBSERVE_ONLY |

### K.3 Lifecycle mode → task coverage

| Mode | Code | Primary tasks |
|------|------|---------------|
| Observe | L4-O | W-ADAPT-1.* |
| Recommend | L4-R | W-ADAPT-2.* |
| Shadow | L4-S | W-ADAPT-3.* |
| Canary | L4-C | W-ADAPT-4.3 |
| Apply | L4-A | W-ADAPT-4.4–4.10 |
| Verify | L4-V | W-ADAPT-5.* |

### K.4 Paydown log

| Date | W-ADAPT ID | Summary |
|------|------------|---------|
| 2026-06-05 | W-ADAPT-1.1–1.12 | Observe (L4-O): contracts, SignalStore, SignalCollector, Nexus/Runtime hooks, `phase_w_adapt_report.py` |
| 2026-06-05 | W-ADAPT-0.2–0.5 | ADR-ADAPT-001 + `intergrax/runtime/adaptive/` scaffold + gate import tests |
| 2026-06-05 | W-ADAPT-0.1 | Phase W-ADAPT register + §6.1t + §6.2ac + Appendix K + Band 2y |
| 2026-06-02 | W-ADAPT-2.1–2.12 | Recommend (L4-R): AdaptationEngine, ProposalBuilder, bandit store, proposal report |
| 2026-06-02 | W-ADAPT-3.1–3.7 | Shadow (L4-S): ProfileVersionStore, shadow executor, integration tests |
| 2026-06-02 | W-ADAPT-4.1–4.10 | Apply (L4-A): canary, apply, rollback, policy-learning HITL |
| 2026-06-02 | W-ADAPT-5.1–5.12 | Verify (L4-V): VerificationLoop, auto-rollback, L4 runtime closeout gate, runbooks |
| 2026-06-02 | W-ADAPT-6.1–6.5 | ProcessPatternMiner, trace sequence reader, pattern report export |
| 2026-06-02 | W-ADAPT-7.1–7.7 | Tier-3 AdaptiveProfile wiring, debug routes, business outcome webhook, acceptance E2E |
| 2026-06-02 | W-ADAPT-OPS | Lab L4-O observe default (`LAB_ADAPTIVE_OBSERVE`); CI/release `--enforce-l4-runtime`; canon §54 + AHIA sync |

---

## Appendix L — LLM completion response envelope traceability (Phase M-LLM-R)

**Source:** Tier-0 LLM adapter audit (2026-06-06) — plain `str` / `Dict[str, Any]` returns insufficient for production observability, replay, cost attribution, and L4 adaptive signals.

**Phase register:** [Phase M-LLM-R](#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06) · **Band 2z** · queue [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

### L.1 Audit finding → remediation map

| # | Audit finding | Remediation | Task IDs |
|---|---------------|-------------|----------|
| 1 | `generate_messages` returns bare `str` | `LLMAdapterResponse` with `content: str` | M-LLM-R.1.1, M-LLM-R.2.1, M-LLM-R.3.*, M-LLM-R.4–6.* |
| 2 | `generate_with_tools` returns `Dict[str, Any]` | Same envelope; `tool_calls: tuple[LLMToolCall, ...]` | M-LLM-R.1.3, M-LLM-R.1.7, M-LLM-R.2.2, M-LLM-R.4.2 |
| 3 | Streaming yields `str` / dict chunks | `LLMStreamEvent` partial/final | M-LLM-R.1.5, M-LLM-R.2.3–2.4, M-LLM-R.3.6 |
| 4 | `generate_structured` return untyped | `LLMStructuredResult[T]` | M-LLM-R.1.6, M-LLM-R.2.5, M-LLM-R.3.7 |
| 5 | SDK `finish_reason` / stop metadata lost | `LLMFinishReason` on response | M-LLM-R.1.1, M-LLM-R.3.1–3.4 |
| 6 | Provider `response_id` / request correlation lost | `response_id: str \| None` on response | M-LLM-R.1.1, M-LLM-R.3.1 |
| 7 | Cached / reasoning tokens discarded | `LLMTokenUsage.cached_input_tokens`, `reasoning_tokens` | M-LLM-R.1.2, M-LLM-R.3.1 |
| 8 | Refusal / content-filter signals lost | `refusal: str \| None` + finish_reason enum | M-LLM-R.1.1, M-LLM-R.3.1–3.2 |
| 9 | Usage only via side-channel (`LLMAdapterUsageLog`) | Per-call `usage` on response + aligned `end_call` | M-LLM-R.1.2, M-LLM-R.2.6, M-LLM-R.7.1 |
| 10 | Inconsistent token counting (estimate vs SDK) | Prefer SDK counts; flag estimate in `LLMProviderExtensions` | M-LLM-R.3.5, M-LLM-R.1.4 |
| 11 | No extensibility without dict bags | `LLMProviderExtensions` tagged union | M-LLM-R.1.4 |
| 12 | Replay `LLMCallInfo` not populated from adapter | Trace bridge from `LLMAdapterResponse` | M-LLM-R.7.2, M-LLM-R.7.3 |
| 13 | `CoreLLMAdapterReturnedDiagV1` tracks `adapter_return_type="str"` | Diagnostics carry finish_reason + tokens | M-LLM-R.7.4 |
| 14 | Conformance enforces `isinstance(text, str)` | Typed conformance helpers | M-LLM-R.8.2 |
| 15 | ~50 call sites assume `str` | Full consumer refactor (Nexus, RAG, agents, websearch) | M-LLM-R.4.*, M-LLM-R.5.*, M-LLM-R.6.* |
| 16 | `make_tool_result` dict factory | Delete; typed `build_adapter_response` | M-LLM-R.1.7 |
| 17 | Public API missing response types | Re-export from `llm_adapters/__init__.py` | M-LLM-R.1.8 |
| 18 | Docs describe two-layer usage but not response envelope | `architecture/LLM_ADAPTERS.md` envelope section | M-LLM-R.8.1 |
| 19 | No CI guard against regression to `str` returns | `check_llm_adapter_typed_returns.py` | M-LLM-R.8.3 |

### L.2 Consumer inventory (must migrate)

| Area | Modules | Task |
|------|---------|------|
| Nexus core LLM | `context_preflight.py + on_next_step` | M-LLM-R.4.1 |
| Tool planning | `tool_planning_service.py` | M-LLM-R.4.2 |
| Planning / history | `plan_sources.py`, `engine_history_layer.py` | M-LLM-R.4.3 |
| Profile services | `user_profile/*`, `organization/*`, `session_memory_consolidation_service.py` | M-LLM-R.4.4 |
| Supervisor | `supervisor.py` | M-LLM-R.4.5 |
| RAG | `query_refiner.py`, `query_expander.py`, `chunk_enricher.py`, `llm_graph_indexer.py` | M-LLM-R.5.1 |
| Websearch | `websearch_context_generator.py`, `websearch_answerer.py` | M-LLM-R.5.2 |
| Legacy RAG | `legacy/rag_answers/pipeline/answer_pipeline.py` | M-LLM-R.5.3 |
| Agents (Tier-2) | `agent cognitive patterns (`on_next_step`)`, `mock_agents.py` | M-LLM-R.6.1 |
| Scaffold / tests | `scaffold/new_agent.py`, `testing_support/builder.py` | M-LLM-R.6.2–6.3 |
| All providers | `llm_adapters/providers/*` | M-LLM-R.3.* |

### L.3 Paydown log

| Date | M-LLM-R ID | Summary |
|------|------------|---------|
| 2026-06-06 | M-LLM-R.0.1 | Phase M-LLM-R register + §6.1v + §6.2ad + Appendix L + Band 2z |
| 2026-06-06 | M-LLM-R.* | Typed `LLMAdapterResponse` envelope; providers + consumers migrated; gate **755** passed |
| — | — | *(append row per merged PR)* |

---

## Appendix M — Full architecture audit traceability (Phase FAUDIT-32)

**Purpose:** 100% mapping from 32-layer [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8 audit to concrete **FAUDIT.\*** remediation IDs. **Canonical phase narrative:** [Phase FAUDIT-32](#phase-faudit-32--full-architecture-audit-closeout).

**Status:** **Done** (2026-06-06) · **23/23 remediation Done** + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed) follow-up · gate **901**

### M.1 Layer → FAUDIT ID matrix (High + Critical only)

| Layer | AUDIT_MAP § | Gap summary | Severity | FAUDIT ID |
|-------|-------------|-------------|----------|-----------|
| Tier boundaries | §2 | `intergrax/runtime/architecture/capability_graph_applications.py` imports `applications.*` | **Critical** | FAUDIT-TIER.1, FAUDIT-TIER.2 |
| Task intake | §3 | No `TaskEnvelope`; worker≡HTTP parity incomplete | High | FAUDIT-INTAKE.1, FAUDIT-INTAKE.2 |
| Identity | §4 | No service/agent identity; delegation scope | High | FAUDIT-ID.1, FAUDIT-ID.2 |
| Policy | §5 | Pre-LLM/pre-output hooks absent | High | FAUDIT-POL.1 |
| LLM adapters | §6 | No policy-driven routing | High | FAUDIT-LLM.1 |
| Cognition | §7 | No `DecisionRecord` per step | High | FAUDIT-COG.1 |
| Orchestration | §9 | No backpressure | High | FAUDIT-ORCH.1 |
| Subagents | §10 | No `SubtaskContract` | High | FAUDIT-SUB.1 |
| Memory | §15 | Entity graph memory; STM retention | High | FAUDIT-MEM.1 |
| Prompts | §17 | No golden prompt CI | High | FAUDIT-PE.1 |
| Registry | §19 | Snapshot omits agents/eval | High | FAUDIT-REG.1 |
| Capability graph | §20 | Missing prompt nodes; no release impact gate | High | FAUDIT-CG.1, FAUDIT-CG.2 |
| Observability | §21 | Missing `LLM_CALL`/`POLICY_DECISION` events | High | FAUDIT-OBS.1 |
| Reliability | §22 | Shallow error taxonomy | High | FAUDIT-REL.1 |
| Security | §23 | No `DataClassification` | High | FAUDIT-SEC.1 |
| Cost | §24 | Tenant attribution not mandatory | High | FAUDIT-COST.1 |
| Evaluation | §25 | Release baseline not CI-enforced | High | FAUDIT-EVAL.1 |
| Lifecycle | §31 | State catalog mismatch; weak adoption | High | FAUDIT-ALG.1 |
| Ops / SLOs | §30 | `release_cycles.json` artifact policy | High | FAUDIT-OPS.1 |

### M.2 Cross-layer themes

| Theme | Layers affected | Risk |
|-------|-----------------|------|
| **Closeout vs maturity** | §17–§25, §31 | Plan **Done** on wiring; AUDIT_MAP **L2** on depth — do not conflate |
| **Dual-path telemetry** | §21, §6 | **L4 Done:** [Phase OBS-BUS](#phase-obs-bus--unified-observability-spine) — unified journal, `ObservabilityEmitter`, typed payloads, emission coverage, journal export |
| **Tier boundary drift** | §2, §28 | Single Critical violation undermines canon §7.4.4 |
| **Identity / intake naming** | §3, §4 | Resolved — `TaskEnvelope` in `intergrax/contracts/task_envelope.py`; parity tests in `test_faudit_remediation.py` |

### M.3 Paydown log

| Date | FAUDIT ID | Summary |
|------|-----------|---------|
| 2026-06-06 | FAUDIT-32.0 | Full 32-layer audit (`scope: C`, `audit-and-fix`); scorecard + §6.1ah queue + Appendix M; gate **893**; boundary scripts OK |
| 2026-06-06 | FAUDIT-TIER.1–OPS.1 | **23/23** remediation implemented; tier gate + intake + observability + registry depth |
| 2026-06-06 | FAUDIT-PE.1+/ALG.1+/MEM.1+ | Golden prompt CI, reference agent lifecycle metadata, STM retention wiring; gate **901** |
| 2026-06-07 | OBS-DEPTH.* + T12 + LEG depth | Unified journal + trace bridge gate + live bus emit + 170-tool catalog + §21 L3 depth gate; gate **967** |
| 2026-06-07 | T13 + CRIT-V-2.* | `eval.judge` + `eval.trajectory`; catalog **172**; doc sync; gate **990** |
| 2026-06-07 | CRIT-V-3.1–3.3 | `CriticOrchestrator`, `L0Gateway`, `L1Gateway`, `CriticEvalToolClient` | gate **996** |

---

## Appendix N — Nexus execution flow traceability (Phase FLOW)

**Source:** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25 · [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md)

**Phase register:** [Phase FLOW](#phase-flow--nexus-execution-depth) · **Band 2aj** · queue [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed) · execution [§6.2aj](#62aj-phase-flow-execution-order-band-2aj--closed-2026-06-07)

**Status:** **Done** (2026-06-09) · **18/18 harness Done** (FLOW-8 product host **Deferred** §6.3)

> **Note:** Distinct from `guides/AGENT_CREATION_GUIDE.md` Appendix N (agent assembly). This appendix maps **orchestration runtime depth** gaps only.

### N.1 FLOW-GAP → FLOW ID matrix (complete)

| Gap ID | Category | Severity | FLOW ID | Deliverable | AUDIT_MAP § |
|--------|----------|----------|---------|-------------|-------------|
| FLOW-GAP-01 | Runtime-core | High | FLOW-1 | Real `EngineBackedNexusPlanner` | §7 |
| FLOW-GAP-02 | Runtime-core | **Critical** | FLOW-2 | ADR-FLOW-001 delegation expansion | §10 |
| FLOW-GAP-03 | Runtime-core | Medium | FLOW-3 | `max_delegation_depth` enforcement | §10 |
| FLOW-GAP-04 | Runtime-core | Medium | FLOW-4 | Opt-in run-level retry | §9, §22 |
| FLOW-GAP-05 | DX | Low | FLOW-5 | `AgentGraph.on_error` wire | §9 |
| FLOW-GAP-06 | Runtime-core | Medium | FLOW-6 | Strict cycle detection | §9 |
| FLOW-GAP-07 | Production-hardening | Medium | FLOW-7 | `MergePolicy` / composer profile | §9 |
| FLOW-GAP-08 | DX / lifecycle | Low | FLOW-10 | Reserved lifecycle states ADR | §8 |
| FLOW-GAP-09 | Production-hardening | Medium | FLOW-11 | Pre-plan policy hooks | §5 |
| FLOW-GAP-10 | Product-proof | Product | FLOW-8 | §42.43 reference Tier-3 app (**Deferred**) | §28 |
| FLOW-GAP-11 | Production-hardening | Medium | FLOW-9 | Multi-agent eval hooks | §25 |
| FLOW-GAP-12 | Runtime-core | Medium | FLOW-13 | `max_inflight_nodes` profile + factory wire | §9 |
| FLOW-GAP-13 | Runtime-core | Medium | FLOW-14 | `SubtaskContract` in delegation expansion | §10 |
| FLOW-GAP-14 | Production-hardening | Medium | FLOW-15 | Subagent budget envelope enforcement | §10 |
| FLOW-GAP-15 | DX | Low | FLOW-16 | `MODIFY_PLAN` reserved semantics ADR | §9 |
| FLOW-GAP-16 | DX | Low | FLOW-17 | `MULTI_AGENT` deterministic ordering policy | §9 |
| §24 / FAUDIT-COG-1 | Cognition | Medium | FLOW-12 | `DecisionRecord` regression gate | §7 |
| — | Docs | Low | FLOW-DOC.* | Flow reference + plan sync | — |

### N.2 Maturity uplift targets

| AUDIT_MAP § | Baseline (FAUDIT-32) | Target | Closing FLOW IDs |
|-------------|----------------------|--------|------------------|
| §5 Policy | L2 partial | **L3** | FLOW-11 |
| §7 Reasoning / planning | L2 | **L3** | FLOW-1, FLOW-12 |
| §8 Execution runtime | L3 | **L3** | FLOW-10 (maintain) |
| §9 Orchestration / graph | L3 partial | **L3+** | FLOW-4–7, FLOW-6, FLOW-13, FLOW-16, FLOW-17 |
| §10 Subagents | L2 | **L3** | FLOW-2, FLOW-3, FLOW-14, FLOW-15 |
| §25 Evaluation | L2 | **L3** | FLOW-9 |

### N.3 Paydown log

| Date | FLOW ID | Summary |
|------|---------|---------|
| 2026-06-07 | — | Phase FLOW scheduled; Appendix N (FLOW) created; §6.1aj + §6.2aj active |
| 2026-06-07 | — | FLOW-GAP-12–16 + FLOW-13–17 added; orchestration plan complete vs flow reference |
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Full Phase FLOW closeout; ADR-FLOW-001/002/003 accepted; gate green |

---

*Plan synced (2026-06-09). **Harness platform** bands 1–2ar **Done**. **Default active queue:** [§6.1](#61-harness-implementation-queue--continuous-gate) maintenance only. Product: [§6.3](#63-end-of-plan--deferred-product-work-only). **Every PR:** §6.1 gate green.*

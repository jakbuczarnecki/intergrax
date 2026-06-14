# Nexus Execution Flow — Implementation Plan

**Architecture (1:1):** [`architecture/NEXUS_EXECUTION_FLOW.md`](../architecture/NEXUS_EXECUTION_FLOW.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §5, §6 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-8.1 | §8 Runtime | Long-running workflow resume E2E on product hosts | P2 | **Done** |
| AUDIT-IDEAL-8.2 | §8 Runtime | Checkpoint introspection API for ops (beyond lab) | P2 | **Done** |
| AUDIT-IDEAL-10.1 | §10 Subagents | Evaluator-loop standard node in product graph specs | P2 | **Done** |
| AUDIT-IDEAL-10.2 | §10 Subagents | Budget delegation enforcement on all delegation paths | P2 | **Done** |
| AUDIT-IDEAL-6.6 | §6 LLM (shared) | ACP `StepLLMRouter` backed by `LLMAdapter` | P1 | **Planned** — [M-LLM-X.5](plan/LLM_ADAPTERS.md) |

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

**Suggested PR order:** See [Phase FLOW — Suggested PR order](#flow--suggested-pr-order).

**Explicitly excluded:** K.1, K.2 (unless FLOW-8 activated), nested harness per child.### 6.1ak Harness implementation queue — Critic & Verification Layer (closed)

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

---

### 6.2aj Phase FLOW execution order (Band 2aj — closed 2026-06-07)

**Status:** **Done** · register: [Phase FLOW](plan/ORCHESTRATION.md) · queue: [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed)

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
| 17 | FLOW-DOC.* | Docs closeout | Low | FLOW-1–17 (except deferred FLOW-8) |### 6.2ak Phase CRIT-V execution order (Band 2ak — active)

**Status:** **Active** · register: [Phase CRIT-V](plan/CRITIC_VERIFICATION.md) · queue: [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-active)

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



**Status:** **Done** (2026-06-07) — **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3 §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **maintenance only** — §6.1 gate (Band 3 §6.3 frozen)  
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
| FLOW-8 | FLOW5 | FLOW-GAP-10 | **Harness CFG simulation** (ORCH-CONFIG.5) + optional Tier-3 §42.43 product host | **Partial** | Harness + Product | `tests/integration/runtime/test_orchestration_cfg_simulation.py` · product §6.3 gate |
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
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Phase FLOW implementation complete: delegation expansion, graph hardening, profile wiring, ADR-FLOW-002/003; gate **906 passed**; **18/18 harness** (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3) |

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



**Status:** **Done** (2026-06-07) — **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3 §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **maintenance only** — §6.1 gate (Band 3 §6.3 frozen)  
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
| FLOW-8 | FLOW5 | FLOW-GAP-10 | **Harness CFG simulation** (ORCH-CONFIG.5) + optional Tier-3 §42.43 product host | **Partial** | Harness + Product | `tests/integration/runtime/test_orchestration_cfg_simulation.py` · product §6.3 gate |
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
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Phase FLOW implementation complete: delegation expansion, graph hardening, profile wiring, ADR-FLOW-002/003; gate **906 passed**; **18/18 harness** (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3) |

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

---

## Phase FLOW — Nexus execution depth

**Status:** **Done** (2026-06-07) — **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3 §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **maintenance only** — §6.1 gate (Band 3 §6.3 frozen)  
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
| FLOW-8 | FLOW5 | FLOW-GAP-10 | **Harness CFG simulation** (ORCH-CONFIG.5) + optional Tier-3 §42.43 product host | **Partial** | Harness + Product | `tests/integration/runtime/test_orchestration_cfg_simulation.py` · product §6.3 gate |
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

## Appendix N


---

## Appendix N — Nexus execution flow traceability (Phase FLOW)

**Source:** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25 · [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md)

**Phase register:** [Phase FLOW](#phase-flow--nexus-execution-depth) · **Band 2aj** · queue [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed) · execution [§6.2aj](#62aj-phase-flow-execution-order-band-2aj--closed-2026-06-07)

**Status:** **Done** (2026-06-07) · **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3; product §6.3)

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
| FLOW-GAP-10 | Product-proof | Harness + Product | FLOW-8 | Harness: `test_orchestration_cfg_simulation.py` **Partial**; Tier-3 §42.43 product **Deferred** §6.3 | §28 |
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

*Plan synced (2026-06-07). **Harness platform** bands 1–2aj **Done** (FAUDIT-32 **23/23** + Phase FLOW **18/18 harness**). **Default active queue:** [§6.1](#61-harness-implementation-queue--continuous-gate) maintenance. Product: [§6.3](#63-end-of-plan--deferred-product-work-only) incl. **FLOW-8**. **Every PR:** §6.1 gate green.*

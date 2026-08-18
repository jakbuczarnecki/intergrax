# Orchestration — Implementation Plan

**Architecture (1:1):** [`architecture/ORCHESTRATION.md`](../../architecture/ORCHESTRATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (ORCHESTRATION plan).

- **Implement / audit default:** Active `### 6.1*` queues with open P0/P1 · Phase AUDIT-IDEAL **Planned** rows. Closed ORCH-* registers — satellite only when re-validating
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/ORCHESTRATION.md`](../../architecture/ORCHESTRATION.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.4 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-3.1 | §3 Intake | Canonical `TaskEnvelope` consolidation (`Task` + `RuntimeRequest`) | P1 | **Done** |
| AUDIT-IDEAL-9.1 | §9 Orchestration | Production queue adapter (beyond SQLite scaffold) | P1 | **Done** |
| AUDIT-IDEAL-9.2 | §9 Orchestration | Swarm + peer-to-peer coordination graph templates | P2 | **Done** |
| AUDIT-IDEAL-9.3 | §9 Orchestration | Dynamic execution strategy selection (L4 / AHI hook) | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

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

**Explicitly excluded:** K.1, K.2, new graph node types, nested harness per child — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1c Harness implementation queue — orchestration strategies (closed)

**Purpose:** Phase **ORCH-STRAT** (Band 2ap) documentation closeout. **Closed 2026-06-08** — all ORCH-STRAT rows **Done**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 1 | **ORCH-STRAT.1–5** | Docs | **Done** | [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#50-orchestration-strategies-catalog) §50–§54 | Strategy catalog + gap register |
| 2 | **ORCH-STRAT.6** | Docs | **Done** | Cross-ref sync | FLOW §27, AUDIT_MAP §9–§10 |

**Runtime backlog:** [Phase ORCH-5](.#phase-orch-5--orchestration-strategy-runtime-gaps-band-2aq--closed) — **Done** (2026-06-09).

### 6.1d Harness implementation queue — orchestration authoring docs (closed)

**Purpose:** Phase **ORCH-DOC.3** (Band 2ar) — posture × pattern matrix for Tier-3 authors. **Closed 2026-06-09**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 1 | **ORCH-DOC.3** | Docs | **Done** | [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#55-interaction-posture--orchestration-matrix) §55 + §53.1 cross-ref | ORCHESTRATION canon + REASONING §9.4 link |

### 6.1g Harness implementation queue — governance audit (closed)

**Purpose:** Phase GOV-AUDIT documentation closeout — **closed 2026-06-05**.

| Order | ID | Status | Deliverable |
|-------|-----|--------|-------------|
| 1 | GOV-DOC.1 | **Done** | Appendix H control plane |
| 2 | GOV-DOC.2 | **Done** | Cross-ref sync |
| 3 | GOV-DOC.3 | **Done** | EXTENSION_AUTHOR §10 |
| — | GOV-PROD.1 | **Deferred** | Product dashboard → §6.3 |

---

### 6.2bb Phase ORCH execution order (Band 2j — closed 2026-06-05)

**Status:** **Done** · register: [Phase ORCH](plan/ORCHESTRATION.md) · queue: [§6.1b](.#61b-harness-implementation-queue--orchestration-closeout-closed)

Work **one ORCH ID per PR**; after each step update the ORCH master table + §6.1b + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | Depends on |
|-------|-----|-------------|----------|------------|
| 1 | ORCH-1 | Planner/classifier kind registry + `nexus_factory` wiring | **Critical** | ORCH-DOC.* |
| 2 | ORCH-2 | `graph_spec_to_plan` + planning runner integration | High | ORCH-1 (shared factory path) |
| 3 | ORCH-3 | `max_parallel_nodes` on `OrchestrationProfile` + `GraphExecutor` | Medium | — (parallel OK after ORCH-1) |
| 4 | ORCH-4 | Docs closeout — Appendix I + plan §0.5 | Low | ORCH-1–3 |### 6.2v Phase V-REM execution order (Band 2i — closed 2026-06-05)

**Status:** **Done** · register: [Phase V-REM](plan/ORCHESTRATION.md) · queue: [§6.1z](.#61z-harness-implementation-queue-consolidated) (closed)

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

---

---

## Cross-domain pasted content removed

| Need | Source |
|------|--------|
| Platform appendices | [`plan/satellites/PLATFORM_FOUNDATION_appendices.md`](plan/satellites/PLATFORM_FOUNDATION_appendices.md) |
| Master registers | [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md) |

---
## Phase ORCH — Orchestration control plane closeout

**Status:** **Done** (2026-06-05) — **6/6** deliverables Done (ORCH-DOC.* + ORCH-1–4); gate **581 passed**  
**Prerequisites:** R-Delegate **Done**, Q+-N.* runners **Done**, H-APP.3.1–3.2 **Done**, V-MA.* **Done**  
**Goal:** Close orchestration audit residuals (AUDIT_MAP §7–§10) — wire declared Tier-3 profile fields to runtime; bridge declarative graph spec to execution plan; cap graph batch concurrency.  
**Priority ladder:** **Band 2j** (§4.0) — **default implementation queue** after §6.1 gate on each PR.  
**Execution order:** [§6.2bb](.#62bb-phase-orch-execution-order-band-2j--active) · queue: [§6.1b](.#61b-harness-implementation-queue--orchestration-closeout-active)
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

**Explicitly out of scope:** Nested full harness per child (use R-Delegate); new graph node types (Tier-1 canon change); product-specific orchestration in `agents`.

### ORCH — Paydown log

| Date | ORCH ID | Summary |
|------|---------|---------|
| 2026-06-05 | ORCH-DOC.1, ORCH-DOC.2 | Governance + orchestration audit docs; Appendix H/I; AUDIT_MAP cross-refs |
| 2026-06-05 | ORCH-1, ORCH-2, ORCH-3 | Orchestration wiring, graph spec plan seed, parallel cap; gate **581** |
| 2026-06-05 | ORCH-4 | Plan + author guide closeout |
| 2026-06-08 | ORCH-STRAT.1–6 | Strategy catalog §50–§54 in [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#50-orchestration-strategies-catalog); FLOW §27 + AUDIT_MAP §9–§10 cross-ref |

**Phase ORCH complete when:** ORCH-1–4 **Done**; §6.1b queue closed; Appendix I has no “planned wiring” gaps; gate **581** green. **Status: complete (2026-06-05).**

---

## Phase ORCH-STRAT — Execution strategies canon (Band 2ap)

**Status:** **Done** (2026-06-08) — strategy sections added to orchestration architecture pair  
**Prerequisites:** Phase ORCH **Done** · Phase V-MA **Done** · Phase FLOW **Done**  
**Goal:** Consolidate coordination patterns, parallelism, resilience, and specialization in [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#50-orchestration-strategies-catalog) §50–§54 — close audit gap “strategies only in FLOW”
**Priority ladder:** **Band 2ap** — **closed** on doc merge  
**ADR:** **No ADR needed** — documentation consolidation; runtime contracts unchanged

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| ORCH-STRAT.1 | **§50** — coordination pattern catalog + collaboration/specialization | **Done** | **Critical** | `architecture/satellites/ORCHESTRATION_production_gates.md` | Maps to `CoordinationPattern` enum |
| ORCH-STRAT.2 | **§51** — parallelism, merge, backpressure | **Done** | **Critical** | same | Cross-ref FLOW §9 |
| ORCH-STRAT.3 | **§52** — resilience (retry layers, checkpoint, failover vs ECP) | **Done** | High | same | Cross-ref FLOW §14, RELIABILITY |
| ORCH-STRAT.4 | **§53** — specialization, delegation, handoff | **Done** | High | same | Cross-ref REASONING, FLOW §13 |
| ORCH-STRAT.5 | **§54** — maturity / gap register | **Done** | Medium | same + this plan | ORCH-5 backlog listed |
| ORCH-STRAT.6 | **Cross-ref sync** — FLOW §27, AUDIT_MAP §9–§10, hub unchanged | **Done** | Medium | `docs/*` | Links resolve |

---

## Phase ORCH-CONFIG — Platform interaction & multi-agent configuration (Band 2ar — closed)

**Status:** **Done** (2026-06-09) — **11/11 Done** (architecture **Done** incl. §59 audit — [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#56-platform-interaction--multi-agent-configuration-canon) §56–§59; reference host CFG presets + harness simulation)
**Prerequisites:** Phase ORCH-STRAT **Done** · Phase H-APP-DOC.1 **Done** · default queue = §6.1 maintenance  
**Goal:** Close every gap in §56.11 so **all CFG-* cases** marked ⚠️/❌ become ✅ without runtime forks  
**Canonical input:** §56.7 case register + §56.11 plan table — do not duplicate elsewhere

**Harness-first rule (2026-06-09):** ORCH-CONFIG validates platform behaviour via **harness integration tests** (`tests/integration/runtime/test_orchestration_cfg_simulation.py`) with abstract stub agents — **not** by implementing Tier-3 business products. Tier-3 reference hosts (FLOW-8 / §6.3) remain product-gated.

**ADR:** [`ADR-FLOW-004`](../../technical/adr/entries/2026-06-09/ADR-FLOW-004.md) (ORCH-CONFIG.2 seed guard); ORCH-CONFIG.3 → no ADR (suffix convention).

| ID | CFG / scope | Deliverable | Status | Priority | Acceptance |
|----|-------------|-------------|--------|----------|------------|
| ORCH-CONFIG.1 | CFG-04 | Rules classifier + `IntentRoute` + orchestration tokens (§56.13) | **Done** | **Critical** | `classifier_kind=rules|llm`; COG-3.2 `LlmTaskClassifier` · `test_llm_task_classifier.py` |
| ORCH-CONFIG.2 | CFG-18 | `ApplicationGraphSpec.trigger_capabilities` + `should_seed` guard | **Done** | **Critical** | ADR-FLOW-004 · `test_graph_spec_to_plan.py` |
| ORCH-CONFIG.3 | CFG-05+ | Pipeline convention: `*.pipeline` → graph_spec seed (no per-product `TaskPlanner` fork) | **Done** | High | `pipeline_capability_suffix` default `.pipeline` |
| ORCH-CONFIG.4 | CFG-03, CFG-14 | Scaffold optional interaction intake + queue consumer wiring | **Done** | High | Product scaffold: `INCLUDE_INTERACTIONS` / `INCLUDE_SCHEDULER`; legal host reference |
| ORCH-CONFIG.5 | CFG-06–08, CFG-17, CFG-20 | Harness CFG simulation + optional Tier-3 product host (FLOW-8) | **Done** (harness) | High | `test_orchestration_cfg_simulation.py` CFG-04/06/07/08/17/18/20 |
| ORCH-CONFIG.6 | CFG-13, CFG-19 | `long_running` profile → task defaults helper / host policy doc | **Done** | Medium | `apply_long_running_from_profile` · `intergrax/applications/USAGE.md` |
| ORCH-CONFIG.7 | CFG-16, CFG-20 | `strict` multi-agent preset on `ApplicationEnvironmentProfile` | **Done** | Medium | `strict_multi_agent_defaults()` · critic + merge bundled |
| ORCH-CONFIG.8 | CFG-17 | Swarm runtime — extends ORCH-5.1 | **Done** | Medium | `swarm_policy.py` · `GraphExecutor` batch guard · CFG-17 sim |
| ORCH-CONFIG.9 | All CFG | `check_orchestration_config_docs.py` — CFG IDs in tests/docs | **Done** | Low | `scripts/maintenance/check_orchestration_config_docs.py` |
| ORCH-CONFIG.10 | CFG-11 | COG-1.* engine planner production path | **Done** | High | `nexus_plan_bridge.py` · `test_cog_depth_residual_gate.py` |
| ORCH-CONFIG.11 | §59 | **Audit canon** — §59 gaps/debt/discrepancies register | **Done** | Medium | [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#59-platform-execution-audit---gaps-technical-debt-discrepancies) §59 + hub index 2026-06-09 |

**Execution order (complete):** ORCH-CONFIG.2 → ORCH-CONFIG.1 → ORCH-CONFIG.3 → ORCH-CONFIG.4 → ORCH-CONFIG.10 → ORCH-CONFIG.6 → ORCH-CONFIG.7 → ORCH-CONFIG.8 → ORCH-CONFIG.9 → ORCH-CONFIG.5 (harness **Done**); FLOW-8 product host **Deferred** §6.3.

**Traceability:**

| Architecture §56 | Plan rows |
|------------------|-----------|
| §56.3 dimensions A–E | ORCH-CONFIG.* acceptance criteria |
| §56.7 CFG register | One ORCH-CONFIG row per ❌/⚠️ case |
| §56.11 gap table | This phase master register |

**Cross-domain ownership (do not duplicate registers):**

| ORCH-CONFIG | Also in plan |
|-------------|--------------|
| ORCH-CONFIG.1 | `REASONING_AND_COGNITION.md` COG-3.* **Done** |
| ORCH-CONFIG.2 | `TIER3_APPLICATION_ENVIRONMENT.md` H-APP-DOC.2 **Done** |
| ORCH-CONFIG.4 | `TIER3_APPLICATION_ENVIRONMENT.md` H-APP-DOC.4 |
| ORCH-CONFIG.5 | `NEXUS_EXECUTION_FLOW.md` FLOW-8 harness sim **Done**; product host **Deferred** §6.3 |
| ORCH-CONFIG.8 | Phase ORCH-5.1 (swarm) |

**Harness tests (canonical):** `tests/unit/applications/test_graph_spec_to_plan.py` · `tests/unit/runtime/nexus/test_orchestration_capabilities.py` · `tests/unit/runtime/nexus/test_intent_routing.py` · `tests/unit/runtime/nexus/test_llm_task_classifier.py` · `tests/integration/runtime/test_orchestration_cfg_simulation.py` · `tests/integration/runtime/test_engine_planner_orchestration_gate.py`

---

## Phase ORCH-5 — Orchestration strategy runtime gaps (Band 2aq — closed)

**Status:** **Done** (2026-06-09) — **5/5 Done**  
**Prerequisites:** Phase ORCH-STRAT **Done** · default queue = §6.1 maintenance  
**Goal:** Close gaps in [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#54-maturity-and-gap-register) §54 — swarm depth, pattern metadata on plans, active redundancy policy

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| ORCH-5.1 | **Swarm runtime profile** — budget envelope + parallel cap for `CoordinationPattern.SWARM` | **Done** | Medium | `validate_swarm_parallel_batch` in `GraphExecutor` · CFG-17 sim |
| ORCH-5.2 | **`coordination_pattern` on `NexusPlan` metadata** — explicit pattern id for trace/audit | **Done** | Medium | `PLAN_CREATED` payload + task metadata |
| ORCH-5.3 | **Wire `select_coordination_pattern()` to lab hosts** — optional advisory in planning trace | **Done** | Low | `planning_coordination_advisory.py` · lab `emit_coordination_advisory` |
| ORCH-5.4 | **Advanced merge strategies** — citation-preserving or structured conflict (IDEAL) | **Done** | Low | `MergeStrategy.CITATION_PRESERVING` · `final_response_composer.py` |
| ORCH-5.5 | **Runbook: orchestration resilience** — link W-OPS SLO to §52 matrix | **Done** | Low | `HARNESS_ENVIRONMENT.md` § orchestration resilience |

**Explicitly out of scope:** active-active duplicate graph nodes (use retry + ECP); K.1/K.2 product graphs (FLOW-8).

### ORCH-STRAT traceability

| Architecture § | Topic | Source |
|----------------|--------|--------|
| §50 | Patterns | V-MA.*, FLOW §27, `multi_agent_coordination.py` |
| §51 | Parallelism | ORCH-3, FLOW-9, FLOW-13 |
| §52 | Resilience | FLOW §14, W-OPS, RELIABILITY |
| §53 | Specialization | REASONING §9–§10, R-Delegate, FLOW §13 |
| §54 | Gaps | ORCH-5.* backlog |
| §57 | Sync/async postures | ORCH-6.* |
| §58 | Platform capabilities index | Cross-domain (hub) |

---

## Phase ORCH-6 — Synchronous and asynchronous execution postures (Band 2au — planned)

**Status:** **Done** (2026-06-09) — architecture canon §57; ORCH-6.1–6.4 implemented.

**Goal:** Document and harden **sync vs async** dispatch as first-class host configuration — same Nexus graph, different client wait semantics.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| ORCH-6-DOC.1 | Canon §57 + §58 index | **Done** | High | FLOW §3.1 cross-ref |
| ORCH-6.1 | **`run_async` host helper** — enqueue + return handle from Tier-3 factory | **Done** | High | `async_task_dispatch.py` + `/v1/tasks/run-async` |
| ORCH-6.2 | **Profile preset** — `async_batch_defaults()` on `ApplicationEnvironmentProfile` | **Done** | Medium | `test_platform_runtime_capabilities.py` |
| ORCH-6.3 | **Agent async contract gate** — default `AgentExecutionMode.ASYNC` on contract | **Done** | Medium | `test_platform_runtime_capabilities.py` |
| ORCH-6.4 | **Author appendix** — sync/async in AGENT_CREATION_GUIDE Appendix X | **Done** | Low | Appendix X §X.4 |
| ORCH-6.5 | **Product host exposure** — `mount_harness_task_routes` + durable queue option beyond lab | **Done** | High | H-APP-WIRING **Done**; legal/research/poc_template |

**Prerequisites:** Queueing plane **Done**; `message_bus.async_runner` skill **Done**.

**Audit note (2026-06-09):** ORCH-6.1–6.4 close **runtime** posture; ORCH-6.5 tracks **Tier-3 surface debt** documented in [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#57-synchronous-and-asynchronous-execution-postures) §57.5, [`§59.2`](../../architecture/satellites/ORCHESTRATION_production_gates.md#59-platform-execution-audit---gaps-technical-debt-discrepancies).

**Explicitly excluded:** New queue transport; nested Nexus per async job.

### 6.1av Harness implementation queue — Orchestration audit maintenance (closed)

**Source:** Layer 3 audit (2026-06-18) — `ORCHESTRATION` layers 3, 9 · [`../audit_results/2026-06-18/ORCHESTRATION.md`](../../../audit_results/2026-06-18/ORCHESTRATION.md)
**Priority ladder:** **Band 1** (§6.1) — P3 harness depth + author DX; runs **in parallel** with gate maintenance; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **ORCH-MAINT-01** | Code/Docs | P3 | **Done** | Scaffold `lab_stack` preset: `INCLUDE_QUEUE_WORKER=true` by default | `new-application` lab preset wires `QueuedNexusExecutionAdapter` path; gate green |
| 2 | **ORCH-MAINT-02** | Docs | P3 | **Done** | LKW hybrid daemon enablement runbook (CFG-14) in `local_workspace_application/ARCHITECTURE.md` | Operator can enable scheduler + interactions without reading source |
| 3 | **ORCH-MAINT-03** | Code | P3 | **Done** | `TaskPriority` in `intergrax/queueing` + broker adapter hook | Priority field on enqueue; unit test; no Nexus fork |
| 4 | **ORCH-MAINT-04** | Code | P3 | **Done** | Durable `AsyncTaskIndex` via integration profile (Redis/SQLite slug) | `async_task_index_resolver.py` selects profile-backed index; lab may keep in-memory fallback |

**Suggested PR order:** none — §6.1av queue closed (2026-06-18).

**Explicitly excluded:** CFG-14 full LKW hybrid E2E (product §6.3); FLOW-8 product host; active-active L0 — [§6.3](PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

### 6.1aw Harness implementation queue — Orchestration audit maintenance (2026-06-19)

**Source:** Interactive layer audit (2026-06-19) — `ORCHESTRATION` layers 3, 9 · [`../audit_results/2026-06-19/ORCHESTRATION.md`](../../../audit_results/2026-06-19/ORCHESTRATION.md) · prior: [`../audit_results/2026-06-18/ORCHESTRATION.md`](../../../audit_results/2026-06-18/ORCHESTRATION.md)
**Priority ladder:** **Band 1** (§6.1) — doc sync + audit artifact; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **ORCH-MAINT-DOC-01** | Docs | P3 | **Done** | Sync [`architecture/satellites/ORCHESTRATION_production_gates.md`](../../architecture/satellites/ORCHESTRATION_production_gates.md#59-platform-execution-audit---gaps-technical-debt-discrepancies) §59.2 async-queue note + §59.4 `run_async` row with ORCH-MAINT-01/04 truth (lab scaffold `INCLUDE_QUEUE_WORKER=true`; `async_task_index_resolver` profile-backed index) | No stale “not scaffold-default” / lab-only in-memory wording; product hosts remain opt-in |
| 2 | **ORCH-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/legacy/2026-06-19` | `ORCHESTRATION.md` + `legacy campaign README` updated; L3 verdict layers 3, 9 |

**Suggested PR order:** none — §6.1aw queue closed (2026-06-19).

**Explicitly excluded:** CFG-14 full LKW hybrid E2E (product §6.3); FLOW-8 product host; active-active L0; new queue transport — [§6.3](PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

---

### Phase B — Extended Nexus



| # | Deliverable | Status |

|---|-------------|--------|

| B.1–B.7 | Classifier, planner, validation, retry, tool policy, composer | **Done** |



---

---

### Phase C — Multi-Agent Readiness



| # | Deliverable | Status |

|---|-------------|--------|

| C.1–C.6 | ExecutionGraph, GraphExecutor, ContextManager, Research pipeline | **Done** |



---

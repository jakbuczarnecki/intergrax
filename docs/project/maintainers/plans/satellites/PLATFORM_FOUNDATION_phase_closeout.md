# Platform Foundation — phase closeout (V-REM, FAUDIT-32, …)

**Parent hub:** [`PLATFORM_FOUNDATION.md`](../PLATFORM_FOUNDATION.md)

## Phase V-REM — Phase V Runtime Remediation (audit closeout)

**Source:** Plan/code audit (2026-06-05) — reconcile Phase V **Done** claims vs runtime evidence; aligned with [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) layers 5, 19, 21, 23, 25, 26.  
**Status:** **Done** (2026-06-05) — **10/10 Done**.  
**Prerequisites:** Phase V contracts **Done**; Phase H-APP **Done** (Tier-3 `ApplicationSecurityProfile` hooks exist).  
**Goal:** Close every **Partial** Phase V row and **A.4** EvalRunner gap — move from governance/evidence-only to **runtime-enforced** behavior. **Achieved 2026-06-05.**  
**Priority ladder:** **Band 2i** (§4.0) — closed.  
**Execution order:** [§6.2v](.#62v-phase-v-rem-execution-order-band-2i--closed-2026-06-05).
**Traceability:** [Appendix J](.#appendix-j--phase-v-remediation-traceability-audit-gap--v-rem-id).

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

---

## Phase FAUDIT-32 — Full architecture audit closeout

**Status:** **Done** (2026-06-06) — 32-layer audit (`scope: C`) + **23/23 FAUDIT remediation** implemented → [§6.1ah](.#61ah-harness-implementation-queue--faudit-32-remediation-closed)
**Source:** [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8  
**Traceability:** **Appendix M** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](.#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](.#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done.

**Post depth bands (2026-06-09):** MEM-DEPTH, COG-DEPTH, ECP-DEPTH, ORCH-CONFIG closeout complete — Appendix M scorecard refreshed. **IDEAL-L3 W2 (2026-06-09):** P0+P1 depth uplift — **32/32 layers L3** (see [Phase IDEAL-L3](plan/IDEAL_HARNESS_L3.md)).

**Gate evidence (verify step):** `uv run pytest -m gate -q` → **901 passed**; `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py` → **OK**.

### FAUDIT-32 — Layer scorecard (summary)

| # | Layer | Score | Crit | High | Plan accurate? |
|---|-------|-------|------|------|----------------|
| 1 | Strategic Harness Model | L3 | 0 | 0 | Yes |
| 2 | Tier Model and Dependency Boundaries | L3 | 0 | 0 | Yes |
| 3 | Interface and Task Intake | L3 | 0 | 1 | Partial |
| 4 | Identity, Trust and Tenancy | L3 | 0 | 0 | Yes |
| 5 | Policy and Governance | L3 | 0 | 2 | Partial |
| 6 | LLM and Model Adapter Layer | L3 | 0 | 1 | Yes |
| 7 | Reasoning, Planning and Cognition | L3 | 0 | 0 | Yes |
| 8 | Execution Runtime and Agent OS | L3 | 0 | 0 | Yes |
| 9 | Orchestration, Scheduler and Execution Graph | L3 | 0 | 0 | Yes |
| 10 | Subagents and Multi-Agent Coordination | L3 | 0 | 0 | Yes |
| 11 | Tool Layer | L3 | 0 | 1 | Yes |
| 12 | Skill Layer | L3 | 0 | 0 | Yes |
| 13 | Integration Layer | L3 | 0 | 0 | Yes |
| 14 | RAG and Retrieval Layer | L3 | 0 | 0 | Yes |
| 15 | Memory Layer | L3 | 0 | 0 | Yes |
| 16 | Context Engineering Layer | L3 | 0 | 0 | Yes |
| 17 | Prompt Engineering and Prompt Registry | L3 | 0 | 0 | Yes |
| 18 | Agent Assembly and Agent Contracts | L2 | 0 | 1 | Yes |
| 19 | Registry Architecture | L2 | 0 | 2 | **No** |
| 20 | Capability Graph Architecture | L3 | 0 | 0 | Yes |
| 21 | Observability and Telemetry | L3 | 0 | 0 | Yes |
| 22 | Error Handling and Reliability | L3 | 0 | 0 | Yes |
| 23 | Security and Data Governance | L3 | 0 | 0 | Yes |
| 24 | Cost and Resource Governance | L3 | 0 | 0 | Yes |
| 25 | Evaluation and Benchmarking | L2 | 0 | 1 | **No** |
| 26 | Testing, CI and Architecture Gates | L3 | 0 | 0 | Yes |
| 27 | Developer Experience, Scaffold and Lab | L3 | 0 | 1 | Yes |
| 28 | Product Environment and Tier-3 Applications | L3 | 0 | 1 | Partial |
| 29 | Modality, Vision, Audio and Dedicated ML | L3 | 0 | 1 | Yes |
| 30 | Operational Excellence and SLOs | L3 | 0 | 1 | Partial |
| 31 | Agent Lifecycle Governance | L3 | 0 | 0 | Yes |
| 32 | Architecture Governance and Documentation Loop | L3 | 0 | 1 | Yes |

**Plan accuracy note:** Rows marked **No** or **Partial** mean the phase closeout register claims **Done** for **wiring/bridge** work, but FAUDIT found **High** gaps vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` / `INTEGRAX_HARNESS_AUDIT_MAP.md` §8 — tracked as **FAUDIT.\*** residuals, not reopening closed closeout phases.

### FAUDIT-32 — Remediation register (implementation queue → §6.1ah)

| ID | Layer | Gap | Severity | Module / acceptance |
|----|-------|-----|----------|-------------------|
| FAUDIT-TIER.1 | §2 | Tier-0 imports `applications/*` in `capability_graph_applications.py` | **Critical** | Move manifest catalog to Tier-3 injection or static metadata; zero `from applications` under `intergrax` |
| FAUDIT-TIER.2 | §2 | No CI gate for `intergrax` → `applications` imports | High | `scripts/maintenance/check_intergrax_no_applications_imports.py` in §6.1 |
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

---

### Phase A — Foundation Stabilization



| # | Deliverable | Status |

|---|-------------|--------|

| A.1 | Unified run lifecycle | **Done** |

| A.2 | Task trace persistence | **Done** |

| A.3 | NexusLoop production path | **Done** |

| A.4 | EvalRunner integration (NexusEvalRunner + gate coverage) | **Done** |

| A.4.1 | NexusEvalRunner integration tests + inclusion in gate | **Done** (2026-06-05 — `tests/integration/eval/test_nexus_eval_runner.py`) |

| A.5-min | Pre-P4.2 regression gate | **Done** |

| A.5 | Full regression suite (Legal E2E, all steps) | **Deferred** |

| A.6 | Shim cleanup | **Done** | Removed `applications/legal_agent`; docs + duplicate `legal_application/tests` cleaned |



**A.5-min completion criteria (gate before P4.2):**



```bash

uv run pytest tests/ -m gate -q

```



| Test area | File |

|-----------|------|

| TaskLifecycle transitions | `tests/unit/runtime/task/test_task_lifecycle.py` |

| TaskTraceEmitter + RuntimeEventBus | `tests/unit/runtime/task/test_task_trace_event_bus.py` |

| trace_bridge mapping | `tests/unit/runtime/events/test_trace_bridge.py` |

| AgentEngine.run / run_with_result | `tests/integration/agents/test_agent_engine_*.py` |

| NexusLoop + Echo (lifecycle + events) | `tests/integration/runtime/test_nexus_loop_echo.py` |

| GraphExecutor sequential stub | `tests/integration/runtime/test_graph_executor_stub.py` |



**Infrastructure fixes included:** circular import (`tool_runtime` ↔ `runtime_state`), missing `RegistryToolExecutor`, `ExecutionGraph` pydantic imports, lazy pipeline imports in `tests/conftest.py`.



**Explicitly not required before P4.2:** Legal through NexusLoop, full Nexus step matrix, E2E with real LLM.



---

---

### Phase K — Hardening & Reference Agents

**Harness prerequisites:** L, Q+, R, S, T, U, and §4.1 **Done** — platform is ready **when** product chooses to start Band 3 (§6.3).

**Scheduling rule (2026-06-02):** K.1/K.2 are **end-of-plan** (§4.0 Band 3, §6.3). Completing harness phases does **not** auto-schedule business agents as the next implementation task.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| K.1 | Problem Radar prototype | **Deferred** | §36 | Wave-1 scaffold frozen (`agents/problem_radar`); resume after harness backlog |
| K.2 | Vendor Discovery prototype | **Deferred** | §37 | After Phase S; product decision |
| K.3 | Policy engine facade | **Done** | §42.11 | `PolicyEngine` + `coerce_replay_policy_engine`; `ExecutionGuard` uses `evaluate_replay` (2026-05-27) |
| K.4 | Dual `AgentDecision` cleanup | **Done** | §42.7 | `ToolPlanDecision` in `tools.core.tool_plan_decision`; no `tools_agent` alias (TYP-06, 2026-06-02) |
| K.5 | ChatAgent / legacy removal | **Done** | §39 | Production paths use Nexus only; `check_production_chat_agent_imports.py` gate (2026-05-27) |
| K.6 | A.5 full Legal E2E gate | **Deferred** | — | Real LLM; not blocking lab — product/CI decision |

---

---

### Phase Q — Harness Quality & Consolidation (audit remediation)

**Source:** Harness implementation audit (2026-06-01) — Nexus, LLM, RAG, memory, observability, legacy, tests, docs.  
**Goal:** Remove bugs, technical debt, dead code, monoliths, dual-path semantics, and documentation drift **without** new business agents or integration catalog breadth.  
**Principle:** evolve, not rewrite · one deliverable per PR · gate green after each step · §0.6 (Tier-1 only when reusable).

**Out of scope for Phase Q:**

- Phase K.1/K.2 business agents (product)
- K.6 / B.15 Legal live LLM E2E (product/CI)
- New integration slugs (Phase M on-demand)
- New Tier-0 universal mechanisms (§5.2.4 human approval)
- Replacing `ToolsAgent` planner (Phase O out of scope)

**Delivery rule:** Same cadence as §6.1 — implement **one Q.* ID** → summarize → update this table + Appendix C status → next ID.

**Phase Q complete when:** All rows below **Done**; Appendix C 100% **Done** or **Won't fix** (documented); §0.5 Harness quality row **Done**; gate unchanged or increased.

---

#### Q.0 — Program governance

| # | Deliverable | Status | Tier | Audit ref | Done when |
|---|-------------|--------|------|-----------|-----------|
| Q.0.1 | Appendix C traceability matrix (audit → Q ID) | **Done** | Docs | C-all | Appendix C below; each row has owner phase |
| Q.0.2 | Phase Q execution order + PR sizing guide | **Done** | Docs | — | §4 + subsection **Q execution order** below |
| Q.0.3 | Gate policy: no Q PR without `pytest -m gate` | **Done** | CI | — | Documented in Q DoD; CI unchanged paths |

---

#### Phase Q-N — Nexus, loops, orchestration, error handling

**Components:** `intergrax/runtime/nexus`, `intergrax/runtime/execution`, `intergrax/runtime/hooks`, `intergrax/runtime/interrupts`, `intergrax/runtime/policy`, `intergrax/runtime/nexus/retry`, `intergrax/agents/agent_engine.py`, `intergrax/agents/uaep.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-N.1 | **Decompose `NexusLoop`** — extract HITL runner, long-running coordinator calls, event publisher, shadow/sandbox cleanup into dedicated modules; `NexusLoop` orchestrates only | **Done** | High | `nexus/orchestration` (`graph_runner`, `task_events`, `lifecycle_bridge`, …) | `nexus_loop.py` ~586 lines; gate green |
| Q-N.2 | **Fix duplicate `_normalize_human_response`** — single call in `_handle_task_impl` | **Done** | High | `nexus_loop.py` L229–231 | Duplicate call removed (2026-06-01) |
| Q-N.3 | **Retry semantics document + facade** — one doc section: `RetryEngine` (graph/validation/alternate agent) vs `RuntimeConfig.max_run_retries` (LLM/tool in `AgentEngine`); optional `RetryCoordinator` delegating both | **Done** | High | `nexus/retry`, `nexus/config.py`, architecture §31.1 | Doc merged; no duplicate retry without trace event |
| Q-N.4 | **Unify policy injection** — `PolicyEngine` only in public Nexus/UAEP APIs; remove `RuntimePolicyEngine` union from external signatures; `coerce_policy_engine` internal | **Done** | Medium | `nexus_loop.py`, `uaep.py`, factories | Type check / mypy clean on factories; gate green |
| Q-N.5 | **§42 hook parity — decision / interrupt / retry** — wire `BEFORE/AFTER_DECISION`, `BEFORE/AFTER_INTERRUPT`, `BEFORE/AFTER_RETRY` in NexusLoop + UAEP + `RetryEngine`; update `hooks/parity.py` to **WIRED** or **Won't fix** with canon amendment | **Done** | Medium | `hooks`, `nexus_loop.py`, `uaep.py`, `retry_engine.py` | `parity.py` no NOT_WIRED for these six OR canon §42.20 amended + tests |
| Q-N.6 | **§42 hook parity — trace persist** — `BEFORE/AFTER_TRACE_PERSIST` **WIRED** at trace finalize path; `parity.py` → **WIRED** | **Done** | Medium | `hooks`, `task_trace.py`, trace emitter | Parity test; hook invoked in integration test |
| Q-N.7 | **Rename Nexus context helpers module** — `nexus/context/tool_context_helpers.py` → `nexus/context/tool_context_helpers.py` (or merge into `tools_step.py`); update imports | **Done** | Low | `tool_context_helpers.py` + shim `tools.py` | Backward-compatible re-export (2026-06-01) |
| Q-N.8 | **Split `RuntimeConfig`** — `ModelRuntimeConfig`, `RetrievalRuntimeConfig`, `ToolsRuntimeConfig`, `PlanningRuntimeConfig`, `TraceRuntimeConfig`; composed `RuntimeConfig`; `validate()` cross-field | **Done** | High | `nexus/config.py` | Backward-compatible properties or migration shim one release; all factories updated |
| Q-N.9 | **Type `integration_profile`** — `IntegrationProfile` from `intergrax.integrations` on `RuntimeConfig` / wiring contexts | **Done** | Medium | `nexus/config.py`, `engine/runtime_context.py` | No `Optional[object]` for profile in public config |
| Q-N.10 | **`production_mode` lab default** — `lab_application` / scaffold sets `production_mode=False`; document in Step 4E | **Done** | Low | Tier-3 factories, `guides/AGENT_CREATION_GUIDE.md` | `harness_production_mode()` in `applications/_shared/runtime_defaults.py` |
| Q-N.11 | **Graph callback typing** — `ExecutionNode` instead of `object` in `GraphExecutor` / NexusLoop node callbacks | **Done** | Low | `execution/graph_executor.py`, `nexus_loop.py` | Mypy/ruff on execution package |
| Q-N.12 | **Interrupt handler hygiene** — remove duplicate `InterruptType` import; add unit test for interrupt → policy path | **Done** | Low | `interrupts/handler.py` | Duplicate import removed (2026-06-01) |
| Q-N.13 | **`AgentEngine` static UAEP** — document or inject `event_bus` for `AgentEngine.run` static path; no silent missing events | **Done** | Low | `agents/agent_engine.py` | `_resolve_static_executor`; `tests/unit/agents/test_agent_engine_event_bus.py` |
| Q-N.14 | **Unit tests for `NexusLoop` helpers** — `_finish_task`, lifecycle transitions, HITL branch stubs (mock deps) | **Done** | High | `tests/unit/runtime/nexus/test_nexus_loop.py` | New file; ≥15 focused tests; marker `gate` |
| Q-N.15 | **`GraphExecutor` unit coverage** — failure recovery, skip completed, handoff edge (beyond stub integration) | **Done** | Medium | `tests/unit/runtime/execution` | `test_graph_executor_coverage.py` + checkpoint skip in `test_runtime_checkpoint.py` |

---

#### Phase Q-L — LLM adapters

**Components:** `intergrax/llm_adapters`, `docs/project/architecture/LLM_ADAPTERS.md`, governance plugin.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-L.1 | **Remove or complete `tracked_llm_call`** — if kept: `finally` calls `usage.end_call`; if removed: delete `tracked_call.py` + references | **Done** | Medium | `_shared/tracked_call.py` | File removed (unused) (2026-06-01) |
| Q-L.2 | **Public API surface** — re-export `LLMAdapter`, `LLMProvider`, `LLMAdapterRegistry`, `LLMProfile` from `llm_adapters/__init__.py` | **Done** | Low | `llm_adapters/__init__.py` | Public re-exports (2026-06-01) |
| Q-L.3 | **Provider catalog table in docs** — 19 rows: slug, adapter class, env vars, tools/stream/structured, native vs compat | **Done** | High | `docs/project/architecture/LLM_ADAPTERS.md` | Table matches `LLMProvider` enum + conformance list |
| Q-L.4 | **Fix `LLMProfile` docstring** — `max_retries` only via `options={}`; align examples in guide | **Done** | Low | `registry/profile.py`, tests | Example fixed (2026-06-01) |
| Q-L.5 | **Per-provider `supports_streaming()` / `supports_structured_output()`** — override defaults (`False` base default for streaming); table in Q-L.3 | **Done** | Medium | Each `providers/*.py`, ABC defaults | Conformance reads flags; no false positives |
| Q-L.6 | **`PolicyEngine` + `llm_cost_evaluation`** — rule hook on `TASK_COMPLETED` or policy replay; or remove “next step” from docs until done | **Done** | Medium | `governance`, `observability_bridge.py`, `policy_engine.py` | Test: over-quota/warn triggers policy decision or structured log contract |
| Q-L.7 | **Usage tracking doc** — distinguish adapter `LLMAdapterUsageLog` vs runtime `LLMUsageTracker` | **Done** | Low | `docs/project/architecture/LLM_ADAPTERS.md` § Observability | Two-layer table |
| Q-L.8 | **Conformance: structured output** — parametrize providers with `supports_structured_output`; mock SDK | **Done** | Medium | `tests/unit/llm_adapters` | Added to gate subset in `llm-adapters-guard.yml` |
| Q-L.9 | **Bedrock `context_window_tokens`** — lookup table or model metadata for common `model_id` | **Done** | Low | `providers/aws_bedrock_adapter.py` | `_CONTEXT_WINDOWS` + prefix fallback; `test_bedrock_context_window.py` |
| Q-L.10 | **OpenAI-compat adapter init** — replace `__dict__.update` with explicit delegation or composition wrapper | **Done** | Low | `openai_compat_providers.py`, factory | `_delegate` + `__getattr__` composition |
| Q-L.11 | **Central env appendix** — single table: `INTERGRAX_LLM_*`, secrets map, per-provider overrides | **Done** | Medium | `architecture/LLM_ADAPTERS.md` appendix | Cross-links from each `providers/*/USAGE.md` |

---

#### Phase Q-R — RAG pipeline & Nexus RAG integration

**Components:** `intergrax/rag`, `runtime/nexus/context/context_builder.py`, `nexus/tools/plan_context_invocation.py`, `context_engineering/providers`, `tools/providers/rag`, `agents/legal/*` plan flags.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-R.1 | **Delete dead code in `ContextBuilder`** — `_build_backend_where`, `_map_hits_to_chunks`, unused `VectorStoreHit` import | **Done** | High | `context_builder.py` | Dead helpers removed (2026-06-01) |
| Q-R.2 | **Single retrieval per turn (design)** — ADR in plan: either (A) retrieval only in `rag.retrieve` (catalog), or (B) only in CE history provider; remove duplicate vector calls | **Done** | High | `context_engineering/providers`, `context_builder.py` | History provider uses `perform_retrieval=False` (2026-06-01) |
| Q-R.3 | **`ContextBuilder` respects plan `use_rag`** — `_should_use_rag` checks plan/engine `use_rag` when present, not only `enable_rag` | **Done** | High | `context_builder.py` | `request.metadata["use_rag"]`; unit test (2026-06-01) |
| Q-R.4 | **Conditional `rag.retrieve` (catalog)** — include `rag.retrieve` only when plan/tool_ids require RAG | **Done** | High | `tool_runtime.py`, `plan_context_invocation.py` | Tool plan matrix |
| Q-R.5 | **Prefetch vs final `top_k`** — `RetrievalRequest.prefetch_k` optional; Nexus passes `max_docs_per_query` as `final_k` only; service uses profile `prefetch_top_k` when unset | **Done** | High | `retrieval_request.py`, `retrieval_service.py` | `test_retrieval_request_prefetch.py` (2026-06-01) |
| Q-R.6 | **Unify RAG config surface** — map `RuntimeConfig.max_docs_per_query` / threshold → `RagProfile` at factory wire time; deprecate duplicate fields with shim + trace | **Done** | High | `nexus/config.py`, `RetrievalRuntimeConfig`, `rag_profile.py` | One source of truth documented |
| Q-R.7 | **`RagProfile.extras`** — use for vendor knobs or remove field | **Done** | Low | `rag_profile.py` | No unused field in frozen profile |
| Q-R.8 | **`INTERGRAX_RAG_METRICS_ENABLED` in `rag_profile_from_env`** or documented exclusion | **Done** | Low | `rag_profile.py`, architecture §7.1.2 | `extras.metrics_enabled` from env (2026-06-01) |
| Q-R.9 | **`rag/answers` deprecation path** — mark package deprecated; redirect doc to `RetrievalService`; no new imports from Nexus | **Done** | Medium | `rag/answers`, `chat_agent` removal (Q-X.1) | Grep: zero imports from `runtime` and `agents` except tests |
| Q-R.10 | **`UserProfileManager` LTM via `RetrievalService`** — same metadata scope / `RagProfile` chunking policy | **Done** | Medium | `memory/user_profile_manager.py` | Unit test with fake `RetrievalService` |
| Q-R.11 | **Naming guide — three “context builders”** — table in `AGENT_CREATION_GUIDE` or `intergrax/rag/README.md`: Nexus `ContextBuilder`, `ContextManager`, `DefaultContextBuilder` | **Done** | Low | Docs | Linked from architecture §28 pointer |
| Q-R.12 | **Legacy `use_rag` plan flags** — migrate Legal/Nexus plans to `tool_ids` including `rag.retrieve`; emit deprecation `RuntimeEvent` on boolean | **Done** | Medium | `plan_context_invocation.py`, `legal/*`, `tool_runtime.py` | Legal tests use `tool_ids`; booleans shim one release |

---

#### Phase Q-M — Memory

**Components:** `intergrax/memory`, `runtime/task_memory`, `runtime/nexus/context`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-M.1 | **Memory architecture one-pager** — four stores: session history, user LTM, task KV (`TaskMemory`), shared graph context; diagram + when to enable SQLite | **Done** | High | `docs` section in plan §0 or `AGENT_CREATION_GUIDE` Appendix | Linked from §0.3 execution path |
| Q-M.2 | **Task memory visibility in scaffold** — `wire_task_memory` in lab/product templates; env `INTERGRAX_TASK_MEMORY_DB` in `.env.example`; Step 4E paragraph | **Done** | Medium | `applications/*`, scaffold, guide | Scaffold acceptance asserts task memory path optional |
| Q-M.3 | **`resolve_task_memory_persistence` defaults** — log warning when None in lab; debug API hint | **Done** | Low | `task_memory/store.py`, `lab_application` factory | Doc + single integration test |

---

#### Phase Q-O — Observability & metrics

**Components:** `runtime/events`, `runtime/nexus/tracing`, `runtime/metrics`, `debug`, `llm_adapters/tracking`, `rag/tracking`, `applications/_shared/platform_wiring.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-O.1 | **Register RAG observability plugin in default bootstrap** — `register_rag_observability_plugin(plugins)` alongside LLM in `platform_wiring.py` | **Done** | **Critical** | `platform_wiring.py` | `test_platform_wiring_observability.py` (2026-06-01) |
| Q-O.2 | **RAG observability bridge tests** — mirror `test_observability_bridge.py` (LLM) | **Done** | High | `tests/unit/rag/tracking` | `test_rag_observability_bridge.py` (2026-06-01) |
| Q-O.3 | **Parser trace export strategy** — route `parser_trace_flush` through `ObservabilityBackend` **or** document intentional bypass + single env table | **Done** | Medium | `parser_trace_flush.py`, `parser_trace_exporter.py`, integrations | Documented in architecture §7.1.2 RAG observability |
| Q-O.4 | **`metrics/export.py` typed trace summary** — use `DiagnosticPayload` / `trace_models` schema ids instead of substring heuristics | **Done** | Medium | `runtime/metrics/export.py` | Unit test with synthetic trace events |
| Q-O.5 | **Lint `metrics/export.py`** — remove duplicate `ExecutionMetrics` import | **Done** | Low | `metrics/export.py` | Ruff clean (2026-06-01) |
| Q-O.6 | **`export_run_metrics` behavioral field** — populate from governance/replay or remove from DTO | **Done** | Low | `metrics/export.py` | `ExecutionMetrics` from trace events in `export_run_metrics` |
| Q-O.7 | **Mount LLM metrics routes on lab** — `register_llm_metrics_routes(app)` when `INTERGRAX_LLM_METRICS_ENABLED` | **Done** | Medium | `lab_application/host/factory.py` | Routes registered at factory (2026-06-01) |
| Q-O.8 | **Observability env profile doc** — one table: trace DB, runtime events DB, LLM/RAG metrics, parser trace, integration observability slug | **Done** | High | New subsection §0 or `infra/README` cross-link | All Tier-3 `.env.example` reference same names |
| Q-O.9 | **RAG metrics parity decision** — implement log-only parity **or** `register_rag_metrics_routes` + optional Pushgateway | **Done** | Medium | `rag/tracking`, architecture §7.1.2 | Matches documented behavior |
| Q-O.10 | **Unify phase mapping** — `trace_bridge` delegates phase to `phase_coverage.py`; single source | **Done** | Medium | `events/trace_bridge.py`, `phase_coverage.py` | Unit test: same `ExecutionPhase` for sample events |
| Q-O.11 | **Debug router type imports** — explicit imports for `DebugHitlResumeService`, `AgentRegistry` in annotations | **Done** | Low | `debug/router.py`, `debug/app.py` | Explicit imports in `debug/router.py` |
| Q-O.12 | **`trace_bridge` unit tests** | **Done** | Medium | `tests/unit/runtime/events/test_trace_bridge.py` | Gate marker |
| Q-O.13 | **Clarify dual Prometheus** — in-process scrape vs `integrations` PromQL backend | **Done** | Low | `docs/project/architecture/LLM_ADAPTERS.md` § Observability | Prevents operator confusion |
| Q-O.14 | **Event/trace store adoption** — SQLite-first default; scale-out criteria for `cassandra` / `elasticsearch` | **Done** | Low | Architecture §33.1 + `cassandra/USAGE.md` | No separate ADR file |

---

#### Phase Q-X — Legacy removal & code hygiene

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-X.1 | **`ChatAgent` removal** — migrate remaining tests to `AgentEngine`/`NexusLoop`; delete `intergrax/chat_agent.py`; keep import guard script as negative test | **Done** | High | `chat_agent.py`, `tests/unit/chat_agent` | Grep zero production imports; gate green |
| Q-X.2 | **`task_metadata_bridge` shrink** — migrate callers to typed `Task` metadata; deprecate flat bridge with warning event | **Done** | Medium | `task_metadata_bridge.py`, `uaep.py` | `execution_options_for_request`; legacy warnings; Task hydrates typed fields |
| Q-X.3 | **Copyright / naming consistency** — `Intergrax` header; fix `Integrax` typo in `chat_agent` (or file deleted in Q-X.1) | **Done** | Low | Affected files from audit | Spot-check script or ruff rule |
| Q-X.4 | **`tools_base` deprecation timeline** — document removal after Q-R.12; no new imports | **Done** | Low | `tools/tools_base.py`, governance script | Module docstring + `DeprecationWarning` on import |
| Q-X.5 | **Sync M.6 “Future” slugs table** — weaviate, milvus, snowflake, vault → **Done (beta)** with paths | **Done** | Low | This plan M.6 P3 section | Table matches repo `integrations/providers` |

---

#### Phase Q-T — Test harness gaps

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-T.1 | NexusLoop unit suite | **Done** | High | See Q-N.14 | — |
| Q-T.2 | `test_rag_profile_from_env` | **Done** | Medium | `tests/unit/rag/profiles` | Gate (2026-06-01) |
| Q-T.3 | `test_context_builder_retrieval` | **Done** | High | `tests/unit/runtime/nexus/context` | `test_context_builder.py` (2026-06-01) |
| Q-T.4 | `test_user_profile_manager` | **Done** | Medium | `tests/unit/memory` | Index + search |
| Q-T.5 | **Catalog vs legacy RAG path** — integration test one pipeline run, retrieval call count ≤1 | **Done** | High | `tests/integration/runtime` | Implements Q-R.2 acceptance |
| Q-T.6 | **Observability wiring E2E** — lab factory bootstraps LLM+RAG plugins | **Done** | High | `tests/integration/runtime/test_platform_wiring_observability.py` | Q-O.1 (2026-06-01) |

---

#### Phase Q-D — Documentation & plan sync

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-D.1 | Update `docs/README.md` current focus → Phase Q | **Done** | High | `docs/README.md` | — |
| Q-D.2 | Canon §52 Phase L status → **Done** (pointer to Phase Q) | **Done** | Low | `intergrax_runtime_architecture.md` §52 | — |
| Q-D.3 | §2 architecture map — §42 row points to Phase Q-N.5–Q-N.6 | **Done** | Low | This file §2 | — |
| Q-D.4 | `AGENT_CREATION_GUIDE` — Q-M.1 memory diagram + Q-R.11 naming | **Done** | Medium | Guide appendices | — |
| Q-D.5 | **§5.2 reuse enforcement** — document existing gates (`check_agents_vendor_imports`, `check_integration_vendor_imports`, `check_production_chat_agent_imports`) in AGENT_CREATION_GUIDE anti-patterns | **Done** | Low | Guide + `scripts` | New agent authors see one list |

---

#### Phase Q — Definition of done (global)

1. Deliverable row **Done** with PR link/date in Appendix C paydown log.
2. **Gate:** `uv run pytest -m gate -q` green.
3. **No new** duplicate Tier-0 mechanism (§5.2).
4. **Tests** for behavior change (unit or integration); not docs-only for code fixes.
5. Update **Appendix C** status column for audit ID.

---

#### Phase Q — Recommended execution order

Execute in order unless a row is marked parallel. Critical path for harness stability:

```text
Wave 1 (bugs + critical):  Q-O.1 → Q-N.2 → Q-R.5 → Q-R.1
Wave 2 (RAG semantics):    Q-R.3 → Q-R.4 → Q-R.2 → Q-T.5 → Q-R.6
Wave 3 (observability):    Q-O.2 → Q-O.4 → Q-O.7 → Q-O.10 → Q-O.12 → Q-O.8
Wave 4 (Nexus structure):  Q-N.14 → Q-N.1 → Q-N.3 → Q-N.8
Wave 5 (LLM docs/debt):    Q-L.3 → Q-L.1 → Q-L.5 → Q-L.8 → Q-L.11
Wave 6 (memory + legacy):  Q-M.1 → Q-M.2 → Q-R.10 → Q-X.1 → Q-R.9
Wave 7 (hooks + policy):   Q-N.5 → Q-N.6 → Q-L.6 → Q-N.4
Wave 8 (cleanup):          Q-N.7 → Q-X.2 → Q-X.3 → Q-X.5 → Q-D.*
Parallel anytime:          Q-L.2, Q-L.4, Q-L.9, Q-L.10, Q-O.5, Q-O.6, Q-O.11, Q-O.13, Q-N.10–Q-N.13, Q-N.15
```

**Historical (Phase Q only):** Do not start Phase K.1/K.2 until Q Waves 1–3 were **Done** — **met** (2026-06-01). Phase S focuses on harness environment; K.1/K.2 wait until S Done.

---

---

### Phase Q+ — Harness Hardening (post-audit 2026-06-01)

**Source:** Technical debt audit after Phase Q — architecture compliance, typing, observability gaps, legacy parallel stacks, Nexus/planning monoliths.  
**Goal:** Intergrax as a **strong, typed, observable harness** comparable in discipline to Cursor / Claude Code / Google ADK-style agent labs — not merely “gate green”.  
**Principle:** evolve, not rewrite · explicit `Protocol` / Pydantic at boundaries · **zero new `getattr` in `runtime/nexus` and `agents`** (integrations/LLM SDK edges exempt) · one Q+.* ID per PR · gate green.

**Relationship to Phase Q:** Phase Q closed the **first** audit (Appendix C). Phase Q+ closes the **second** audit (Appendix D). Do not reopen Q.* rows unless a regression is found.

**Out of scope for Phase Q+:**

- Phase K.1/K.2 product agents (unless explicitly prioritized — record in Appendix D)
- K.6 / B.15 Legal live LLM E2E
- New integration catalog slugs (Phase M on-demand)
- Rewriting all LLM provider adapters (only isolate SDK reflection — Q+-I.*)
- Mandatory Cassandra / multi-tenant scale-out (architecture §33.1 criteria only)

**Phase Q+ complete when:** All Q+ rows **Done** or **Won't fix** (canon amendment); Appendix D 100%; §0.5 Harness hardening **Done**; gate unchanged or increased; grep gate: no new `getattr` in `runtime/nexus` + `agents` (CI script Q+.0.3).

---

#### Q+.0 — Program governance

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+.0.1 | **Appendix D** — audit topic → Q+ ID matrix (P0–P3) | **Done** | High | This file Appendix D | Every audit section mapped |
| Q+.0.2 | **Q+ execution order** — Waves 1–5 below | **Done** | High | §4 Priority Order | Team follows wave sequence |
| Q+.0.3 | **CI grep gate** — fail on new `getattr`/`setattr` in `intergrax/runtime/nexus`, `intergrax/agents` | **Done** | High | `scripts/maintenance/check_harness_no_getattr.py` + gate workflow | Zero grandfathered harness paths (2026-06-01) |

---

#### Q+-T — Typing & explicit contracts (P0)

**Audit:** loose coupling, `getattr`, `Any` on harness paths, classes not implementing Protocols.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-T.1 | **`UAEPAgent` Protocol** — `get_steps`, `run_step`, optional `resume_step`, `decide_after_step`; replace `supports_uaep()` duck typing | **Done** | **Critical** | `agents/uaep_protocol.py`, `agents/uaep.py` | Standalone `@runtime_checkable` Protocol; no `getattr` in UAEP |
| Q+-T.2 | **`ToolInvokerProtocol`** — explicit `registry`; remove `catalog_context` invoker chain `getattr` | **Done** | **Critical** | `runtime/nexus/tools`, `catalog_context.py` | Typed invoker only |
| Q+-T.3 | **`RuntimeState` trace hook** — `trace_event: Optional[TraceEmitterFn]`; remove `getattr(state, "trace_event")` | **Done** | High | `tool_access_policy.py` | `TraceEmittingRuntimeState` Protocol |
| Q+-T.4 | **`Agent.can_handle(TaskContext)`** — replace `task_context: Any` on `Agent` ABC | **Done** | High | `agents/agent_contract.py`, product agents | Production agents use `TaskContext` |
| Q+-T.5 | **`EnginePlan` / tool plan union** — `tool_runtime` reads `tool_ids` without `getattr(source, …)` | **Done** | High | `tool_runtime.py`, `engine_plan_models.py` | `ToolPlanLike` + `EnginePlan.resolved_tool_ids()` |
| Q+-T.6 | **`long_running_bridge`** — `RuntimeEventPublisher` accepts `RuntimeEvent` only (not `object`) | **Done** | Medium | `orchestration/long_running_bridge.py` | Align with `NexusRuntimeEventPublisher` |
| Q+-T.7 | **`context_builder` session snapshot** — typed session view; no `getattr(session, attr)` loop | **Done** | Medium | `context/context_builder.py` | `ChatSession` fields directly |
| Q+-T.8 | **`rag_step_policy`** — use `NexusPlan` / `EnginePlan` fields only | **Done** | Low | `agent on_next_step policy` | `isinstance(plan, EnginePlan)` |

---

#### Q+-N — Nexus decomposition & retry (P0–P1)

**Audit:** `nexus_loop` still owns intake/classification/planning; no `RetryCoordinator`; thin graph tests.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-N.1 | **`NexusIntakeRunner`** — resume/long-running preamble + HITL verdict branches extracted from `nexus_loop` | **Done** | High | `orchestration/intake_runner.py` | `nexus_loop` delegates; behavior unchanged |
| Q+-N.2 | **`NexusPlanningRunner`** — classify → plan → pre-graph HITL; hooks + runtime events | **Done** | High | `orchestration/planning_runner.py` | `nexus_loop` slimmed; graph phase unchanged |
| Q+-N.3 | **`RetryCoordinator`** (optional facade) — delegate `RetryEngine` + `RuntimeConfig.max_run_retries` with `RETRY_SCHEDULED` events | **Done** | Medium | `nexus/retry/coordinator.py`, architecture §31.1 | Graph emits `RETRY_SCHEDULED`; run retries use coordinator |
| Q+-N.4 | **`GraphExecutor` integration tests** — handoff edge, validation retry + alternate agent | **Done** | Medium | `tests/integration/runtime/test_graph_executor_handoff_retry.py` | Handoff + alternate-agent retry |
| Q+-N.5 | **Planner failure observability** — `engine_planner` errors → `RuntimeEventType.PLAN_FAILED` (narrow exceptions) | **Done** | Medium | `planning/engine_planner.py`, `planner_events.py` | `test_engine_planner_plan_failed.py` |

---

#### Q+-O — Observability parity (P1)

**Audit:** metrics heuristics, RAG HTTP metrics asymmetry, lab `production_mode` not wired.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-O.1 | **`export_run_metrics` typed-only** — remove getattr/substring fallbacks; require `DiagnosticPayload` / schema ids | **Done** | High | `runtime/metrics/export.py` | `TraceEvent` / `SerializedTraceEvent` only |
| Q+-O.2 | **Wire `harness_production_mode()`** in lab + scaffold factories | **Done** | Medium | `scaffold/new_agent.py`, Tier-2 lab agents | Lab/scaffold agents use `harness_production_mode()` |
| Q+-O.3 | **RAG metrics HTTP decision** — implement `register_rag_metrics_routes` **or** document Won't fix + unified `/metrics` scrape | **Won't fix** (core) | Medium | architecture §7.1.2 | No default `/metrics/rag`; log + plugin scrape |
| Q+-O.4 | **Ingestion path events** — consistent `RuntimeEvent` on ingest failures | **Done** | Low | `ingestion_events.py`, `ingestion_service.py` | `INGESTION_FAILED` + gate test |

---

#### Q+-L — Legacy & duplicate stacks (P0–P2)

**Audit:** `tools_agent`, `supervisor`, `chains`, `openai/rag`, `rag/answers` parallel Tier-0 paths.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-L.1 | **`tools_agent` deprecation enforcement** — extend `check_*_imports`; zero new production imports outside `agents/legal` migration | **Done** | **Critical** | `scripts/check_tools_agent_imports.py` | CI fails on new imports |
| Q+-L.2 | **Legal agent → catalog `ToolRuntime`** — remove runtime dependency on `ToolsAgent` / `run_bounded_tool_loop` / `ctx.invoke_tool` planner loop | **Done** | **Critical** | `agents/legal`, `catalog_tool_planner.py` | Legal uses `CatalogToolPlanner` + `tool_planner` |
| Q+-L.3 | **`RuntimeConfig` default tools** — no default `ToolsAgent` in `config` / `config_sections` | **Done** | High | `nexus/config.py`, `config_sections.py` | `tool_planner: ToolPlannerProtocol` only |
| Q+-L.4 | **`supervisor` boundary** — move to `experiments/supervisor` or hard-deprecate with import guard | **Done** | Medium | `intergrax/supervisor/__init__.py`, gate import test | Not imported from runtime/applications |
| Q+-L.5 | **`chains/langchain_qa_chain`** — removed from harness (package deleted) | **Done** | Medium | — | No `intergrax.chains` imports |
| Q+-L.6 | **`rag/answers` e2e** — migrate `tests/e2e/rag` to `RetrievalService`; package import guard | **Done** | Medium | `tests/e2e/rag/test_rag_full_runtime_e2e.py` | No `rag.answers` import |
| Q+-L.7 | **`openai/rag/rag_openai.py`** — bridge to `RetrievalService` or delete if unused | **Won't fix** | Low | `openai/rag/rag_openai.py` | Zero production imports; legacy sample only |

---

#### Q+-M — Task metadata & bridge (P1)

**Audit:** automatic legacy hydrate on every `Task()`; bridge still central.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-M.1 | **Opt-in metadata hydrate** — `Task.from_metadata()` / factory; remove automatic `model_validator` hydrate | **Done** | High | `task/task.py`, `task_metadata_bridge.py` | Hydrate only when legacy keys / `_hydrate_legacy` |
| Q+-M.2 | **Tier-3 uses typed `Task.options` only** — lab/scaffold run path sets contract without flat keys | **Done** | Medium | `task_intake.py`, lab `fastapi_router.py` | `graph_id` via orchestration state |

---

#### Q+-P — Planning monoliths (P2)

**Audit:** `step_planner.py` ~683 lines, `engine_planner.py` ~623 lines — hard to extend.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-P.1 | **Split `engine_planner`** — parse / validate / LLM call modules; each &lt; ~300 lines | **Done** | Medium | `engine_planner_parse.py`, `nexus_llm_plan_builder.py`, `engine_planner_diagnostics.py`, `nexus_llm_plan_builder.py` | Orchestration + traces extracted |
| Q+-P.2 | **Split `step_planner`** — strategy registry vs executor | **Done** | Medium | `planning/step_planner` (`config`, `step_factory`, `assembly`, `strategies`, `planner`) | Package import stable; gate tests |
| Q+-P.3 | **Structured plan parse errors** — no silent `except Exception: pass` without trace | **Done** | Medium | `engine_planner_parse.py` | Narrow `ValueError` / `JSONDecodeError` only |

---

#### Q+-S — Session monolith (P2)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-S.1 | **Decompose `session_manager`** — storage vs summarization vs org instructions | **Done** | Low | `session_profile_instructions.py`, `session_consolidation.py`, `session_lifecycle.py` | Profile, consolidation, lifecycle coordinators |

---

#### Q+-I — Integration / LLM SDK edges (P3)

**Audit:** acceptable `getattr` inside provider SDK shims — isolate, do not spread.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-I.1 | **SDK reflection quarantine** — document per-provider `*_sdk_bridge.py`; no new getattr in `runtime` | **Done** | Low | Architecture §5.2.2 | Vendor SDK bridges quarantined to provider modules |

---

#### Q+-D — Documentation (Phase Q+)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-D.1 | Canon §9 — orchestration module list includes intake/planning runners (when done) | **Done** | Low | `intergrax_runtime_architecture.md` | — |
| Q+-D.2 | `AGENT_CREATION_GUIDE` — anti-pattern: `getattr`, `ToolsAgent`, flat metadata | **Done** | Medium | Guide § anti-patterns | Linked from §0.6 |
| Q+-D.3 | `docs/README.md` focus → Phase Q+ Wave 1 | **Done** | High | `docs/README.md` | Wave 2 focus |

---

#### Phase Q+ — Definition of done

1. Q+ row **Done** with date in Appendix D paydown log.
2. **Gate:** `uv run pytest -m gate -q` green.
3. **No new** `getattr`/`setattr` in harness paths (Q+.0.3).
4. **Tests** for each behavior change.
5. Update Appendix D status.

---

#### Phase Q+ — Recommended execution order

```text
Wave 1 (P0 contracts):     Q+.0.3 → Q+-T.1 → Q+-T.2 → Q+-T.3 → Q+-T.4 → Q+-T.5
Wave 2 (P0 legacy):      Q+-L.1 → Q+-L.2 → Q+-L.3 → Q+-M.1
Wave 3 (P1 Nexus+obs):   Q+-N.1 → Q+-N.2 → Q+-O.1 → Q+-O.2 → Q+-N.3 → Q+-N.4 → Q+-N.5
Wave 4 (P2 monoliths):     Q+-P.1 → Q+-P.2 → Q+-S.1 → Q+-L.4 → Q+-L.5 → Q+-L.6
Wave 5 (P3 + docs):        Q+-L.7 → Q+-I.1 → Q+-O.3 → Q+-O.4 → Q+-D.*
Parallel anytime:         Q+-T.6, Q+-T.7, Q+-T.8, Q+-M.2
```

**Gate before Phase K scale:** Waves 1–3 **Done** (typing + Legal off ToolsAgent + Nexus intake/planning split + metrics typed).

---

---

### Phase S — Harness Environment GA (post-R 2026-06-01)

**Source:** Architecture audit (2026-06-01); strategic pivot — **full harness environment** before business agents.  
**Status:** **Done** (2026-06-01). **Prerequisites met:** Phase L, Q, Q+, R (MVP).  
**Goal:** Make the **Harness AI environment** (Tier-0 + Tier-1 + lab/product wiring) **ops-ready and complete** — stable integration paths, observability, platform skills, operator docs — using **existing** reference agents (echo, research, legal, signoff_probe), not new product agents.  
**Principle:** evolve, not rewrite · Tier-1 only via §0.6 · one S.* ID per PR · gate green.

**Explicitly out of scope for Phase S:**

- **K.1 Problem Radar / K.2 Vendor Discovery** — **Phase K** (after U Done)
- Multi-tenant SaaS (canon §50 — future)
- Nested full harness per child — graph delegation remains default (R-Delegate)
- `stable` on all **135** integration slugs — only the **lab harness stack** (see S-Ops.1)

**Deferred from old Phase S scope → Phase K:** S-K.* (reference business agent proof).

#### S.0 — Canon & strategy sync

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S.0.1 | **Development strategy** document + docs index | **Done** | Critical | `INTERGRAX_DEVELOPMENT_STRATEGY.md`, `docs/README.md` | Linked from plan + root README |
| S.0.2 | **Canon §2 / §50–§51** — laboratory + harness narrative | **Done** | Critical | `intergrax_runtime_architecture.md` | No contradiction with strategy |
| S.0.3 | **Canon §52** — Phase S harness question | **Done** | High | Canon §52 | Environment GA, not K.1/K.2 |
| S.0.4 | **Plan pivot** — Phase S = harness only; K.1/K.2 deferred | **Done** | Critical | This file §0, §4, Phase K, Appendix F | 2026-06-01 |

#### S-Ops — Integration & observability (harness stack)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-Ops.1 | **Integration stable track** — lab harness stack (`sqlite`, `redis`, `qdrant`, `slack`, `sentry`, …) marked `stable` in catalog | **Done** | **Critical** | `harness_lab_stack.py`, `architecture/INTEGRATIONS.md` | `test_harness_lab_stable_stack.py` |
| S-Ops.2 | **OTLP / observability** — lab profile wires `otel` when `LAB_OTEL_ENABLED`; document noop vs export | **Done** | High | `IntegrationProfile.harness_environment()`, `.env.example` | `test_lab_harness_environment_wiring.py` |
| S-Ops.3 | **Harness-smoke CI** — expand M.12+ coverage for stable stack (network optional) | **Done** | Medium | `.github/workflows/unit-tests.yml` | harness-smoke includes S unit tests |
| S-Ops.4 | **Legal live LLM E2E** | **Deferred** | Low | K.6 / B.15 | Not blocking harness environment |

#### S-H — Platform harness capabilities (no business agents)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-H.1 | **Platform skill bundle `harness`** — ≥3 skills (e.g. `harness.tool_smoke`, `harness.context_demo`, `harness.trace_read`) | **Done** | **Critical** | `intergrax/skills/providers/harness`, `architecture/SKILLS.md`, bootstrap | `test_harness_skill_bundle.py` |
| S-H.2 | **Lab wiring** — `SkillProfile` + `ToolProfile` + policy bundle documented as canonical harness preset | **Done** | High | `skill_wiring.py`, `guides/HARNESS_ENVIRONMENT.md` | lab enables `harness` bundle |
| S-H.3 | **Cursor SKILL.md importer** in gate | **Done** | Medium | `tests/unit/skills/importers/test_cursor_skill_md.py` | `pytest.mark.gate` |
| S-H.4 | **`rag.answers` test migration** — no deprecation warnings in gate | **Done** | Low | `tests/integration/rag/answers` | `RetrievalService` only |
| S-H.5 | **Echo/signoff path** — lab run proves skills + trace + policy bundle (existing agents) | **Done** | High | `tests/acceptance/agent_os/test_lab_application.py` | gate + harness wiring tests |

#### S-Doc — Operator & author surfaces

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-Doc.1 | **`guides/HARNESS_ENVIRONMENT.md`** — lab stack, env vars, stable integrations, OTLP, policy bundle read order | **Done** | **Critical** | `docs/project/technical/guides/HARNESS_ENVIRONMENT.md`, `docs/README.md` index | Linked from plan §6 |
| S-Doc.2 | **Context / trace operator section** — `CONTEXT_*` events, debug API, metrics routes | **Done** | Medium | `guides/HARNESS_ENVIRONMENT.md` | Pointers to canon §28.1 |

#### Phase S — Definition of done

1. **Stable** integration list for lab harness stack published and tested (S-Ops.1).
2. **OTLP path** documented and wired for lab when env configured (S-Ops.2).
3. **≥ 3** `harness.*` platform skills + legal/research bundles registered (S-H.1).
4. **`guides/HARNESS_ENVIRONMENT.md`** complete; lab wiring matches doc (S-H.2, S-Doc.1).
5. Gate: `uv run pytest -m gate -q` green; `python scripts/maintenance/check_harness_no_getattr.py` OK.
6. §0.5 **Harness environment GA** row **Done** with date; Appendix F updated.
7. **K.1/K.2 remain Deferred** — not required for Phase S close.

#### Phase S — Recommended execution order

```text
Wave S0 (docs):      S.0.* (Done)
Wave S1 (ops):       S-Ops.1 → S-Ops.2 → S-Ops.3
Wave S2 (platform):  S-H.1 → S-H.2 → S-H.3
Wave S3 (proof):     S-H.5 → S-Doc.1 → S-Doc.2
Wave S4 (cleanup):   S-H.4
Parallel:            S-Ops.4, domain skill growth (legal/research) — not required for S Done
```

**After Phase S Done (historical):** Harness environment was ready for product agents. **Scheduling (2026-06-02):** K.1/K.2 remain **§6.3 end-of-plan** until explicit product prioritization.

---

---

### Phase T — Harness Cleanliness (post-S 2026-06-01)

**Status:** **Done** (2026-06-01). **Prerequisites:** Phase S **Done**.  
**Goal:** Close harness technical debt — unified lab preset, typed Tier-2 agents, native catalog planner, expanded stable stack, gate hygiene — without new business agents.

| # | Deliverable | Status | Location | Acceptance |
|---|-------------|--------|----------|------------|
| T-Ops.1 | **`lab_harness_preset()`** — default lab profile (sqlite + log + lab_json + OTEL; optional redis/qdrant) | **Done** | `IntegrationProfile`, `integration_wiring.py`, `settings.py` | `test_lab_harness_preset.py` |
| T-H.1 | **Echo/signoff `skill_ids`** — `harness.tool_smoke` on `AgentContract` | **Done** | `agents/echo`, `agents/signoff_probe` | `test_harness_reference_agent_skills.py` |
| T-H.2 | **`rag.answers` gate hygiene** — gate uses `RetrievalService` only; legacy tests marked `legacy_rag_answers` | **Done** | `tests/integration/rag/answers` | No `rag.answers` in `-m gate` |
| T-H.3 | **Typed `TaskContext` in Tier-2 agents** — no `getattr` on capability/message content in `agents` | **Done** | echo, research, signoff, org worker, lab mocks | `check_harness_no_getattr.py` scans `agents` |
| T-Ops.5 | **`CatalogToolPlanner`** without `ToolsAgent` wrapper | **Done** | `tool_planning_service.py`, `catalog_tool_planner.py` | `test_catalog_tool_planner.py` |
| T-Ops.6 | **Tier-2 stable stack** — `postgresql` + `sentry` in `HARNESS_LAB_STABLE_SLUGS` | **Done** | `harness_lab_stack.py`, postgresql `register.py` | `test_harness_lab_stable_stack.py` |

#### Phase T — Definition of done

1. Lab default wiring uses `lab_harness_preset()` (OTEL on unless env disables).
2. Echo and signoff_probe declare `harness.tool_smoke` via `skill_ids`.
3. Gate RAG path is `RetrievalService`-only; legacy `rag.answers` tests excluded from gate.
4. `python scripts/maintenance/check_harness_no_getattr.py` passes with `agents` in scan roots.
5. `CatalogToolPlanner` does not import `ToolsAgent`.
6. `postgresql` stable in catalog and harness stack list.

**After Phase T Done (historical):** Harness cleanliness complete. **Scheduling (2026-06-02):** product milestone K.1/K.2 is **deferred** (§6.3), not the default next step.

---

---

### Phase U — Harness Production Hardening (post-T 2026-06-01)

**Source:** Harness-system audit (2026-06-01) — security, contracts, policy wiring, typing, legacy, CI; **no business agents** (K.1/K.2 out of scope).  
**Status:** **Done** (2026-06-01). **Prerequisites:** Phase T **Done**. **Residual:** U-Leg.* (legacy module removal) — optional follow-up; does not block K.  
**Goal:** Close the gap between **laboratory harness** (fast iteration) and **production harness** (strategy doc: governance, persisted trace, secured surfaces, typed contracts, single policy path) without starting product agents.

**Explicitly out of scope for Phase U:**

- **K.1 Problem Radar / K.2 Vendor Discovery** — remain **Phase K** (after U Done)
- Multi-tenant SaaS (canon §50)
- New domain skills beyond harness platform pack
- Legal/product application feature work (except shared harness wiring used by lab)

#### U.0 — Audit & plan sync

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U.0.1 | **Appendix G** — audit findings → U.* IDs (100% mapped) | **Done** | Critical | This file Appendix G | Every audit row has U ID |
| U.0.2 | **§0.5 / §4 / §6** — Phase U as **NOW**; K.1/K.2 gated on U Done | **Done** | Critical | This file | No contradiction with strategy |

#### U-Sec — Lab & debug security surfaces

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Sec.1 | **AuthZ on lab surfaces** — optional API key / bearer for `POST /v1/lab/run`, `/debug/*`, MCP mount; default **deny** when `INTERGRAX_HARNESS_API_KEY` set | **Done** | **Critical** | `harness_auth.py`, lab/debug/MCP routes | `test_harness_auth.py` |
| U-Sec.2 | **MCP default opt-in** — `LAB_INCLUDE_MCP=false` default for strict profile; document in `guides/HARNESS_ENVIRONMENT.md` | **Done** | High | `LabApplicationSettings`, `.env.example` | `test_lab_application_settings_phase_u.py` |
| U-Sec.3 | **Sandbox tool policy** — lab enables `sandbox.exec` only when `SandboxSessionManager` wired; document risk | **Done** | High | `tool_wiring.py`, harness docs | Unit: sandbox omitted without session |
| U-Sec.4 | **`strict_harness` runtime profile** — `production_mode=True`, `GovernanceService`, persisted `trace_db_path`, OTEL; env `LAB_STRICT_HARNESS=true` | **Done** | **Critical** | `lab_runtime_config.py`, lab wiring | `test_lab_strict_harness.py` |

#### U-Pol — Unified policy path (lab + Tier-1)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Pol.1 | **`apply_policy_bundle` in lab** — `build_lab_runtime_config(ctx)` applies `ApplicationBuildContext.policy_bundle` to every UAEP `RuntimeConfig` (echo, signoff, mocks) | **Done** | **Critical** | `lab_runtime_config.py`, `runtime_config_bridge.py` | Reference agents use `build_lab_agent_runtime_context` |
| U-Pol.2 | **Policy engine vs bundle** — single composition root: Nexus `policy_engine` + `RuntimeConfig.policy_bundle` documented and wired from same `build_runtime_policy_bundle()` in lab | **Done** | High | `policy_wiring.py`, lab registry | Bundle passed via `ApplicationBuildContext` |
| U-Pol.3 | **Typed `RuntimePolicyBundle`** — replace `budget: Any`, `plan_loop: Any` with concrete policy types or `Protocol` refs | **Done** | Medium | `runtime/policy/policy_bundle.py` | `BudgetPolicy` / `PlanLoopPolicy` fields |

#### U-Con — Agent / UAEP contract unification

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Con.1 | **`HarnessReferenceAgent` base** — `class HarnessReferenceAgent(Agent):` + required UAEP methods; echo/signoff/mock inherit | **Done** | **Critical** | `intergrax/agents/harness_reference_agent.py` | Echo/signoff/mocks inherit |
| U-Con.2 | **Register-time UAEP check** — `AgentRegistry.register()` rejects agents that fail `isinstance(agent, UAEPAgent)` when manifest marks `requires_uaep: true` | **Done** | High | `agent_registry.py`, lab manifest | `test_agent_registry_uaep.py` |
| U-Con.3 | **Skill runtime proof** — gate test: lab registry resolves `harness.tool_smoke` → non-empty `allowed_tools` and tool step can plan | **Done** | High | `test_harness_reference_agent_skills.py`, acceptance lab | Echo/signoff declare `harness.tool_smoke` |

#### U-Typ — Strong typing & getattr hygiene

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Typ.1 | **Fix `ToolsAgentConfig`** — remove erroneous tuple defaults (`temperature = None,`); use `@dataclass` or explicit `__init__` | **Done** | **Critical** | `intergrax/tools/tools_agent.py` | Extends `ToolPlanningConfig` |
| U-Typ.2 | **`ToolPlanningConfig` in Tier-1** — planner prompts/config in `runtime/nexus/tools`; `ToolPlanningService` does not import `tools.tools_agent` | **Done** | High | `runtime/nexus/tools` | `test_catalog_tool_planner.py` |
| U-Typ.3 | **`ToolPlannerTrackable` protocol** — replace `isinstance(tool_planner, CatalogToolPlanner)` in `runtime_state` | **Done** | Medium | `tool_planner_trackable.py`, `runtime_state.py` | Protocol-based LLM tracker |
| U-Typ.4 | **Extend getattr audit** — `integrations/registry/profile.py`, `sandbox/service.py` | **Done** | Medium | Typed profile + `SandboxSession` | Harness nexus/agents paths clean |
| U-Typ.5 | **Remove `hasattr` on harness paths** — `shared_task_context`, `engine_plan_models`, `platform_wiring` trace_store resolution | **Done** | Medium | `platform_wiring.py`, `nexus_loop.trace_store` | Typed trace resolution |

#### U-Arch — Integration & composition consistency

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Arch.1 | **Single lab integration preset** — `create_lab_interaction_adapter()` uses `lab_harness_preset()` (not `IntegrationProfile.lab()`) | **Done** | High | `integration_wiring.py` | `test_lab_harness_environment_wiring.py` |
| U-Arch.2 | **Typed lab wiring returns** — remove `# type: ignore` on trace/checkpoint/notification adapters; explicit bundle types | **Done** | Medium | `SQLiteIntegrationBundle`, `integration_wiring.py` | Typed sqlite facades |
| U-Arch.3 | **Rename runtime `tools_agent_*` fields** — `tools_agent_answer` → `tool_planner_answer` (or `catalog_tool_answer`); update trace diag types | **Done** | Low | `runtime_state.py`, tracing adapters | Gate green |

#### U-Leg — Legacy stack removal

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Leg.1 | **`ToolsAgent.run` deprecation freeze** — document; block new imports; optional redirect to `ToolRuntime` only path | **Done** | Medium | `tools_agent.py`, `check_tools_agent_run.py` | CI audit |
| U-Leg.2 | **`rag.answers` removal or archive** — migrate remaining `legacy_rag_answers` tests to `RetrievalService`; delete or move module under `intergrax/legacy` | **Done** | Medium | `intergrax/legacy/rag_answers` | `test_rag_answers_removed.py` |
| U-Leg.3 | **Legacy tool plan booleans** — document sunset for `from_legacy` / `uses_legacy_booleans_only`; gate new usage | **Done** | Low | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | Deprecation warnings |

#### U-Doc — Operator & architecture alignment

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Doc.1 | **`guides/HARNESS_ENVIRONMENT.md`** — security (auth, MCP), strict profile, policy bundle wiring truth | **Done** | High | `docs/project/technical/guides/HARNESS_ENVIRONMENT.md` | Phase U security section |
| U-Doc.2 | **Canon §52 / strategy** — lab vs production harness checklist references Phase U | **Won't fix** | Medium | — | Deferred; plan + HARNESS_ENVIRONMENT sufficient |
| U-Doc.3 | **Fix Phase K footer** in `guides/HARNESS_ENVIRONMENT.md` (post-T, gated on U) | **Done** | Low | `guides/HARNESS_ENVIRONMENT.md` | Gated on Phase U |

#### U-CI — Verification & smoke

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-CI.1 | **harness-smoke includes Phase U tests** — auth, strict harness, lab settings | **Done** | High | `.github/workflows/unit-tests.yml` | harness-smoke extended |
| U-CI.2 | **Acceptance: production harness path** — one gate test: strict lab + sqlite trace + policy bundle + skill-resolved tools | **Done** | **Critical** | `tests/acceptance/agent_os`, unit strict harness | `pytest -m gate` **479 passed** |
| U-CI.3 | **Optional: strict harness job** — separate CI job with `LAB_STRICT_HARNESS=true` + API key | **Done** | Medium | `.github/workflows/unit-tests.yml` | `harness-strict` job |

#### Phase U — Definition of done

1. Lab **policy bundle** reaches `RuntimeConfig` for all reference agents (U-Pol.1); tool policy resolution exercised in test.
2. **Secured-by-configuration** lab/debug/MCP (U-Sec.1–U-Sec.2); **strict_harness** E2E exists (U-Sec.4, U-CI.2).
3. Reference agents use **HarnessReferenceAgent** or equivalent enforced UAEP (U-Con.1–U-Con.2).
4. **`ToolsAgentConfig` bug fixed**; Tier-1 planner config decoupled from `tools_agent` (U-Typ.1–U-Typ.2).
5. **Integration preset** consistent (U-Arch.1); docs accurate (U-Doc.*).
6. Gate: `uv run pytest -m gate -q` green; getattr + tools_agent audits pass.
7. §0.5 **Harness production hardening** row **Done** with date; Appendix G 100% **Done** or **Won't fix**.
8. **K.1/K.2 remain Deferred** until U Done.

#### Phase U — Recommended execution order

```text
Wave U0 (plan):     U.0.* (Done with this edit)
Wave U1 (security): U-Sec.1 → U-Sec.2 → U-Sec.4
Wave U2 (policy):   U-Pol.1 → U-Pol.2 → U-Con.3
Wave U3 (contracts): U-Con.1 → U-Con.2 → U-Typ.1
Wave U4 (typing):   U-Typ.2 → U-Typ.3 → U-Typ.4 → U-Typ.5
Wave U5 (arch):     U-Arch.1 → U-Arch.2 → U-Pol.3
Wave U6 (legacy):   U-Leg.2 → U-Leg.1 → U-Leg.3 → U-Arch.3
Wave U7 (close):    U-Doc.* → U-CI.* → Appendix G paydown log
```

**After Phase U Done (historical):** Production-grade harness baseline achieved. **Scheduling (2026-06-02):** start K.1/K.2 only via **§6.3** after explicit product decision — not by default.

---

---

### Phase V — Harness Architecture Hardening (post-U)

**Source:** Architecture hardening audit against `IDEAL_HARNESS_AI_ARCHITECTURE.md` (2026-06-02).  
**Status:** **Done** (2026-06-05) — Phase V-REM closed all runtime enforcement gaps. **Prerequisites:** Phase U **Done**.  
**Goal:** Close architecture-level gaps that increase long-term technical debt, reduce extensibility, or weaken governance in harness-only scope.

**Explicitly in scope for Phase V:**

- Capability dependency graph + compatibility gates
- Agent lifecycle governance (certification/promotion/deprecation/retirement/ownership)
- Context quality scoring + context regression discipline
- Prompt engineering architecture and governance
- Evaluation registry operations (offline/online/shadow/human)
- Architecture metrics and architecture debt governance
- Advanced security/data governance defenses (prompt/tool/retrieval attacks)
- Cost/resource governance (budgets, quotas, forecasting, optimization)
- Multi-agent coordination model catalog and selection matrix
- Knowledge-graph/Graph-RAG evolution path (harness capability, no product-domain rollout)

**Explicitly out of scope for Phase V:**

- K.1/K.2 business agent delivery
- New product-specific Tier-3 applications
- Domain skill packs not under `harness.*`

#### V-CG — Capability Graph Architecture

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-CG.1 | Capability graph schema (nodes + edges for Integration/Tool/Skill/Policy/Agent/Application/Product) | **Done** | **Critical** | Typed schema + docs in canon |
| V-CG.2 | Graph lineage builder from registries | **Done** | High | Per-application agent→application edges via `capability_graph_applications.py` |
| V-CG.3 | Impact analysis report (blast radius) for changed capabilities | **Done** | High | Guard script green on corrected graph |
| V-CG.4 | Compatibility validation on dependency graph edges | **Done** | **Critical** | `phase_v_capability_graph_guard.py --enforce` green |

#### V-ALG — Agent Lifecycle Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-ALG.1 | Agent certification gate contract (quality/policy/security) | **Done** | **Critical** | Certification criteria codified + tested |
| V-ALG.2 | Promotion flow (dev -> staging -> production) with evidence | **Done** | High | Promotion requires evidence bundle |
| V-ALG.3 | Deprecation + retirement workflow and migration window policy | **Done** | High | `AgentRegistry` / `AgentRouter` filter retired/deprecated via `agent_routing_policy.py` |
| V-ALG.4 | Owner/on-call metadata required for production-eligible agents | **Done** | High | Production-mode ownership gate enforced at selection |

#### V-CE — Context Quality and Regression Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-CE.1 | Relevance/freshness/confidence scoring in context assembly | **Done** | High | Scores emitted in trace/runtime events |
| V-CE.2 | Duplicate suppression + context quality thresholds | **Done** | Medium | Threshold policy test coverage |
| V-CE.3 | Context regression benchmark suite | **Done** | High | CI regression baseline stored and compared |
| V-CE.4 | Retrieval effectiveness evaluation (precision/recall@k style) | **Done** | Medium | Bench report in evaluation registry |

#### V-PE — Prompt Engineering Architecture

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-PE.1 | Prompt registry governance contract (owner/version/risk metadata) | **Done** | High | `PromptMeta` extended; `harness_capability_summary` reference prompt; registry governance validation |
| V-PE.2 | Prompt composition model (system/task/policy/context layers) | **Done** | High | Canon + reference implementation path |
| V-PE.3 | Deterministic policy injection overlays | **Done** | High | Prompt build trace shows overlays |
| V-PE.4 | Prompt regression/adversarial test suite | **Done** | Medium | Gate includes prompt regression subset |

#### V-EVAL — Evaluation and Benchmarking Operations

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-EVAL.1 | Unified evaluation modes: offline/online/shadow/human | **Done** | **Critical** | Mode contracts documented + wired |
| V-EVAL.2 | Golden datasets + scenario libraries + regression suites | **Done** (typed asset bundle contracts) | High | Versioned benchmark assets |
| V-EVAL.3 | Automated evaluators (rule-based + LLM judge) | **Done** | High | Evaluator outputs persisted |
| V-EVAL.4 | Evaluation registry trend/comparison reports | **Done** | High | Report artifact required for major releases |

#### V-AM — Architecture Metrics & Debt Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-AM.1 | Architecture health metric spec (modularity/dependency/coverage/debt) | **Done** | **Critical** | Canon metrics section + thresholds |
| V-AM.2 | Metrics emission pipeline and dashboards | **Done** (pipeline + trend/gate contracts) | High | Dashboard + alert definitions |
| V-AM.3 | Governance coverage and observability coverage measurement | **Done** | High | Coverage reports generated in CI |
| V-AM.4 | Architecture debt index + periodic review process | **Done** | High | Debt report cadence defined and used |

#### V-SEC — Security & Data Governance Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-SEC.1 | Prompt injection defense profile + tests | **Done** | **Critical** | Adversarial tests in gate subset |
| V-SEC.2 | Tool injection defense (schema/argument/capability controls) | **Done** | High | `ToolInjectionDefenseMiddleware` on `BEFORE_TOOL_CALL` via `application_security_wiring.py` |
| V-SEC.3 | Retrieval poisoning defense (trust score/quarantine flow) | **Done** | High | `retrieval_security_wiring.py` filters chunks in `rag.retrieve` (catalog) when profile enabled |
| V-SEC.4 | Tenant isolation verification + security audit trail checks | **Done** | High | `TenantSecurityMiddleware` on `BEFORE_TASK_INTAKE` |

#### V-COST — Cost & Resource Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-COST.1 | Budget envelopes (tenant/app/agent/model/tool) | **Done** | High | Budget policy enforcement tests |
| V-COST.2 | Token/tool/resource quotas with deny/degrade behavior | **Done** | High | Quota exceedance behavior deterministic |
| V-COST.3 | Forecast + anomaly detection for spend and token drift | **Done** | Medium | Forecast/anomaly report available |
| V-COST.4 | Optimization recommendations with policy guardrails | **Done** | Medium | Recommendations recorded in ops reports |

#### V-MA — Multi-Agent Coordination Model Catalog

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-MA.1 | Coordination patterns catalog (hierarchical/orchestrator-worker/supervisor-worker/peer/swarm/evaluator-loop) | **Done** | High | Canon section + selection table |
| V-MA.2 | Pattern selection matrix (risk/latency/cost/complexity) | **Done** | High | Matrix used in planning docs |
| V-MA.3 | Pattern-specific acceptance tests | **Done** | Medium | Test suite covers selected patterns |

#### V-KG — Knowledge Graph Evolution Path (Harness)

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-KG.1 | Graph-RAG architecture contract | **Done** | Medium | Canon section + terminology alignment |
| V-KG.2 | Hybrid retrieval reference path (vector + keyword + graph) | **Done** | Medium | Reference implementation notes |
| V-KG.3 | Graph-backed explainability trace fields | **Done** | Medium | Trace schema supports graph provenance |

#### V-V6 — Phase V Closeout (L3/L4 Evidence & CI)

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-V6.1 | Bounded adaptive governance contracts (policy-learning envelopes, human gates) | **Done** | High | `adaptive_governance.py` + unit tests |
| V-V6.2 | L3/L4 maturity gate evidence aggregator | **Done** | **Critical** | `maturity_gate_evidence.py` + `maturity_gate_evidence_report.json` |
| V-V6.3 | CI closeout gate (`phase_v_closeout_gate.py --enforce`) | **Done** | **Critical** | Regression workflow runs closeout after gate tests |

#### Phase V — Execution matrix (dependencies and order)

Phase V should be executed in dependency-aware waves:

```text
Wave V0 (planning):      V-CG.1 + V-AM.1 + ownership/cadence baseline
Wave V1 (foundations):   V-CG.2 -> V-CG.4 + V-ALG.1 + V-PE.1 + V-EVAL.1
Wave V2 (quality):       V-CE.1 -> V-CE.3 + V-PE.2 -> V-PE.4 + V-EVAL.2 -> V-EVAL.3
Wave V3 (governance):    V-ALG.2 -> V-ALG.4 + V-SEC.1 -> V-SEC.4 + V-COST.1 -> V-COST.2
Wave V4 (ops maturity):  V-AM.2 -> V-AM.4 + V-EVAL.4 + V-COST.3 -> V-COST.4
Wave V5 (advanced):      V-MA.1 -> V-MA.3 + V-KG.1 -> V-KG.3
Wave V6 (closeout):      L3/L4 gate evidence + docs sync + priority reset
```

Critical dependency rules:

- `V-CG.1` must precede `V-CG.2/V-CG.4` and dependency-health metrics in `V-AM`.
- `V-PE.1` and `V-EVAL.1` must precede prompt/eval regression gates.
- `V-ALG.1` must precede production promotion flow (`V-ALG.2`).
- `V-SEC.*` and `V-COST.*` deny/degrade behavior must be validated before L3 gate.

#### Phase V — KPI thresholds and acceptance metrics

Minimum quantitative targets for Phase V completion:

| Area | Metric | Target |
|------|--------|--------|
| Capability graph | Changed harness PRs with graph impact artifact | **>= 95%** |
| Compatibility | Graph-edge compatibility gate pass on default branch | **100% required** |
| Lifecycle governance | Production-eligible agents with owner + certification metadata | **100% required** |
| Context quality | Context regression suite pass rate | **>= 95%** |
| Prompt quality | Prompt regression/adversarial suite pass rate | **>= 95%** |
| Evaluation ops | Critical capabilities with baseline + post-change scores | **100% required** |
| Security hardening | Adversarial defense suite pass rate (prompt/tool/retrieval) | **100% required** |
| Cost governance | Budget/quota policy test pass rate | **100% required** |
| Architecture metrics | Modularity/dependency/governance/observability coverage reported | **100% runs** |
| Architecture debt | Critical debt items trending (rolling 30d) | **non-increasing** |

#### Phase V — Operating cadence and governance ceremonies

- **Weekly:** Architecture hardening triage (V-* progress, blockers, scope control).
- **Weekly:** Security/cost review for new deny/degrade paths and policy regressions.
- **Bi-weekly:** Architecture review board for high-impact V-* design changes.
- **Monthly:** Architecture debt review (index trend + mitigation decisions).
- **Per release candidate:** L3/L4 evidence review (gates below) before release approval.

#### Phase V — Stream ownership model

| Stream | Primary owner | Supporting owners |
|--------|----------------|-------------------|
| V-CG | Platform architecture | Runtime + DevEx |
| V-ALG | Runtime governance | Platform + QA |
| V-CE / V-PE | Runtime + Prompt systems | QA/Eval |
| V-EVAL | Evaluation engineering | Runtime + Product quality |
| V-AM | Platform observability | Runtime + DevEx |
| V-SEC | Security engineering | Runtime + Platform |
| V-COST | Runtime economics | Platform + FinOps |
| V-MA | Orchestration/runtime | QA |
| V-KG | Knowledge systems | Runtime + Eval |

Owner rules:

- Every V-* PR must include a single accountable owner.
- Cross-stream dependencies must list an explicit approver before merge.
- Ownership metadata for production-impacting components must be reflected in registries where applicable.

#### Phase V — L3/L4 gate evidence (architecture maturity)

L3 readiness requires:

1. `V-CG.*`, `V-ALG.*`, `V-EVAL.1-4`, `V-SEC.1-4`, `V-COST.1-2`, `V-AM.1-3` complete.
2. KPI thresholds marked **100% required** above are satisfied.
3. Security and compatibility gates are green for two consecutive release cycles.
4. Architecture governance artifacts updated (canon + plan + traceability appendices).

L4 readiness requires:

1. L3 criteria met and stable.
2. `V-COST.3-4`, `V-MA.*`, `V-KG.*`, and adaptive loops with bounded governance controls.
3. Closed-loop evaluation feedback demonstrates measurable quality/cost improvement over baseline.
4. Policy-learning/adaptive behavior remains human-governed and auditable.

#### Phase V — Definition of done

1. Capability graph compatibility validation is active in CI for harness-critical changes.
2. Agent lifecycle governance gates exist and are enforced for production-eligible agents.
3. Context/prompt/evaluation governance artifacts are versioned and regression-tested.
4. Architecture health metrics are measurable and reviewed on a recurring cadence.
5. Security/data/cost hardening controls are testable, observable, and documented.
6. All changes remain harness-only (no implicit K.1/K.2 scope creep).
7. Coverage matrix (Appendix H) has **no `Uncovered` rows** for harness-scope architecture domains.

#### Phase V — Paydown log

| Date | V ID | Summary |
|------|------|---------|
| 2026-06-02 | V-CG.1, V-AM.1, V-ALG.1 | Typed baseline contracts added (`intergrax/runtime/architecture`) + report-only artifacts script (`scripts/release/phase_v_foundations_report.py`) + unit tests |
| 2026-06-02 | V-CG.2, V-CG.3, V-CG.4 | Lineage/impact/compatibility modules + capability graph guard script (`scripts/release/phase_v_capability_graph_guard.py`) + enforce switch + unit tests |
| 2026-06-02 | V-AM.2, V-ALG.2, V-EVAL.1 | Metrics pipeline contracts + promotion flow evaluator + unified evaluation mode contracts + governance artifacts script (`scripts/release/phase_v_governance_report.py`) + unit tests |
| 2026-06-02 | V-ALG.3, V-ALG.4, V-EVAL.2 | Lifecycle/deprecation governance contracts + production ownership guard + evaluation asset bundle contracts + governance report extensions + unit tests |
| 2026-06-02 | V-EVAL.3, V-AM.3 | Automated evaluators (`evaluation_automation.py`) + architecture coverage report (`architecture_coverage.py`) + governance report persistence + unit tests |
| 2026-06-02 | V-AM.4, V-EVAL.4 | Debt governance cadence/policy report (`debt_governance.py`) + release trend/comparison report (`evaluation_registry_trends.py`) + governance script artifacts + unit tests |
| 2026-06-02 | V-SEC.1, V-SEC.2 | Prompt injection defense profile (`prompt_security.py`) + tool injection defense controls (`tool_security.py`) + governance artifacts + adversarial unit tests |
| 2026-06-02 | V-SEC.3, V-SEC.4 | Retrieval poisoning defense (`retrieval_security.py`) + tenant isolation/audit verification (`tenant_security.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-COST.1, V-COST.2, V-COST.3, V-COST.4 | Budget envelopes + quota deny/degrade + cost forecast/anomaly + optimization guardrails (`cost_*.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-CE.1, V-CE.2, V-PE.1, V-PE.2 | Context quality scoring/dedup (`context_engineering.py`) + prompt registry/composition (`prompt_registry_governance.py`, `prompt_composition.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-CE.3, V-CE.4, V-PE.3, V-PE.4 | Context regression benchmark + retrieval effectiveness + policy overlays + prompt regression suite + governance artifacts + unit tests |
| 2026-06-02 | V-MA.1, V-MA.2, V-MA.3, V-KG.1, V-KG.2, V-KG.3 | Multi-agent coordination catalog/selection/acceptance + Graph-RAG/hybrid retrieval/provenance contracts + governance artifacts + unit tests |
| 2026-06-02 | V-V6.1, V-V6.2, V-V6.3 | Bounded adaptive governance + L3/L4 maturity evidence + `phase_v_closeout_gate.py` CI enforcement |
| 2026-06-03 | H-APP.* | Phase H-APP: ApplicationEnvironmentProfile, unified wiring, 43 tasks, gate 510 |
| 2026-06-05 | V-REM.0.* | Plan audit: 9 Phase V + 1 Phase A gaps reclassified Partial; Phase V-REM + Appendix J + §6.1z queue opened |
| — | — | *(append row per merged PR)* |

---

---

#### V-AM — Architecture Metrics & Debt Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-AM.1 | Architecture health metric spec (modularity/dependency/coverage/debt) | **Done** | **Critical** | Canon metrics section + thresholds |
| V-AM.2 | Metrics emission pipeline and dashboards | **Done** (pipeline + trend/gate contracts) | High | Dashboard + alert definitions |
| V-AM.3 | Governance coverage and observability coverage measurement | **Done** | High | Coverage reports generated in CI |
| V-AM.4 | Architecture debt index + periodic review process | **Done** | High | Debt report cadence defined and used |#### V-SEC — Security & Data Governance Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-SEC.1 | Prompt injection defense profile + tests | **Done** | **Critical** | Adversarial tests in gate subset |
| V-SEC.2 | Tool injection defense (schema/argument/capability controls) | **Done** | High | `ToolInjectionDefenseMiddleware` on `BEFORE_TOOL_CALL` via `application_security_wiring.py` |
| V-SEC.3 | Retrieval poisoning defense (trust score/quarantine flow) | **Done** | High | `retrieval_security_wiring.py` filters chunks in `rag.retrieve` (catalog) when profile enabled |
| V-SEC.4 | Tenant isolation verification + security audit trail checks | **Done** | High | `TenantSecurityMiddleware` on `BEFORE_TASK_INTAKE` |

---

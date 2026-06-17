# Agent Contracts And Assembly — Implementation Plan

**Architecture (1:1):** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Agent architecture completion — executive summary (2026-06-11)

**Phases ACP + ACP-CLOSE + ACP-FINISH + AUDIT-IDEAL (§12–§20):** **Done** (2026-06-13) — platform runtime, fleet migration, production gates, token budget depth, CI matrix, registry snapshot, cap-graph blast-radius, lifecycle on-call.  
**Parallel track:** [Phase AUDIT-IDEAL](#phase-audit-ideal--ideal-architecture-gap-register-2026-06-09) — **10/10 Done** (incl. 19.1 · 20.1 · 31.1).

| Track | Scope | Status | Remaining IDs |
|-------|-------|--------|---------------|
| **ACP runtime depth** | §25.4–§25.5 token usage, limits, reactions | **Done** | — |
| **Architecture doc truth** | §28.3 GAP register · §36.4 · §40.13 tables | **Done** | **ACP-FINISH-DOC-1** **Done** (2026-06-13) |
| **AUDIT-IDEAL (§12–§20)** | Registry snapshot · cap-graph CI · lifecycle owner | **Done** | — |
| **Gate maintenance** | §6.1 continuous | **Active** | `pytest -m gate` on every PR |

**Architecture-complete DoD (ACP-FINISH):** GAP-ACP-36/37 **Closed** · §28.3 **37 Closed · 0 Open** · ACP-TOK-* green · one implementation journal entry · domain audit prompt regenerated.

**Ordered queue:** [§6.1bc](#61bc-harness-implementation-queue--acp-finish-closed) — **Done** (2026-06-13).

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §17–§19, §31 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-13) — 10/10 **Done** · master register [`AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) synced

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-17.1 | §17 Prompts | Prompt approval workflow (beyond registry metadata) | P2 | **Done** |
| AUDIT-IDEAL-17.2 | §17 Prompts | Prompt diff / compare API for all managed prompts | P2 | **Done** |
| AUDIT-IDEAL-18.1 | §18 Assembly | `ModalityProfile` mandatory on certified agents | P1 | **Done** |
| AUDIT-IDEAL-18.2 | §18 Assembly | Cross-host agent reuse certification test suite | P2 | **Done** |
| AUDIT-IDEAL-19.1 | §19 Registry | Durable cross-host registry snapshot store (DEBT-19-01) | **P0** | **Done** |
| AUDIT-IDEAL-19.2 | §19 Registry | Capability negotiation at runtime resolve | P2 | **Done** |
| AUDIT-IDEAL-20.1 | §20 Cap. graph | Product CI blast-radius check on tool/skill changes | P1 | **Done** |
| AUDIT-IDEAL-20.2 | §20 Cap. graph | Policy change impact visualization CLI | P2 | **Done** |
| AUDIT-IDEAL-31.1 | §31 Lifecycle | Owner/on-call mandatory on all certified agents | P1 | **Done** |
| AUDIT-IDEAL-31.2 | §31 Lifecycle | Evaluation required before production promotion (enforce) | P1 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

## Phase ACP — Agent Cognitive Patterns (ACP)

**Status:** **Done** (2026-06-11) — Waves **0–8** delivered; master register **80/80** ACP-* rows **Done**; fleet migration **100%** Runtime dimension  
**Architecture:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40 (incl. **§32.0** readability & typed-only contracts)  
**ADR:** [ADR-AGENT-001](../adr/entries/2026-06-11/ADR-AGENT-001.md) · [ADR-AGENT-002](../adr/entries/2026-06-11/ADR-AGENT-002.md) · [ADR-AGENT-003](../adr/entries/2026-06-11/ADR-AGENT-003.md)  
**Author guide:** [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix AC (sync with §32.0)  
**Audit:** [`guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md) · domain audit **2026-06-11**  
**Priority ladder:** **Band 2aw** — **closed** · **Band 2bb (ACP-CLOSE)** — **closed** · **Band 2bc (ACP-FINISH)** — **closed** (2026-06-13)

**Strategic outcome (delivered):** Tier-2 authors use **`agent.run(AgentRunRequest)`** + typed **`on_next_step` → `StepOutcome`**; environment merges per-agent memory/tools/RAG/LLM from Tier-3 profile; Nexus remains `Task` entry for multi-agent prod.

**Remaining (ACP-FINISH):** none — §25.4–§25.5 **Done** via **ACP-TOK-*** (2026-06-11 implementation · 2026-06-13 doc sync).

**Explicit production gate:** mutating agents — **ACP-CLOSE-PROD-*** **Done**; token budget depth **Done** (**ACP-TOK-2** · **ACP-TOK-3**) for hosts using `AgentBinding.budget_slice`.

**Doc canon status (2026-06-13):** architecture §13–§40 **accepted and implementation-complete**; GAP-ACP-36/37 **Closed** via **ACP-TOK-*** + **ACP-FINISH-DOC-1**.

**Explicitly excluded:** Nexus refactor; moving `GraphExecutor`/`PolicyEngine` into agents; Phase K business agents; new Tier-0 execution engine.

**Full-domain scope:** Phase **ACP** (§13–§40) **implemented** at platform level; **§12–§20** normative — maintain via gate + **AUDIT-IDEAL** residuals. See [§12–§20 scope mapping](#acp-scope-mapping-12-20-vs-acp-waves).

---

### ACP scope mapping §12–§20 vs ACP waves

**Purpose:** Prevent treating registry, contract, prompt, and lifecycle canon as “closed trivia” while only shipping ACP runtime. Every new agent MUST satisfy **§12 contract** *and* **§13+ runtime** after Wave 0–2.

| Arch § | Topic | Baseline in code | Plan status | Verified by |
|--------|-------|------------------|-------------|-------------|
| **§12** | Agent contract (capabilities, schemas, tools, risk, validation, failure modes) | **Done** — ACP-CON-4 register gate | ACP-CON-4 · ACP-0b · ACP-DX-5 **Done** | `test_agent_assembly_resolver.py` + register rejection |
| **§14** | Agent execution result | **Done** — typed `AgentRunResult` + trace | ACP-DX-1 · ACP-OBS-1 **Done** | Typed `AgentRunResult` + trace |
| **§15** | Agent registry | **Done** (REG-*) | ACP-CON-6 · AUDIT-IDEAL-19.1 **Done** | `check_harness_registry_resolution.py` · `registry_snapshot_store.py` |
| **§16** | Capability model | **Done** (CG-*) | ACP-CON-6 **Done** | Capability routing integration test |
| **§17** | Prompt registry | **Done** (PE-*) | PE-* **Done** | PE wiring tests |
| **§18** | Registry architecture | **Done** (REG-*) | REG-* **Done** | `test_registry_wiring.py` |
| **§19** | Capability graph | **Done** (CG-*) | CG-* · AUDIT-IDEAL-20.1 **Done** | `phase_v_capability_graph_guard.py` |
| **§20** | Lifecycle governance | **Done** (V-ALG, AS-2) | ACP-PROD-9 · AUDIT-IDEAL-31.1 **Done** | `check_agents_lifecycle_metadata.py` · `check_on_call_ownership_model.py` |
| **§13–§40** | ACP runtime (run, step loop, env, prod) | **Done** — incl. §25.4–§25.5 | ACP · ACP-CLOSE · **ACP-FINISH Done** | ACP-TOK-* · `test_acp_token_*` |

**Rule:** An agent PR MUST pass **§12 assembly validation** (**ACP-CON-4**) and applicable **ACP-CLOSE** rows when touching mutating prod paths or legacy surfaces.

---

### ACP implementation principles (normative for every PR)

| # | Principle | Architecture | Enforcement |
|---|-----------|--------------|-------------|
| P1 | **Readability first** | §32.0 | Every `on_next_step` ends with one `StepOutcome.*` factory; reviewer understands continue/complete/fail without running app |
| P2 | **Typed-only author surface** | §32.0 · §37.1 | Pydantic `extra=forbid`; no `dict` state keys in `agents/`; CI `check_agent_typed_state.py` |
| P3 | **READ → UPDATE → DECIDE** | §32.0.1 | `load_session_state` → `state_delta` → `StepOutcome` — no in-place mutation |
| P4 | **One engine, two entries** | §29 · §38 | Direct `run()` and `Task→Nexus` share `AgentRunRequest` merge + step loop — no divergent code paths |
| P5 | **Harness executes, agent decides** | §38 | Domain planning in `on_next_step`; `HarnessKernel` owns policy/trace/budget/state — no planning in kernel |
| P5b | **Runtime is glue only** | §38 · §32.4 | `AgentRuntime.advance_step` calls `on_next_step` then `HarnessKernel.execute_step` — **no** policy logic, trace append, or state merge in runtime |
| P6 | **Legacy removal is deliverable** | §13.5 · ACP-CLOSE-LEG | DEBT register → zero; no new author-visible UAEP/AgentEngine after ACP-CLOSE |
| P9 | **§12 contract is not optional** | §12 · §45 | Register-time gate: schemas, risk, validation_rules, failure_modes — **ACP-CON-4** |
| P10 | **Fleet migration is a program** | Wave 8 · §40.15 | Tiered batches `ACP-MIG-*`; not one-off ACP-LEG-2 PR |
| P11 | **Prod decision = scoreboard** | §40.15 · ACP-PROD-12 | Single report; thresholds binding for roster promotion |
| P7 | **Cross-layer contracts** | §30 · matrix below | Agent PRs that touch tools/memory/RAG/policy MUST cite paired domain plan row |
| P8 | **Architecture = acceptance spec** | §45 · waves | Wave DoD = architecture behaviors demonstrable in tests — not “types exist” alone |

---

### ACP legacy & technical debt register (must shrink to zero)

**Audit 2026-06-11:** **18/18 Closed** in code — DEBT register fully closed; **ACP-CLOSE complete** (CI-2 Done).

| Debt ID | Legacy surface | Replacement | Status | Closed by / Open row |
|---------|----------------|-------------|--------|----------------------|
| DEBT-ACP-01 | `Agent.run()` without `AgentRunRequest` | §29 typed `run()` | **Closed** | ACP-DX-3 |
| DEBT-ACP-02 | `RuntimeRequest` opaque metadata I/O | `AgentRunRequest` §30.9 | **Closed** | ACP-DX-1 |
| DEBT-ACP-03 | Raw `dict` agent state | `AcpSessionState` §32.0 | **Closed** | ACP-0 · ACP-DX-6 |
| DEBT-ACP-04 | `decide_after_step` + `AgentDecision` author API | `StepOutcome` factories §32.0.4 | **Closed** | ACP-CLOSE-LEG-2 · CognitiveAgent UAEP shim only |
| DEBT-ACP-05 | `get_steps` / `run_step` as primary author API | `on_next_step` §32.5 | **Closed** | ACP-STEP-3 · ACP-8 |
| DEBT-ACP-06 | `AgentEngine.run` fallback in `AgentEngine` | `advance_step` + kernel only | **Closed** | ACP-CLOSE-LEG-1 |
| DEBT-ACP-07 | `build_context` duplicating `RuntimeConfig` | `merge_environment` §30 | **Closed** | ACP-DX-2 · ACP-CFG |
| DEBT-ACP-08 | No `AgentRunTrace` on result | Plane B §31 | **Closed** | ACP-OBS-1 |
| DEBT-ACP-09 | No `ApplicationRunSummary` | Plane A §31 | **Closed** | ACP-OBS-2 |
| DEBT-ACP-10 | Single LLM model per run | `StepLLMRouter` §33 | **Closed** | ACP-LLM-1 |
| DEBT-ACP-11 | Ad-hoc graph handoff via metadata | `SharedContextView` §34 | **Closed** | ACP-STATE-1 |
| DEBT-ACP-12 | Capability routing by class name | Registry token §37.6 | **Closed** | ACP-CON-6 |
| DEBT-ACP-13 | Free-text errors / terminal reasons | Enums §37.4–§37.5 | **Closed** | ACP-CON-1 |
| DEBT-ACP-14 | Full state replace / in-place mutation | `state_delta` §37.2 | **Closed** | ACP-CON-2 |
| DEBT-ACP-15 | Scaffold UAEP-first only | Typed scaffold `--pattern` | **Closed** | ACP-8 |
| DEBT-ACP-16 | Roster on legacy patterns | Typed loop fleet-wide | **Closed** | ACP-MIG-* · ACP-LEG-2 |
| DEBT-ACP-17 | No prod checkpoint / idempotency (platform) | §40 persistence modules | **Closed** — platform + host depth + compensation queue + cross-run idempotency | ACP-PROD-1..3 · **ACP-CLOSE-PROD-1..8 Done** |
| DEBT-ACP-18 | ReAct loop split from TOOL-ENG-6 | Unified budget §25.2 | **Closed** | **ACP-CLOSE-PAT-1** + **TOOL-ENG-6** |

**Removal policy:** ACP + ACP-CLOSE + **ACP-FINISH closed** (2026-06-13). Extend via ADR if scope expands.

---

### ACP cross-domain coupling matrix

Agent layer is **not isolated**. Each ACP wave may require coordinated delivery or read-only integration with:

| Intergrax domain | Architecture | Plan | ACP touchpoints | Sync rule |
|------------------|--------------|------|-----------------|-----------|
| **Unified execution / policy** | `UNIFIED_EXECUTION_RUNTIME` §42 | `plan/UNIFIED_EXECUTION_RUNTIME.md` | Policy pre/post inside **`HarnessKernel.execute_step`**; `PolicyVerdictRecord` | ACP-STEP-2b · ACP-ORG-3 before STRICT prod |
| **Orchestration / Nexus** | `ORCHESTRATION` · `NEXUS_EXECUTION_FLOW` | `plan/ORCHESTRATION.md` · `plan/NEXUS_EXECUTION_FLOW.md` | Graph node → `AgentRunRequest`; HITL pause | ACP-DX-4 must not fork Task lifecycle |
| **Tools** | `TOOLS` §gateway | `plan/TOOLS.md` **TOOL-ENG-6** | `tool_gateway`; declarative `StepActionRequest`; idempotency | ACP-3 + TOOL-ENG-6 same sprint; ACP-PROD-2..3 |
| **Skills** | `SKILLS` | `plan/SKILLS.md` | `skill_ids` → allowed_tools on contract | AS-3 CI already; verify on binding slices |
| **LLM adapters** | `LLM_ADAPTERS` | `plan/LLM_ADAPTERS.md` | `StepLLMRouter`; per-step `LlmStepResult` on trace | ACP-LLM-1 + LLM profile merge |
| **Memory** | `MEMORY` | `plan/MEMORY.md` | `memory_view`; `memory_scope` user vs org §30.9 | ACP-DX-2 identity namespace tests |
| **RAG** | `RAG` | `plan/RAG.md` | `rag_gateway`; collection binding | ACP-DX-5 host slice tests |
| **Observability** | `OBSERVABILITY` | `plan/OBSERVABILITY.md` | `AgentRunTrace`; redaction §40.8 | ACP-OBS-1 + OBS trace spine |
| **Reliability / HITL** | `RELIABILITY_FAILURE_AND_HITL` | `plan/RELIABILITY_FAILURE_AND_HITL.md` | `pause_hitl`; checkpoint resume | ACP-PROD-1 + acceptance 04/05 |
| **Tier-3 applications** | `TIER3_APPLICATION_ENVIRONMENT` | `plan/TIER3_APPLICATION_ENVIRONMENT.md` | `AgentBinding`; profile merge; intake `RequestIdentity` | ACP-DX-5 · ACP-ORG-1 |
| **Critic / verification** | `CRITIC_VERIFICATION` | `plan/CRITIC_VERIFICATION.md` | `ReflectionAgent` critic hooks | ACP-6 — no critic SDK in Tier-2 |
| **DX / scaffold / CI** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | `plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | Scaffold; conformance scripts | ACP-8 · ACP-PROD-10 |
| **Integrations** | `INTEGRATIONS` | `plan/INTEGRATIONS.md` | `required_integration_slugs`; gateway-only | ACP-CON-7 |
| **Elastic / scale** | `ELASTIC_CAPACITY_AND_SCALING` | `plan/ELASTIC_CAPACITY_AND_SCALING.md` | Budget guards §32.6 | **ACP-STEP-2b** kernel enforcement |

**PR rule:** If an ACP PR imports or changes Tier-1 Nexus semantics beyond the bridge listed in the wave, **stop** and split — agent layer composes; it does not fork Nexus.

---

### ACP architecture traceability (§ → deliverables → verification)

| Architecture § | Capability delivered | Primary plan IDs | Verification |
|----------------|---------------------|------------------|--------------|
| §12 | Full `AgentContract` at register | **ACP-CON-4** · AS-1 | Incomplete contract → `AgentAssemblyError` |
| §14 | Typed execution result | ACP-DX-1 · ACP-OBS-1 | `AgentRunResult` fields + trace |
| §15–§16 | Registry + capabilities | REG/CG **Done** · ACP-CON-6 · AUDIT-IDEAL-19.1 **Done** | Capability token routing |
| §17–§19 | Prompt + registry + graph | PE/REG/CG **Done** · AUDIT-IDEAL-20.1 **Done** | `phase_v_capability_graph_guard.py` |
| §20 | Lifecycle governance | V-ALG/AS **Done** · ACP-PROD-9 · AUDIT-IDEAL-31.1 **Done** | `check_on_call_ownership_model.py` |
| §13 | `run()` + `on_next_step` author API | ACP-DX-3 · ACP-STEP-1 | Direct `run()` test + agent_os 01 |
| §21–§28 | Cognitive patterns + gaps closed | ACP-1..6 · ACP-9 | Pattern unit + reference agents |
| §29 | `AgentRunRequest` / `Result` | ACP-DX-1 · ACP-CON-1 | Round-trip JSON; enum tests |
| §30 · §30.9 | Environment merge + identity | ACP-DX-2 · ACP-DX-5 | Merge order; user_id gate |
| §25.4 | Invocation token metering (agent + environment) | **ACP-TOK-1** · GAP-ACP-36 | `invocation_usage` + `budget.tokens_*` live in kernel |
| §25.5 | Per-agent limits + exceed reactions | **ACP-TOK-2** · **ACP-TOK-3** · GAP-ACP-37 | Hard cap blocks LLM; `BudgetReactionProfile` paths |
| §31 | Dual observability planes | ACP-OBS-1 · ACP-OBS-2 | Trace on result; multi-agent summary |
| §32 · **§32.0** | Step loop + readability | ACP-STEP-* · ACP-DX-6 · ACP-0 | Factory tests; typed-state CI |
| §33 | Per-step LLM | ACP-LLM-1 | Model hint within profile |
| §34 | Shared graph state | ACP-STATE-1 · ACP-PROD-5 | Two-agent handoff |
| §35 | UC-1..11 without rewrite | Waves 2–6 integration | UC mapping acceptance |
| §37 | Operational contracts | ACP-CON-* | Merge, enums, routing, security CI |
| §38 | NexusLoop vs HarnessKernel split | ACP-STEP-2 · ACP-STEP-2b | Runtime glue-only test; kernel owns policy/trace/budget/state |
| §39 | Org policy envelope | ACP-ORG-* | UC-11 fixture |
| §40 | Production reliability + scoreboard | ACP-PROD-* **Done** · **ACP-CLOSE-PROD-1..8 Done** | Per-roster `production_mode` via §40.15 scoreboard thresholds |
| §45 | New agent checklist | ACP-8 · ACP-11..13 | Scaffold + conformance CI |
| **Fleet** | Roster migration | **ACP-MIG-*** Wave 8 | Tracker 100% Runtime dimension |

### ACP — Master register (ACP-DOC · ACP-DX · ACP-PROD · …)

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| ACP-DOC.1 | ACP0 | **Architecture canon §21–§28** — ACP spec, flows, patterns, gaps | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | This document + ADR-AGENT-001 |
| ACP-DOC.2 | ACP0 | **Appendix AC** — cognitive patterns author guide in `AGENT_CREATION_GUIDE.md` | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + pattern selection table + skeleton |
| ACP-DOC.3 | ACP0 | **Audit prompt** — ACP dimensions in domain audit | **Done** | `guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Regenerated via `generate_domain_audit_prompts.py` |
| ACP-ADR.1 | ACP0 | **ADR-AGENT-001** accepted | **Done** | `docs/adr/entries/2026-06-11/ADR-AGENT-001.md` | Linked from architecture §21 |
| ACP-ADR.2 | ACP0 | **ADR-AGENT-002** accepted — `run()` facade | **Done** | `docs/adr/entries/2026-06-11/ADR-AGENT-002.md` | Linked from architecture §29 |
| ACP-DOC.4 | ACP0 | **Architecture §29–§30** — run facade + per-agent environment binding | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §29–§30 + ADR-AGENT-002 |
| ACP-DOC.5 | ACP0 | **Architecture §31–§36** — dual observability, step loop, LLM routing, UC catalog | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §31–§36 + ADR-AGENT-003 |
| ACP-ADR.3 | ACP0 | **ADR-AGENT-003** accepted — `on_next_step` + dual observability | **Done** | `docs/adr/entries/2026-06-11/ADR-AGENT-003.md` | Linked from architecture §31–§32 |
| ACP-DX-1 | ACP-DX | **`AgentRunRequest` / `AgentRunResult` / `RequestIdentity` / `AgentEnvironmentOverrides`** Pydantic contracts | **Done** | `intergrax/contracts/agent_run.py` | Round-trip + user_id required when memory_scope=user |
| ACP-DX-2 | ACP-DX | **`merge_environment`** + `EffectiveAgentRunEnvironment` + **memory_scope resolution** §30.9 | **Done** | `intergrax/agents/run_environment.py` | Unit test merge order + user vs org namespace |
| ACP-DX-3 | ACP-DX | **`IntergraxAgent.run` upgrade** — uses merge + typed result; hooks `configure_run`, `on_run_start/end` | **Done** | `intergrax/agents/authoring/base.py`, `acp_run.py` | Test direct run without Nexus |
| ACP-DX-4 | ACP-DX | **Nexus node bridge** — Task metadata → AgentRunRequest same merge path | **Done** | `runtime_request_bridge.py`, `agent_engine.py`, `acp_checkpoint_task_enricher.py` | Harness hosts set `acp.session.v1` via task enricher |
| ACP-DX-5 | ACP-DX | **`AgentBinding` profile slices** — tool/memory/integration per roster entry | **Done** | `applications/contracts/manifest.py` | Binding slice merge tests |
| ACP-DX-6 | ACP-DX | **Author readability kit** — `StepOutcome` factories, `load_session_state` / `session_state_delta`, `check_agent_typed_state.py` | **Done** | `intergrax/agents/authoring/step_outcome.py`, `state_access.py`, `scripts/` | Factories set consistent enums; CI fails raw dict state in agents |
| ACP-DOC.10 | ACP0 | **Architecture §32.0** — author readability & typed-only contracts (READ/UPDATE/DECIDE) | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §32.0 + ACP-AP-11..15 + checklist §45 |
| ACP-STEP-1 | ACP-STEP | **`AgentStepContext` / `StepOutcome` / author `on_next_step`** on `IntergraxAgent` | **Done** | `intergrax/agents/authoring/step_loop.py`, `agent_step_context.py`, `base.py` | Unit: terminal + continue via factories §32.0; no dict author surface |
| ACP-STEP-2 | ACP-STEP | **`AgentRuntime.advance_step`** — glue only: `on_next_step` → `HarnessKernel.execute_step`; **no policy/trace/state logic** | **Done** | `intergrax/agents/authoring/step_loop.py` | Unit: advance_step contains no policy imports; delegates 100% to kernel |
| ACP-STEP-2b | ACP-STEP | **`HarnessKernel.execute_step`** — L1 harness cycle: policy pre/post, state merge §37.2, gateways, budgets §32.6, trace/`AgentStepRecord`, declarative actions §32.8 | **Done** | `intergrax/runtime/kernel/step_kernel.py` | Integration: policy deny + trace record + budget exceeded from kernel only |
| ACP-CON-4 | ACP-CON | **§12 full contract gate at register** — `input_schema`, `output_schema`, `risk_level`, `validation_rules`, `failure_modes`, budgets; reject incomplete contracts | **Done** | `agent_assembly_resolver.py` | Register with stub contract → `AgentAssemblyError`; roster agents pass |
| ACP-STEP-3 | ACP-STEP | **UAEP legacy bridge** — `run_step` → advance_step + kernel | **Done** | `uaep_step_bridge.py`, `uaep.py` | UAEP unit tests green; kernel trace on session |
| ACP-OBS-1 | ACP-OBS | **`AgentRunTrace` / `AgentStepRecord`** on `AgentRunResult` | **Done** | `intergrax/contracts/agent_run_trace.py` | `test_acp_wave3_trace`: `steps[0].llm_calls` populated |
| ACP-OBS-2 | ACP-OBS | **`ApplicationRunSummary`** from Nexus task completion | **Done** | `application_run_summary_builder.py`, `task_finisher.py` | `test_application_run_summary_builder` multi-agent |
| ACP-LLM-1 | ACP-LLM | **`StepLLMRouter`** on step context | **Done** | `intergrax/agents/authoring/llm_router.py` | Per-step model hint + trace record |
| ACP-STATE-1 | ACP-STATE | **`SharedContextView`** for graph handoffs | **Done** | `intergrax/contracts/shared_context.py`, `shared_context_bridge.py` | Two-node handoff unit test |
| ACP-CON-1 | ACP-CON | **`AgentRunErrorCode` / `TerminalReason` enums** + Pydantic on run contracts | **Done** | `intergrax/contracts/agent_run.py`, `agent_run_enums.py` | extra=forbid; enum round-trip |
| ACP-CON-2 | ACP-CON | **`state_delta` merge-patch** + `_version` + checkpoint/resume | **Done** | `intergrax/agents/authoring/state_merge.py` | Unit: merge, delete null, conflict |
| ACP-CON-3 | ACP-CON | **Side-effect mode** immediate vs declarative enforcement | **Done** | `intergrax/agents/authoring/side_effect_validation.py` | Reject mixed mode per step |
| ACP-CON-6 | ACP-CON | **Capability routing** — registry query by token not class | **Done** | Nexus selection path + test | Integration: two impls same capability |
| ACP-CON-7 | ACP-CON | **Security CI guards** — gateway-only I/O, STRICT widen deny | **Done** | `scripts/check_agent_step_security.py` | CI green on roster |
| ACP-DOC.6 | ACP0 | **Architecture §37** — pre-implementation operational contracts | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Audit gaps A–G closed in canon |
| ACP-DOC.7 | ACP0 | **Architecture §38** — NexusLoop vs HarnessKernel execution stack | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §38 + ACP-INV-11 |
| ACP-DOC.8 | ACP0 | **Architecture §39** — organizational policy envelope & virtual workforce | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §39 + UC-11 |
| ACP-ORG-1 | ACP-ORG | **`OrganizationalPolicyEnvelope`** on ApplicationEnvironmentProfile | **Done** | `intergrax/applications/contracts/org_policy.py` | Pydantic extra=forbid |
| ACP-ORG-2 | ACP-ORG | **`OrganizationalPolicyContext`** in merge_environment | **Done** | `intergrax/agents/run_environment.py` | Role + envelope merge test |
| ACP-ORG-3 | ACP-ORG | **Kernel org enforcement** — channel/tool/playbook overlays | **Done** | `intergrax/runtime/kernel/step_kernel.py` | Block denied channel tool |
| ACP-ORG-4 | ACP-ORG | **`PolicyVerdictRecord` + compliance_summary** on trace/result | **Done** | `intergrax/contracts/agent_run_trace.py` | Step trace assertion |
| ACP-ORG-5 | ACP-ORG | **Reference org fixture + golden compliance eval** | **Done** | `lab_org_virtual_workforce_defaults` + gate tests | Zero POLICY_DENIED on happy path |
| ACP-DOC.9 | ACP0 | **Architecture §40** — production reliability, safety, persistence, release gates | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §40 canon accepted; ACP-PROD delivered; depth = ACP-CLOSE |
| ACP-PROD-1 | ACP-PROD | **Checkpoint / resume / replay** — step store + crash recovery | **Done** | `checkpoint_store.py`, acceptance `05c`/`05d` | Resume smoke; no double mutating tool |
| ACP-PROD-2 | ACP-PROD | **Side-effect idempotency ledger** — dedupe + declarative execute/commit in kernel | **Done** | `side_effect_ledger.py`, `declarative_tool_executor.py`, `step_kernel.py` | Resume replay skip + commit on invoke |
| ACP-PROD-3 | ACP-PROD | **ToolExecutionProfile + compensation enqueue** | **Done** | `tool_execution_profile.py`, `compensation_enqueue.py`, `step_kernel.py` | Policy deny after commit triggers recall |
| ACP-PROD-4 | ACP-PROD | **ReliabilityProfile in HarnessKernel** — retry/CB/timeout | **Done** | `intergrax/runtime/kernel/session_reliability.py` | REL profile wired |
| ACP-PROD-5 | ACP-PROD | **SharedContextView CAS + conflict policy** | **Done** | `intergrax/contracts/shared_context.py` | Parallel graph conflict test |
| ACP-PROD-6 | ACP-PROD | **`ArtifactRef` contract** on result/step | **Done** | `intergrax/contracts/artifact_ref.py` | Typed artifacts in test |
| ACP-PROD-7 | ACP-PROD | **Agent threat model CI** — matrix §40.7 | **Done** | `scripts/check_agent_threat_model.py` | CI-02..03 + threat cases |
| ACP-PROD-8 | ACP-PROD | **Privacy/redaction on trace/memory** | **Done** | `intergrax/contracts/privacy_redaction.py` | PII redaction test |
| ACP-PROD-9 | ACP-PROD | **Release eval gates** — golden/regression/policy suites | **Done** | `scripts/check_agent_release_gates.py` | Staging gate green |
| ACP-PROD-10 | ACP-PROD | **CI conformance matrix §40.10** | **Done** | `scripts/check_acp_ci_conformance_matrix.py` | CI-01..16 applicable rows |
| ACP-PROD-11 | ACP-PROD | **Schema version registry + migration adapters** | **Done** | `intergrax/contracts/migrations/` | check_contract_schema_versions |
| ACP-0 | ACP1 | **`AcpSessionState` / `acp.state.v1` schema** — Pydantic envelope + agent subclass pattern §32.0 | **Done** | `intergrax/contracts/acp_state.py` | Unit test round-trip; extra=forbid |
| ACP-0b | ACP1 | **`cognitive_pattern` on AgentContract** — optional field + validation | **Done** | `agent_contract_meta.py`, `agent_assembly_resolver.py` | `test_cognitive_patterns` validation |
| ACP-1 | ACP1 | **`CognitiveAgent` ABC** — perceive/reason/act/evaluate + `on_next_step` | **Done** | `patterns/base.py` | Pattern probe runs |
| ACP-2 | ACP2 | **`ReflexAgent`** | **Done** | `patterns/reflex.py` | `PatternReflexProbe` |
| ACP-3 | ACP2 | **`ReActAgent`** — bounded loop, budget in `acp.state.v1` | **Done** | `patterns/react.py`, `patterns/states.py` | `PatternReActProbe` |
| ACP-4 | ACP2 | **`PlanExecuteAgent`** — multi-step + phase machine | **Done** | `patterns/plan_execute.py` | `PatternPlanExecuteProbe` |
| ACP-5 | ACP2 | **`DecompositionAgent`** — sub-question queue + convergence | **Done** | `patterns/decomposition.py` | `PatternDecompositionProbe` |
| ACP-6 | ACP2 | **`ReflectionAgent`** — draft/critique/revise phases | **Done** | `patterns/reflection.py` | `PatternReflectionProbe` (CVL hook Wave 6+) |
| ACP-7 | ACP3 | **Decision helpers** — legacy UAEP bridge; new code uses `StepOutcome` factories §32.0 (ACP-DX-6) | **Done** | `intergrax/agents/authoring/decisions.py` | Primary `finish`/`continue_with`/…; UAEP helpers deprecated; `to_step_outcome` bridge |
| ACP-8 | ACP3 | **Scaffold `--pattern`** flag on `new-agent` | **Done** | `scaffold/new_agent.py`, `scaffold/cli.py` | `test_acp_pattern_scaffold` |
| ACP-9 | ACP4 | **Harness reference probes** — one per pattern | **Done** | `patterns/reference.py` | Pattern probe unit tests |
| ACP-10 | ACP4 | **Unit test package** `tests/unit/agents/authoring/patterns/` | **Done** | tests | 32 gate tests — one probe run per pattern + contracts + phase machines |
| ACP-11 | ACP5 | **Gate: ACP pattern scaffold** — no UAEP boilerplate | **Done** | `scripts/check_scaffold_acp_pattern.py` | Scaffold smoke script |
| ACP-13 | ACP5 | **Pattern conformance** — contract vs class | **Done** | `scripts/check_agent_pattern_conformance.py` | AST check on `agents/*/contract.py` |
| ACP-12 | ACP5 | **Acceptance: pattern agent in agent_os suite** | **Done** | `tests/acceptance/agent_os/test_acp_pattern_agents.py` | NexusLoop + `acp.session.v1` per pattern (mock LLM) |
| ACP-CFG | ACP6 | **`build_context` profile injection** — reduce per-agent `RuntimeConfig` duplication | **Done** | `intergrax/agents/reference_harness.py` | `build_lab_agent_runtime_config_from_merged` |
| ACP-LEG-1 | ACP-LEG | **Deprecate AgentEngine path** — `DeprecationWarning` in `AgentEngine` fallback | **Done** | `intergrax/agents/agent_engine.py` | `test_agent_engine_legacy_deprecation` |
| ACP-LEG-2 | ACP-LEG | **Fleet migration complete** — superseded by **Wave 8** `ACP-MIG-*` program (not ad-hoc per-agent) | **Done** | `agents/*` | Scoreboard Runtime ≥100% roster-wide; typed-state CI allowlist empty |
| ACP-MIG-1 | ACP-MIG | **Fleet inventory auditor** — legacy surface per agent (`uaep`/`runtime_engine`/`dict state`) | **Done** | `scripts/audit_agent_fleet_legacy.py` | JSON report for all `agents/*` packages |
| ACP-MIG-2 | ACP-MIG | **Migration tiers + batch order** — harness → staging read-only → staging mutating → prod-eligible | **Done** | plan §6.1aw Wave 8 · `agents/README.md` | Documented tiers match roster table |
| ACP-MIG-3 | ACP-MIG | **Pilot batch (3 agents)** — echo, signoff_probe, research → typed `on_next_step` | **Done** | `agents/echo`, `signoff_probe`, `research` | Direct `run()` + agent_os green per agent |
| ACP-MIG-4 | ACP-MIG | **Product batch** — legal, summary, LKW trio, DSW quartet | **Done** | product `agents/*` | Host wiring tests unchanged; scoreboard Runtime ≥80% each |
| ACP-MIG-5 | ACP-MIG | **Remaining roster** — org_worker, assistant, K-path agents; lab mocks excluded | **Done** | `agents/*` | Zero UAEP-only new code; bridge allowlist shrinking |
| ACP-MIG-6 | ACP-MIG | **Fleet migration CI gate** — `check_agent_fleet_migration.py` blocks regression | **Done** | `scripts/` | CI fails if migrated agent reintroduces legacy surface |
| ACP-MIG-7 | ACP-MIG | **Per-host binding verification** after each batch | **Done** | `applications/*/manifest.py` tests | AgentBinding slices + capability routing per host |
| ACP-PROD-12 | ACP-PROD | **`AgentProductionReadinessReport`** scoreboard — 10 dimensions 0–100% per agent | **Done** | `intergrax/contracts/agent_readiness.py`, `scripts/report_agent_production_readiness.py` | Report generated for roster; prod promotion uses thresholds §6.1az |
| ACP-LEG-3 | ACP-LEG | **Document AgentEngine internal-only** | **Done** | `runtime.py` module docstring + architecture §13 | INTERNAL ONLY banner |
| ACP-LEG-4 | ACP-LEG | **Remove author UAEP from `--pattern` scaffold** — typed hooks only | **Done** | `scaffold/new_agent.py` | `--pattern` agents have no `get_steps` |
| ACP-DOC.11 | ACP0 | **Detailed implementation waves §6.1aw** + debt/coupling matrix | **Done** | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §6.1aw |
| ACP-DOC.12 | ACP0 | **Plan correction** — §12–§20 scope map, runtime/kernel split, ACP-CON-4 | **Done** | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Wave 1 + scope mapping |
| ACP-DOC.13 | ACP0 | **Wave 8 fleet migration** + **§6.1az production readiness scoreboard** | **Done** | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Operational closure |

---

## Phase ACP-CLOSE — Architecture compliance closeout

**Status:** **Done** (2026-06-11) — post-ACP domain audit · **Band 2bb** · queue **[§6.1bb](#61bb-harness-implementation-queue--acp-close-done)**  
**Source:** Architecture ↔ plan ↔ code audit (2026-06-11)  
**Goal (achieved):** **DEBT-ACP register → zero**; §40 host depth; §40.12 evidenced; CI-1..3 wired.

**Explicitly excluded:** Nexus graph/orchestration refactor; Phase K agents; new Tier-0 engines.

### ACP-CLOSE — Master register

| ID | Area | Deliverable | Status | Arch § | Modules / scripts | Acceptance |
|----|------|-------------|--------|--------|-------------------|------------|
| ACP-CLOSE-DOC-1 | DOC | Plan Phase ACP header + scope map sync | **Done** | plan | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | This update |
| ACP-CLOSE-DOC-2 | DOC | Architecture §28.3 GAP register — Closed/Open truth table | **Done** | §28.3 | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | 32 Closed · 3 Open (03/04/07) |
| ACP-CLOSE-DOC-3 | DOC | Architecture §36.4 · §40.13 · implementation status tables | **Done** | §36.4 · §40.13 | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Code maps Done; §40 platform implemented |
| ACP-CLOSE-DOC-4 | DOC | Regenerate domain audit prompt | **Done** | audit | `guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | `generate_domain_audit_prompts.py` |
| ACP-CLOSE-LEG-1 | LEG | **Remove** `AgentEngine` fallback from `AgentEngine` | **Done** | §13.5 · §38 | `agent_engine.py` | `ValueError` on non-UAEP/non-ACP agents; DEBT-ACP-06 **Closed** |
| ACP-CLOSE-LEG-2 | LEG | **Remove** author-visible UAEP (`decide_after_step` on `IntergraxAgent`) | **Done** | §13.3–13.4 | `uaep_linear_bridge.py`, `uaep.py` | `linear_agent_decide_after_step`; DEBT-ACP-04 **Closed** for linear agents |
| ACP-CLOSE-LEG-3 | LEG | Retire public `uaep_pipeline.py` bridge | **Done** | §13.5 | ADR-FLOW-005 | Public module removed; superseded by ACP-only `AgentEngine` |
| ACP-CLOSE-LEG-4 | LEG | §45 checklist — UAEP internal-only wording | **Done** | §45 | `AGENT_CREATION_GUIDE.md`, `check_agent_creation_guide_acp_canon.py` | No author UAEP-first path; CI grep gate |
| ACP-CLOSE-LEG-5 | LEG | **Delete** Tier-1 `RuntimeEngine` pipeline stack (`pipelines/`, `runtime_steps/`, engine planner) | **Done** | §13.5 · ADR-FLOW-005 | `nexus/tools/tool_loop.py`, `plan_context_invocation.py` | ACP-only scaffold; `RuntimeConfig.pipeline` removed; docs scrubbed |
| ACP-CLOSE-PROD-1 | PROD | `AgentCheckpointStore` on **all mutating product hosts** | **Done** | §40.1 | `acp_checkpoint_host_wiring.py`, `harness_host_runtime.py` | Auto-resolve store on harness hosts; exposed on `HarnessHostRuntime` |
| ACP-CLOSE-PROD-2 | PROD | `acp_checkpoint_task_enricher` on product hosts (lab pattern) | **Done** | §40.1.4 | `task_control_wiring.py`, `applications/*/host/factory.py` | `build_reliability_task_enricher(..., agent_checkpoint_store=)` |
| ACP-CLOSE-PROD-3 | PROD | `CatalogDeclarativeToolInvoker` — real execution context (no `MagicMock` shim) | **Done** | §32.8 · §40.3 | `catalog_declarative_invoker.py` | `_CatalogDispatchLLMStub` + direct `RuntimeContext`; preserves host invoker |
| ACP-CLOSE-PROD-4 | PROD | Nexus **E2E** acceptance — catalog declarative invoker (not callable mock) | **Done** | §27 · §40.12 | `test_acp_nexus_catalog_declarative_resume.py` | `build_harness_host_runtime` + `NexusLoop` resume green |
| ACP-CLOSE-PROD-5 | PROD | **Durable compensation queue** for `enqueued` requests | **Done** | §40.3.3 | `compensation_queue_store.py`, `compensation_queue_worker.py` | Host + kernel persist `enqueued` jobs |
| ACP-CLOSE-PROD-6 | PROD | `ReliabilityProfile.idempotency_store` ↔ ledger replay §40.2.2 | **Done** | §40.2 | `idempotency_ledger_bridge.py`, `idempotency_store_wiring.py`, kernel replay | `test_acp_declarative_mutating_cross_run_dedupe.py` |
| ACP-CLOSE-PROD-7 | PROD | §40.12 checklist **green** — reference mutating agent artifact | **Done** | §40.12 | `section_40_12_checklist.py`, `check_acp_section_40_12_checklist.py` | `build/acp_section_40_12_reference.json` |
| ACP-CLOSE-PROD-8 | PROD | Scoreboard mutating agents **100%** checkpoint + idempotency dimensions | **Done** | §40.15 | `readiness/scoreboard.py` | `--require-mutating-checkpoint-idempotency-100` green |
| ACP-CLOSE-PAT-1 | PAT | ReAct ↔ **TOOL-ENG-6** unified tool loop + budget keys | **Done** | §26.3 · §25.2 | `patterns/react.py`, `react_budget.py`, `tool_loop_step.py` | 2-iteration integration; DEBT-ACP-18 **Closed** |
| ACP-CLOSE-PAT-2 | PAT | `ReflectionAgent` → CVL critic hooks (gateway only) | **Done** | §26.6 | `critic_gateway.py`, `patterns/reflection.py`, `acp_session_host_wiring.py` | Gateway-only CVL; host `critic_graph_hooks` on ACP run |
| ACP-CLOSE-PAT-3 | PAT | Author terminology — single canonical §29 entry | **Done** | §28.3 GAP-07 · §29.0 | `AGENT_CREATION_GUIDE.md`, architecture §22–§23 · §27 | §29 single entry; GAP-ACP-07 **Closed** |
| ACP-CLOSE-ORG-1 | ORG | STRICT **configure_run widen deny** per-agent | **Done** | §39.4 | `configure_run_strict.py`, `merge_environment`, `acp_run` | `test_configure_run_strict.py` |
| ACP-CLOSE-ORG-2 | ORG | UC-11 compliance golden per **product host** | **Done** | §39.5 | `product_host_org_envelope`, `uc11_compliance_golden.py` | `test_uc11_product_host_compliance.py` (6 hosts) |
| ACP-CLOSE-CI-1 | CI | Post-LEG grep + fleet migration gate — zero Tier-2 `AgentEngine` | **Done** | §40.10 CI-04 | `check_agent_fleet_migration.py`, `check_agent_acp_close_ci.py` | `.github/workflows/unit-tests.yml` |
| ACP-CLOSE-CI-2 | CI | Anti-pattern ACP-AP-02 after TOOL-ENG-6 | **Done** | §28.4 | `check_agent_acp_ap02_tool_loop_boundary.py` | CI-17 · Nexus does not schedule tool iterations |
| ACP-CLOSE-CI-3 | CI | `check_agent_production_readiness.py --fail-on-blockers` in gate workflow | **Done** | §40.15 | `check_agent_acp_close_ci.py` | CI-16 + gate workflow |

**Cross-plan (not ACP-CLOSE IDs — deliver in owning domain):**

| ID | Domain plan | Deliverable | Status |
|----|-------------|-------------|--------|
| AUDIT-IDEAL-19.1 | this file §AUDIT-IDEAL | Durable cross-host registry snapshot | **Done** |
| AUDIT-IDEAL-20.1 | this file §AUDIT-IDEAL | Product CI blast-radius on tool/skill changes | **Done** |
| AUDIT-IDEAL-31.1 | this file §AUDIT-IDEAL | Owner/on-call mandatory on certified agents | **Done** |
| TOOL-ENG-6 | `plan/TOOLS.md` | Tool loop step — sync with ACP-CLOSE-PAT-1 | **Done** |
| ACP-TOK-1..3 | this file §ACP-FINISH | Token metering, limits, reactions | **Done** |
| ACP-TOK-CI | this file §ACP-FINISH | Token budget CI gate | **Done** |
| ACP-FINISH-DOC-1 | this file §ACP-FINISH | GAP-ACP-36/37 Closed + §40.13 sync | **Done** (2026-06-13) |

**ACP-CLOSE DoD:** DEBT-ACP **3/3 Open → Closed**; architecture §28.3 synced; mutating scoreboard dimensions **≥100%**; §40.12 evidenced; `pytest -m gate` green.

**Delivery rule:** One **ACP-CLOSE-\*** ID per PR → update this register → journal on phase completion.

---

## Phase ACP-FINISH — Agent architecture completion

**Status:** **Done** (2026-06-13) — GAP-ACP-36/37 **Closed** · architecture §28.3 **37 Closed · 0 Open** · **Band 2bc closed**  
**Goal:** Close **GAP-ACP-36** (invocation token rollups) and **GAP-ACP-37** (per-agent limits + application reaction policies). After this phase, agent architecture canon is **decision-complete and implementation-complete** for §13–§40 (AUDIT-IDEAL §12–§20 residuals remain parallel).

**Explicitly excluded:** New cognitive patterns; Nexus orchestration refactor; Phase K business agents.

### ACP-FINISH — Master register

| ID | Area | Deliverable | Status | Arch § | Modules / scripts | Acceptance |
|----|------|-------------|--------|--------|-------------------|------------|
| ACP-TOK-1 | TOK | **Metering** — agent + environment token rollups in invocation state | **Done** | §25.4 · §33.4 | `HarnessKernel`, `acp_run.py`, `merge_environment` | `test_acp_token_usage_metering.py` |
| ACP-TOK-2 | TOK | **Limits** — per-agent caps from application + hard/advisory enforcement | **Done** | §25.5.1–§25.5.2 | `AgentBinding.budget_slice`, `AgentExecutionOptions.max_total_tokens`, kernel pre-LLM check | `test_acp_token_budget_enforcement.py` |
| ACP-TOK-3 | TOK | **Reactions** — environment policies on threshold/exceed | **Done** | §25.5.3 · §30.8 | `CostProfile.budget_reaction`, host hooks, notify wiring | `abort` · `hitl` · `degrade_model` · `notify_only` · `custom_hook` paths tested |
| ACP-TOK-CI | CI | Token budget contract gate | **Done** | §25.4–§25.5 · §40.10 | `check_agent_token_budget_contract.py` | CI-18 row; fails if kernel bypasses metering or agents increment budget in state_delta |
| ACP-FINISH-DOC-1 | DOC | Architecture status tables + GAP-ACP-36/37 → Closed | **Done** (2026-06-13) | §28.3 · §36.4 · §40.13 | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`, audit prompt | 37 Closed · 0 Open; §40.13 declares architecture **complete** |

### ACP-FINISH — Sub-task breakdown (implementation guide)

| Parent | Sub | Work package | Depends on |
|--------|-----|--------------|------------|
| ACP-TOK-1 | 1a | After each LLM call: increment `AcpBudgetState.tokens_in/out/total` from `LlmCallRecord` / router drain | — |
| ACP-TOK-1 | 1b | Build `AcpInvocationUsageView` (agent mirror + environment rollup from task / `ApplicationRunSummary`) | 1a |
| ACP-TOK-1 | 1c | `merge_environment`: materialize `tokens_limit` / `tokens_remaining` when limits resolved (no enforcement yet) | 1a |
| ACP-TOK-1 | 1d | Tests: `test_acp_token_usage_metering.py`, adaptive `model_hint` downgrade fixture (§33.4) | 1a–1c |
| ACP-TOK-2 | 2a | Resolve limits: `CostProfile` → `AgentBinding.budget_slice` → `execution_options` (merge §25.5.1) | 1c |
| ACP-TOK-2 | 2b | `HarnessKernel` pre-LLM hard enforcement when `enforcement=hard` | 2a |
| ACP-TOK-3 | 2c | Wire `BudgetReactionProfile` reactions + `RuntimeEvent` `BUDGET_THRESHOLD` / `BUDGET_EXCEEDED` | 2b |
| ACP-TOK-3 | 2d | Tier-3 reference: one product host with `budget_slice` + `budget_reaction` + notify hook | 2c |
| ACP-TOK-CI | — | Static + smoke gate; wire `check_agent_acp_close_ci.py` or matrix CI-18 | 1d + 2d |
| ACP-FINISH-DOC-1 | — | Close GAP register; refresh §40.13.1 audit acceptance | all above |

**Cross-domain:** [`plan/LLM_ADAPTERS.md`](LLM_ADAPTERS.md) (metering source) · [`plan/UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) (`RunBudget`, `BudgetEnforcer`) · [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) (COST-1 · `cost_runtime_bridge.py` baseline).

**ACP-FINISH DoD:** ACP-TOK-1..3 + ACP-TOK-CI + ACP-FINISH-DOC-1 **Done**; `pytest -m gate` green; architecture §28.3 **0 Open** GAPs for ACP.

**Delivery rule:** One **ACP-TOK-\*** or **ACP-FINISH-DOC-1** per PR → update register → journal on phase completion.

---

## Phase ACP-DEPTH — (merged into ACP-FINISH)

Historical alias for token-depth work. **Closed register:** [Phase ACP-FINISH](#phase-acp-finish--agent-architecture-completion) **Done** (2026-06-13).

---

### 6.1av Harness implementation queue — Agent Cognitive Patterns (ACP) — closed

**Status:** **Done** (2026-06-11) · **Follow-on:** [§6.1bc ACP-FINISH](#61bc-harness-implementation-queue--acp-finish-closed) **Done** (2026-06-13)

**Purpose:** Historical wave order (Band 2aw). **Detailed steps:** [§6.1aw](#61aw-acp-detailed-implementation-waves).

| Wave | IDs | Closes architecture | Legacy removed |
|------|-----|---------------------|----------------|
| **0** | ACP-DX-1 · ACP-CON-1 · ACP-CON-4 · ACP-0 · ACP-DX-6 · ACP-CON-2 | **§12** gate · §29 · §37.1–§37.2 · **§32.0** types | DEBT-ACP-02/03/13/14 ✓ |
| **1** | ACP-STEP-1 · ACP-STEP-2 · ACP-STEP-2b · ACP-CON-3 | §32 · §38 · §32.8 | DEBT-ACP-04 partial → **ACP-CLOSE-LEG-2** |
| **2** | ACP-DX-2 · ACP-DX-3 · ACP-DX-4 · ACP-DX-5 · ACP-CFG | §29–§30 · §36 | DEBT-ACP-01/07 ✓ |
| **3** | ACP-OBS-1 · ACP-OBS-2 · ACP-LLM-1 · ACP-STATE-1 | §31–§34 | DEBT-ACP-08..11 ✓ |
| **4** | ACP-STEP-3 · ACP-LEG-1 · ACP-LEG-3 | §13.4 UAEP bridge | DEBT-ACP-05 ✓ · 06 → **ACP-CLOSE-LEG-1** |
| **5** | ACP-0b · ACP-1..13 · ACP-8 · ACP-LEG-4 | §21–§28 patterns + scaffold | DEBT-ACP-15 ✓ · 18 → **ACP-CLOSE-PAT-1** |
| **8** | **ACP-MIG-1..7** · **ACP-LEG-2** | Fleet migration | DEBT-ACP-16 ✓ |
| **6** | ACP-CON-6 · ACP-CON-7 · ACP-ORG-1..5 | §37.6–§37.7 · §39 | DEBT-ACP-12 ✓ |
| **7** | ACP-PROD-1..12 | §40 platform + scoreboard | DEBT-ACP-17 platform ✓ · depth → **ACP-CLOSE-PROD-*** |

**Continuous:** §6.1 gate maintenance · `pytest -m gate` green every PR.

---

### 6.1aw ACP detailed implementation waves

Each wave lists **PR-sized steps** in order. A step is **Done** only when acceptance tests pass and listed debt IDs are closed or explicitly bridged with deprecation.

#### Wave 0 — Typed contracts foundation (architecture §29 · §37 · §32.0)

**Goal:** All run/step/state types exist before loop wiring. **No** author-facing `dict` in new code after this wave.

| Step | ID | Files / modules | Tasks | Acceptance | Debt closed |
|------|-----|-----------------|-------|------------|-------------|
| 0.1 | ACP-DX-1 | `intergrax/contracts/agent_run.py` | Define `AgentRunRequest`, `AgentRunResult`, `RequestIdentity`, `AgentEnvironmentOverrides`, `AgentExecutionOptions`, `GovernanceSnapshot`, `AgentRunCost` — all `extra=forbid` | `tests/unit/contracts/test_agent_run_roundtrip.py` | DEBT-ACP-02 |
| 0.2 | ACP-CON-1 | same + enums module | `AgentRunErrorCode`, `TerminalReason`, `StepNextAction`, `AgentRunError`; wire to result/outcome fields | Enum round-trip; reject free-text in validation | DEBT-ACP-13 |
| 0.2b | ACP-CON-4 | `agent_assembly_resolver.py` | Extend `validate_contract_metadata` for §12 required fields: `input_schema`, `output_schema`, `risk_level`, `validation_rules` (≥1), `failure_modes` (≥1), `max_steps` or contract budgets; wire `AgentRegistry.register` | `test_agent_assembly_resolver.py`: incomplete contract raises `AgentAssemblyError`; reference agents pass | §12 |
| 0.3 | ACP-0 | `intergrax/contracts/acp_state.py` | `AcpSessionState`, `AcpBudgetState`, `ACP_STATE_KEY` constant; document subclass pattern §32.0.2 | Serialize/deserialize; `_version` field | DEBT-ACP-03 |
| 0.4 | ACP-DX-6a | `intergrax/agents/authoring/step_outcome.py` | `StepOutcome` model + factories: `continue_with`, `complete`, `fail`, `pause_hitl`, `replan` — set enums consistently | Factory unit tests per §32.0.4 | DEBT-ACP-04 |
| 0.5 | ACP-DX-6b | `intergrax/agents/authoring/state_access.py` | `load_session_state(agent, step_ctx)`, `session_state_delta(model, *, include=...)` on `IntergraxAgent` | Typed load + delta from Pydantic dump | DEBT-ACP-03 |
| 0.6 | ACP-CON-2 | `intergrax/agents/authoring/state_merge.py` | RFC 7396 shallow merge; `null` delete; `_version` increment; resume conflict → `VALIDATION_FAILED` | Merge unit matrix §37.2 | DEBT-ACP-14 |
| 0.7 | ACP-DX-6c | `scripts/check_agent_typed_state.py` | Fail CI on `state.get(` / `state[` in `agents/` (allowlist bridge files until Wave 4) | Script in CI workflow | — |

**Wave 0 DoD:** `uv run pytest tests/unit/contracts/ tests/unit/runtime/registry/test_agent_assembly_resolver.py -q` green; **incomplete `AgentContract` cannot register**; architecture §37.1 + §12 gate provable without Nexus.

---

#### Wave 1 — Step loop & kernel (architecture §32 · §38)

**Goal:** One iteration = `advance_step` (glue) → `on_next_step` (domain) → `HarnessKernel.execute_step` (harness).  
**Invariant (normative):** `AgentRuntime.advance_step` has **no** policy engine calls, trace writers, budget counters, or state-merge logic — those live **only** in `HarnessKernel.execute_step` (architecture §13 table · §38.1 L1 · §38.3).

```text
AgentRuntime.advance_step(agent, step_ctx):
    outcome = await agent.on_next_step(step_ctx)     # L2 — domain only
    await HarnessKernel.execute_step(outcome, step_ctx)  # L1 — all harness work
    return outcome

HarnessKernel.execute_step(outcome, step_ctx) -> StepExecutionRecord:
    1. policy pre-check (tools, budget, autonomy, org overlays when §39 wired)
    2. validate + apply state_delta §37.2 (_version bump)
    3. run declarative requested_actions if mode=declarative §32.8
    4. policy post-check on outcome + side effects
    5. enforce step/session budgets §32.6
    6. emit RuntimeEvents; append AgentStepRecord to run trace (Plane B)
    7. optional checkpoint hook when enabled
    DOES NOT: call on_next_step • domain replan • choose next graph agent
```

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 1.1 | ACP-STEP-1a | `intergrax/contracts/agent_step.py` (or `agent_run.py`) | `AgentStepContext` typed: gateways as protocols, `state_snapshot` internal, `load_session_state` path; gateways policy-bound at context build | Context construction test | TOOLS gateway protocol |
| 1.2 | ACP-STEP-1b | `intergrax/agents/authoring/step_loop.py` | `IntergraxAgent.on_next_step` default; `@step` driver mapping; forbid override `advance_step` | Continue + terminal factory tests | — |
| 1.3 | ACP-STEP-2 | `step_loop.py` | `AgentRuntime.advance_step`: **exactly two awaits** — `on_next_step` then `kernel.execute_step`; static check / test that module imports no policy or trace sink | `test_advance_step_is_glue_only.py`: no `PolicyEngine` / `TraceWriter` in advance_step body | — |
| 1.4 | ACP-STEP-2b | `intergrax/runtime/kernel/step_kernel.py` | `HarnessKernel.execute_step` implements full L1 cycle above; **zero** imports from `agents/` domain packages | Integration: policy deny, budget exceeded, trace step record — all attributed to kernel | UAEP · OBSERVABILITY |
| 1.5 | ACP-CON-3 | `step_kernel.py` or kernel helper | Enforce immediate vs declarative mutual exclusion §32.8 at kernel validation | Mixed-mode step rejected before actions run | TOOLS trace |

**Wave 1 DoD:** Glue-only test green; kernel integration test proves policy + trace + state merge without `advance_step` containing harness logic; reference `on_next_step` agent runs 3-step loop.

**Anti-pattern (reject in review):** policy pre/post or `trace.append` inside `AgentRuntime.advance_step` — violates §38 and duplicates L1.

---

#### Wave 2 — Run facade, environment merge, Nexus bridge (architecture §29–§30 · §36)

**Goal:** Same path for `agent.run()` and graph node execution.

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 2.1 | ACP-DX-2 | `intergrax/agents/run_environment.py` | `merge_environment(platform, app_profile, org_envelope, binding, request)` → `EffectiveAgentRunEnvironment`; `memory_scope` resolution §30.9 | Merge order unit test | MEMORY · TIER3 |
| 2.2 | ACP-DX-3 | `intergrax/agents/authoring/base.py` | `run(request: AgentRunRequest)` loop using Wave 1; hooks `configure_run`, `on_run_start/end`; typed `AgentRunResult` | Direct run without Nexus | — |
| 2.3 | ACP-DX-4 | `intergrax/agents/agent_engine.py`, graph executor bridge | Task metadata → `AgentRunRequest`; same `merge_environment` + `run()` | `agent_os` test 01 parity | NEXUS_EXECUTION_FLOW |
| 2.4 | ACP-DX-5 | `applications/contracts/`, host wiring | `AgentBinding` tool/memory/RAG/LLM slices per roster entry | Legal or research host test | TIER3 |
| 2.5 | ACP-CFG | `reference_harness.py`, migrate 1–2 reference agents | Remove duplicated `RuntimeConfig` from `build_context`; profile injection only | Reference agent diff shrinks | INTEGRATIONS |

**Wave 2 DoD:** `await agent.run(AgentRunRequest(...))` in pytest; Nexus single-agent test unchanged behavior; DEBT-ACP-01/07 closed.

---

#### Wave 3 — Observability, LLM routing, shared state (architecture §31–§34)

**Goal:** Plane A + Plane B journals; per-step model; graph handoffs typed.

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 3.1 | ACP-OBS-1 | `intergrax/contracts/agent_run_trace.py` | `AgentRunTrace`, `AgentStepRecord` (tool/RAG/LLM/decision/error codes); attach to `AgentRunResult` | Trace assertion on 2-step run | OBSERVABILITY |
| 3.2 | ACP-LLM-1 | `intergrax/agents/authoring/llm_router.py` | `StepLLMRouter` on context; policy-bound `model_hint`; record in step trace | Per-step model within profile | LLM_ADAPTERS |
| 3.3 | ACP-STATE-1 | `intergrax/contracts/shared_context.py` | `SharedContextView` read/write for graph handoffs | Two-node handoff unit test | ORCHESTRATION |
| 3.4 | ACP-OBS-2 | Nexus task completion path | `ApplicationRunSummary` on Task terminal; `trace_id` join | `agent_os` 02 multi-agent | OBSERVABILITY |

**Wave 3 DoD:** **Met** — `result.trace.steps[0].llm_calls` populated (`test_acp_wave3_trace`); `ApplicationRunSummary` on `TaskResult.metadata` via `build_nexus_task_result` + builder unit test.

---

#### Wave 4 — Legacy bridge & deprecation (architecture §13.4–§13.5)

**Goal:** Existing UAEP agents keep working **through bridge**; new code uses typed loop only.

| Step | ID | Files / modules | Tasks | Acceptance | Debt closed |
|------|-----|-----------------|-------|------------|-------------|
| 4.1 | ACP-STEP-3 | `intergrax/agents/uaep.py` | Map `run_step`/`decide_after_step` → `advance_step` + typed `StepOutcome` translation | Existing UAEP unit tests green | DEBT-ACP-05 bridge |
| 4.2 | ACP-LEG-1 | `agent_engine.py` | `DeprecationWarning` on `AgentEngine` fallback path | Warning in test | DEBT-ACP-06 |
| 4.3 | ACP-LEG-3 | docs + `runtime.py` docstring | Mark `AgentEngine` internal-only in canon | No author guide references | — |

**Wave 4 DoD:** **Met** — UAEP steps route through `HarnessKernel` (`uaep_step_bridge`); `DeprecationWarning` on `AgentEngine` fallback; `AgentEngine` marked internal-only. Fleet body migration = Wave 8.

---

#### Wave 5 — Cognitive patterns & scaffold (architecture §21–§28 · §32.0)

**Goal:** Pattern library demonstrates **readable** typed agents; scaffold emits correct skeleton.

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 5.1 | ACP-0b | `agent_contract_meta.py` | `cognitive_pattern` enum on contract; assembly validation | Resolver test | AS |
| 5.2 | ACP-1 | `patterns/base.py` | `CognitiveAgent` ABC → `on_next_step` delegates perceive/reason/act/evaluate | Base unit test | — |
| 5.3 | ACP-2 | `patterns/reflex.py` | Single-shot `StepOutcome.complete` | Unit test | — |
| 5.4 | ACP-3 | `patterns/react.py` | Budget in `AcpBudgetState`; tool loop | Integration + **TOOL-ENG-6** sync | TOOLS |
| 5.5 | ACP-4..6 | plan_execute, decomposition, reflection | Each uses typed state + factories; reflection uses CVL hook | Pattern tests | CRITIC_VERIFICATION |
| 5.6 | ACP-8 · ACP-LEG-4 | `scaffold/new_agent.py` | `--pattern`; emit state subclass + `on_next_step` skeleton §32.0.5; **no** UAEP boilerplate | Scaffold smoke | DX |
| 5.7 | ACP-9..10 | `pattern_reference_*.py`, tests package | One harness reference per pattern | Lab wiring smoke | — |
| 5.8 | ACP-11..13 | CI scripts | UAEP-only gate; pattern conformance; extend `agent_os` | CI green | DX |

**Wave 5 DoD:** **Met** — `new-agent` defaults to typed `reflex`; `--pattern` selects other patterns; `--uaep` legacy only; pattern library + probes + CI scripts; DEBT-ACP-05/15 closed.

**Prerequisite for Wave 8:** Waves 0–5 Done (typed loop, `run()`, patterns, scaffold target exist).

---

#### Wave 8 — Fleet migration program (operational — full roster)

**Goal:** Migrate **all** Tier-2 agents in `agents/` from legacy UAEP/`AgentEngine`/dict-state surfaces to typed **`on_next_step` + `AcpSessionState` + `agent.run(AgentRunRequest)`** — in controlled batches, without breaking hosts.

**Scope (~16 product agents + harness probes; excludes `lab/mock_agents.py` fixtures unless listed):**

| Tier | Agents (initial roster) | Risk | Migration target | Scoreboard min before next tier |
|------|-------------------------|------|------------------|--------------------------------|
| **T0 Harness** | echo, signoff_probe | Low | Reflex / typed loop | Runtime ≥80% |
| **T1 Staging read-only** | research, summary, local_search | Low–med | Pattern base + typed state | Runtime ≥80% |
| **T2 Staging mutating** | legal, local_indexer, local_synthesizer, DSW×4 | Med | Full §32.0 + host tests | Runtime ≥90%; Checkpointing N/A until Wave 7 |
| **T3 Prod-eligible** | echo (prod), future promoted | High | Scoreboard **overall ≥90%**, no dimension below 80% | Per §6.1az |
| **T4 Experimental** | problem_radar, vendor_discovery, org_worker, assistant | Variable | Best-effort; may stay bridge longer with ADR | Documented waiver |

**Per-agent migration checklist (every agent in a batch):**

```text
1. Inventory  — ACP-MIG-1 report row: legacy flags, host bindings, mutating tools
2. Contract   — §12 complete (ACP-CON-4); cognitive_pattern set (ACP-0b)
3. State      — typed AcpSessionState subclass; remove dict keys
4. Runtime    — on_next_step + StepOutcome factories; remove author UAEP unless bridge-only shim
5. Tests      — agents/<slug>/tests: await agent.run(AgentRunRequest); agent_os if applicable
6. Host       — manifest AgentBinding unchanged or updated; ACP-MIG-7 host test
7. Scoreboard — generate report; tier gate before merge
8. CI         — check_agent_fleet_migration.py + typed-state (allowlist -= agent)
```

| Step | ID | Tasks | Acceptance |
|------|-----|-------|------------|
| 8.1 | ACP-MIG-1 | `audit_agent_fleet_legacy.py` → `build/agent_fleet_inventory.json` | All packages listed; legacy flags accurate |
| 8.2 | ACP-MIG-2 | Migration tiers in plan + `agents/README.md` migration table | Operator can pick next batch from table |
| 8.3 | ACP-MIG-3 | **Pilot PR batch** (≤3 agents): echo, signoff_probe, research | 3 scoreboard reports; Runtime ≥80% each |
| 8.4 | ACP-MIG-4 | **Product PR batch**: legal, summary, LKW×3, DSW×4 (may split 2 PRs) | Host tests green; Runtime ≥80% |
| 8.5 | ACP-MIG-5 | Remaining agents + shrink typed-state allowlist to zero | `check_agent_typed_state.py` full roster |
| 8.6 | ACP-MIG-6 | CI regression gate on fleet | Re-introducing `get_steps`-only agent fails CI |
| 8.7 | ACP-MIG-7 | Post-batch host binding verification | legal + research + lab smoke per batch |
| 8.8 | ACP-LEG-2 | Close fleet migration — DEBT-ACP-16 | **Done** — 100% roster Runtime; `check_agent_production_readiness.py --require-fleet-migration-closure` |

**Wave 8 DoD:** No production agent on UAEP-only author path; fleet inventory clean; **ACP-LEG-2 Done**; scoreboard generated for every roster agent.

**Delivery rule:** One **batch PR** (ACP-MIG-3 or MIG-4) may migrate ≤5 agents; each agent row updated in [fleet tracker](#acp-fleet-migration-tracker) below.

##### ACP fleet migration tracker

| Agent | Tier | Host(s) | Status | Batch | Runtime % | Blocker |
|-------|------|---------|--------|-------|-----------|---------|
| echo | T0/T3 | lab, poc | **Done** | MIG-3 | — | — |
| signoff_probe | T0 | lab | **Done** | MIG-3 | — | — |
| research | T1 | research, lab | **Done** | MIG-3 | — | — |
| summary | T1 | research | **Done** | MIG-4 | — | — |
| legal | T2 | legal, lab | **Done** | MIG-4 | — | — |
| local_indexer | T2 | LKW | **Done** | MIG-4 | — | — |
| local_search | T1 | LKW | **Done** | MIG-4 | — | — |
| local_synthesizer | T2 | LKW | **Done** | MIG-4 | — | — |
| dispute_* (×4) | T2 | DSW | **Done** | MIG-4 | — | — |
| organization_worker | T4 | lab | **Done** | MIG-5 | — | — |
| intergrax_assistant | T4 | assistant | **Done** | MIG-5 | — | — |
| problem_radar | T4 | K.1 path | **Done** | MIG-5 | — | — |
| vendor_discovery | T4 | K.2 path | **Done** | MIG-5 | — | — |

*Update **Status** → In progress / Done per PR; **Runtime %** from ACP-PROD-12 report.*

---

#### Wave 6 — Routing, security, organizational policy (architecture §37.6–§37.7 · §39)

| Step | ID | Tasks | Acceptance | Cross-domain |
|------|-----|-------|------------|--------------|
| 6.1 | ACP-CON-6 | Nexus resolves `required_capability` → registry token; ban class name in task payload | **Done** — `capability_routing.py` + gate test | ORCHESTRATION · REG |
| 6.2 | ACP-CON-7 | `check_agent_step_security.py` — gateway-only I/O, STRICT widen deny | **Done** — static roster gate | UNIFIED_EXECUTION_RUNTIME |
| 6.3 | ACP-ORG-1..2 | `OrganizationalPolicyEnvelope` + merge context | **Done** — merge + host profile preset | TIER3 |
| 6.4 | ACP-ORG-3..4 | Kernel org overlays; `PolicyVerdictRecord` on trace | **Done** — channel deny + compliance_summary | UAEP policy |
| 6.5 | ACP-ORG-5 | Lab org fixture + compliance eval | **Done** — happy-path kernel gate | V-EVAL |

**Wave 6 DoD:** UC-11 path demonstrable; capability routing test passes.

---

#### Wave 7 — Production reliability gate (architecture §40)

**Blocks:** mutating / customer-facing agents until minimum **ACP-PROD-1..3** + **ACP-PROD-9..10** Done.

| Step | ID | Delivers §40 capability | Acceptance | Cross-domain |
|------|-----|-------------------------|------------|--------------|
| 7.1 | ACP-PROD-1 | Checkpoint / resume / replay | **Done** — store + host wiring + `test_acceptance_05c`/`05d` resume smoke | RELIABILITY |
| 7.2 | ACP-PROD-2 | Idempotency ledger | **Done** — ledger dedupe + replay skip + kernel execute/commit + host catalog invoker wiring | TOOLS |
| 7.3 | ACP-PROD-3 | `ToolExecutionProfile` + compensation | **Done** — mutating tool gate + step-failure compensation enqueue | TOOLS |
| 7.4 | ACP-PROD-4 | ReliabilityProfile in kernel | **Done** — circuit breaker + checkpoint interval | RELIABILITY |
| 7.5 | ACP-PROD-5 | SharedContext CAS | **Done** — per-key publish/CAS + graph test | ORCHESTRATION |
| 7.6 | ACP-PROD-6 | `ArtifactRef` | **Done** — `artifact_refs` on `AgentRunResult` | OBSERVABILITY |
| 7.7 | ACP-PROD-7..8 | Threat CI + privacy redaction | **Done** — threat gate + PII redaction | OBSERVABILITY · MEMORY |
| 7.8 | ACP-PROD-9..11 | Release gates + CI matrix + schema registry | **Done** — aggregate CI scripts | DX |
| 7.9 | ACP-PROD-12 | Production readiness scoreboard — aggregate §6.1az gates into one report | **Done** — `report_agent_production_readiness.py` · `check_agent_production_readiness.py` | DX · V-EVAL |

**Wave 7 DoD:** §40.12 checklist green for reference mutating agent; scoreboard emitted for roster; architecture §40 maturity gate **unblocks** prod roster promotion via scoreboard thresholds.

---

### 6.1az Agent Production Readiness Scoreboard (ACP-PROD-12)

**Purpose:** Single **operator-facing artifact** — replaces hunting across scattered CI scripts when deciding if an agent may enter **production roster** (`production_mode`, `production_eligible`).

**Artifact:** `AgentProductionReadinessReport` (typed, `extra=forbid`) — per agent, per generation run.

```text
AgentProductionReadinessReport:
    agent_id: str
    contract_id: str
    generated_at: datetime
    overall_pct: float                    # 0–100 weighted mean (see weights)
    production_eligible_recommendation: bool
    dimensions: list[AgentReadinessDimensionScore]

AgentReadinessDimensionScore:
    dimension: AgentReadinessDimension    # enum — 10 values below
    pct: float                            # 0–100
    status: pass | partial | fail | not_applicable
    weight: float                         # for overall_pct
    evidence: list[str]                   # test names, CI script ids, plan rows
    blockers: list[str]                   # human-readable gaps
```

| # | Dimension | Architecture / plan source | Scoring inputs (automated where possible) | Default weight |
|---|-----------|---------------------------|-------------------------------------------|----------------|
| 1 | **Contract** | §12 · ACP-CON-4 | Assembly resolver; schemas; validation_rules; failure_modes | 10% |
| 2 | **Runtime** | §13 · §32 · §32.0 · Wave 8 | `on_next_step`; StepOutcome factories; typed state; no UAEP author surface | 15% |
| 3 | **Policy** | §37.7 · §39 · ACP-ORG | PolicyVerdictRecord in trace; org envelope test; STRICT deny cases | 10% |
| 4 | **Observability** | §31 · ACP-OBS | AgentRunTrace on result; step records; trace_id join | 10% |
| 5 | **Checkpointing** | §40.1 · ACP-PROD-1 | Resume smoke; checkpoint store wired | 10% |
| 6 | **Idempotency** | §40.2 · ACP-PROD-2 | Mutating tools have idempotency_key; ledger dedupe test | 10% |
| 7 | **Security** | §40.7 · ACP-CON-7 · ACP-PROD-7 | Gateway-only I/O; threat matrix rows; vendor import CI | 10% |
| 8 | **Evaluation** | §40.9 · ACP-PROD-9 · V-EVAL | Golden/regression suites registered; staging green | 10% |
| 9 | **Lifecycle** | §20 · V-ALG · AS-2 | owner_team; runbook_ref; promotion evidence when prod-eligible | 5% |
| 10 | **Capability routing** | §37.6 · ACP-CON-6 | Task routes by capability token; binding resolves impl | 10% |

**Production roster promotion thresholds (normative — no compromise for mutating/customer-facing):**

| Profile | `overall_pct` | Per-dimension floor | Extra |
|---------|---------------|---------------------|-------|
| **Read-only staging** | ≥70% | Runtime ≥80% | Checkpointing/Idempotency may be `not_applicable` |
| **Mutating staging** | ≥80% | Checkpointing + Idempotency **100%** | ACP-CLOSE-PROD-1..4 · §40.12 reference |
| **Production roster** | **≥90%** | **No dimension below 80%** (except N/A) | ACP-PROD-9..10 green · §40.12 checklist |
| **Waiver** | — | — | ADR + operator sign-off only |

**Commands (target):**

```bash
uv run python scripts/report_agent_production_readiness.py --agent legal
uv run python scripts/report_agent_production_readiness.py --roster --format markdown
uv run python scripts/check_agent_production_readiness.py --min-overall 90 --fail-on-blockers
```

**Integration:** `check_agent_release_gates.py` (ACP-PROD-9) **consumes** scoreboard output — not duplicate logic. CI matrix §40.10 row **CI-16**: scoreboard generation on roster in gate workflow.

**Architecture canon:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §40.15.

---

### 6.1ax Suggested PR sequence (single-ID commits)

```text
Wave 0:  ACP-DX-1 → ACP-CON-1 → ACP-CON-4 → ACP-0 → ACP-DX-6 (0.4+0.5+0.7) → ACP-CON-2
Wave 1:  ACP-STEP-1 → ACP-STEP-2b → ACP-STEP-2 → ACP-CON-3   # kernel first, then glue wiring
Wave 2:  ACP-DX-2 → ACP-DX-3 → ACP-DX-4 → ACP-DX-5 → ACP-CFG
Wave 3:  ACP-OBS-1 → ACP-LLM-1 → ACP-STATE-1 → ACP-OBS-2
Wave 4:  ACP-STEP-3 → ACP-LEG-1 → ACP-LEG-3
Wave 5:  ACP-0b → ACP-1 → … → ACP-13 → ACP-8+LEG-4
Wave 8:  ACP-MIG-1 → ACP-MIG-2 → ACP-MIG-3 → ACP-MIG-4 → ACP-MIG-5 → ACP-MIG-6 → ACP-MIG-7 → ACP-LEG-2
Wave 6:  ACP-CON-6 → ACP-CON-7 → ACP-ORG-1 → … → ACP-ORG-5
Wave 7:  ACP-PROD-1 → … → ACP-PROD-11 → ACP-PROD-12
ACP-CLOSE:  DOC-2..4 → LEG-1 → LEG-2 → LEG-3 → PROD-1 → PROD-2 → PROD-3 → PROD-4 → PROD-5 → PROD-6 → PAT-1 (+ TOOL-ENG-6) → ORG-1 → PROD-7 → PROD-8 → PAT-2 → CI-1..3
ACP-FINISH:  TOK-1 → TOK-2 → TOK-3 → TOK-CI → FINISH-DOC-1
```

**Note:** ACP + ACP-CLOSE + **ACP-FINISH complete** (2026-06-13) — §25.4–§25.5 token depth closed.

**Journal:** ACP waves journaled; one entry on **ACP-CLOSE** phase completion, per [`implementation-journal/README.md`](../guides/implementation-journal/README.md).

---

### 6.1bb Harness implementation queue — ACP-CLOSE (done)

**Purpose:** Single ordered backlog for **full architecture ↔ implementation compliance** after ACP waves. **Band 2bb**. **Closed 2026-06-11.**

| Order | ID | Priority | Arch § | Status |
|-------|-----|----------|--------|--------|
| 1 | ACP-CLOSE-DOC-2 | P0 | §28.3 | **Done** |
| 2 | ACP-CLOSE-DOC-3 | P0 | §36.4 · §40.13 | **Done** |
| 3 | ACP-CLOSE-DOC-4 | P1 | audit | **Done** |
| 4 | ACP-CLOSE-LEG-1 | **P0** | §13.5 | **Done** |
| 5 | ACP-CLOSE-LEG-2 | **P0** | §13.4 | **Done** |
| 6 | ACP-CLOSE-LEG-3 | P1 | §13.5 | **Done** |
| 7 | ACP-CLOSE-LEG-4 | P2 | §45 | **Done** |
| 8 | ACP-CLOSE-PROD-1 | **P0** | §40.1 | **Done** |
| 9 | ACP-CLOSE-PROD-2 | **P0** | §40.1.4 | **Done** |
| 10 | ACP-CLOSE-PROD-3 | P1 | §32.8 | **Done** |
| 11 | ACP-CLOSE-PROD-4 | **P0** | §27 · §40.12 | **Done** |
| 12 | ACP-CLOSE-PROD-5 | P1 | §40.3.3 | **Done** |
| 13 | ACP-CLOSE-PROD-6 | P1 | §40.2.2 | **Done** |
| 14 | ACP-CLOSE-PAT-1 + TOOL-ENG-6 | P1 | §26.3 | **Done** |
| 15 | ACP-CLOSE-ORG-1 | P1 | §39.4 | **Done** |
| 16 | ACP-CLOSE-PROD-7 | **P0** | §40.12 | **Done** |
| 17 | ACP-CLOSE-PROD-8 | **P0** | §40.15 | **Done** |
| 18 | ACP-CLOSE-PAT-2 | P2 | §26.6 | **Done** |
| 19 | ACP-CLOSE-PAT-3 | P2 | §28.3 | **Done** |
| 20 | ACP-CLOSE-ORG-2 | P2 | §39.5 | **Done** |
| 21 | ACP-CLOSE-CI-1 | P1 | §40.10 | **Done** |
| 22 | ACP-CLOSE-CI-2 | P2 | §28.4 | **Done** |
| 23 | ACP-CLOSE-CI-3 | P1 | §40.15 | **Done** |

**Parallel (owning plan):** AUDIT-IDEAL-19.1 · 20.1 · 31.1 — **Done** (2026-06-13); see [Phase AUDIT-IDEAL](#phase-audit-ideal--ideal-architecture-gap-register-2026-06-09).

**Minimum viable close (P0 only):** DOC-1..4 · LEG-1 · LEG-2 · PROD-1 · PROD-2 · PROD-4 · PROD-7 · PROD-8 · CI-2 → **Done** — ACP-CLOSE wave complete.

---

### 6.1bc Harness implementation queue — ACP-FINISH (closed)

**Purpose:** Final tasks to declare **agent architecture (§13–§40) implementation-complete**. **Band 2bc**.  
**Status:** **Done** (2026-06-13) — GAP-ACP-36 · GAP-ACP-37 **Closed** · architecture §28.3 open count → **0**.

| Order | ID | Priority | Arch § | Status | Depends |
|-------|-----|----------|--------|--------|---------|
| 1 | ACP-TOK-1 | **P1** | §25.4 · §33.4 | **Done** | — |
| 2 | ACP-TOK-2 | **P1** | §25.5.1–§25.5.2 | **Done** | TOK-1 |
| 3 | ACP-TOK-3 | **P1** | §25.5.3 · §30.8 | **Done** | TOK-2 |
| 4 | ACP-TOK-CI | P2 | §40.10 CI-18 | **Done** | TOK-1..3 |
| 5 | ACP-FINISH-DOC-1 | P1 | §28.3 · §40.13 | **Done** (2026-06-13) | TOK-CI |

**Parallel (ideal-architecture depth — not blocking ACP-FINISH DoD):**

| ID | Priority | Arch § | Status | Notes |
|----|----------|--------|--------|-------|
| AUDIT-IDEAL-19.1 | **P0** | §15 Registry | **Done** | `registry_snapshot_store.py` · `check_registry_snapshot_diff.py` |
| AUDIT-IDEAL-20.1 | P1 | §19 Cap. graph | **Done** | `phase_v_capability_graph_guard.py` · `check_capability_graph_strict_deploy.py` |
| AUDIT-IDEAL-31.1 | P1 | §20 Lifecycle | **Done** | `check_agents_lifecycle_metadata.py` · `check_on_call_ownership_model.py` |

**Suggested PR order:**

```text
ACP-TOK-1 (metering) → ACP-TOK-2 (limits) → ACP-TOK-3 (reactions + reference host)
  → ACP-TOK-CI → ACP-FINISH-DOC-1
```

**Journal:** [`entries/2026-06-13/acp-finish-doc-1-gap-register-closeout.md`](../guides/implementation-journal/entries/2026-06-13/acp-finish-doc-1-gap-register-closeout.md) (ACP-FINISH phase completion).

---

### 6.1bd Layer backlog — post-maturity (P2–P4, non-blocking)

**Status:** Active maintenance — does **not** block layer completion (ACP + AUDIT-IDEAL **Done**).

| ID | Priority | Topic | Status | Notes |
|----|----------|-------|--------|-------|
| COST-1 | P2 | Nexus `RunBudget` graph env cap | Partial | Per-agent ACP-TOK enforcement **Done**; graph-level cap deferred |
| ROSTER-PROD | P2 | Individual agent `production_mode` promotion | Ongoing | §40.15 thresholds per agent; platform gates **Done** |
| FAUDIT-REG.1 | P2 | Extend `HarnessRegistrySnapshot` with eval registry depth | Planned | `PLATFORM_FOUNDATION` master register |

---

### 6.1l Harness implementation queue — registry architecture closeout (closed)

**Purpose:** Single ordered list for **Phase REG** (Band 2r). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REG-DOC.1** | Docs | **Done** | Appendix O + cross-refs | Author map complete |
| 2 | **REG-1** | Code | **Done** | `HarnessRegistrySnapshot` + `registry_wiring` | `test_registry_wiring.py` |
| 3 | **REG-2** | Code | **Done** | `registry_assembly_resolver` wire | `test_registry_wiring.py` |
| 4 | **REG-3** | CI | **Done** | `check_harness_registry_resolution.py` | CI green |

**Suggested PR order (complete):** REG-DOC.1 → REG-1 → REG-2 → REG-3.### 6.1m Harness implementation queue — capability graph closeout (closed)

**Purpose:** Single ordered list for **Phase CG** (Band 2s). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CG-DOC.1** | Docs | **Done** | Appendix P + cross-refs | Author map complete |
| 2 | **CG-1** | Code | **Done** | `capability_graph_wiring` | `test_capability_graph_wiring.py` |
| 3 | **CG-2** | Code | **Done** | `capability_graph_assembly_resolver` | wire-time validation tests |
| 4 | **CG-3** | CI | **Done** | `check_harness_capability_graph_wiring.py` | CI green |

**Suggested PR order (complete):** CG-DOC.1 → CG-1 → CG-2 → CG-3.

---

### 6.2bj Phase CG execution order (Band 2s — closed 2026-06-02)

**Status:** **Done** · register: [Phase CG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · queue: [§6.1m](#61m-harness-implementation-queue--capability-graph-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CG-DOC.1 | Appendix P + plan sync | High |
| 2 | CG-1 | `capability_graph_wiring` | Critical |
| 3 | CG-2 | `capability_graph_assembly_resolver` | High |
| 4 | CG-3 | `check_harness_capability_graph_wiring.py` | Medium |### 6.2bi Phase REG execution order (Band 2r — closed 2026-06-02)

**Status:** **Done** · register: [Phase REG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · queue: [§6.1l](#61l-harness-implementation-queue--registry-architecture-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REG-DOC.1 | Appendix O + plan sync | High |
| 2 | REG-1 | `HarnessRegistrySnapshot` + `registry_wiring` | Critical |
| 3 | REG-2 | `registry_assembly_resolver` | High |
| 4 | REG-3 | `check_harness_registry_resolution.py` | Medium |

---

### 6.2bg Phase AS execution order (Band 2q — closed 2026-06-02)

**Status:** **Done** · register: [Phase AS](plan/ORCHESTRATION.md) · queue: [§6.1k](#61k-harness-implementation-queue--agent-assembly-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | AS-DOC.1 | Appendix N + plan sync | High |
| 2 | AS-1 | `agent_assembly_resolver` | Critical |
| 3 | AS-2 | Lifecycle state on `AgentContract` | High |
| 4 | AS-3 | `skill_ids` resolution audit script | Medium |### 6.2bh Phase CLEAN execution order (closed 2026-06-02)

**Status:** **Done** · register: [Phase CLEAN](plan/ORCHESTRATION.md) · queue: [§6.1j](#61j-harness-implementation-queue--legacy-module-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CLEAN-1 | Remove `chat_router.py` | Critical |
| 2 | CLEAN-2 | Remove `tools_agent.py` | Critical |
| 3 | CLEAN-3 | `check_legacy_modules_removed.py` in CI | High |
| 4 | CLEAN-4 | Docs sync | Low |

---

## Phase AS — Agent assembly control plane closeout

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

---

## Phase PE — Prompt registry control plane closeout

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

---

## Phase REG — Registry architecture control plane closeout

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

---

## Phase CG — Capability graph control plane closeout

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

---

### Phase L — Agent OS Certification

**Directive:** L1 certification recorded in Appendix A. K.1/K.2 are **Phase K product work** — **last** in the plan (§6.3), not concurrent with harness bands 1–2.  
**Agent workflow:** [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md)

| # | Deliverable | Status | Req | Notes |
|---|-------------|--------|-----|-------|
| L.1 | UAEP-first agent scaffold | **Done** | R2 | `python -m intergrax.scaffold new-agent` |
| L.2 | Agent creation guide | **Done** | R2 | Single canonical how-to |
| L.3 | Lab application (Tier-3) | **Done** | R1 | `applications/lab_application/` |
| L.4 | Reference technical agents | **Done** | R5 | Echo + `agents/lab/mock_agents.py` |
| L.5 | Agent OS acceptance suite | **Done** | R1 | `tests/acceptance/agent_os/` (+ `05b` mid-step UAEP) |
| L.6 | Runtime independence verification | **Done** | R5 | Register + run without Nexus edits |
| L.7 | Application composition verification | **Done** | R5 | Agents ≠ applications |
| L.8 | Certification checklist | **Done** | R1 | Appendix A (this file) |
| L.9 | **Sign-off exercise** | **Done** | — | `agents/signoff_probe/` — Appendix A record |

**Acceptance tests (L.5):**

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
```

| # | Scenario | Test |
|---|----------|------|
| 1 | Single agent | `test_acceptance_01_single_agent_execution` |
| 2 | Sequential multi-agent | `test_acceptance_02_sequential_multi_agent` |
| 3 | Parallel multi-agent | `test_acceptance_03_parallel_multi_agent` |
| 4 | HITL approve/resume | `test_acceptance_04_human_approval_flow` |
| 5 | Checkpoint recovery | `test_acceptance_05_checkpoint_recovery` · `test_acceptance_05c` · `test_acceptance_05d` |
| 6 | Retry / alternate agent | `test_acceptance_06_retry_flow` |
| 7 | Partial results | `test_acceptance_07_partial_results` |
| 8 | Memory / shared context | `test_acceptance_08_memory_handoff` |
| 9 | Sandbox tools | `test_acceptance_09_sandbox_tool_execution` |
| 10 | Shadow workspace | `test_acceptance_10_shadow_workspace` |

---

---

#### V-ALG — Agent Lifecycle Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-ALG.1 | Agent certification gate contract (quality/policy/security) | **Done** | **Critical** | Certification criteria codified + tested |
| V-ALG.2 | Promotion flow (dev -> staging -> production) with evidence | **Done** | High | Promotion requires evidence bundle |
| V-ALG.3 | Deprecation + retirement workflow and migration window policy | **Done** | High | `AgentRegistry` / `AgentRouter` filter retired/deprecated via `agent_routing_policy.py` |
| V-ALG.4 | Owner/on-call metadata required for production-eligible agents | **Done** | High | Production-mode ownership gate enforced at selection |#### V-CE — Context Quality and Regression Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-CE.1 | Relevance/freshness/confidence scoring in context assembly | **Done** | High | Scores emitted in trace/runtime events |
| V-CE.2 | Duplicate suppression + context quality thresholds | **Done** | Medium | Threshold policy test coverage |
| V-CE.3 | Context regression benchmark suite | **Done** | High | CI regression baseline stored and compared |
| V-CE.4 | Retrieval effectiveness evaluation (precision/recall@k style) | **Done** | Medium | Bench report in evaluation registry |

---

#### V-PE — Prompt Engineering Architecture

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-PE.1 | Prompt registry governance contract (owner/version/risk metadata) | **Done** | High | `PromptMeta` extended; `harness_capability_summary` reference prompt; registry governance validation |
| V-PE.2 | Prompt composition model (system/task/policy/context layers) | **Done** | High | Canon + reference implementation path |
| V-PE.3 | Deterministic policy injection overlays | **Done** | High | Prompt build trace shows overlays |
| V-PE.4 | Prompt regression/adversarial test suite | **Done** | Medium | Gate includes prompt regression subset |#### V-EVAL — Evaluation and Benchmarking Operations

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-EVAL.1 | Unified evaluation modes: offline/online/shadow/human | **Done** | **Critical** | Mode contracts documented + wired |
| V-EVAL.2 | Golden datasets + scenario libraries + regression suites | **Done** (typed asset bundle contracts) | High | Versioned benchmark assets |
| V-EVAL.3 | Automated evaluators (rule-based + LLM judge) | **Done** | High | Evaluator outputs persisted |
| V-EVAL.4 | Evaluation registry trend/comparison reports | **Done** | High | Report artifact required for major releases |

---

## Phase ACP-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — formal Full Harness LC closeout; no open P0/P1 in domain scope  
**Prerequisites:** Phase ACP + ACP-CLOSE + ACP-FINISH + AUDIT-IDEAL **Done**  
**Goal:** Reconcile audit prompt + ACP-INV-02 canon; confirm gates green; journal + progress tracker  
**ADR:** **No ADR needed** — documentation and process closeout only

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ACP-LC-S1 | **Audit prompt sync** — AUDIT-IDEAL rows Done in known gaps | **Done** | High | `docs/guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | No Planned drift |
| ACP-LC-S2 | **ACP-INV-02 canon** — remove stale „until ACP-LEG” wording | **Done** | High | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §21 | Matches ACP-CLOSE-LEG-5 |
| ACP-LC-S3 | **Gate verification** — `check_agent_acp_close_ci.py` green | **Done** | High | `scripts/check_agent_acp_close_ci.py` | Fleet 17/17 · migration complete |
| ACP-LC-S4 | **Full Harness LC journal** + `layer_completion_progress.json` | **Done** | High | implementation-journal | mature status |

**Deferred P2 (not blocking LC):** `boundary_demo` ReflexAgent migration · COST-1 graph RunBudget cap · FAUDIT-REG.1

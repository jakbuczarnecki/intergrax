# Intergrax — Runtime Implementation Plan

**The single implementation map** — phases, status, gaps, priority, and readiness checklist.

Status: Working draft (2026-06-01, Phase Q Harness Quality — audit remediation)  
Architecture canon: [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)  
Agent workflow: [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md)  
Navigation: [`README.md`](README.md)  

Principle: **evolve, not rewrite** · **reuse Tier-0** (canon §5.2)

---

## Documentation model

Do not maintain separate status/readiness/roadmap files. This plan is the **only** live implementation document:

| Topic | Where |
|-------|--------|
| Full architecture specification | `intergrax_runtime_architecture.md` |
| Phase status, gaps, priority | **This file** |
| Tier-0 integration catalog (what / where) | Architecture canon §7.1.1–§7.1.5 |
| Tier-0 integration implementation (how) | **This file** Phase M |
| Tier-0 tool catalog (what / where) | Architecture canon §7.1.6–§7.1.7, §22 |
| Tier-0 tool implementation (how) | **This file** Phase O |
| Agent creation workflow | `AGENT_CREATION_GUIDE.md` |
| Tier-3 application environment (self-contained deploy) | Architecture canon §7.4.8–§7.4.10 |
| Tier-3 composition engine (manifest, wiring API) | [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) |
| Tier-3 application hosts (`applications/<app>/`) | [`applications/USAGE.md`](../applications/USAGE.md) |
| Application scaffold & deploy plan | **This file** Phase N |
| Business-agent go/no-go checklist | **Appendix A** (below) |
| Technical debt backlog (analysis only) | **Appendix B** (below) |
| Harness quality audit → Phase Q tracker | **This file** Phase Q + **Appendix C** |
| Harness GA / consolidation (no new OS features) | **This file** Phase Q |

---

## 0. Architecture at a glance

Condensed from the canon. For full contracts and forbidden patterns, read `intergrax_runtime_architecture.md`.

### 0.1 Strategic objective

Intergrax is an **Agent Operating System / Harness AI runtime** — not a collection of business agents.

Current optimization targets:

- experimentation speed · agent creation speed · runtime stability
- orchestration quality · observability · composability

**Harness GA (Phase L):** Agent OS certified — Appendix A **20/20**. New agents ship via scaffold without Nexus edits.

**Product agents (Phase K):** Problem Radar, Vendor Discovery, Legal expansion — **product decision**, not a runtime gate. May run **in parallel** with Phase Q.

**Platform quality (Phase Q):** Mandatory consolidation from harness implementation audit (2026-06-01) — bugs, debt, monoliths, dual-path RAG, observability gaps. **Execute before scaling business agents** unless a K item is explicitly prioritized.

### 0.6 When Tier-1 (Nexus) changes are required

**Default (Tier-2 agent):** register + `AgentContract` + UAEP steps — **no** edits to `intergrax/runtime/`.

**Extend Tier-1** only when the need is **reusable across many future agents**, not one product:

| Situation | Action |
|-----------|--------|
| New agent with existing capabilities, memory, graph, HITL, sandbox | **Tier-2 only** — `agents/<slug>/` |
| New capability id, prompts, domain tools | **Tier-2** (+ Tier-0 adapter if new external integration) |
| New orchestration primitive (e.g. new graph node type, new lifecycle state) | **Tier-1** — must serve multiple agents; update canon §42 first |
| New platform concern (new store, queue, notification channel) | **Tier-0** — `intergrax/` shared module |
| Agent-specific product wiring (routes, env, which agents active) | **Tier-3** — `applications/<product>/` |
| One agent needs special-case branch in `NexusLoop` | **Anti-pattern** — refactor to contract/metadata or Tier-0 |

If the answer to “will another agent need this?” is **no**, it does not belong in Nexus.

### 0.2 Four tiers

| Tier | Folder | Role | Analogy |
|------|--------|------|---------|
| **Tier-0** | `intergrax/` | Platform — LLM, storage, queues, logging, adapters | Kernel drivers |
| **Tier-1** | `intergrax/runtime/` | **Nexus Agent OS** — orchestration, lifecycle, trace, memory, HITL | Operating system |
| **Tier-2** | `agents/` | Reusable agent capabilities — domain logic, prompts, tools | Applications |
| **Tier-3** | `applications/` | Self-contained execution environments — env, Docker, wiring, routes | Deployable product/lab host |

### 0.3 Execution path

```text
HTTP / CLI / Worker
    → Tier-3 Application (optional)
    → UnifiedTaskRunner
    → NexusLoop (Tier-1)
    → AgentEngine / UAEP
    → Tier-2 Agent (get_steps → run_step → decide_after_step)
    → ToolRuntime / MemoryView / Validation
    → Trace + RuntimeEvents + TaskResult
```

### 0.4 Agent OS rule

New agents integrate via **`AgentRegistry.register()`** — never by editing `NexusLoop`, `GraphExecutor`, or task lifecycle code.

### 0.5 Maturity dashboard

| Scope | Score | Blocks new agent? | Notes |
|-------|-------|-------------------|-------|
| **Harness GA (functional)** | **Done** | **No** | L certified; scaffold + lab + gate |
| **Harness quality (Phase Q)** | **In progress** (Waves 1–3) | **No** (recommended before K scale) | See Appendix C — Wave 1–3 landed 2026-06-01 |
| Canon §1–41 (tiers, Nexus, graph, repo split) | **~88–92%** → target **≥95%** post-Q | No | Q-N, Q-R, Q-X |
| §42 Unified Execution Runtime | **~92–95%** → target **≥98%** post-Q-N.5–Q-N.6 | No | Optional hooks + trace persist |
| Laboratory workflow | **~95%** → target **≥98%** post-Q-O | No | RAG metrics bootstrap |
| Agent OS certification (Phase L) | **Done** | No | Appendix A |
| Regression gate | **397 passed** | No | Must stay green after each Q.* PR |

---



## 1. Plan Objective



Transform Intergrax into an **internal agent experimentation laboratory** (§2, §35) aligned with the canonical architecture:



```text

hypothesis → capability → contract → registration → Nexus → trace → evaluation → decision

```



**Success metric:** time from idea to first running experiment **< 1 hour**.



**Current alignment:**



| Scope | Score | Notes |

|-------|-------|-------|

| Architecture §1–41 (tiers, Nexus, graph, repo split) | **~88–92%** | Phases A–F; typed task contract |

| §42 Unified Execution Runtime | **~92–95%** | Hooks B.06 + product Slack/Teams intake on legal/research (2026-05-27) |
| Laboratory workflow (inspect, decide) | **~95%** | D.1–D.5 done |
| Pre-P4.2 regression gate | **Done** | **397 tests**, marker `gate` |



---



## 2. Map: Architecture → Implementation Status



| Section | Requirement | Status | Location |

|---------|-------------|--------|----------|

| §5.1 Four tiers | Tier-0..3 model | **Done** | architecture doc + `agent_kit/tiers.py` |

| §5.2 Reuse Tier-0 | No redundant platform | **Doc + process** | §5.2, §8.8, §39.8 |

| §9.1 Nexus Loop | Global orchestration | **Done** | `nexus_loop.py` |

| §9.2 Local agent loop | Bounded UAEP steps | **Done** | Echo, Research, Legal `thin_steps` / `dynamic_steps` |

| §12–16 Contracts / Registry | AgentContract, capabilities | **Done** | `intergrax/contracts/`, `runtime/registry/` |

| §22 ToolRuntime | Policy gateway | **Done** | `tool_runtime.py`, `ToolAccessPolicy` |

| §23 Task lifecycle | States + trace + typed contract | **Done** | `task/`, `task_contract.py`, `TaskContextAssemblyOptions`, `task_metadata_bridge.py` |

| §24–25 Execution graph | Multi-agent | **Done** | `execution/`, `GraphExecutor` |

| §29 Validation | Nexus + agent | **Done** | `NexusValidationEngine` |

| §31 Retry | Runtime-managed | **Done** | `RetryEngine` |

| §33 Observability | Trace + events | **Done** (lab scope) | Trace + runtime events + metrics export (`B.08`–`B.11`); OTel provider beta |

| §42 Execution runtime | UAEP, hooks, governance, tool gateway | **Partial (~92–95%)** → **Phase Q-N.5–Q-N.6** | Tool + agent-selection hooks; Q closes decision/interrupt/retry + trace persist |
| §19 Debug / experiments | CLI, API, registry, cost | **Done** | D.1–D.5 ✅ |

| §7.4 Repo split | agents / applications | **Done** | `agents/legal`, `applications/legal_application` |
| §7.1 Integration Library | Catalog + contracts + providers | **Done** (beta) | Phase M core + M.6 P1/P2/P3; on-demand slugs only |

| §19 Debug surface | CLI / API | **Done** | D.1 CLI + D.2 API ✅ |

| §32 HITL | Approval / reject / escalate | **Done** | F.3 + `runtime/human/` |

| §26 Long-running tasks | Checkpoint / resume | **Done** (baseline) | Scheduler + partial results API + UAEP mid-step (`B.01`, `B.02`) |
| §18 Slack / Teams | Interaction adapters | **Done** (product baseline) | Outbound + `POST /v1/interactions/intake` on lab/legal/research/poc; verifier via env |
| §27 Memory model | Bounded task / agent memory | **Done** | I.1–I.5: TaskMemory, MemoryView, SharedTaskContext, handoff, ContextManager v2 |
| §42.9 Pause / Resume | `RuntimeCheckpoint` | **Done** (baseline) | HITL pause + full snapshot (`plan_snapshot`, `graph_snapshot`, UAEP cursor) |
| §41 Unified entry | Single run lifecycle | **Done** (lab scope) | `UnifiedTaskRunner` on all Tier-3 hosts; legacy `AgentEngine` opt-out removed |

| §20–21 Shadow / Sandbox | Isolated exec | **Done** | F.1 ShadowWorkspace + F.2 SandboxRuntime ✅ |



---



## 3. Implementation Phases



### Phase A — Foundation Stabilization



| # | Deliverable | Status |

|---|-------------|--------|

| A.1 | Unified run lifecycle | **Done** |

| A.2 | Task trace persistence | **Done** |

| A.3 | NexusLoop production path | **Done** |

| A.4 | EvalRunner integration | **Done** |

| A.5-min | Pre-P4.2 regression gate | **Done** |

| A.5 | Full regression suite (Legal E2E, all steps) | **Deferred** |

| A.6 | Shim cleanup | **Done** | Removed `applications/legal_agent/`; docs + duplicate `legal_application/tests/` cleaned |



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



### Phase B — Extended Nexus



| # | Deliverable | Status |

|---|-------------|--------|

| B.1–B.7 | Classifier, planner, validation, retry, tool policy, composer | **Done** |



---



### Phase C — Multi-Agent Readiness



| # | Deliverable | Status |

|---|-------------|--------|

| C.1–C.6 | ExecutionGraph, GraphExecutor, ContextManager, Research pipeline | **Done** |



---



### Phase D — Observability and Experiments



**Goal:** §19, §35 — laboratory tooling (not SaaS UI).



| # | Deliverable | Status | Notes |

|---|-------------|--------|-------|

| D.0 | §42 P4.1 Event Bus wiring | **Done** | `RuntimeEventBus`, `trace_bridge`, NexusLoop |

| D.1 | Debug CLI | **Done** | `python -m intergrax.debug tasks list\|show\|trace` |

| D.2 | Minimal debug API | **Done** | FastAPI `GET /debug/tasks` on trace store |

| D.3 | Experiment registry | **Done** | SQLite registry; CLI + `GET/POST /debug/experiments` |

| D.4 | Notebook templates | **Done** | `notebooks/experiments/`, `experiments/workflow.py` |

| D.5 | Cost in trace | **Done** | `AgentExecutionResult.cost` from LLM usage / runtime stats |



---



### Phase E — Legal Agent Refactoring (parallel)



| # | Deliverable | Status |

|---|-------------|--------|

| E.1 | Thin sequential Legal — domain steps as UAEP `AgentStep` list | **Done** |

| E.2 | ToolRuntime via gateway (no direct Nexus step imports in bridge) | **Done** (P4.4) |

| E.3 | Governance on UAEP decision path | **Done** (P4.3) |

| E.4 | Thin dynamic Legal (`LegalDynamicPipeline` routing) | **Done** |



**E.4 delivered (2026-05-27):** `agents/legal/uaep/dynamic_steps.py` — 5 UAEP macro-steps (setup → tool plan → route → waves → finalize); `legal_execution_loop` phase functions extracted. Gate: 34 tests.



**E.1 delivered (2026-05-27):** `agents/legal/uaep/thin_steps.py` — 8 UAEP steps (setup → finalize); `LegalAnalysisPipeline` reuses same runners; dynamic mode keeps single pipeline boundary. Gate: 33 tests.



---



### Phase F — Advanced / On-Demand

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| F.1 | ShadowWorkspace | **Done** | `runtime/workspace/`; UAEP + NexusLoop integration |
| F.2 | SandboxRuntime | **Done** | `runtime/sandbox/`; `sandbox.exec` via BoundToolGateway |
| F.3 | Advanced HITL (reject/escalation store) | **Done** | `runtime/human/` store + NexusLoop reject/escalate |
| F.4 | Long-running tasks / Slack-Teams | **Done (partial)** | Checkpoints ✅; Slack/Teams = notification stub only |

| F.5 | Typed task contract | **Done** | `TaskExecutionOptions`, `TaskRuntimeState`, `TaskResultSummary`, bridge |

Long-running **full** §26 (scheduler, UAEP mid-step) and Slack/Teams **full** §18 — see Phase G–H below.



---



### Phase P4 — §42 Unified Execution Runtime



| Step | Deliverable | Status |

|------|-------------|--------|

| P4.1 | Event bus + trace bridge | **Done** |

| P4.2 | UAEP in AgentEngine | **Done** |

| P4.3 | Governance (interrupt, HITL) | **Done** |

| P4.4 | Tool gateway unification | **Done** |

| P4.5 | Agent migration (Echo, Research, Legal) | **Done** |



**P4.5 delivered (2026-05-27):** `uaep_pipeline.py`; Research, Summary, Legal agents on UAEP (`get_steps` / `run_step` / `decide_after_step`); integration tests + NexusLoop research. Gate: 31 tests.



**P4.4 delivered (2026-05-27):** `RuntimeToolGateway`, `ToolRuntime.invoke_request`, Legal bridge via `ToolRequest`; UAEP `BoundToolGateway`. Gate: 25 tests.



**P4.3 delivered (2026-05-27):** `runtime/interrupts/`, `runtime/human/`, policy in UAEP + NexusLoop.



---

### Phase G — §42 Runtime Convergence

**Goal:** Close largest gaps vs §42.9, §42.10, §42.24, §42.40 (evolve, not rewrite).

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| G.1 | `RuntimeCheckpoint` contract | **Done** | §42.9.2 | Plan + graph node states + UAEP step index |
| G.2 | UAEP mid-execution resume | **Done** | §42.9.3 | Skip re-run paused step on resume |
| G.3 | HITL middleware hooks | **Done** | §42.10 | `BEFORE/AFTER_HUMAN_APPROVAL` in NexusLoop |
| G.4 | `HumanRequest` v2 fields | **Done** | §42.10.1 | Typed urgency, deadline propagation, timeout stub |
| G.5 | RuntimeEvent-first observability | **Done** | §42.24 | `RuntimeEventPersistence` + `store.py` (`open_runtime_event_store`, env `INTERGRAX_RUNTIME_EVENTS_DB` only) |
| G.6 | Debug API: HITL + checkpoints | **Done** | §19 | Pluggable stores; events/checkpoints/HITL resume |
| G.7 | Graph failure recovery | **Done** | §42.40, §30 | Skip completed nodes; checkpoint on graph fail |
| G.8 | Cooperative cancellation | **Done** | §42.26 | Cancel propagation through graph / UAEP |

---

### Phase H — Interaction Surfaces (§18)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| H.1 | Outbound webhook delivery | **Done** | §18 | Pluggable delivery + formatters; HTTP opt-in |
| H.2 | `InteractionAdapter` protocol | **Done** | §18 | Inbound → normalized `Task` |
| H.3 | Slack inbound lab path | **Done** | §18 | Debug API intake + signature stub |
| H.4 | HITL notification templates | **Done** | §42.10 | Reusable template + `notify_hitl_pause`; Slack/Teams formatters |
| H.5 | Teams parity | **Done** | §18 | Activity parser + HMAC verifier + debug intake tests |
| H.6 | Organization Worker demo | **Done** | §38 | E2E lab: intake → HITL → notification → resume |

---

### Phase I — Memory & Context (§27–28)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| I.1 | `TaskMemory` store | **Done** | §27 | Contract + coordinator; `store.py` (`open_task_memory_store`, env `INTERGRAX_TASK_MEMORY_DB` only) |
| I.2 | `MemoryView` gateway | **Done** | §42.35 | `PolicyScopedMemoryView` + UAEP wiring + `MEMORY_*` events |
| I.3 | `SharedTaskContext` | **Done** | §42.14 | Contract + `ContextManager` + graph merge + memory bridge |
| I.4 | Agent handoff | **Done** | §42.15 | `AgentHandoff` + `HandoffCoordinator` + graph path + `HANDOFF_*` events |
| I.5 | ContextManager v2 | **Done** | §28 | Provenance + summary tiers + `TaskContextAssemblyOptions` on `TaskExecutionOptions.context` |

---

### Phase J — Unified Execution Entry (§41)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| J.1 | NexusLoop default in apps | **Done** | §41 | Legal + Research: `UnifiedTaskRunner` only (legacy `AgentEngine` removed, B.14) |
| J.2 | RunService → UnifiedTaskRunner | **Done** | §41 | `NexusTaskExecutionAdapter` + `CreateRunRequest.payload` → Task |
| J.3 | Worker queue Task v2 | **Done** | §41 | `QueuedNexusExecutionAdapter`, `nexus.task.v2` Celery handler, checkpoint resume |
| J.4 | Long-running scheduler | **Done** | §26 | `LongRunningScheduler`, delayed resume + HITL timeout enforcement |
| J.5 | Partial results API | **Done** | §26 | `GET /debug/tasks/{id}/progress`, `TASK_PROGRESS` events, notification template |

---

### Phase K — Hardening & Reference Agents

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| K.1 | Problem Radar prototype | **Blocked** | §36 | After Phase L sign-off |
| K.2 | Vendor Discovery prototype | **Blocked** | §37 | After Phase L sign-off |
| K.3 | Policy engine facade | **Done** | §42.11 | `PolicyEngine` + `coerce_replay_policy_engine`; `ExecutionGuard` uses `evaluate_replay` (2026-05-27) |
| K.4 | Dual `AgentDecision` cleanup | **Done** | §42.7 | `ToolPlanDecision`; deprecated `tools_agent.AgentDecision` alias (2026-05-27) |
| K.5 | ChatAgent / legacy removal | **Done** | §39 | Production paths use Nexus only; `check_production_chat_agent_imports.py` gate (2026-05-27) |
| K.6 | A.5 full Legal E2E gate | **Deferred** | — | Real LLM; not blocking lab — product/CI decision |

---

### Phase L — Agent OS Certification

**Directive:** L1 certification recorded in Appendix A. Phase K is a **product** decision, not a runtime gate.  
**Agent workflow:** [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md)

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
| 5 | Checkpoint recovery | `test_acceptance_05_checkpoint_recovery` |
| 6 | Retry / alternate agent | `test_acceptance_06_retry_flow` |
| 7 | Partial results | `test_acceptance_07_partial_results` |
| 8 | Memory / shared context | `test_acceptance_08_memory_handoff` |
| 9 | Sandbox tools | `test_acceptance_09_sandbox_tool_execution` |
| 10 | Shadow workspace | `test_acceptance_10_shadow_workspace` |

---

### Phase M — Integration Library (Tier-0 Catalog)

**Canon:** §7.1.1–§7.1.5  
**Goal:** One discoverable integration catalog so platform teams ship adapters and agent teams compose them in Tier-3 — without duplicating Redis/Postgres/Slack clients per agent.

**Principle:** evolve existing modules (`queueing/`, `distributed/`, `websearch/`, …) into catalog providers; do not fork parallel stacks.

**Out of scope:** `intergrax/llm_adapters/` — LLM providers are **not** part of the Integration Library (§7.1.2).

### Phase M-LLM — LLM Adapter Layer (Tier-0)

**Canon:** §5.2.2 · **Doc:** [LLM_ADAPTERS.md](LLM_ADAPTERS.md)  
**Goal:** One `LLMAdapter` contract, lazy registry, streaming + native tools + structured output across commercial and self-hosted providers.

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-LLM.1 | Shared `_shared/` (messages, tools, retry, conformance) | **Done** | 2026-05-30 |
| M-LLM.2 | Seven core providers hardened | **Done** | OpenAI, Claude, Azure, Gemini, Mistral, Bedrock, Ollama |
| M-LLM.3 | Groq + vLLM (OpenAI-compatible) | **Done** | `openai_compat_providers.py` |
| M-LLM.4 | Bedrock Converse + tools + stream | **Done** | `INTERGRAX_BEDROCK_USE_CONVERSE`, `converse_stream` |
| M-LLM.5 | Conformance tests in CI gate | **Done** | `tests/unit/llm_adapters/` |
| M-LLM.6 | `LLM_ADAPTERS.md` + README section | **Done** | 19 providers |
| M-LLM.7 | OpenAI-compat expansion + Vertex + `LLMProfile` | **Done** | Together, Fireworks, OpenRouter, DeepSeek, xAI, llama.cpp, Cohere, Vertex |
| M-LLM.8 | Optional network smoke workflow | **Done** | Weekly schedule + `workflow_dispatch` |
| M-LLM.9 | Azure refactor (Chat Completions base) | **Done** | Thin `AzureOpenAIChatAdapter` |
| M-LLM.10 | Production hardening | **Done** | Metrics, builtin conformance, `LLMProfile`, Bedrock tools stream, `cohere_native`, `azure_ai_inference` |
| M-LLM.11 | Production ops layer | **Done** | OTLP/Prometheus routes, tenant metrics, rate limit + circuit breaker, secrets map, PR guard, extended network smoke |
| M-LLM.12 | Nexus + governance wiring | **Done** | `llm_tenant_scope`, runtime metrics plugin, `INTERGRAX_LLM_TENANT_MAX_TOKENS` quota |
| M-LLM.13 | Observability + secrets + distributed limits | **Done** | Pushgateway, `LLM_OBSERVABILITY.md`, Vault loader, Redis rate limit, governance warn |

### Phase M-RAG — RAG Engine (Tier-0)

**Canon:** §5.2.2 · **Architecture:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) (RAG stack)  
**Goal:** One configurable retrieval path for `rag.retrieve`, Nexus `ContextBuilder`, and ingest — no duplicate dense-only shortcuts; parsers/chunkers/rerankers selected via profile and Integration Library slugs (never hardcoded to a single vendor).

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-RAG.1 | `RagProfile` + env (`INTERGRAX_RAG_*`) | **Done** | `intergrax/rag/profiles/rag_profile.py` |
| M-RAG.2 | `RetrievalService` (route → retrieve → rerank) | **Done** | `intergrax/rag/retrieval/`; wired to `rag.retrieve` + Nexus |
| M-RAG.3 | Adaptive `QueryRouter` (fast / standard / deep) | **Done** | `intergrax/rag/routing/query_router.py` |
| M-RAG.4 | `IngestPipeline` + configurable chunking strategy | **Done** | `intergrax/rag/ingest/`; `rag.ingest_document` |
| M-RAG.5 | Contextual chunk enricher (optional LLM) | **Done** | `INTERGRAX_RAG_CONTEXTUAL_ENRICH`; injected `LLMAdapter` |
| M-RAG.6 | Query expansion (`deterministic` / `llm`) | **Done** | `MultiQueryRetriever` + `query_expander.py` |
| M-RAG.7 | Evaluation metrics (`recall@k`, MRR) | **Done** | `intergrax/rag/evaluation/metrics.py` |
| M-RAG.8 | `create_default_rag_stack()` bootstrap | **Done** | `intergrax/rag/bootstrap/rag_stack_bootstrap.py` |
| M-RAG.9 | Tool/Nexus wiring (`retrieval_service`, profile on `ToolWiringContext`) | **Done** | `RuntimeConfig.retrieval_service` |
| M-RAG.10 | Native sparse / BM25 in vector backends | **Done** | `LexicalHybridSupport` + `query_hybrid` on InMemory/Qdrant/Weaviate; RRF fusion |
| M-RAG.11 | RAG eval CI gate + golden datasets | **Done** | `tests/fixtures/rag_golden/`, `golden_harness.py`, `rag-guard.yml` |
| M-RAG.12 | GraphRAG (`GraphStore` contract) | **Done** (beta) | `graph/` + `graph_rag` retriever + heuristic indexer |
| M-RAG.13 | Platform agentic retrieval loop (budgeted) | **Done** | `AgenticRetrievalLoop` on deep tier + `INTERGRAX_RAG_AGENTIC_*` |
| M-RAG.14 | Qdrant native sparse vectors + RRF fusion | **Done** | `INTERGRAX_RAG_QDRANT_SPARSE`, `bm25_sparse_encoder.py` |
| M-RAG.15 | Weaviate native `query.hybrid` | **Done** | Live client + `INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID`; fallback to in-memory |
| M-RAG.16 | LLM graph indexer (optional adapter) | **Done** | `INTERGRAX_RAG_GRAPH_INDEXER_MODE=llm\|heuristic_then_llm` |
| M-RAG.17 | LLM agentic query refinement | **Done** | `INTERGRAX_RAG_AGENTIC_QUERY_MODE=llm` + injected `LLMAdapter` |
| M-RAG.18 | Neo4j GraphRAG backend | **Done** | `Neo4jRagGraphStore` + `INTERGRAX_RAG_GRAPH_STORE=neo4j` |
| M-RAG.19 | SPLADE / learned sparse encoder | **Done** | `sparse_encoder.py`; `INTERGRAX_RAG_SPARSE_ENCODER=splade` (optional `fastembed`) |
| M-RAG.20 | Weaviate prod hardening | **Done** | `schema.py` — migration, multi-tenant, metadata filters |
| M-RAG.21 | Extended golden datasets | **Done** | graph_rag, multi_hop, agentic scenarios in `retrieval_cases.json` |
| M-RAG.22 | RAG observability metrics | **Done** | `INTERGRAX_RAG_METRICS_ENABLED`, runtime plugin on `TASK_COMPLETED` |

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M.0 | Integration backlog + categories approved | **Done** | Canon §7.1.3 catalog table |
| M.1 | Scaffold `intergrax/integrations/` package | **Done** | `contracts/`, `registry/`, `_shared/`, `providers/` |
| M.2 | Category contracts (P0 set) | **Done** | 7 P0 contracts + re-exports for queueing/notifications/interactions |
| M.3 | `IntegrationRegistry` + `IntegrationProfile` | **Done** | `catalog.register_integration`, `resolve`, env/mapping profile |
| M.4 | P0 providers — wrap existing | **Done** | See **M.4 provider tracker** below |
| M.5 | Provider conformance test harness | **Done** | `tests/unit/integrations/`, `_shared/conformance.py` |
| M.6 | P1 providers (on demand) | **Done** (beta) | postgresql, mysql, jira, confluence, prometheus, ms365_graph, aws, azure, gcp — see M.4/M.6 trackers |
| M.6 P2 | Extended providers (on demand) | **Done** (beta) | All P2/P3 slugs shipped 2026-05-30 — see **M.6 P2 tracker**; `_shared/p2/` + thin `providers/<slug>/` shells |
| M.7 | Agent Creation Guide § integrations | **Done** | Appendix E — capabilities/tools vs `IntegrationProfile` / `wire_lab_integrations()` |
| M.8 | Lab `IntegrationProfile` example | **Done** | `applications/lab_application/` — `wire_lab_integrations()` + `log` provider |

**M.4 delivery workflow (one provider per iteration):**

1. Implement `providers/<category>/<slug>/` (wrap legacy module — no fork).
2. Register via `register_<slug>_integration()` + `register_default_integrations()`.
3. Unit tests under `tests/unit/integrations/providers/`.
4. Add `providers/<slug>/USAGE.md` — English usage guide (factory + `IntegrationProfile` + API invoke example). Extend `scripts/generate_integration_usage_docs.py` and run `uv run python scripts/generate_integration_usage_docs.py`.
5. Update canon §7.1.3 status + this tracker + migration map row.
6. Next slug in priority order.

#### M.4 provider tracker

| Slug | Category | Status | Package | Legacy source |
|------|----------|--------|---------|---------------|
| `redis` | key_value_cache | **Done** | `providers/redis/` — `create_redis_integration()` (KV, idempotency, rate limit, semaphore, rerank) |
| `sqlite` | relational_store | **Done** | `providers/sqlite/` — `create_sqlite_integration()` (trace, events, checkpoints, HITL, …) |
| `kafka` | message_bus | **Done** (+ adopcja) | `providers/kafka/` — runtime transport delegates here |
| `celery` | message_bus | **Done** | `providers/celery/` — `create_celery_integration()` (inject `app` or broker/backend env) |
| `google_cse` | search_provider | **Done** | `providers/google_cse/` — `create_google_cse_integration()` (legacy `GOOGLE_CSE_*` env) |
| `bing` | search_provider | **Done** | `providers/bing/` — `create_bing_integration()` (legacy `BING_SEARCH_V7_API_KEY`) |
| `slack` | notification + interaction | **Done** (+ adopcja) | `providers/slack/` — runtime wiring delegates here |
| `teams` | notification + interaction | **Done** (+ adopcja) | `providers/teams/` — runtime wiring delegates here |
| `webhook` | notification_channel | **Done** (+ adopcja) | `providers/webhook/` — generic HTTP + `GenericJsonPayloadFormatter` |
| `lab_json` | interaction_surface | **Done** (+ adopcja) | `providers/lab_json/` — lab intake; runtime channel ``lab`` |
| `rabbitmq` | message_bus | **Done** (+ adopcja) | `providers/rabbitmq/` — `create_rabbitmq_integration()` (requires `kv_store`) |
| `log` | notification_channel | **Done** (+ adopcja) | `providers/log/` — wraps `LoggingNotificationAdapter`; lab profile default |
| `postgresql` | relational_store | **Done** (beta) | `providers/postgresql/` — `RelationalStore` via psycopg3; only `opens.py` connects |
| `mysql` | relational_store | **Done** (beta) | `providers/mysql/` — `RelationalStore` via pymysql; only `opens.py` connects |
| `databricks` | relational_store | **Done** (beta) | `providers/databricks/` — SQL Warehouse via databricks-sql-connector; only `opens.py` connects |
| `mongodb` | document_store | **Done** (beta) | `providers/mongodb/` — flexible JSON `DocumentStore`; PyMongo only in `opens.py` |
| `pinecone` | vector_store | **Done** (beta) | `providers/pinecone/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `qdrant` | vector_store | **Done** (beta) | `providers/qdrant/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `chroma` | vector_store | **Done** (beta) | `providers/chroma/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `s3` | object_storage | **Done** (beta) | `providers/s3/` — put/get/delete/presigned_url; boto3 only in `opens.py` |
| `jira` | issue_tracker | **Done** (beta) | `providers/jira/` — REST v3; only `opens.py` creates httpx client |
| `confluence` | wiki_knowledge | **Done** (beta) | `providers/confluence/` — REST wiki; only `opens.py` creates httpx client |
| `prometheus` | observability_backend | **Done** (beta) | `providers/prometheus/` — PromQL query API; only `opens.py` creates httpx client |
| `elasticsearch` | observability_backend | **Done** (beta) | `providers/elasticsearch/` — `_search` aggregations; only `opens.py` creates httpx client |
| `ms365_graph` | collaboration_suite | **Done** (beta) | `providers/ms365_graph/` — Graph mail/calendar/directory; only `opens.py` creates httpx client |
| `cassandra` | document_store | **Done** (beta) | `providers/cassandra/` — CQL get/put/delete/query; only `opens.py` creates driver session |
| `aws` | cloud_platform | **Done** (beta) | `providers/aws/` — IAM/STS auth + category defaults; only `opens.py` creates boto3 session |
| `azure` | cloud_platform | **Done** (beta) | `providers/azure/` — MI / service principal + category defaults; only `opens.py` creates credential |
| `gcp` | cloud_platform | **Done** (beta) | `providers/gcp/` — ADC / service account + category defaults; only `opens.py` creates credentials |

#### M.6 P2 — Extended provider tracker (canon §7.1.3 P2)

Deliver after M.6 P1 priorities unless a product app blocks on a specific slug. Each P2 provider follows the same workflow as M.4 (contract → `providers/<slug>/` → tests → catalog row).

| Slug | Category | Status | Rationale / notes |
|------|----------|--------|-------------------|
| **`cassandra`** | **document_store** | **Done** (beta) | High-volume log / event retention; CQL driver via `opens.py` single entry |
| **`elasticsearch`** | **observability_backend** | **Done** (beta) | Log search / aggregations (`_search` + Lucene `query_string` via ObservabilityBackend); complements `prometheus` |
| **`databricks`** | **relational_store** | **Done** (beta) | Lakehouse SQL Warehouse; PAT via `opens.py`; `execute` / `fetch_all` for analytics agents |
| **`mongodb`** | **document_store** | **Done** (beta) | Flexible JSON documents; partition-scoped get/put/delete/query via PyMongo |
| **`pinecone`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/pinecone_vector_store.py` |
| **`qdrant`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/qdrant_vector_store.py` |
| **`chroma`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/chroma_vector_store.py` |
| **`s3`** | **object_storage** | **Done** (beta) | AWS S3 blobs; boto3 only in `opens.py` |
| **`azure_blob`** | **object_storage** | **Done** (beta) | Azure Blob; `providers/azure_blob/` + shared `CatalogObjectStorage` |
| **`gcs`** | **object_storage** | **Done** (beta) | GCS via `_shared/p2/gcs_blob.py` |
| **`dynamodb`** | **document_store** | **Done** (beta) | boto3 table facade in `_shared/p2/factories.py` |
| **`oracle`** / **`mssql`** / **`azure_sql`** / **`cloud_sql`** | **relational_store** | **Done** (beta) | SQL adapters via `_shared/p2/clients.py` |
| **`memcached`** / **`elasticache`** | **key_value_cache** | **Done** (beta) | pymemcache / Redis-compatible duck client |
| **`sqs`** / **`service_bus`** / **`pubsub`** | **message_bus** | **Done** (beta) | `CloudTaskQueue` over cloud SDK facades |
| **`email_smtp`** | **notification_channel** | **Done** (beta) | stdlib SMTP in factory open path |
| **`otel`** | **observability_backend** | **Done** (beta) | OTLP-oriented metrics facade (beta noop exporter default) |
| **`github`** / **`linear`** / **`azure_devops`** | **issue_tracker** | **Done** (beta) | REST issue trackers via httpx |
| **`notion`** / **`sharepoint`** | **wiki_knowledge** | **Done** (beta) | REST wiki adapters |
| **`google_workspace`** | **collaboration_suite** | **Done** (beta) | Gmail / Calendar REST |
| **`brave`** / **`serpapi`** | **search_provider** | **Done** (beta) | Shared `_shared/rest_search.py` hit mappers |
| **`playwright`** | **browser_automation** | **Done** (beta) | `contracts/browser_automation.py` + Playwright factory |

#### M.6 P3 / M.7 — Harness integrations (Done beta, 2026-05-29)

**M.11 harness defaults (Done beta):** default `notify_channel` injection from lab wiring (`task_defaults.py`, `LAB_HARNESS` enricher on lab run + interaction intake).

**M.10 harness Tier A (Done beta):** composite observability (`observability_backends` + role-based `resolve_observability_backend`), HITL→PagerDuty runtime path (`create_harness_notification_adapter`, `LAB_HARNESS`), integration tests.

**M.9 harness depth (Done beta):** full adapters (LangSmith, OpenSearch, Vespa, GitLab, PagerDuty, Braintrust), tools (`gitlab.create_issue`, `pagerduty.trigger_incident`, `braintrust.log_eval`), `slash_command`, lab harness profile, CI harness-smoke job. Catalog: **99**.

**M.8 harness gap (Done beta):** +14 slugs via `_shared/p4/factories.py`

**M.7 harness (Done beta):** +21 slugs via `_shared/p3/factories.py` (incl. **sentry**).

#### M.7 — Document parser catalog bridge (2026-05-30)

Vendor document parsing moved from `intergrax/rag/document_loaders/parsers/` into `integrations/providers/document_parser/`. RAG uses `CatalogDocumentParser` + `resolve_document_parser()`.

**Wave 2 (2026-05-30):** `openpyxl`, `whisper`, `yt_dlp`; `cohere_rerank` / `jina_rerank`; Bing/Google CSE implementations under `integrations/.../web_client.py` (websearch re-exports); `ParserPipeline` ingestion trace; tool `rag.ingest_document`; `IntegrationProfile.legal_product()` / `research_product()` / `lab()` with `document_parser=docling`; lab `GET /v1/lab/integrations/docling/health`.

**Wave 3 (2026-05-30):** `reddit`, `google_places` search providers; Chroma/Qdrant/Pinecone SDK in `integrations/.../rag_store.py` (RAG shims); runtime SQLite delivery ledger via `sqlite/opens`; `rag.ingest_document` env flags for legal/research; parser trace export to Langfuse/Sentry.

**Wave 4 (2026-05-30):** `inmemory` vector store SDK in `integrations/.../inmemory/rag_store.py`; SQLite observability via `integration_profile_wiring` + `wire_nexus_observability(integration_profile=…)` with default-path fallback; parser pipeline spans appended to `RunTraceWriter` (`parser_trace_span.py`); vendor import governance script + CI gate; Phase Q scaffold defaults (`IntegrationProfile`, `ToolProfile` with `websearch.read_url`).

**Wave 5 (2026-05-30):** Phase P wave 3 tools (`websearch.fetch_batch`, `rag.list_collections`, `observability.query_traces`); full `IntegrationProfile` on legal/research products; Weaviate/Milvus `rag_store.py`; Redis SDK cleanup in distributed/rag shims; governance extended to `agents/` + `rag/`; parser trace export on `RunTraceWriter.finalize_run`; Phase Q scaffold wave 2 (lab vs product ToolProfile, env profile override).

| Slug | Status | Notes |
|------|--------|-------|
| `docling` | **Done** (beta) | local + server; `opens.py` only Docling/httpx imports |
| `pymupdf` | **Done** (beta) | PDF + optional Tesseract OCR |
| `unstructured` | **Done** (beta) | HTML loader |
| `python_docx` | **Done** (beta) | Word `.docx` |
| `openpyxl` | **Done** (beta) | Excel/CSV via pandas |
| `whisper` | **Done** (beta) | Audio + YouTube (uses yt_dlp opens) |
| `yt_dlp` | **Done** (beta) | YouTube audio/video download |
| `cohere_rerank` | **Done** (beta) | RAG rerank via integration resolver |
| `jina_rerank` | **Done** (beta) | RAG rerank via integration resolver |
| `reddit` | **Done** (beta) | Reddit OAuth2 search |
| `google_places` | **Done** (beta) | Google Places text search |

#### M.6 P3 — Legacy backlog note (superseded)

Slugs below were **already in** `IntegrationSlug` unless marked *proposed*. Prioritize when a product app blocks; otherwise deliver after P2.

| Priority | Slug(s) | Category | Why agents/apps need it |
|----------|---------|----------|-------------------------|
| **High** | `mongodb` | document_store | Session state, flexible agent memory, JSON artifacts at scale |
| **High** | `pinecone`, `qdrant`, `chroma` | vector_store | Production RAG — unify Tier-3 `IntegrationProfile.vector_store` with existing `rag/` backends |
| **High** | `s3`, `azure_blob`, `gcs` | object_storage | Checkpoint blobs, sandbox exports, document ingestion pipelines |
| **High** | `email_smtp` | notification_channel | HITL and report delivery without Slack/Teams |
| **Medium** | `notion`, `sharepoint` | wiki_knowledge | Runbooks and internal docs (Confluence complement) |
| **Medium** | `github`, `linear` | issue_tracker | Dev workflows, PR/issue-aware agents |
| **Medium** | `google_workspace` | collaboration_suite | Google-tenant mail/calendar parity with MS365 |
| **Medium** | `otel` | observability_backend | Export runtime traces/metrics to Grafana Cloud, Datadog, etc. |
| **Medium** | `playwright` | browser_automation | JS-heavy sites, authenticated flows beyond static fetch |
| **Medium** | `brave`, `serpapi` | search_provider | Rate-limit / vendor diversity for research agents |
| **Low** | `oracle`, `mssql`, `azure_sql`, `cloud_sql` | relational_store | Enterprise DB deployments |
| **Low** | `dynamodb`, `memcached`, `elasticache` | document_store / KV | AWS-native persistence tiers |
| **Future** | *weaviate*, *milvus*, *snowflake*, *vault* *proposed* | vector_store / relational / secrets | Add slug + category only after human approval (§5.2.4) |

**Vector-store rule (pinecone / qdrant / chroma):** implementation **stays** in `intergrax/rag/vectorstore/`. Integration Library adds `providers/<slug>/` as a **thin registry adapter**: `opens.py` is the only module that imports vendor SDK; `bundle.create_*_vector_store()` delegates to the existing RAG provider. Tier-3 selects slug via `IntegrationProfile.vector_store`; RAG pipeline code unchanged.

**MongoDB — suggested implementation sketch (greenfield):**

```text
providers/mongodb/
├── config.py                   # INTERGRAX_MONGODB_URI, DATABASE, COLLECTION_PREFIX
├── client.py                   # PyMongo collection wrapper (internal — no driver outside opens.py)
├── adapter.py                  # MongoDocumentStore implements DocumentStore
├── opens.py                    # ONLY place that constructs MongoClient
├── bundle.py                   # create_mongodb_document_store()
├── register.py
└── tests/                      # mocked collection; integration_live optional
```

**Prerequisite (mongodb):** `DocumentStore` contract — **Done** (`contracts/document_store.py`). Partition key maps to MongoDB `_id` or compound `{tenant_id, key}` index.

**Pinecone — suggested implementation sketch (catalog bridge):**

```text
providers/pinecone/
├── config.py                   # INTERGRAX_PINECONE_API_KEY, INDEX, NAMESPACE, ENV
├── adapter.py                  # Thin VectorStore registry facade (delegates to rag/)
├── opens.py                    # ONLY place that imports pinecone SDK / builds Pinecone client
├── bundle.py                   # create_pinecone_vector_store() → rag PineconeVectorStore
├── register.py
└── tests/                      # mocked delegate; guard: no pinecone import outside opens.py
```

**Prerequisite (pinecone):** `contracts/vector_store.py` — **Done** (re-exports `rag/vectorstore/contracts/vector_store.py`). Registered under `IntegrationCategory.VECTOR_STORE`.

**Cassandra — suggested implementation sketch (greenfield):**

```text
contracts/document_store.py     # DocumentStore — get/put/delete/query by partition key
providers/cassandra/
├── config.py                   # INTERGRAX_CASSANDRA_CONTACT_POINTS, KEYSPACE, USER, PASSWORD
├── client.py                   # CQL session (internal — no direct driver import outside opens.py)
├── adapter.py                  # CassandraDocumentStore implements DocumentStore
├── opens.py                    # ONLY place that constructs cassandra driver session
├── bundle.py                   # create_cassandra_integration()
├── register.py
└── tests/                      # testcontainers or mocked session; integration_live optional
```

**Prerequisite (cassandra):** `DocumentStore` contract — **Done** (`contracts/document_store.py`). Runtime event / trace backends remain SQLite-first until an explicit adoption milestone names Cassandra as a target store.

**Elasticsearch — suggested implementation sketch (greenfield):**

```text
providers/elasticsearch/
├── config.py                   # INTERGRAX_ELASTICSEARCH_URL, USER, PASSWORD, INDEX_PREFIX
├── client.py                   # REST search client (internal — no httpx outside opens.py)
├── adapter.py                  # ElasticsearchObservabilityBackend implements ObservabilityBackend
├── opens.py                    # ONLY place that constructs httpx client / ES connection
├── bundle.py                   # create_elasticsearch_observability_backend()
├── register.py
└── tests/                      # mocked _search / ES|QL responses; integration_live optional
```

**Contract note:** start with `ObservabilityBackend` (`query_instant` / `query_range`) mapped to ES\|QL or index-scoped aggregations where feasible; add optional `search_logs(query, *, limit)` on the contract in a follow-up if PromQL-shaped methods prove awkward for log-only clusters.

**Databricks — suggested implementation sketch (greenfield):**

```text
providers/databricks/
├── config.py                   # INTERGRAX_DATABRICKS_HOST, HTTP_PATH, TOKEN, CATALOG, SCHEMA
├── client.py                   # SQL connection wrapper (internal — no driver import outside opens.py)
├── adapter.py                  # DatabricksRelationalStore implements RelationalStore
├── opens.py                    # ONLY place that opens databricks-sql-connector / REST session
├── bundle.py                   # create_databricks_relational_store()
├── register.py
└── tests/                      # mocked cursor / Statement Execution API; integration_live optional
```

**Contract note:** implements existing `RelationalStore` (`connect`, `execute`, `fetch_all`, `close`). Optional `tenant_schema` maps to Unity Catalog ``catalog.schema`` (default schema per connection). Not a replacement for domain runtime stores (SQLite-first) — target is analytics / reporting agents and batch read paths.


1. Create package skeleton:

```text
intergrax/integrations/
├── __init__.py
├── contracts/
│   ├── __init__.py
│   └── base.py              # IntegrationMetadata, HealthStatus, IntegrationError
├── registry/
│   ├── __init__.py
│   ├── catalog.py           # slug → provider entry (lazy import)
│   └── factory.py           # resolve(category, slug | env)
├── _shared/
│   ├── config.py            # pydantic BaseIntegrationConfig
│   └── health.py
└── providers/
    └── .gitkeep
```

2. Add `IntegrationMetadata` dataclass: `slug`, `categories`, `status` (`stable` | `beta` | `deprecated`), `env_prefix`.

3. Register package in `pyproject.toml` / existing import paths (no new top-level dependency unless provider-specific).

#### M.2 — Category contracts (step-by-step)

For each category in §7.1.2, implement a **minimal** Protocol in `integrations/contracts/`:

| Contract | Minimum methods | Notes |
|----------|-----------------|-------|
| `RelationalStore` | `connect()`, `execute()`, `fetch_all()`, `close()` | **Done** — `contracts/relational_store.py`; sqlite/postgresql/mysql/**databricks** (beta) |
| `KeyValueCache` | `get`, `set`, `delete`, `set_if_absent` | Maps to existing `IdempotencyStore` / Redis helpers |
| `MessageBus` | `enqueue`, `get_status`, `get_result` | Re-export / implement `queueing.contracts.TaskQueue` |
| `SearchProvider` | `search(query, *, limit)` → `SearchResult[]` | Align with `websearch/providers/base.py` |
| `NotificationChannel` | `notify(message)` | Align with `runtime/notifications/adapter_contract.py` |
| `InteractionSurface` | `can_handle`, `to_inbound`, `channel` | Align with `runtime/interactions/adapter_contract.py` |
| `CloudPlatform` | `slug`, `default_region`, `resolve(category)`, `health` | **Done** — `contracts/cloud_platform.py`; **`aws`**, **`azure`**, **`gcp`** providers (beta) |
| `CollaborationSuite` | `get_message`, `list_messages`, `send_mail`, `list_calendar_events`, `get_user` | **Done** — `contracts/collaboration_suite.py`; `ms365_graph` provider |
| `DocumentStore` | `get`, `put`, `delete`, `query` (partition-scoped) | **Done** — `contracts/document_store.py`; `cassandra`, **`mongodb`** (beta) providers |
| `VectorStore` | `add_documents`, `query`, `delete`, … | **Done** — `contracts/vector_store.py` re-exports `rag/`; **`pinecone`**, **`qdrant`**, **`chroma`** (beta) |
| `ObjectStorage` | `put`, `get`, `delete`, `presigned_url` | **Done** — `contracts/object_storage.py`; **`s3`** (beta) |
| `IssueTracker` | `get_issue`, `add_comment`, `search_issues` | **Done** — `contracts/issue_tracker.py`; `jira` provider |
| `WikiKnowledge` | `get_page`, `search_pages` | **Done** — `contracts/wiki_knowledge.py`; `confluence` provider |
| `ObservabilityBackend` | `query_instant`, `query_range` | **Done** — `contracts/observability_backend.py`; `prometheus`, **`elasticsearch`** (beta) providers |

**Rule:** if a contract already exists elsewhere, **re-export or inherit** — do not define a third variant.

#### M.3 — IntegrationRegistry (step-by-step)

1. `catalog.py` — static registry:

```python
INTEGRATION_ENTRIES: dict[str, IntegrationEntry] = {
    "sqlite": IntegrationEntry(categories=("relational_store",), factory="..."),
    "redis": IntegrationEntry(categories=("key_value_cache",), factory="..."),
    # ...
}
```

2. `factory.py`:

```python
def resolve(category: str, slug: str | None = None, *, config: Mapping[str, Any] | None = None) -> Any:
    """slug defaults from env INTERGRAX_INTEGRATION_<CATEGORY> or IntegrationProfile."""
```

3. `IntegrationProfile` — pydantic model loaded from env or YAML in Tier-3 `settings.py`.

4. `health_check_all(profile)` — optional startup probe for lab/production.

#### M.4 — Adding a new provider (checklist for implementers)

Copy this checklist into every `providers/<slug>/README.md`:

```text
[ ] 1. Pick category contract(s) from integrations/contracts/
[ ] 2. Create providers/<slug>/ with adapter.py, config.py, config.example.yaml
[ ] 3. Implement contract — no business logic, no Nexus imports
[ ] 4. Register slug in registry/catalog.py
[ ] 5. Add unit tests with fakes or testcontainers (default: no live vendor)
[ ] 6. Optional: pytest -m integration_live with CI secrets
[ ] 7. Wire in one Tier-3 application as reference (lab or product)
[ ] 8. Update canon §7.1.3 status column
```

**Example — wrapping existing Redis idempotency store:**

```text
providers/redis/
├── adapter.py       # RedisKeyValueCache implements KeyValueCache
├── config.py        # REDIS_URL, REDIS_PREFIX
└── tests/
    └── test_redis_cache.py  # fakeredis or mock
```

Delegate to `intergrax/distributed/providers/redis_idempotency_store.py` internally.

**Example — new Jira provider (greenfield):**

```text
providers/jira/
├── adapter.py       # JiraIssueTracker implements IssueTracker
├── config.py        # JIRA_BASE_URL, JIRA_API_TOKEN
├── config.example.yaml
├── README.md
└── tests/
    └── test_jira_issue_tracker.py  # responses mocked from fixtures/
```

Expose agent tools via Tier-0 tool registration (`jira.get_issue`, `jira.create_comment`) — ToolRuntime policy in Tier-1.

#### M.4b — Cloud platform providers (aws / azure / gcp)

Each platform folder exposes **one auth entry point** and registers sub-service slugs:

```text
providers/aws/
├── adapter.py       # CloudPlatform: IAM profile, region, resolve("object_storage") → S3
├── config.py        # AWS_REGION, AWS_PROFILE, AWS_ROLE_ARN
├── services/        # thin wrappers delegating to category contracts
│   ├── s3.py
│   ├── sqs.py
│   └── dynamodb.py
└── tests/

providers/azure/
├── adapter.py       # Managed identity + service principal
├── services/
│   ├── blob.py
│   └── service_bus.py
└── ...

providers/gcp/
├── adapter.py       # ADC + service account
├── services/
│   ├── gcs.py
│   └── pubsub.py
└── ...
```

**Checklist:** implement infrastructure services (S3, SQS, Blob, GCS, Pub/Sub, …) only. LLM wiring stays in `intergrax/llm_adapters/` — do not register Bedrock, Azure OpenAI, or Vertex under `integrations/`.

#### M.5 — Migration map (legacy → catalog)

| Legacy location | Target slug | Action |
|-----------------|-------------|--------|
| `distributed/providers/redis_kv_store.py` (+ siblings) | `redis` | **Done** — single entry `integrations/providers/key_value_cache/redis/create_redis_integration()` |
| `queueing/providers/kafka/` | `kafka` | **Done** — runtime transport + tests delegate to `integrations/providers/message_bus/kafka/` |
| `queueing/providers/celery/` | `celery` | **Done** — `integrations/providers/message_bus/celery/create_celery_integration()` |
| `queueing/providers/rabbitmq/` | `rabbitmq` | **Done** — runtime transport + tests delegate to `integrations/providers/message_bus/rabbitmq/` |
| `websearch/providers/google_cse_provider.py` | `google_cse` | **Done** — `integrations/providers/search_provider/google_cse/create_google_cse_integration()` |
| `websearch/providers/bing_provider.py` | `bing` | **Done** — `integrations/providers/search_provider/bing/create_bing_integration()` |
| `runtime/notifications/adapters/webhook_adapter.py` | `webhook` | **Done** — `integrations/providers/notification_channel/webhook/create_webhook_integration()` |
| `runtime/notifications/adapters/logging_adapter.py` | `log` | **Done** — `integrations/providers/notification_channel/log/`; factory delegates |
| `runtime/notifications/adapters/` | `slack`, `teams` | **Done** — runtime delegates |
| `runtime/interactions/adapters/lab_json_adapter.py` | `lab_json` | **Done** — `integrations/providers/interaction_surface/lab_json/create_lab_json_integration()` |
| `runtime/*/stores/sqlite_*.py` (+ store openers) | `sqlite` | **Done** — single entry `integrations/providers/relational_store/sqlite/create_sqlite_integration()` |
| (new) | `postgresql` | **Done** — `integrations/providers/relational_store/postgresql/`; **only** `opens.py` calls `psycopg.connect` |
| (new) | `mysql` | **Done** — `integrations/providers/relational_store/mysql/`; **only** `opens.py` calls `pymysql.connect` |
| (new) | `jira` | **Done** — `integrations/providers/issue_tracker/jira/`; **only** `opens.py` creates httpx client |
| (new) | `confluence` | **Done** — `integrations/providers/wiki_knowledge/confluence/`; **only** `opens.py` creates httpx client |
| (new) | `prometheus` | **Done** — `integrations/providers/observability_backend/prometheus/`; **only** `opens.py` creates httpx client |
| (new) | `ms365_graph` | **Done** — `integrations/providers/collaboration_suite/ms365_graph/`; **only** `opens.py` creates httpx client + token fetch |
| (new) | `cassandra` | **Done** — `integrations/providers/document_store/cassandra/`; **only** `opens.py` creates driver session |
| (new) | `aws` | **Done** — `integrations/providers/cloud_platform/aws/`; **only** `opens.py` creates boto3 session |
| (new) | `azure` | **Done** — `integrations/providers/cloud_platform/azure/`; **only** `opens.py` creates Azure credential |
| (new) | `gcp` | **Done** — `integrations/providers/cloud_platform/gcp/`; **only** `opens.py` creates Google credentials |
| (new) | `elasticsearch` | **Done** — `integrations/providers/observability_backend/elasticsearch/`; **only** `opens.py` creates httpx client |
| (new) | `databricks` | **Done** — `integrations/providers/relational_store/databricks/`; **only** `opens.py` calls `databricks.sql.connect` |
| (new) | `mongodb` | **Done** — `integrations/providers/document_store/mongodb/`; **only** `opens.py` calls `pymongo.MongoClient` |
| `rag/vectorstore/providers/pinecone_*` | `pinecone` | **Done** — `providers/pinecone/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/providers/qdrant_*` | `qdrant` | **Done** — `providers/qdrant/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/providers/chroma_*` | `chroma` | **Done** — `providers/chroma/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/bootstrap/vectorstore_bootstrap.py` | integration catalog | **Done** — `create_default_vectorstore_manager()` resolves via `IntegrationProfile.vector_store` |
| `rag/vectorstore/providers/*` | other vector slugs | Catalog entry only until bridge provider ships |

**Not migrated to `integrations/`:** `intergrax/llm_adapters/` — LLM providers are a separate Tier-0 concern (§7.1.2 out-of-scope table).

#### M.6 — Testing strategy

| Layer | Location | Marker |
|-------|----------|--------|
| Contract unit tests | `tests/unit/integrations/` | default gate |
| Provider unit tests | `intergrax/integrations/providers/<slug>/tests/` | default gate |
| Registry / factory | `tests/unit/integrations/test_registry.py` | gate |
| Live vendor smoke | `tests/integration/integrations/` | `integration_live` (CI optional) |

Conformance test pattern: given a fake backend, assert all Protocol methods behave consistently (including error types).

#### M.7 — Agent Creation Guide (Appendix E)

Documented in [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) Appendix E:

- Agents: `capabilities`, `allowed_tools`, `ToolRequest` — no integration slug imports.
- Applications: `IntegrationProfile`, `wire_lab_integrations()`, `register_default_integrations()`.
- Env: `INTERGRAX_INTEGRATION_<CATEGORY>` overrides.

Tier-3 composition example (product factory):

```python
# applications/my_app/factory.py
from intergrax.integrations import (
    IntegrationCategory,
    IntegrationProfile,
    register_default_integrations,
)

def create_app():
    register_default_integrations()
    profile = IntegrationProfile.lab()  # or build_profile_from_env()

    cloud = profile.resolve(IntegrationCategory.CLOUD_PLATFORM)       # aws | azure | gcp
    db = profile.resolve(IntegrationCategory.RELATIONAL_STORE)        # sqlite | postgresql
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    storage = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
    notifier = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
    # wire into Nexus factories, not into agents/
```

Agents reference capabilities in `AgentContract` (e.g. `allowed_tools=["websearch.query"]`) — not integration slugs.

#### M.8 — Definition of done (Phase M incremental)

Each provider PR is **done** when:

1. Contract conformance tests pass.
2. Registered in `catalog.py` with metadata.
3. `providers/<slug>/USAGE.md` — English: env vars, factory call, `IntegrationProfile` resolve, minimal invoke example.
4. At least one Tier-3 app or lab factory can select it via `IntegrationProfile`.
5. No new direct vendor imports added under `agents/`.

Szablony utrzymywane przez `scripts/generate_integration_usage_docs.py` (regeneracja po dodaniu providera).

---

### Phase N — Application Environment & Deploy Scaffold (Tier-3)

**Canon:** §7.4.8–§7.4.10  
**Goal:** From agent POC to **docker-pushable** dedicated lab/product host in minutes — same ergonomics as `new-agent`, with isolated `.env.example`, manifest, and Docker.

**Prerequisite:** Phase L complete; Phase M.3 (`IntegrationProfile`) available.

**Delivery rule (this phase):** One step per iteration — implement → summarize → update docs → present next step (see **§6.1**).

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| N.0 | Architecture & plan documented | **Done** | §7.4.8–§7.4.10 | This section + runtime canon (2026-05-30) |
| N.1 | `ApplicationManifest` + `AgentBinding` models | **Done** | §7.4.10 | `intergrax/applications/contracts/manifest.py` |
| N.2 | Manifest conformance harness + unit tests | **Done** | §7.4.10 | `intergrax/applications/_shared/wiring.py` |
| N.2.1 | Unified agent initialization (builders / factories / context) | **Done** | §7.4.10 | `ApplicationBuildContext`, `build_application_registry`; lab + legal migrated |
| N.2.2 | Strongly typed `AgentBinding.mount(AgentClass, factory=...)` | **Done** | §7.4.10 | `type[Agent]` + callable factory; `deserialize()` for scaffold strings only |
| N.3 | `python -m intergrax.scaffold new-application` (profile `lab`) | **Done** | §7.4.8 | `new_application.py`, `agent_catalog.py`, `cli.py`; lab templates + smoke |
| N.4 | Scaffold profile `product` (fastapi_core skeleton) | **Done** | §7.4.8 | `new_application_product.py`; FastAPI Core + auth stub + `/health`; `--agents` list |
| N.5 | Docker templates under `applications/<app>/docker/` | **Done** | §7.4.8 | Dockerfile + `.dockerignore` + `docker-compose.yml` + `build-docker.sh` / `.bat`; monorepo-root context |
| N.6 | Reference app `poc_template_application` (committed example) | **Done** | §7.4.8 | `applications/poc_template_application/`; README three-command quickstart; gate smoke |
| N.7 | Backfill `.env.example` on existing apps | **Done** | §7.4.8 | `lab_application`, `legal_application`, `research_application`, `poc_template_application` |
| N.8 | `AGENT_CREATION_GUIDE.md` Step 4E (dedicated application) | **Done** | — | Step 4E + Appendix F cross-links; gate doc test |
| N.9 | Acceptance `test_scaffold_application` (gate) | **Done** | — | `test_scaffold_acceptance.py` — lab/product E2E, CLI profiles, docker scripts |
| N.10 | Optional `new-stack` (agent + application in one CLI) | **Done** | — | `intergrax/scaffold/new_stack.py`; gate test in `test_scaffold_acceptance.py` |

#### N — Step-by-step implementation sequence

Execute **strictly in order**; do not skip ahead without completing acceptance for the current step.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | N.1 | Add `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` (Pydantic) | Unit tests pass; no scaffold yet |
| 2 | N.2 | Add `applications/_shared/conformance.py` (or mirror integrations pattern) | Manifest load + minimal registry build test |
| 3 | N.3 | Implement `new_application.py` + `lab` profile templates | `uv run python -m intergrax.scaffold new-application test_lab --profile lab --agents echo` creates tree; smoke test green |
| 4 | N.3b | Wire `build_parser()` subcommand; post-create hints (uvicorn, pytest, docker) | CLI prints next commands; gate test added (N.9 partial) |
| 5 | N.5 | Add Docker/docker-compose + build scripts to scaffold | `applications/<app>/docker/build-docker.sh` (or `.bat`) builds image from repo root |
| 6 | N.6 | Commit `applications/poc_template_application/` from scaffold | README three-command quickstart verified |
| 7 | N.7 | Add per-app `.env.example` to legal, research, lab | Vars match each `settings.py`; no secrets committed |
| 8 | N.4 | Add `product` profile to scaffold | **Done** — `test_scaffold_product_application.py`; FastAPI Core + `/health` |
| 9 | N.8 | Update agent guide Step 4E | **Done** — scaffold lab/product, Docker scripts, three-command quickstart |
| 10 | N.9 | Full acceptance + `pytest -m gate` | **Done** — runtime E2E + `test_scaffold_acceptance.py` |

**Scaffold CLI (target interface):**

```bash
python -m intergrax.scaffold new-application my_lab \
  --profile lab \
  --agents echo,my_agent \
  --port 8091 \
  --prefix /v1/my_lab
```

**Out of scope for Phase N:**

- Separate `pyproject.toml` per application (stay monorepo + `pythonpath`)
- Auto-discovery of agents in `lab_application` (keep explicit wiring; manifest is declarative, not magic)
- Runtime sandbox (Tier-1) changes — only document distinction (§7.4.9)

#### Tier-3 application layer — readiness (2026-05-30)

**Status: ready** to generate new applications via scaffold. Checklist: [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md).

| Track | ID | Status | Notes |
|-------|-----|--------|-------|
| Engine | N.1–N.2.2 | **Done** | manifest, `build_application_registry`, conformance |
| Scaffold | N.3–N.4, N.10 | **Done** | `lab` + `product` + `new-stack` |
| Deploy | N.5–N.7 | **Done** | Docker scripts, `BUILD_AND_DEPLOY`, `.env.example` |
| Docs + gate | N.8–N.9 | **Done** | Step 4E, `test_scaffold_acceptance`, legal/research/lab manifest tests |
| Hardening | A.1–A.2 | **Done** | `test_legal_manifest_wiring`, tool_wiring assertions on scaffold |
| Optional CI Docker | B.1 | **Done** | `tests/integration/applications/test_poc_template_docker_build.py` (not in gate) |
| Product maturity | — | **Reference** | `legal_application` chat routes — extend scaffold `product` manually |

**Verify:**

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest -m gate -q
```

---

### Phase O — Tool Library & Unified Tool Model (Tier-0)

**Canon:** §7.1.6–§7.1.7, §22, §42.12  
**Goal:** Ship a reusable **Tool Library** catalog (mirror Integration Library) and migrate legacy pipeline flags (`use_rag`, `use_websearch`) to explicit catalog tools.

**Prerequisite:** Phase M.3 (`IntegrationProfile`) available; tool engine (`ToolRegistry`, `RuntimeToolInvoker`) exists.

**Catalog reference:** [`TOOLS.md`](TOOLS.md)

**Delivery rule:** One domain or migration slice per iteration — implement → gate → update `TOOLS.md` → next step.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| O.0 | Architecture & catalog documented | **Done** | §7.1.6–§7.1.7, §22 | Runtime canon + `TOOLS.md` + this section (2026-05-30) |
| O.1 | Extended `ToolContract` | **Done** | §22 | `ToolRiskLevel`, `ToolRetryPolicy`, metadata fields; invoker timeout/retry/trace (2026-05-30) |
| O.2 | `ToolCatalog` + `ToolProfile` + `ToolWiringContext` | **Done** | §7.1.6 | `intergrax/tools/registry/`; `build_registry_from_profile`; RuntimeConfig wiring (2026-05-30) |
| O.3 | Context tools: `rag.retrieve`, `websearch.query` | **Done** | §7.1.7, §22.1 | `providers/rag/`, `providers/websearch/` (2026-05-30) |
| O.4 | Reference domain: `jira.*` tools | **Done** | §7.1.6 | `get_issue`, `add_comment`, `search_tasks` over `IssueTracker` (2026-05-30) |
| O.4b | Catalog domain bundles: `confluence.*`, `notify.send`, observability, `sandbox.exec` | **Done** | §7.1.6 | All first-party catalog tools registered (2026-05-30) |
| O.5 | **Unified tool model migration** | **Done** | §7.1.7, §22.2 | `tool_ids` on plans; RagStep/WebsearchStep → catalog shims (2026-05-30) |
| O.6 | Schema exporters (OpenAI + MCP) | **Done** | §7.1.6 | `tools/exporters/`; MCP catalog mount on lab/poc_template (2026-05-30) |
| O.7 | Migrate legacy `ToolBase` → `ToolContract` | **Done** | §5.2.2 | `ChatAgent` → registry; `tools_base` deprecated (2026-05-30) |
| O.8 | `ToolProfile` in Tier-3 scaffold | **Done** | §7.4.8 | `tool_wiring.py` template; lab + poc_template reference (2026-05-30) |
| O.9 | Agent Creation Guide Appendix E update | **Done** | — | Unified model + ToolProfile examples (2026-05-30) |
| O.10 | Gate tests for catalog conformance | **Done** | — | `tests/unit/tools/providers/` — all catalog bundles (2026-05-30) |
| O.11 | Phase P wave 2 context tools: `websearch.read_url`, `confluence.search` | **Done** | §7.1.7, §22.1 | `providers/websearch/read_url_*`, confluence alias (2026-05-30) |
| O.12 | Phase P wave 3 tools: `websearch.fetch_batch`, `rag.list_collections`, `observability.query_traces` | **Done** | §7.1.7, §22.1 | Extended `ObservabilityBackend.query_traces`, vector `list_collections` (2026-05-30) |

#### O — Step-by-step implementation sequence

Execute **strictly in order** for foundation (O.1–O.4); O.5–O.10 may overlap after O.4 reference tools land.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | O.1 | Extend `ToolContract` + update `RuntimeToolInvoker` for new fields | Unit tests pass; backward compatible defaults |
| 2 | O.2 | Add `tools/registry/catalog.py`, `profile.py`, `ToolWiringContext` dataclass | `register_default_tools()` no-op registry; profile enables subset |
| 3 | O.3 | Implement `providers/rag/` and `providers/websearch/` handlers | **Done** — `rag.retrieve`, `websearch.query` + tests |
| 4 | O.4 | Implement `providers/jira/` bundle (3 tools) | **Done** — conformance tests with mocked `IssueTracker` |
| 4b | O.4b | Implement remaining catalog bundles (`confluence`, `notify`, `observability`, `sandbox`) | **Done** — all tool_ids in `register_default_tools()` |
| 5 | O.5a | Add `tool_ids` to plan models; map legacy booleans → tool_ids | **Done** — `ToolInvocationPlan`, `LegalToolPlan` |
| 6 | O.5b | `RagStep` / `WebsearchStep` delegate to catalog tools | **Done** — `catalog_context.py` shim |
| 7 | O.5c | Update `LegalToolPlan` / engine plans to tool list | **Done** — bridge passes `tool_ids` |
| 8 | O.6 | MCP + OpenAI exporters from single catalog | **Done** — `tools/exporters/` |
| 9 | O.7 | Remove `ToolBase` usage from production paths | **Done** — `ChatAgent` uses registry `ToolRegistry` |
| 10 | O.8–O.10 | Scaffold, docs, gate | **Done** |

#### O.4 — Adding a new tool provider (checklist)

Copy into every `tools/providers/<domain>/USAGE.md`:

```text
[ ] 1. Define Input/Output Pydantic models (LLM-friendly field names)
[ ] 2. Implement ToolHandler — compose integration contract(s), no vendor SDK
[ ] 3. Build ToolContract per tool (description tuned for model selection)
[ ] 4. register_<domain>_tools(registry, ctx: ToolWiringContext)
[ ] 5. Register in tools/registry/catalog.py
[ ] 6. Unit tests with fakes (no live vendor in default gate)
[ ] 7. Wire in lab or poc_template via ToolProfile
[ ] 8. Update TOOLS.md status + this plan tracker
```

#### O.5 — Unified tool model (migration design)

**Problem:** Two parallel mechanisms — boolean plan flags dispatching pipeline steps vs `ToolRegistry` for function tools.

**Target:** One registry, one invoker, one policy surface.

```text
BEFORE (legacy):
  plan.use_rag=True        → RagStep (direct)
  plan.use_websearch=True  → WebsearchStep (direct)
  plan.use_tools=True      → ToolsStep → ToolRegistry

AFTER (canonical):
  plan.tool_ids=["rag.retrieve", "websearch.query", "jira.search_tasks"]
      → ToolRuntime.invoke_request (per id)
      → RuntimeToolInvoker → handler
      → integration / RAG module
```

**Compatibility (O.5a):** `ToolInvocationPlan.from_legacy(use_rag=…)` maps booleans to default tool_ids. Emit deprecation trace when legacy fields used.

**Context injection:** `rag.retrieve` and `websearch.query` set `injects_context=true`; invoker callback or Nexus hook merges bounded output into prompt assembly (§22.1).

**Out of scope for Phase O:**

- Domain-specific tools inside `agents/` (stay Tier-2; register via `ToolProvider` if reusable)
- Replacing `ToolsAgent` planner — it remains the LLM loop over `ToolRegistry`
- New integration categories (still Phase M / §5.2.4)

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
| Q.0.1 | Appendix C traceability matrix (audit → Q ID) | **Open** | Docs | C-all | Appendix C below; each row has owner phase |
| Q.0.2 | Phase Q execution order + PR sizing guide | **Open** | Docs | — | §4 + subsection **Q execution order** below |
| Q.0.3 | Gate policy: no Q PR without `pytest -m gate` | **Open** | CI | — | Documented in Q DoD; CI unchanged paths |

---

#### Phase Q-N — Nexus, loops, orchestration, error handling

**Components:** `intergrax/runtime/nexus/`, `intergrax/runtime/execution/`, `intergrax/runtime/hooks/`, `intergrax/runtime/interrupts/`, `intergrax/runtime/policy/`, `intergrax/runtime/nexus/retry/`, `intergrax/agents/agent_engine.py`, `intergrax/agents/uaep.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-N.1 | **Decompose `NexusLoop`** — extract HITL runner, long-running coordinator calls, event publisher, shadow/sandbox cleanup into dedicated modules; `NexusLoop` orchestrates only | **Open** | High | `nexus_loop.py` → `nexus/orchestration/` or `nexus/task_runner/` | `nexus_loop.py` &lt; ~600 lines; behavior unchanged; integration tests green |
| Q-N.2 | **Fix duplicate `_normalize_human_response`** — single call in `_handle_task_impl` | **Done** | High | `nexus_loop.py` L229–231 | Duplicate call removed (2026-06-01) |
| Q-N.3 | **Retry semantics document + facade** — one doc section: `RetryEngine` (graph/validation/alternate agent) vs `RuntimeConfig.max_run_retries` (LLM/tool in `RuntimeEngine`); optional `RetryCoordinator` delegating both | **Open** | High | `nexus/retry/`, `nexus/config.py`, `docs/` or canon §31 pointer | Doc merged; no duplicate retry without trace event |
| Q-N.4 | **Unify policy injection** — `PolicyEngine` only in public Nexus/UAEP APIs; remove `RuntimePolicyEngine` union from external signatures; `coerce_policy_engine` internal | **Open** | Medium | `nexus_loop.py`, `uaep.py`, factories | Type check / mypy clean on factories; gate green |
| Q-N.5 | **§42 hook parity — decision / interrupt / retry** — wire `BEFORE/AFTER_DECISION`, `BEFORE/AFTER_INTERRUPT`, `BEFORE/AFTER_RETRY` in NexusLoop + UAEP + `RetryEngine`; update `hooks/parity.py` to **WIRED** or **Won't fix** with canon amendment | **Open** | Medium | `hooks/`, `nexus_loop.py`, `uaep.py`, `retry_engine.py` | `parity.py` no NOT_WIRED for these six OR canon §42.20 amended + tests |
| Q-N.6 | **§42 hook parity — trace persist** — `BEFORE/AFTER_TRACE_PERSIST` **WIRED** at trace finalize path; `parity.py` → **WIRED** | **Open** | Medium | `hooks/`, `task_trace.py`, trace emitter | Parity test; hook invoked in integration test |
| Q-N.7 | **Rename Nexus context helpers module** — `runtime_steps/tools.py` → `runtime_steps/tool_context_helpers.py` (or merge into `tools_step.py`); update imports | **Done** | Low | `tool_context_helpers.py` + shim `tools.py` | Backward-compatible re-export (2026-06-01) |
| Q-N.8 | **Split `RuntimeConfig`** — `ModelRuntimeConfig`, `RetrievalRuntimeConfig`, `ToolsRuntimeConfig`, `PlanningRuntimeConfig`, `TraceRuntimeConfig`; composed `RuntimeConfig`; `validate()` cross-field | **Open** | High | `nexus/config.py` | Backward-compatible properties or migration shim one release; all factories updated |
| Q-N.9 | **Type `integration_profile`** — `IntegrationProfile` from `intergrax.integrations` on `RuntimeConfig` / wiring contexts | **Open** | Medium | `nexus/config.py`, `engine/runtime_context.py` | No `Optional[object]` for profile in public config |
| Q-N.10 | **`production_mode` lab default** — `lab_application` / scaffold sets `production_mode=False`; document in Step 4E | **Open** | Low | Tier-3 factories, `AGENT_CREATION_GUIDE.md` | Lab smoke test expects non-production defaults |
| Q-N.11 | **Graph callback typing** — `ExecutionNode` instead of `object` in `GraphExecutor` / NexusLoop node callbacks | **Open** | Low | `execution/graph_executor.py`, `nexus_loop.py` | Mypy/ruff on execution package |
| Q-N.12 | **Interrupt handler hygiene** — remove duplicate `InterruptType` import; add unit test for interrupt → policy path | **Done** | Low | `interrupts/handler.py` | Duplicate import removed (2026-06-01) |
| Q-N.13 | **`AgentEngine` static UAEP** — document or inject `event_bus` for `AgentEngine.run` static path; no silent missing events | **Open** | Low | `agents/agent_engine.py` | Test: static run emits expected lifecycle events when bus configured |
| Q-N.14 | **Unit tests for `NexusLoop` helpers** — `_finish_task`, lifecycle transitions, HITL branch stubs (mock deps) | **Open** | High | `tests/unit/runtime/nexus/test_nexus_loop.py` | New file; ≥15 focused tests; marker `gate` |
| Q-N.15 | **`GraphExecutor` unit coverage** — failure recovery, skip completed, handoff edge (beyond stub integration) | **Open** | Medium | `tests/unit/runtime/execution/` | Complements `test_graph_executor_stub.py` |

---

#### Phase Q-L — LLM adapters

**Components:** `intergrax/llm_adapters/`, `docs/LLM_ADAPTERS.md`, `docs/LLM_OBSERVABILITY.md`, governance plugin.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-L.1 | **Remove or complete `tracked_llm_call`** — if kept: `finally` calls `usage.end_call`; if removed: delete `tracked_call.py` + references | **Done** | Medium | `_shared/tracked_call.py` | File removed (unused) (2026-06-01) |
| Q-L.2 | **Public API surface** — re-export `LLMAdapter`, `LLMProvider`, `LLMAdapterRegistry`, `LLMProfile` from `llm_adapters/__init__.py` | **Done** | Low | `llm_adapters/__init__.py` | Public re-exports (2026-06-01) |
| Q-L.3 | **Provider catalog table in docs** — 19 rows: slug, adapter class, env vars, tools/stream/structured, native vs compat | **Open** | High | `docs/LLM_ADAPTERS.md` | Table matches `LLMProvider` enum + conformance list |
| Q-L.4 | **Fix `LLMProfile` docstring** — `max_retries` only via `options={}`; align examples in guide | **Done** | Low | `registry/profile.py`, tests | Example fixed (2026-06-01) |
| Q-L.5 | **Per-provider `supports_streaming()` / `supports_structured_output()`** — override defaults (`False` base default for streaming); table in Q-L.3 | **Open** | Medium | Each `providers/*.py`, ABC defaults | Conformance reads flags; no false positives |
| Q-L.6 | **`PolicyEngine` + `llm_cost_evaluation`** — rule hook on `TASK_COMPLETED` or policy replay; or remove “next step” from docs until done | **Open** | Medium | `governance/`, `observability_bridge.py`, `policy_engine.py` | Test: over-quota/warn triggers policy decision or structured log contract |
| Q-L.7 | **Usage tracking doc** — distinguish adapter `LLMAdapterUsageLog` vs runtime `LLMUsageTracker` in `LLM_OBSERVABILITY.md` | **Open** | Low | `docs/LLM_OBSERVABILITY.md` | One diagram, two names |
| Q-L.8 | **Conformance: structured output** — parametrize providers with `supports_structured_output`; mock SDK | **Open** | Medium | `tests/unit/llm_adapters/` | Added to gate subset in `llm-adapters-guard.yml` |
| Q-L.9 | **Bedrock `context_window_tokens`** — lookup table or model metadata for common `model_id` | **Open** | Low | `providers/aws_bedrock_adapter.py` | Unit test known model ids |
| Q-L.10 | **OpenAI-compat adapter init** — replace `__dict__.update` with explicit delegation or composition wrapper | **Open** | Low | `openai_compat_providers.py`, factory | Conformance unchanged |
| Q-L.11 | **Central env appendix** — single table: `INTERGRAX_LLM_*`, secrets map, per-provider overrides | **Open** | Medium | `LLM_ADAPTERS.md` appendix | Cross-links from each `providers/*/USAGE.md` |

---

#### Phase Q-R — RAG pipeline & Nexus RAG integration

**Components:** `intergrax/rag/`, `runtime/nexus/context/context_builder.py`, `runtime_steps/rag_step.py`, `history_step.py`, `pipelines/no_planner_pipeline.py`, `tools/providers/rag/`, `agents/legal/*` plan flags.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-R.1 | **Delete dead code in `ContextBuilder`** — `_build_backend_where`, `_map_hits_to_chunks`, unused `VectorStoreHit` import | **Done** | High | `context_builder.py` | Dead helpers removed (2026-06-01) |
| Q-R.2 | **Single retrieval per turn (design)** — ADR in plan: either (A) retrieval only in `RagStep`/`rag.retrieve`, or (B) only in `HistoryStep`; remove duplicate vector calls | **Done** | High | `history_step.py`, `context_builder.py` | `HistoryStep` uses `perform_retrieval=False` (2026-06-01) |
| Q-R.3 | **`ContextBuilder` respects plan `use_rag`** — `_should_use_rag` checks plan/engine `use_rag` when present, not only `enable_rag` | **Done** | High | `context_builder.py` | `request.metadata["use_rag"]`; unit test (2026-06-01) |
| Q-R.4 | **`NoPlannerPipeline` conditional `RagStep`** — include `RagStep` only when plan/tool_ids require RAG | **Open** | High | `no_planner_pipeline.py`, `pipeline_factory.py` | Pipeline test matrix |
| Q-R.5 | **Prefetch vs final `top_k`** — `RetrievalRequest.prefetch_k` optional; Nexus passes `max_docs_per_query` as `final_k` only; service uses profile `prefetch_top_k` when unset | **Done** | High | `retrieval_request.py`, `retrieval_service.py` | `test_retrieval_request_prefetch.py` (2026-06-01) |
| Q-R.6 | **Unify RAG config surface** — map `RuntimeConfig.max_docs_per_query` / threshold → `RagProfile` at factory wire time; deprecate duplicate fields with shim + trace | **Open** | High | `nexus/config.py`, `RetrievalRuntimeConfig`, `rag_profile.py` | One source of truth documented |
| Q-R.7 | **`RagProfile.extras`** — use for vendor knobs or remove field | **Open** | Low | `rag_profile.py` | No unused field in frozen profile |
| Q-R.8 | **`INTERGRAX_RAG_METRICS_ENABLED` in `rag_profile_from_env`** or documented exclusion in `RAG_OBSERVABILITY.md` | **Done** | Low | `rag_profile.py` | `extras.metrics_enabled` from env (2026-06-01) |
| Q-R.9 | **`rag/answers/` deprecation path** — mark package deprecated; redirect doc to `RetrievalService`; no new imports from Nexus | **Open** | Medium | `rag/answers/`, `chat_agent` removal (Q-X.1) | Grep: zero imports from `runtime/` and `agents/` except tests |
| Q-R.10 | **`UserProfileManager` LTM via `RetrievalService`** — same metadata scope / `RagProfile` chunking policy | **Open** | Medium | `memory/user_profile_manager.py` | Unit test with fake `RetrievalService` |
| Q-R.11 | **Naming guide — three “context builders”** — table in `AGENT_CREATION_GUIDE` or `intergrax/rag/README.md`: Nexus `ContextBuilder`, `ContextManager`, `DefaultContextBuilder` | **Open** | Low | Docs | Linked from architecture §28 pointer |
| Q-R.12 | **Legacy `use_rag` plan flags** — migrate Legal/Nexus plans to `tool_ids` including `rag.retrieve`; emit deprecation `RuntimeEvent` on boolean | **Open** | Medium | `engine_plan_models.py`, `legal/*`, `tool_runtime.py` | Legal tests use `tool_ids`; booleans shim one release |

---

#### Phase Q-M — Memory

**Components:** `intergrax/memory/`, `runtime/task_memory/`, `runtime/nexus/context/`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-M.1 | **Memory architecture one-pager** — four stores: session history, user LTM, task KV (`TaskMemory`), shared graph context; diagram + when to enable SQLite | **Open** | High | `docs/` section in plan §0 or `AGENT_CREATION_GUIDE` Appendix | Linked from §0.3 execution path |
| Q-M.2 | **Task memory visibility in scaffold** — `wire_task_memory` in lab/product templates; env `INTERGRAX_TASK_MEMORY_DB` in `.env.example`; Step 4E paragraph | **Open** | Medium | `applications/*`, scaffold, guide | Scaffold acceptance asserts task memory path optional |
| Q-M.3 | **`resolve_task_memory_persistence` defaults** — log warning when None in lab; debug API hint | **Open** | Low | `task_memory/store.py`, `lab_application` factory | Doc + single integration test |

---

#### Phase Q-O — Observability & metrics

**Components:** `runtime/events/`, `runtime/nexus/tracing/`, `runtime/metrics/`, `debug/`, `llm_adapters/tracking/`, `rag/tracking/`, `applications/_shared/platform_wiring.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-O.1 | **Register RAG observability plugin in default bootstrap** — `register_rag_observability_plugin(plugins)` alongside LLM in `platform_wiring.py` | **Done** | **Critical** | `platform_wiring.py` | `test_platform_wiring_observability.py` (2026-06-01) |
| Q-O.2 | **RAG observability bridge tests** — mirror `test_observability_bridge.py` (LLM) | **Done** | High | `tests/unit/rag/tracking/` | `test_rag_observability_bridge.py` (2026-06-01) |
| Q-O.3 | **Parser trace export strategy** — route `parser_trace_flush` through `ObservabilityBackend` **or** document intentional bypass + single env table | **Open** | Medium | `parser_trace_flush.py`, `parser_trace_exporter.py`, integrations | ADR paragraph in `RAG_OBSERVABILITY.md` or refactor |
| Q-O.4 | **`metrics/export.py` typed trace summary** — use `DiagnosticPayload` / `trace_models` schema ids instead of substring heuristics | **Open** | Medium | `runtime/metrics/export.py` | Unit test with synthetic trace events |
| Q-O.5 | **Lint `metrics/export.py`** — remove duplicate `ExecutionMetrics` import | **Done** | Low | `metrics/export.py` | Ruff clean (2026-06-01) |
| Q-O.6 | **`export_run_metrics` behavioral field** — populate from governance/replay or remove from DTO | **Open** | Low | `metrics/export.py` | API contract test |
| Q-O.7 | **Mount LLM metrics routes on lab** — `register_llm_metrics_routes(app)` when `INTERGRAX_LLM_METRICS_ENABLED` | **Done** | Medium | `lab_application/host/factory.py` | Routes registered at factory (2026-06-01) |
| Q-O.8 | **Observability env profile doc** — one table: trace DB, runtime events DB, LLM/RAG metrics, parser trace, integration observability slug | **Open** | High | New subsection §0 or `infra/README` cross-link | All Tier-3 `.env.example` reference same names |
| Q-O.9 | **RAG metrics parity decision** — implement log-only parity **or** `register_rag_metrics_routes` + optional Pushgateway; update `RAG_OBSERVABILITY.md` | **Open** | Medium | `rag/tracking/` | Matches documented behavior |
| Q-O.10 | **Unify phase mapping** — `trace_bridge` delegates phase to `phase_coverage.py`; single source | **Open** | Medium | `events/trace_bridge.py`, `phase_coverage.py` | Unit test: same `ExecutionPhase` for sample events |
| Q-O.11 | **Debug router type imports** — explicit imports for `DebugHitlResumeService`, `AgentRegistry` in annotations | **Open** | Low | `debug/router.py`, `debug/app.py` | Mypy/ruff on debug package |
| Q-O.12 | **`trace_bridge` unit tests** | **Open** | Medium | `tests/unit/runtime/events/test_trace_bridge.py` | Gate marker |
| Q-O.13 | **Clarify dual Prometheus** — in `LLM_OBSERVABILITY.md`: in-process scrape vs `integrations` PromQL backend | **Open** | Low | Docs | Prevents operator confusion |
| Q-O.14 | **Event/trace store adoption ADR** — document SQLite-first default; criteria to wire `cassandra` / `elasticsearch` for runtime events at scale | **Open** | Low | Plan + `integrations/providers/cassandra/USAGE.md` cross-link | ADR in Phase Q or canon pointer; no mandatory Cassandra migration in Q |

---

#### Phase Q-X — Legacy removal & code hygiene

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-X.1 | **`ChatAgent` removal** — migrate remaining tests to `RuntimeEngine`/`NexusLoop`; delete `intergrax/chat_agent.py`; keep import guard script as negative test | **Open** | High | `chat_agent.py`, `tests/unit/chat_agent/` | Grep zero production imports; gate green |
| Q-X.2 | **`task_metadata_bridge` shrink** — migrate callers to typed `Task` metadata; deprecate flat bridge with warning event | **Open** | Medium | `task_metadata_bridge.py`, factories | Fewer bridge calls each PR until removed |
| Q-X.3 | **Copyright / naming consistency** — `Intergrax` header; fix `Integrax` typo in `chat_agent` (or file deleted in Q-X.1) | **Open** | Low | Affected files from audit | Spot-check script or ruff rule |
| Q-X.4 | **`tools_base` deprecation timeline** — document removal after Q-R.12; no new imports | **Open** | Low | `tools/tools_base.py`, governance script | CI grep gate extended |
| Q-X.5 | **Sync M.6 “Future” slugs table** — weaviate, milvus, snowflake, vault → **Done (beta)** with paths | **Open** | Low | This plan M.6 P3 section | Table matches repo `integrations/providers/` |

---

#### Phase Q-T — Test harness gaps

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-T.1 | NexusLoop unit suite | **Open** | High | See Q-N.14 | — |
| Q-T.2 | `test_rag_profile_from_env` | **Done** | Medium | `tests/unit/rag/profiles/` | Gate (2026-06-01) |
| Q-T.3 | `test_context_builder_retrieval` | **Done** | High | `tests/unit/runtime/nexus/context/` | `test_context_builder.py` (2026-06-01) |
| Q-T.4 | `test_user_profile_manager` | **Open** | Medium | `tests/unit/memory/` | Index + search |
| Q-T.5 | **Catalog vs legacy RAG path** — integration test one pipeline run, retrieval call count ≤1 | **Open** | High | `tests/integration/runtime/` | Implements Q-R.2 acceptance |
| Q-T.6 | **Observability wiring E2E** — lab factory bootstraps LLM+RAG plugins | **Done** | High | `tests/integration/runtime/test_platform_wiring_observability.py` | Q-O.1 (2026-06-01) |

---

#### Phase Q-D — Documentation & plan sync

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-D.1 | Update `docs/README.md` current focus → Phase Q | **Open** | High | `docs/README.md` | — |
| Q-D.2 | Canon §52 Phase L status → **Done** (pointer to Phase Q) | **Open** | Low | `intergrax_runtime_architecture.md` §52 | — |
| Q-D.3 | §2 architecture map — §42 row points to Phase Q-N.5–Q-N.6 | **Open** | Low | This file §2 | — |
| Q-D.4 | `AGENT_CREATION_GUIDE` — Q-M.1 memory diagram + Q-R.11 naming | **Open** | Medium | Guide appendices | — |
| Q-D.5 | **§5.2 reuse enforcement** — document existing gates (`check_agents_vendor_imports`, `check_integration_vendor_imports`, `check_production_chat_agent_imports`) in AGENT_CREATION_GUIDE anti-patterns | **Open** | Low | Guide + `scripts/` | New agent authors see one list |

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

**Do not start Phase K.1/K.2** until Wave 1–3 are **Done** unless product explicitly overrides (record in Appendix C).

---

## 4. Priority Order



```text

NOW:     Phase Q — Harness Quality & Consolidation (audit 2026-06-01) — Waves 1–8 in Phase Q execution order

DONE:    Phase L, M, M-LLM, M-RAG, N, O — harness GA (functional)

DONE:    Phase K hardening K.3–K.5; Appendix B paydown (except B.15)

PARALLEL: M.6 additional provider slugs (on demand) — only when a Q wave needs a slug

AFTER Q: Product decision — Phase K.1/K.2 (Problem Radar / Vendor Discovery)

DEFERRED: K.6 / B.15 Legal live LLM E2E (product/CI; not Phase Q)

RULE:    Tier-1 changes only via §0.6; Phase Q-N.1/N.5/N.8 are approved harness consolidation

```

**Rationale:** Phase Q removes audit findings (RAG double-fetch, RAG metrics bootstrap gap, Nexus monolith, config drift, dead code, doc drift) so Phase K agents inherit a **single semantic path** for RAG, retry, observability, and memory — not a second platform build.



---



## 5. Definition of Done (Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)



---



## 6. Recommended Next Step

### 6.1 Implementation cadence (Phase Q — current)

Each iteration follows **one deliverable at a time** (same as Phase N):

```text
1. Implement   — single Q.* ID (e.g. Q-O.1 only)
2. Summarize   — what changed, tests run, acceptance criteria met
3. Document    — update Phase Q status + Appendix C row + paydown log (B.0)
4. Next step   — next ID from Phase Q execution order (Wave 1 first)
```

Do not batch Wave 1 items in one PR unless explicitly agreed. **Start with Q-O.1** (RAG metrics plugin registration).

### 6.1b Phase N (complete)

Phase N cadence remains the reference for Tier-3 scaffold work; new applications use `new-stack` without waiting for Phase Q completion, but **lab defaults** should pick up Q-M.2/Q-O.7 when those land.

### 6.2 Tier-3 application layer — **ready for new applications**

Phase N deliverables **N.0–N.10** are complete. Use [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md) before scaffolding.

**Generate:**

```bash
python -m intergrax.scaffold new-stack <slug> --profile lab --capability <slug>.basic
```

**Verify:**

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest tests/unit/applications/test_scaffold_acceptance.py -q
uv run pytest -m gate -q
```

**Phase L — still required for any new agent work:**

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
```

**Do not start:** Problem Radar, Vendor Discovery (K.1/K.2) until product priority is set — Phase N does not block K.

**Recently completed:**

- **L.1–L.8:** Phase L Agent OS readiness — UAEP scaffold, `AGENT_CREATION_GUIDE.md`, `lab_application`, mock agents, acceptance suite (`tests/acceptance/agent_os/`), readiness docs; gate **228 passed**.
- **J.5:** Partial results API — `GET /debug/tasks/{task_id}/progress` aggregates checkpoint partial snapshots; `RuntimeEventType.TASK_PROGRESS`; `partial_result.v1` notification template; gate **228 passed** (includes Phase L acceptance).
- **J.4:** Long-running scheduler — in-process `LongRunningScheduler` polls checkpoint store for delayed resumes and expired HITL deadlines; `UnifiedTaskResumeExecutor` resumes via `UnifiedTaskRunner`; SQLite ledger for idempotency.
- **J.3:** Worker queue Task v2 — `QueuedNexusExecutionAdapter` enqueues `ExecutionRequest` via Tier-0 Celery (`nexus.task.v2`); `create_nexus_celery_worker_app` bootstrap; checkpoint resume through worker payload; gate **207 passed**.
- **J.2:** RunService → UnifiedTaskRunner — `NexusTaskExecutionAdapter` delegates to `UnifiedTaskRunner`; Legal host shares one runner for `/runs` and `/legal/chat`; `POST /runs` forwards `CreateRunRequest.payload` to Task intake.
- **J.1:** NexusLoop default in apps — Legal and Research HTTP use `UnifiedTaskRunner` only; legacy `AgentEngine` opt-out removed from Legal (B.14).

- **I.5:** ContextManager v2 — provenance, summary tiers, typed `TaskContextAssemblyOptions` (`TaskExecutionOptions.context`), metadata bridge sync.
- **I.4:** Agent handoff — `AgentHandoff`, `HandoffCoordinator`, graph executor path, `HANDOFF_*` events.
- **I.3:** `SharedTaskContext` — formal payload on `Task`, `ContextManager` merge, memory read bridge.
- **I.2:** `MemoryView` gateway — `PolicyScopedMemoryView`, UAEP wiring, `MEMORY_READ`/`MEMORY_WRITE` events.
- **I.1:** `TaskMemory` store — `TaskMemoryPersistence`, `TaskMemoryCoordinator`, in-memory + SQLite backends.
- **H.6:** Organization Worker demo — `OrganizationWorkerAgent`, `create_organization_worker_lab_app`, E2E Slack/Teams intake → HITL → resume.
- **H.5:** Teams parity — `TeamsActivityInteractionAdapter`, `TeamsSignatureVerifier`, debug intake tests mirroring H.3.
- **H.4:** HITL notification templates — `HitlPauseNotificationTemplate`, `notify_hitl_pause`, Slack/Teams formatter parity for actions/urgency.
- **H.3:** Debug API `POST /debug/interactions/intake` — JSON/form body, optional `execute`, Slack signature verifier (opt-in).
- **H.2:** `InteractionAdapter` — inbound parsers, factory, `ChainedInteractionAdapter`.
- **G.6:** Debug API — `GET …/events`, `GET …/checkpoints`, `POST …/human-response` with injectable `RuntimeEventPersistence` / `TaskCheckpointPersistence`.
- **G.5:** pluggable `RuntimeEventPersistence` + memory/SQLite backends.
- **F.5:** typed task contract — `TaskExecutionOptions` / `TaskRuntimeState` / `TaskResultSummary` + metadata bridge.
- **F.4:** long-running task snapshots + notification adapter stubs (Slack/Teams **not** real integration).
- **F.1–F.3:** Shadow, Sandbox, advanced HITL.



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

### D.4 Notebook templates (Done)

Interactive §35 workflow under `notebooks/experiments/`:

| File | Purpose |
|------|---------|
| `00_experiment_template.ipynb` | Blank template — copy for new capabilities |
| `01_echo_experiment.ipynb` | Deterministic Echo smoke test |

Shared API: `intergrax.experiments.workflow.ExperimentSession`.

```python
from intergrax.experiments.workflow import ExperimentSession, ensure_repo_root_on_path
ensure_repo_root_on_path()
session = ExperimentSession(trace_db=Path("build/notebooks/trace.db"))
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

**Verdict:** **L1 Agent Operating System certified** (technical). Phase K opens on product decision — not automatic runtime work.

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
Decision:       L1 certified — GO Phase K when product priority set
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
| 2026-05-30 | M.7-agent-guide-integrations | `AGENT_CREATION_GUIDE.md` Appendix E — agents vs Tier-3 wiring |
| 2026-05-30 | N.2.1-unified-wiring | `ApplicationBuildContext`, `builder_key`/`factory_path`, lab+legal on `build_application_registry` |
| 2026-05-30 | N.2-conformance | `build_registry_from_manifest`, `load_agent_from_binding` + unit tests |
| 2026-05-30 | N.1-manifest | `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` + unit tests |
| 2026-05-30 | N.10-new-stack | `scaffold new-stack` — agent + application; `TIER3_READINESS.md` |
| 2026-05-30 | N.9-scaffold-acceptance | `test_scaffold_acceptance.py` — lab/product runtime E2E; fix product `agent_factories.py` indent |
| 2026-05-30 | N.8-agent-guide-4e | `AGENT_CREATION_GUIDE.md` Step 4E — `new-application`, Docker scripts, §7.4.8 links |
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
| B.04 | **Dual `AgentDecision` cleanup** — converge tools-agent variant with canonical §42.7 enum | §42.7 | **Medium** | **Done** | Agents emitting decisions must use one contract | Tier-1 | `ToolPlanDecision` / `ToolsAgentRunResult`; deprecated `tools_agent` aliases (2026-05-27) |
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

### B.6 Suggested priority order (for planning)

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
11. **Phase Q** — Harness audit remediation (Waves 1–8) — **NOW**
12. Phase K — Problem Radar / Vendor Discovery (**after Q Waves 1–3**)
```

**Note:** Phase K business agents remain **product-gated** — Phase Q is the recommended technical prerequisite (Waves 1–3).

---

## Appendix C — Harness audit traceability (Phase Q)

**Purpose:** Every finding from the harness implementation audit (2026-06-01) maps to exactly one Phase Q deliverable. Update **Status** when the deliverable is **Done** / **Won't fix** (with reason).

**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### C.1 Nexus, loops, orchestration, errors

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| N-01 | `NexusLoop` monolith ~1200 lines | Q-N.1 | Open |
| N-02 | Duplicate `_normalize_human_response` | Q-N.2 | Open |
| N-03 | Dual retry (`RetryEngine` vs `max_run_retries`) | Q-N.3 | Open |
| N-04 | `PolicyEngine` \| `RuntimePolicyEngine` union | Q-N.4 | Open |
| N-05 | Hooks NOT_WIRED: decision, interrupt, retry | Q-N.5 | Open |
| N-06 | Hooks PARTIAL: trace persist | Q-N.6 | Open |
| N-07 | `runtime_steps/tools.py` misleading name | Q-N.7 | Open |
| N-08 | `RuntimeConfig` monolith | Q-N.8 | Open |
| N-09 | `integration_profile: object` | Q-N.9 | Open |
| N-10 | `production_mode` default in lab | Q-N.10 | Open |
| N-11 | Graph callbacks typed `object` | Q-N.11 | Open |
| N-12 | Duplicate import `InterruptType` | Q-N.12 | Open |
| N-13 | `AgentEngine` static UAEP / event_bus | Q-N.13 | Open |
| N-14 | No unit tests `nexus_loop.py` | Q-N.14 | Open |
| N-15 | Thin `GraphExecutor` unit coverage | Q-N.15 | Open |

### C.2 LLM adapters

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| L-01 | Dead `tracked_llm_call` | Q-L.1 | Open |
| L-02 | Empty `llm_adapters/__init__.py` | Q-L.2 | Open |
| L-03 | `LLM_ADAPTERS.md` missing provider table | Q-L.3 | Open |
| L-04 | `LLMProfile` docstring `max_retries` wrong | Q-L.4 | Open |
| L-05 | `supports_streaming()` default True | Q-L.5 | Open |
| L-06 | PolicyEngine ignores `llm_cost_evaluation` | Q-L.6 | Open |
| L-07 | Dual usage tracking naming | Q-L.7 | Open |
| L-08 | No structured-output conformance | Q-L.8 | Open |
| L-09 | Bedrock context_window TODO | Q-L.9 | Open |
| L-10 | OpenAI-compat `__dict__.update` fragility | Q-L.10 | Open |
| L-11 | Env vars scattered | Q-L.11 | Open |

### C.3 RAG

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| R-01 | Dead `_build_backend_where` / `_map_hits_to_chunks` | Q-R.1 | Open |
| R-02 | Four parallel retrieval paths | Q-R.2 | Open |
| R-03 | `enable_rag` vs `use_rag` in ContextBuilder | Q-R.3 | Open |
| R-04 | `NoPlannerPipeline` always `RagStep` | Q-R.4 | Open |
| R-05 | `top_k` collapses prefetch | Q-R.5 | Open |
| R-06 | `RuntimeConfig` vs `RagProfile` dual config | Q-R.6 | Open |
| R-07 | Unused `RagProfile.extras` | Q-R.7 | Open |
| R-08 | RAG metrics env not in profile | Q-R.8 | Open |
| R-09 | `rag/answers/` parallel stack | Q-R.9 | Open |
| R-10 | `UserProfileManager` bypasses `RetrievalService` | Q-R.10 | Open |
| R-11 | Three “context builder” names | Q-R.11 | Open |
| R-12 | Legacy `use_rag` plan booleans | Q-R.12 | Open |

### C.4 Memory

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| M-01 | No single memory architecture doc | Q-M.1 | Open |
| M-02 | Task memory not visible in scaffold | Q-M.2 | Open |
| M-03 | Silent default when task memory None | Q-M.3 | Open |

### C.5 Observability & metrics

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| O-01 | RAG plugin not in `platform_wiring` | Q-O.1 | Open |
| O-02 | No RAG bridge tests | Q-O.2 | Open |
| O-03 | Parser trace bypasses `ObservabilityBackend` | Q-O.3 | Open |
| O-04 | `metrics/export` substring heuristics | Q-O.4 | Open |
| O-05 | Duplicate import in `metrics/export.py` | Q-O.5 | Open |
| O-06 | `behavioral` never set in export | Q-O.6 | Open |
| O-07 | `/metrics/llm` not on lab host | Q-O.7 | Open |
| O-08 | Observability env scattered | Q-O.8 | Open |
| O-09 | RAG metrics asymmetry vs LLM | Q-O.9 | Open |
| O-10 | `trace_bridge` vs `phase_coverage` drift | Q-O.10 | Open |
| O-11 | Debug router missing type imports | Q-O.11 | Open |
| O-12 | No `trace_bridge` unit tests | Q-O.12 | Open |
| O-13 | Two Prometheus concepts unclear | Q-O.13 | Open |
| O-14 | Runtime events SQLite-first; Cassandra adoption undefined | Q-O.14 | Open |

### C.6 Legacy, style, docs

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| X-01 | Deprecated `ChatAgent` | Q-X.1 | Open |
| X-02 | `task_metadata_bridge` legacy | Q-X.2 | Open |
| X-03 | Copyright / Integrax typo | Q-X.3 | Open |
| X-04 | `tools_base` deprecation | Q-X.4 | Open |
| X-05 | M.6 Future slugs table stale | Q-X.5 | Open |
| D-01 | `docs/README` focus outdated | Q-D.1 | Open |
| D-02 | Canon §52 still “Active” | Q-D.2 | Open |
| D-03 | §0.1 “blocked until L” stale | Q-D.1 (§0.1 fix) | Done |
| D-04 | Guide missing memory/RAG naming | Q-D.4 | Open |
| D-05 | §5.2 process gates not listed for agent authors | Q-D.5 | Open |

### C.7 Tests (cross-cutting)

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| T-01 | NexusLoop unit suite | Q-T.1 / Q-N.14 | Open |
| T-02 | `rag_profile_from_env` tests | Q-T.2 | Open |
| T-03 | `ContextBuilder` tests | Q-T.3 | Open |
| T-04 | `UserProfileManager` tests | Q-T.4 | Open |
| T-05 | Single retrieval per turn test | Q-T.5 | Open |
| T-06 | Platform wiring observability E2E | Q-T.6 | Open |

### C.8 Phase Q paydown log

| Date | Q ID | Summary |
|------|------|---------|
| 2026-06-01 | Q-D.3 | §0.1 strategic objective — Harness GA vs Phase K vs Phase Q |
| 2026-06-01 | Q-O.1,Q-O.2,Q-O.5,Q-O.7 | RAG plugin bootstrap, tests, metrics lint, lab `/metrics/llm` |
| 2026-06-01 | Q-N.2,Q-N.7,Q-N.12 | Duplicate HITL normalize; tool_context_helpers; interrupt import |
| 2026-06-01 | Q-R.1–Q-R.5,Q-R.8 | RAG dead code, single retrieval path, use_rag metadata, prefetch_k |
| 2026-06-01 | Q-L.1,Q-L.2,Q-L.4 | Remove tracked_llm_call; llm_adapters exports; LLMProfile docstring |
| 2026-06-01 | Q-T.2,Q-T.3,Q-T.6 | New unit/integration tests; gate **399 passed** (+2) |
| — | — | *(append row per merged PR)* |

**Coverage:** 58 audit rows → 49 unique Q deliverables (some Q IDs satisfy multiple rows). **Target:** 100% **Done** or **Won't fix** before declaring Phase Q complete.

**Appendix B relationship:** Open items B.06 (hook remainder), B.07 (baseline beyond), and observability beta notes are **closed by Phase Q** when corresponding Q-N / Q-O rows are **Done**. B.15 and K.* remain outside Phase Q.

---

*Plan synced after harness audit (2026-06-01). **Phase Q** open — see Appendix C. Gate: **397 passed** (must stay green). Phase K.1/K.2 after Q Waves 1–3 unless product override. Legal live LLM E2E (K.6/B.15) deferred.*


# Implementation Phases — Historical (A–V)

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## 3. Implementation Phases



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

**Harness prerequisites:** L, Q+, R, S, T, U, and §4.1 **Done** — platform is ready **when** product chooses to start Band 3 (§6.3).

**Scheduling rule (2026-06-02):** K.1/K.2 are **end-of-plan** (§4.0 Band 3, §6.3). Completing harness phases does **not** auto-schedule business agents as the next implementation task.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| K.1 | Problem Radar prototype | **Deferred** | §36 | Wave-1 scaffold frozen (`agents/problem_radar/`); resume after harness backlog |
| K.2 | Vendor Discovery prototype | **Deferred** | §37 | After Phase S; product decision |
| K.3 | Policy engine facade | **Done** | §42.11 | `PolicyEngine` + `coerce_replay_policy_engine`; `ExecutionGuard` uses `evaluate_replay` (2026-05-27) |
| K.4 | Dual `AgentDecision` cleanup | **Done** | §42.7 | `ToolPlanDecision` in `tools.core.tool_plan_decision`; no `tools_agent` alias (TYP-06, 2026-06-02) |
| K.5 | ChatAgent / legacy removal | **Done** | §39 | Production paths use Nexus only; `check_production_chat_agent_imports.py` gate (2026-05-27) |
| K.6 | A.5 full Legal E2E gate | **Deferred** | — | Real LLM; not blocking lab — product/CI decision |

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

**Catalog (2026-06-02):** **167** slugs in `layout.py` · **12** core / **167** full preset · timeline: pre-P4 **99** → M.6 P4 **127** (+28) → M.6 P5 **135** (+8 greenfield, 25 hardened) → M.6 P6 **167** (+32).

**Out of scope:** `intergrax/llm_adapters/` — LLM providers are **not** part of the Integration Library (§7.1.2).

### Phase M-LLM — LLM Adapter Layer (Tier-0)

**Canon:** §5.2.2 · **Doc:** [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md)  
**Goal:** One `LLMAdapter` contract, lazy registry, streaming + native tools + structured output across commercial and self-hosted providers.

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-LLM.1 | Shared `_shared/` (messages, tools, retry, conformance) | **Done** | 2026-05-30 |
| M-LLM.2 | Seven core providers hardened | **Done** | OpenAI, Claude, Azure, Gemini, Mistral, Bedrock, Ollama |
| M-LLM.3 | Groq + vLLM (OpenAI-compatible) | **Done** | `openai_compat_providers.py` |
| M-LLM.4 | Bedrock Converse + tools + stream | **Done** | `INTERGRAX_BEDROCK_USE_CONVERSE`, `converse_stream` |
| M-LLM.5 | Conformance tests in CI gate | **Done** | `tests/unit/llm_adapters/` |
| M-LLM.6 | `architecture/LLM_ADAPTERS.md` + README section | **Done** | 19 providers |
| M-LLM.7 | OpenAI-compat expansion + Vertex + `LLMProfile` | **Done** | Together, Fireworks, OpenRouter, DeepSeek, xAI, llama.cpp, Cohere, Vertex |
| M-LLM.8 | Optional network smoke workflow | **Done** | Weekly schedule + `workflow_dispatch` |
| M-LLM.9 | Azure refactor (Chat Completions base) | **Done** | Thin `AzureOpenAIChatAdapter` |
| M-LLM.10 | Production hardening | **Done** | Metrics, builtin conformance, `LLMProfile`, Bedrock tools stream, `cohere_native`, `azure_ai_inference` |
| M-LLM.11 | Production ops layer | **Done** | OTLP/Prometheus routes, tenant metrics, rate limit + circuit breaker, secrets map, PR guard, extended network smoke |
| M-LLM.12 | Nexus + governance wiring | **Done** | `llm_tenant_scope`, runtime metrics plugin, `INTERGRAX_LLM_TENANT_MAX_TOKENS` quota |
| M-LLM.13 | Observability + secrets + distributed limits | **Done** | Pushgateway, `architecture/LLM_ADAPTERS.md` § Observability, Vault loader, Redis rate limit, governance warn |
| M-LLM.14 | Typed completion envelope (`LLMAdapterResponse`) | **Done** | Phase M-LLM-R — [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed) · gate **776** |

### Phase M-LLM-R — LLM Completion Response Envelope (audit 2026-06-06)

**Source:** Tier-0 LLM adapter audit (2026-06-06) — `generate_messages` returns `str`; `generate_with_tools` returns untyped dict via `make_tool_result`; SDK metadata (`finish_reason`, `response_id`, cached/reasoning tokens, refusal) discarded; usage only via side-channel `LLMAdapterUsageLog`; replay `LLMCallInfo` not fed from adapter returns.  
**Canon:** §5.2.2 · **Doc:** [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) · **Traceability:** [Appendix L](#appendix-l--llm-completion-response-envelope-traceability-phase-m-llm-r)  
**Status:** **Done** (2026-06-06) — **39/39 Done**  
**Priority ladder:** **Band 2z** (§4.0) — **parallel with W-ADAPT waves 5–7** (Tier-0; no Nexus primitive changes beyond consumer wiring)  
**Execution order:** [§6.2ad](#62ad-phase-m-llm-r-execution-order-band-2z--closed-2026-06-06) · queue: [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)  
**Goal:** Replace plain `str` and `Dict[str, Any]` LLM adapter returns with a **single strongly typed envelope** — `LLMAdapterResponse` — carrying `content: str` plus production-standard metadata, extensible without dict soup.

**Hard rules (non-negotiable):**

- **No** public adapter method returns bare `str` or `Dict[str, Any]` for completions.
- **No** `make_tool_result` dict factory — delete after migration; use typed builders only.
- **No** untyped `tool_calls: list[dict]` — use frozen `LLMToolCall` (+ `LLMToolCallArgument` where needed).
- Per-call `usage` **must** be present on every `LLMAdapterResponse` (sync with `LLMAdapterUsageLog.end_call`; prefer SDK counts over estimates when available).
- `LLMAdapterUsageLog` remains for run-level aggregation; response envelope is the **per-call source of truth** for callers.
- One **M-LLM-R.\*** ID per PR → update master table + Appendix L + paydown log → `pytest -m gate` + `tests/unit/llm_adapters/` green.

**Canonical type (target contract):**

| Type | Role |
|------|------|
| `LLMAdapterResponse` | Primary return for `generate_messages`, `generate_with_tools`, final stream event |
| `LLMTokenUsage` | `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, `reasoning_tokens` |
| `LLMFinishReason` | Enum: `completed`, `length`, `tool_calls`, `content_filter`, `refusal`, `error`, … |
| `LLMToolCall` | Typed native tool call (`id`, `name`, `arguments_json` or validated args model) |
| `LLMStreamEvent` | Streaming partial/final chunks (`event_kind`, `delta_content`, optional `completion` on final) |
| `LLMStructuredResult[T]` | `generate_structured` → `(parsed: T, response: LLMAdapterResponse)` |
| `LLMProviderExtensions` | Tagged optional extensions (OpenAI / Anthropic / Gemini slices) — **no** open `dict` bag |

**Naming note:** `LLMAdapterResponse` (not bare `LLMResponse`) — Tier-0 adapter return type; avoids collision with HTTP transport and product API DTOs.

#### M-LLM-R — Traceability (audit gap → task ID)

| Audit gap | Task IDs |
|-----------|----------|
| `generate_messages` → `str` | M-LLM-R.2.1, M-LLM-R.3.*, M-LLM-R.4.*, M-LLM-R.5.*, M-LLM-R.6.* |
| `generate_with_tools` → `Dict[str, Any]` | M-LLM-R.1.7, M-LLM-R.2.2, M-LLM-R.3.*, M-LLM-R.4.2 |
| `stream_messages` → `Iterable[str]` | M-LLM-R.1.5, M-LLM-R.2.3, M-LLM-R.3.* |
| `stream_with_tools` → `Iterable[Dict]` | M-LLM-R.1.5, M-LLM-R.2.4, M-LLM-R.3.* |
| `generate_structured` untyped | M-LLM-R.1.6, M-LLM-R.2.5 |
| SDK metadata discarded (`finish_reason`, `response_id`, refusal) | M-LLM-R.1.1, M-LLM-R.3.1–3.6 |
| Usage only side-channel | M-LLM-R.1.2, M-LLM-R.2.6, M-LLM-R.7.1 |
| Inconsistent token counting (estimate vs SDK) | M-LLM-R.3.5, M-LLM-R.3.6 |
| Replay `LLMCallInfo` not fed from adapter | M-LLM-R.7.2, M-LLM-R.7.3 |
| `CoreLLMAdapterReturnedDiagV1.adapter_return_type="str"` | M-LLM-R.7.4 |
| Conformance asserts `isinstance(text, str)` | M-LLM-R.8.2 |
| Public API missing response types | M-LLM-R.1.8, M-LLM-R.8.1 |

#### Wave M-LLM-R-0 — Planning and canon sync

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.0.1 | **Plan register** — Phase M-LLM-R, §4.0 Band 2z, §6.1v, §6.2ad, Appendix L; M-LLM follow-up pointer | **Done** | **Critical** | This section | Cross-links from `architecture/LLM_ADAPTERS.md` |
| M-LLM-R.0.2 | **`docs/adr/ADR-LLM-001.md`** — typed completion envelope vs plain string; two-layer usage model preserved | **Done** | High | `docs/adr/` | ADR linked from plan + `architecture/LLM_ADAPTERS.md` |
| M-LLM-R.0.3 | **Canon §5.2.2 addendum** — `LLMAdapterResponse` contract paragraph in `intergrax_runtime_architecture.md` | **Done** | Medium | Architecture canon | No duplicate full spec in README |

#### Wave M-LLM-R-1 — Contract types (Tier-0)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.1.1 | **`LLMAdapterResponse`** — frozen dataclass: `content`, `finish_reason`, `usage`, `model`, `provider`, `response_id`, `refusal`, `tool_calls`, `provider_extensions` | **Done** | **Critical** | `llm_adapters/contracts/adapter_response.py` | Unit: construction + immutability |
| M-LLM-R.1.2 | **`LLMTokenUsage`** — frozen dataclass with cached/reasoning token fields | **Done** | **Critical** | same module | `total_tokens` derived or validated |
| M-LLM-R.1.3 | **`LLMFinishReason`** enum + **`LLMToolCall`** (+ argument typing) | **Done** | **Critical** | `llm_adapters/contracts/tool_call.py` or same package | No raw tool dicts in public API |
| M-LLM-R.1.4 | **`LLMProviderExtensions`** — tagged union slices (OpenAI / Anthropic / Gemini / Bedrock) | **Done** | High | `llm_adapters/contracts/provider_extensions.py` | Extensibility without `Dict[str, Any]` |
| M-LLM-R.1.5 | **`LLMStreamEvent`** — partial/final streaming envelope | **Done** | High | `llm_adapters/contracts/stream_event.py` | Final event carries full `LLMAdapterResponse` |
| M-LLM-R.1.6 | **`LLMStructuredResult[T]`** generic wrapper for structured output | **Done** | High | `llm_adapters/contracts/structured_result.py` | Typed generic; mypy/pyright clean |
| M-LLM-R.1.7 | **Typed builders** — replace `make_tool_result` with `build_adapter_response(...)` / `merge_stream_events(...)` | **Done** | **Critical** | `llm_adapters/_shared/adapter_response_builders.py` | Delete `tool_results.py` dict factory |
| M-LLM-R.1.8 | **Public re-exports** — response types from `llm_adapters/__init__.py` | **Done** | Medium | `llm_adapters/__init__.py` | Import smoke test in gate |

#### Wave M-LLM-R-2 — `LLMAdapter` ABC refactor

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.2.1 | **`generate_messages` → `LLMAdapterResponse`** | **Done** | **Critical** | `contracts/llm_adapter.py` | ABC + all stubs updated |
| M-LLM-R.2.2 | **`generate_with_tools` → `LLMAdapterResponse`** | **Done** | **Critical** | same | `tool_calls` on response, not dict key |
| M-LLM-R.2.3 | **`stream_messages` → `Iterable[LLMStreamEvent]`** | **Done** | High | same | Final event mandatory |
| M-LLM-R.2.4 | **`stream_with_tools` → `Iterable[LLMStreamEvent]`** | **Done** | High | same | Tool deltas typed |
| M-LLM-R.2.5 | **`generate_structured` → `LLMStructuredResult[T]`** | **Done** | High | same | Return type annotated |
| M-LLM-R.2.6 | **`_finalize_call` helper** — unify `begin_call`/`end_call` + populate `LLMTokenUsage` on response from same counters | **Done** | **Critical** | `llm_adapter.py` or `_shared/call_lifecycle.py` | Single path; no duplicate counting |

#### Wave M-LLM-R-3 — Provider adapters (all 19 slugs)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.3.1 | **OpenAI Responses + Chat Completions** — map SDK usage, `finish_reason`, `response.id` / choice metadata | **Done** | **Critical** | `openai_responses_adapter.py`, `openai_chat_completions_adapter.py` | Mocked unit tests per method |
| M-LLM-R.3.2 | **Claude + Mistral + Cohere native** — SDK usage where available; map stop_reason / refusal | **Done** | **Critical** | `claude_adapter.py`, `mistral_adapter.py`, `cohere_native_adapter.py` | Stop using estimate-only when SDK exposes usage |
| M-LLM-R.3.3 | **Gemini + Vertex** — candidate finish reason, usage metadata, typed tool calls | **Done** | High | `gemini_adapter.py`, `vertex_gemini_adapter.py` | Conformance green |
| M-LLM-R.3.4 | **AWS Bedrock** — Converse + legacy paths; map stopReason, usage, toolUse blocks | **Done** | High | `aws_bedrock_adapter.py` | Existing bedrock tool tests updated |
| M-LLM-R.3.5 | **Ollama + OpenAI-compat family** — best-effort usage; document estimate fallback in `provider_extensions` | **Done** | Medium | `ollama_adapter.py`, `openai_compat_*` | Explicit `usage.source` flag on extensions |
| M-LLM-R.3.6 | **Streaming parity** — all `supports_streaming()` adapters emit typed `LLMStreamEvent` | **Done** | High | all streaming providers | No `yield str` remaining |
| M-LLM-R.3.7 | **Structured output parity** — return `LLMStructuredResult[T]` with raw completion preserved | **Done** | Medium | adapters with `supports_structured_output()` | JSON parse failures attach to response metadata |

#### Wave M-LLM-R-4 — Nexus runtime consumers (Tier-1)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.4.1 | **`CoreLLMStep`** — `state.raw_answer = completion.content`; trace finish_reason + token snapshot | **Done** | **Critical** | `runtime_steps/core_llm_step.py` | `test_core_llm_step.py` updated |
| M-LLM-R.4.2 | **`ToolPlanningService`** — native tools path uses `completion.tool_calls`; planner text path uses `completion.content` | **Done** | **Critical** | `tools/tool_planning_service.py` | Tool plan tests green |
| M-LLM-R.4.3 | **`plan_sources` + `engine_history_layer`** — consume `.content` | **Done** | High | `planning/plan_sources.py`, `context/engine_history_layer.py` | Unit tests updated |
| M-LLM-R.4.4 | **User/org profile services + session consolidation** — all `generate_messages` call sites | **Done** | High | `runtime/user_profile/*`, `runtime/organization/*` | Grep: zero `.generate_messages` → str assignment |
| M-LLM-R.4.5 | **`supervisor.py`** — all LLM call sites | **Done** | Medium | `intergrax/supervisor/supervisor.py` | Supervisor unit tests |
| M-LLM-R.4.6 | **Optional: store last adapter response on `RuntimeState`** — `last_llm_adapter_response: LLMAdapterResponse \| None` for trace/replay | **Done** | Medium | `engine/runtime_state.py` | Enables per-step cost attribution |

#### Wave M-LLM-R-5 — RAG, websearch, legacy (Tier-0 consumers)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.5.1 | **RAG LLM paths** — `query_refiner`, `query_expander`, `chunk_enricher`, `llm_graph_indexer` | **Done** | **Critical** | `intergrax/rag/` | RAG unit tests use typed mocks |
| M-LLM-R.5.2 | **Websearch** — `websearch_context_generator`, `websearch_answerer` | **Done** | High | `intergrax/websearch/` | Tests updated |
| M-LLM-R.5.3 | **Legacy `rag_answers`** — migrate or mark deprecated path to `.content` | **Done** | Low | `legacy/rag_answers/` | No str assumption in active Nexus paths |

#### Wave M-LLM-R-6 — Agents, scaffold, test support (Tier-2)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.6.1 | **Agent pipeline mocks** — echo, legal, research, problem_radar, signoff_probe, organization_worker, lab mocks | **Done** | High | `agents/*/steps/pipeline.py`, `agents/lab/mock_agents.py` | Agent unit tests green |
| M-LLM-R.6.2 | **`scaffold/new_agent.py` template** — generated stub returns `LLMAdapterResponse` | **Done** | High | `intergrax/scaffold/new_agent.py` | New-agent scaffold test |
| M-LLM-R.6.3 | **`testing_support/builder.py` fake adapter** | **Done** | Medium | `testing_support/builder.py` | Shared test helper |
| M-LLM-R.6.4 | **Tier-2 rule check** — agents must not assume `str` from adapter | **Done** | Low | `scripts/check_agents_llm_adapter_response.py` | CI script in §6.1 maintenance list |

#### Wave M-LLM-R-7 — Observability, replay, trace bridge

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.7.1 | **Align `LLMAdapterUsageLog.end_call` with response `usage`** — same integers; optional validation assert in debug | **Done** | High | `llm_adapter.py` | Metrics unchanged; no double-count |
| M-LLM-R.7.2 | **Emit `LLM_CALL` trace events from runtime** — populate `LLMCallInfo` fields from `LLMAdapterResponse` | **Done** | **Critical** | `core_llm_call_recorded.py`, `trace_replay_bridge.py`, `persisted_trace_event_store.py` | Gate: `test_trace_replay_bridge.py` |
| M-LLM-R.7.3 | **`LLMCallInfo` typed bridge** — map `LLMAdapterResponse` → replay model (no loose dict payloads) | **Done** | High | `runtime/replay/models.py` + mapper | Frozen mapper function |
| M-LLM-R.7.4 | **Update diagnostics** — `CoreLLMAdapterReturnedDiagV1`: `finish_reason`, token fields, drop `adapter_return_type="str"` | **Done** | Medium | `tracing/adapters/core_llm_adapter_returned.py` | PII-safe payload |
| M-LLM-R.7.5 | **Adaptive harness signal hook (optional)** — expose per-call tokens/refusal for W-ADAPT cost/quality signals | **Done** | Low | `llm_call_summary.py`, `signal_collector.py`, `HarnessOutcomeSignal.last_llm_*` | Optional `SignalAssemblyInput.last_llm_call` |

#### Wave M-LLM-R-8 — Docs, conformance, CI closeout

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.8.1 | **`architecture/LLM_ADAPTERS.md` rewrite** — response envelope section; migration guide; two-layer usage clarified | **Done** | **Critical** | `docs/architecture/LLM_ADAPTERS.md` | Examples use `.content` |
| M-LLM-R.8.2 | **Conformance suite** — `assert_generate_messages_returns_completion`; tools/stream/structured typed asserts | **Done** | **Critical** | `_shared/conformance.py`, `tests/unit/llm_adapters/` | Gate + `llm-adapters-guard.yml` |
| M-LLM-R.8.3 | **`check_llm_adapter_typed_returns.py`** — CI guard: no `-> str` / `-> Dict[str, Any]` on adapter public methods | **Done** | High | `scripts/` | Added to §6.1 maintenance |
| M-LLM-R.8.4 | **Phase closeout** — Appendix L paydown complete; M-LLM table row M-LLM.14 **Done**; remove audit follow-up pointer | **Done** | Medium | This plan | All M-LLM-R.* Done |

**Suggested PR order:**

```text
Wave 0:  M-LLM-R.0.2 → 0.3
Wave 1:  M-LLM-R.1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8
Wave 2:  M-LLM-R.2.6 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
Wave 3:  M-LLM-R.3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7  (may split 1 PR per provider family)
Wave 4:  M-LLM-R.4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
Wave 5:  M-LLM-R.5.1 → 5.2 → 5.3
Wave 6:  M-LLM-R.6.1 → 6.2 → 6.3 → 6.4
Wave 7:  M-LLM-R.7.1 → 7.2 → 7.3 → 7.4 → 7.5
Wave 8:  M-LLM-R.8.1 → 8.2 → 8.3 → 8.4
```

**Explicitly out of scope:** K.1/K.2, new product Tier-3 apps, rewriting provider SDK clients, HTTP API response DTOs for product routes (Tier-3 owns those separately).

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
| M.6 P4 | Harness platform expansion | **Done** (beta) (28/28) | `_shared/p5/` · `bootstrap_m6_p4.py` · [M.6 P4 register](#m6-p4--harness-platform-expansion-done) |
| M.6 P5 | Harness integration depth (audit 2026-06-02) | **Done** (33/34) | Harden 25 STABLE + health · 8 greenfield · `trivy` → [M.6 P6](#m6-p6--harness-integration-expansion-planned) · [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334) |
| M.6 P6 | Harness integration expansion (audit 2026-06-02) | **Done** (32/32) | Security, sandbox, identity, GitOps CI, speech catalog, enterprise ops, data/workflow, modality reserve · [M.6 P6 register](#m6-p6--harness-integration-expansion-planned) · Band **2ac** |
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

**M.9 harness depth (Done beta):** full adapters (LangSmith, OpenSearch, Vespa, GitLab, PagerDuty, Braintrust), tools (`gitlab.create_issue`, `pagerduty.trigger_incident`, `braintrust.log_eval`), `slash_command`, lab harness profile, CI harness-smoke job. Catalog: **99** (M.9 closeout; **135** after M.6 P5).

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

#### M.6 P4 — Harness platform expansion (Done)

**Status:** **Done** (2026-06-02) — **28/28 Done** · catalog **127** slugs  
**Source:** Integration harness ROI audit (2026-06-02)  
**Queue:** [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed) · **Execution order:** [§6.2ae](#62ae-phase-m6-p4-execution-order--done)  
**Priority ladder:** **Band 2aa** (§4.0) — **Done**  
**Implementation:** `intergrax/integrations/_shared/p5/` + thin shells via `scripts/wire_p5_m6_p4_providers.py` · `register_m6_p4_integrations()` in `bootstrap_extended.py`

**Hard rules:**

- **No** LLM API slugs — use `llm_adapters/` (canon §7.1.2).
- **New categories** (`feature_flag`, `ci_cd`) require canon §5.2.4 review before merge — track **M-P4-CAT.\*** first.
- Reuse M.4 workflow: contract (or extend existing) → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → gate green.
- `ollama` bridges existing `infra/integration` Docker + `LLMAdapter` Ollama path — not a duplicate LLM stack.

**New category proposals (M-P4-CAT):**

| ID | Category | Slugs | Status | Acceptance |
|----|----------|-------|--------|------------|
| M-P4-CAT.1 | `feature_flag` | `unleash`, `launchdarkly` | **Done** | `FeatureFlagBackend` + `IntegrationCategory.FEATURE_FLAG` |
| M-P4-CAT.2 | `ci_cd` | `github_actions` | **Done** | `CiCdBackend` + `IntegrationCategory.CI_CD` |

##### M.6 P4 — Master register (28 slugs)

| Wave | ID | Slug | Category | Priority | Status | Harness ROI | Acceptance |
|------|-----|------|----------|----------|--------|-------------|------------|
| H-INT-1 | M-P4.1 | `pgvector` | vector_store | **P0** | **Done** (beta) | Unify PostgreSQL (stable) + RAG vectors + adaptive stores | `IntegrationProfile.vector_store=pgvector`; RAG hybrid query; gate unit tests |
| H-INT-1 | M-P4.2 | `duckdb` | relational_store | **P0** | **Done** (beta) | Local OLAP for `phase_w_adapt_report`, eval trends, golden scenarios | `RelationalStore` read path; CI-friendly file DB; report script optional backend |
| H-INT-1 | M-P4.3 | `influxdb` | observability_backend | **P1** | **Done** (beta) | Time-series utility U, cost, latency — adaptive KPIs | `ObservabilityBackend` query_range; W-ADAPT signal export optional |
| H-INT-1 | M-P4.4 | `timescaledb` | relational_store | **P1** | **Done** (beta) | Hypertables for adaptive + eval registry trends on Postgres | Extends `postgresql` contract; migration note in USAGE |
| H-INT-2 | M-P4.5 | `grafana` | observability_backend | **P0** | **Done** (beta) | W-OPS.4 SLO dashboards; L3 release visibility | HTTP API health + dashboard URL artifact; lab stack doc |
| H-INT-2 | M-P4.6 | `loki` | observability_backend | **P0** | **Done** (beta) | Log query for RuntimeEvents / structured logs | LogQL query adapter; complements `prometheus` |
| H-INT-2 | M-P4.7 | `tempo` | observability_backend | **P0** | **Done** (beta) | Trace backend for OTEL (`otel` slug exists; dedicated store) | Trace query by `trace_id`; lab compose profile |
| H-INT-3 | M-P4.8 | `aws_secrets_manager` | secrets_store | **P0** | **Done** (beta) | Prod harness secrets; complements `aws` facade | `SecretsStore` get/list; no secrets in agent code |
| H-INT-3 | M-P4.9 | `azure_key_vault` | secrets_store | **P0** | **Done** (beta) | Azure prod parity | MI / SP auth via `azure` patterns |
| H-INT-3 | M-P4.10 | `gcp_secret_manager` | secrets_store | **P0** | **Done** (beta) | GCP prod parity | ADC / SA via `gcp` patterns |
| H-INT-3 | M-P4.11 | `doppler` | secrets_store | **P1** | **Done** (beta) | Dev/prod secret sync for harness authors | Project/config scoped fetch; lab `.env` bridge |
| H-INT-4 | M-P4.12 | `unleash` | feature_flag | **P0** | **Done** (beta) | Gradual `AdaptiveProfile` rollout (observe→recommend) | Requires **M-P4-CAT.1**; tenant-scoped flags |
| H-INT-4 | M-P4.13 | `launchdarkly` | feature_flag | **P1** | **Done** (beta) | Enterprise feature flags + canary | Requires **M-P4-CAT.1** |
| H-INT-4 | M-P4.14 | `github_actions` | ci_cd | **P0** | **Done** (beta) | Harness release gate status; `harness-release.yml` evidence | Requires **M-P4-CAT.2**; workflow run + check suite read |
| H-INT-4 | M-P4.15 | `redpanda` | message_bus | **P1** | **Done** (beta) | Kafka-compatible async `AdaptationScheduler` / pattern miner | Lab compose; consumer/producer contract tests |
| H-INT-4 | M-P4.16 | `cloudflare_r2` | object_storage | **P1** | **Done** (beta) | S3-compatible cheap eval/adaptive artifacts | `ObjectStorage` put/get; reuse S3 adapter patterns |
| H-INT-5 | M-P4.17 | `memgraph` | graph_store | **P1** | **Done** (beta) | GraphRAG alternative; lighter lab footprint | `GraphStore` contract; RAG `INTERGRAX_RAG_GRAPH_STORE` option |
| H-INT-5 | M-P4.18 | `falkordb` | graph_store | **P2** | **Done** (beta) | Redis-module graph — reuse lab `redis` stack | Bolt/Redis protocol adapter |
| H-INT-5 | M-P4.19 | `incident_io` | notification_channel | **P1** | **Done** (beta) | Ops runbooks (`runbook/adaptive/*`) → real incidents | Outbound incident create; HITL escalation path |
| H-INT-5 | M-P4.20 | `kubernetes` | cloud_platform | **P1** | **Done** (beta) | Prod harness host deploy; health probes at scale | Extend `CloudPlatform` — namespace/workload health |
| H-INT-5 | M-P4.21 | `servicenow` | issue_tracker | **P2** | **Done** (beta) | Enterprise change approval for policy learning | `IssueTracker` search/get; HITL change ticket |
| H-INT-5 | M-P4.22 | `bitbucket` | issue_tracker | **P2** | **Done** (beta) | Atlassian stack beside `jira` | REST issues/PRs |
| H-INT-5 | M-P4.23 | `asana` | issue_tracker | **P2** | **Done** (beta) | PM human task queue beside `linear` | Task search/create |
| H-INT-5 | M-P4.24 | `sendgrid` | notification_channel | **P2** | **Done** (beta) | Deliverability beyond raw `email_smtp` | Transactional send API |
| H-INT-5 | M-P4.25 | `mailgun` | interaction_surface | **P2** | **Done** (beta) | Inbound email → interaction intake | Webhook verify + payload normalize |
| H-INT-5 | M-P4.26 | `mlflow` | observability_backend | **P2** | **Done** (beta) | Experiment tracking beside wandb/braintrust | Run/metric log API; lab workflow §35 |
| H-INT-5 | M-P4.27 | `huggingface_hub` | object_storage | **P2** | **Done** (beta) | W-ML model artifact pull (ONNX/YOLO) | Model file get/list; modality plane bridge |
| H-INT-5 | M-P4.28 | `ollama` | interaction_surface | **P2** | **Done** (beta) | Local inference host (`infra/integration` ollama service) | Health probe + model list; cross-link [architecture/MODALITY.md](architecture/MODALITY.md) · not LLM catalog slug |

**Explicitly excluded from M.6 P4:** CRM (Salesforce, HubSpot), payment rails, blockchain, duplicate vector SaaS, LLM vendor APIs.

##### M.6 P4 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-02 | M-P4.0 | Register 28 harness-ROI integration slugs + §6.1w + §6.2ae + Band 2aa (audit → plan) |
| 2026-06-02 | M-P4.1–M-P4.28 | All 28 M.6 P4 providers: `_shared/p5/`, layout **127**, tests `test_p5_m6_p4_providers.py`, gate green |
| 2026-06-02 | M-P4.FU | Tier-3 follow-up (no business agents): `harness_production_stack` / `harness_production_defaults`, lab env (`LAB_OBSERVABILITY_GRAFANA_STACK`, `LAB_ADAPTIVE_FEATURE_FLAG`, `LAB_SECRETS_BACKEND`), adaptive feature-flag gate, pgvector persistence + health, M6 P4 stable promotion (8 slugs), `health_check_harness_m6_p4_probes`, docs sync |
| 2026-06-02 | M-P4.FU.2 | Adaptive runtime bridge uses gated `wiring.profile`; debug `GET /debug/integrations/health`; remove `getattr` from P5 health probes (`IntegrationHealthProbe`); W-OPS integration health debug gate; gate **790** |

#### M.6 P5 — Harness integration depth (Done — 33/34)

**Deferred:** `trivy` — absorbed into **M.6 P6** [M-P6.1](#m6-p6--master-register-32-slugs) with `security_scanner` category (**M-P6-CAT.1**).

**Delivered (2026-06-02):**

- `_shared/p6/factories.py` — 8 greenfield harness slugs
- `bootstrap_m6_p5.py` + `layout.py` (+8 slugs → **135** catalog slugs)
- Health probes on harden adapters; **STABLE** promotion (25 slugs)
- Tier-3 presets: `harness_metrics_stack`, `harness_eval_stack`, `harness_async_stack`, `harness_ci_stack`
- `HARNESS_M6_P5_PROBE_SLUGS` + `health_check_harness_m6_p5_probes()` + debug API `stack=m6_p5`
- `integrations-pick` presets: `harness_metrics`, `harness_eval`, `harness_async`, `harness_ci`
- Tests: `tests/unit/integrations/providers/test_p6_m6_p5_providers.py`

#### M.6 P5 — Harness integration depth (register archive)

**Status:** **Done** (2026-06-02) — **33/34** · catalog **135** slugs in layout.py (**136** when `trivy` ships)  
**Source:** Harness integration re-audit (2026-06-02) — post M.6 P4 follow-up  
**Queue:** [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-planned) · **Execution order:** [§6.2af](#62af-phase-m6-p5-execution-order-band-2ab--planned)  
**Priority ladder:** **Band 2ab** (§4.0) — runs **in parallel** with §6.1 maintenance; **does not** unblock Band 3 product work

**Scope split:**

| Kind | Count | Meaning |
|------|-------|---------|
| **Harden** | 25 | Slug already in catalog (`beta`) — health probe, STABLE promotion, harness preset wiring, tests |
| **Greenfield** | 9 | New slug + provider tree — same M.4 workflow as P4 |

**Hard rules (inherit M.6 P4):**

- **No** LLM vendor API slugs — use `llm_adapters/` (canon §7.1.2).
- **No** CRM, payments, blockchain, or duplicate vector SaaS without explicit harness ROI.
- Reuse `_shared/p5/` HTTP patterns or existing provider packages — **do not** fork RAG/runtime stores.
- One slug (or one harden wave) per PR; gate green after each.
- `infra/integration` Docker profile must be documented in slug `USAGE.md` when a local service exists.

**New category proposals (M-P5-CAT):**

| ID | Category | Slugs | Status | Acceptance |
|----|----------|-------|--------|------------|
| M-P5-CAT.1 | `ci_cd` (extend) | `gitlab_ci`, `circleci`, `azure_pipelines`, `codecov` | **Done** | Read-only workflow/check/coverage APIs on existing `CiCdBackend` |
| M-P5-CAT.2 | `security_scanner` *(proposed)* | `trivy` | **Deferred** | `SecurityScannerBackend` with `scan_image(ref) -> ScanReport`; canon §5.2.4 review before merge |
| M-P5-CAT.3 | — *(use existing)* | `mailpit`, `localstack`, `grafana_oncall`, `opentelemetry_collector` | **Done** | Map to existing categories (`notification_channel`, `cloud_platform`, `notification_channel`, `observability_backend`) |

**Tier-3 named presets (deliver with H-INT-6 closeout):**

| Preset function | Slugs (primary) | Harness use |
|-----------------|-----------------|-------------|
| `harness_metrics_stack()` | `prometheus` + `grafana` + `otel` | W-OPS.4 SLO / metrics-first lab |
| `harness_eval_stack()` | `langfuse` + `minio` + `duckdb` | EVAL export + experiment traces |
| `harness_async_stack()` | `redpanda` or `kafka` + `redis` + optional `temporal` | W-ADAPT async / long-running |
| `harness_ci_stack()` | `github_actions` + `gitlab_ci` + optional `circleci` | Multi-CI release evidence |

##### M.6 P5 — Master register (34 slugs)

| Wave | ID | Slug | Category | Kind | Priority | Status | Harness ROI | Acceptance |
|------|-----|------|----------|------|----------|--------|-------------|------------|
| H-INT-6 | M-P5.1 | `prometheus` | observability_backend | harden | **P0** | **Done** | Metrics SLO backbone (W-OPS.4); complements Grafana stack | Health probe; `harness_metrics_stack`; infra `:9090` |
| H-INT-6 | M-P5.2 | `clickhouse` | observability_backend | harden | **P0** | **Done** | OLAP eval/adaptive trends at scale | Query adapter; infra `:8123` |
| H-INT-6 | M-P5.3 | `vault` | secrets_store | harden | **P0** | **Done** | Prod secrets alt in `harness_production_stack` | Health probe; STABLE; infra `:8200` |
| H-INT-6 | M-P5.4 | `pagerduty` | notification_channel | harden | **P0** | **Done** | HITL / incident escalation (tool already wired) | Integration health + lab smoke |
| H-INT-6 | M-P5.5 | `github` | issue_tracker | harden | **P0** | **Done** | PR/issue context for release board | Read API; links to `github_actions` evidence |
| H-INT-6 | M-P5.6 | `gitlab_ci` | ci_cd | greenfield | **P0** | **Done** | GitLab pipeline status for harness release | **M-P5-CAT.1**; `CiCdBackend` read |
| H-INT-6 | M-P5.7 | `circleci` | ci_cd | greenfield | **P0** | **Done** | Multi-CI release evidence | **M-P5-CAT.1** |
| H-INT-6 | M-P5.8 | `azure_pipelines` | ci_cd | greenfield | **P0** | **Done** | Azure DevOps CI parity | **M-P5-CAT.1**; pairs with `azure_devops` issue tracker |
| H-INT-6 | M-P5.9 | `mailpit` | notification_channel | greenfield | **P0** | **Done** | Local SMTP/HITL without SaaS | Infra `:1025`/`:8025`; email capture tests |
| H-INT-6 | M-P5.10 | `localstack` | cloud_platform | greenfield | **P0** | **Done** | S3/SQS/DynamoDB smoke in CI | Infra `:4566`; pairs with `s3`/`sqs`/`dynamodb` slugs |
| H-INT-7 | M-P5.11 | `langfuse` | observability_backend | harden | **P0** | **Done** | LLM trace + eval export (EVAL/W-ADAPT) | Infra `:3000`; `harness_eval_stack` |
| H-INT-7 | M-P5.12 | `phoenix` | observability_backend | harden | **P0** | **Done** | Arize OSS trace UI for lab | Infra `:6006` |
| H-INT-7 | M-P5.13 | `braintrust` | observability_backend | harden | **P1** | **Done** | Online eval registry bridge | API read + export hook |
| H-INT-7 | M-P5.14 | `mlflow` | observability_backend | harden | **P1** | **Done** | Experiment tracking (M.6 P4 beta hardening) | STABLE promotion path |
| H-INT-7 | M-P5.15 | `influxdb` | observability_backend | harden | **P1** | **Done** | Adaptive KPI time-series (M.6 P4 beta) | STABLE + W-ADAPT optional export |
| H-INT-7 | M-P5.16 | `timescaledb` | relational_store | harden | **P1** | **Done** | Eval/adaptive hypertables on Postgres | Extends `postgresql` patterns |
| H-INT-7 | M-P5.17 | `temporal` | message_bus | harden | **P1** | **Done** | Long-running harness workflows | Infra `heavy` profile `:7233` |
| H-INT-7 | M-P5.18 | `redpanda` | message_bus | harden | **P1** | **Done** | Kafka-compat async adaptive bus (M.6 P4 beta) | STABLE + `harness_async_stack` |
| H-INT-7 | M-P5.19 | `minio` | object_storage | harden | **P1** | **Done** | Local S3 for eval/adaptive artifacts | Infra `:9000`; preset with `harness_eval_stack` |
| H-INT-7 | M-P5.20 | `s3` | object_storage | harden | **P1** | **Done** | Prod checkpoint/eval blob store | `harness_production_stack` option |
| H-INT-8 | M-P5.21 | `neo4j` | graph_store | harden | **P1** | **Done** | GraphRAG harness eval | Infra `:7687`; health probe |
| H-INT-8 | M-P5.22 | `mongodb` | document_store | harden | **P1** | **Done** | MEM platform JSON artifacts | Infra `:27017` |
| H-INT-8 | M-P5.23 | `elasticsearch` | observability_backend | harden | **P1** | **Done** | Log search for RuntimeEvents | Infra `:9200` |
| H-INT-8 | M-P5.24 | `nats` | message_bus | harden | **P2** | **Done** | Lightweight async bus | Infra `:4222` |
| H-INT-8 | M-P5.25 | `chroma` | vector_store | harden | **P2** | **Done** | RAG lab alternative | Infra `:8000`; thin RAG bridge |
| H-INT-8 | M-P5.26 | `weaviate` | vector_store | harden | **P2** | **Done** | RAG lab alternative | Infra `:8080` |
| H-INT-8 | M-P5.27 | `launchdarkly` | feature_flag | harden | **P2** | **Done** | Enterprise canary beside Unleash | Adaptive gate smoke |
| H-INT-8 | M-P5.28 | `signoz` | observability_backend | harden | **P2** | **Done** | Self-hosted OTEL APM | Optional Grafana stack alt |
| H-INT-9 | M-P5.29 | `codecov` | ci_cd | greenfield | **P2** | **Done** | Coverage gate in release evidence | **M-P5-CAT.1** |
| H-INT-9 | M-P5.30 | `trivy` | security_scanner | greenfield | **P2** | **→ M-P6.1** | Image/SBOM scan before STABLE promote | Absorbed into [M.6 P6](#m6-p6--harness-integration-expansion-planned) (**M-P6-CAT.1**) |
| H-INT-9 | M-P5.31 | `grafana_oncall` | notification_channel | greenfield | **P2** | **Done** | On-call beside Grafana stack | Webhook/API incident create |
| H-INT-9 | M-P5.32 | `opentelemetry_collector` | observability_backend | greenfield | **P2** | **Done** | Collector admin/health (export via `otel`) | Distinct from app OTEL export slug |
| H-INT-9 | M-P5.33 | `snowflake` | relational_store | harden | **P2** | **Done** | Enterprise eval analytics | Existing beta hardening only |
| H-INT-9 | M-P5.34 | `supabase` | relational_store | harden | **P2** | **Done** | Postgres+auth lab shortcut | Existing beta hardening only |

**Explicitly excluded from M.6 P5:** CRM (Salesforce, HubSpot), payment rails, blockchain, `vespa`/`selenium` (heavy lab only), `servicenow`/`asana`/`notion`/`sharepoint`/`google_workspace` (business PM/collab), duplicate vector SaaS without infra smoke (`pinecone`, `milvus` until explicitly requested).

**Per-slug checklist (harden):** health probe → STABLE promotion → harness preset slot (if applicable) → `HARNESS_M6_P5_PROBE_SLUGS` or W-OPS extension → gate green → paydown log row.

**Per-slug checklist (greenfield):** contract/category gate → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → bootstrap register → gate green → paydown log row.

##### M.6 P5 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-02 | M-P5.0 | Register 34 harness-depth slugs from integration re-audit; §6.1x + §6.2af + Band 2ab |
| 2026-06-02 | M-P5.1–34 | Implement 33/34 slugs: health + STABLE harden, p6 greenfield, presets, W-OPS probes; `trivy` deferred (M-P5-CAT.2) |
| 2026-06-02 | M-P5.FU | W-OPS `harness_m6_p5_health_gate`; `IntegrationBinding` JSON dict roundtrip fix; register status sync |

#### M.6 P6 — Harness integration expansion (Done — 32/32)

**Status:** **Done** (2026-06-02) — **32/32** · catalog **167** slugs in layout.py  
**Source:** Harness integration gap audit (2026-06-02) — post M.6 P5; all **32** proposed slugs registered below (includes `trivy` migrated from M-P5.30, plus `modal`, `daytona`, `workos`, `hubspot` from audit waves)  
**Queue:** [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-planned) · **Execution order:** [§6.2ag](#62ag-phase-m6-p6-execution-order-band-2ac--planned)  
**Priority ladder:** **Band 2ac** (§4.0) — runs **in parallel** with §6.1 maintenance; **does not** unblock Band 3 product work

**Scope:** **32 greenfield** slugs — new provider trees + category contracts where noted. No business-agent logic.

**Hard rules (inherit M.6 P4/P5):**

- **No** LLM vendor API slugs — use `llm_adapters/` (canon §7.1.2).
- Reuse `_shared/p6/` / `_shared/p7/` HTTP patterns — **do not** fork RAG/runtime stores.
- One slug (or one category CAT wave) per PR; gate green after each.
- `infra/integration` Docker profile documented in slug `USAGE.md` when a local service exists.
- **`salesforce` / `hubspot` / `stripe`:** harness-platform slugs only (metering, CRM context for support agents) — **not** Band 3 product agents.

**New category proposals (M-P6-CAT — canon §5.2.4 review before first slug in category):**

| ID | Category | Slugs | Status | Acceptance |
|----|----------|-------|--------|------------|
| M-P6-CAT.1 | `security_scanner` | `trivy`, `snyk`, `semgrep` | **Done** | `SecurityScannerBackend`: `scan_image(ref)`, `scan_repo(path)` → `ScanReport`; completes **M-P5-CAT.2** |
| M-P6-CAT.2 | `sandbox_host` | `e2b`, `modal`, `daytona` | **Done** | `SandboxHostBackend`: `create_session()`, `exec()`, `upload_artifact()`; bridges Tier-1 `sandbox.exec` tool |
| M-P6-CAT.3 | `identity_provider` | `auth0`, `keycloak`, `workos` | **Done** | `IdentityProviderBackend`: `verify_token()`, `userinfo()`, optional `list_tenants()` |
| M-P6-CAT.4 | `speech_provider` | `elevenlabs`, `deepgram` | **Done** | `SpeechProviderBackend`: TTS/STT; unifies `speech_adapters/` with Integration Library ([architecture/MODALITY.md](architecture/MODALITY.md)) |
| M-P6-CAT.5 | `workflow_orchestrator` | `prefect`, `airflow` | **Done** | `WorkflowOrchestratorBackend`: trigger run, poll status, fetch logs (eval/RAG batch jobs) |
| M-P6-CAT.6 | `vision_serving` | `triton` | **Done** | Remote CV inference host ([architecture/MODALITY.md](architecture/MODALITY.md) W-ML.4) |
| M-P6-CAT.7 | `ml_inference_host` | `replicate` | **Done** | Managed model endpoint (`predict`, health) |
| M-P6-CAT.8 | `billing_meter` | `stripe` | **Done** | Usage metering hook for harness SaaS path (canon §50 future) |
| M-P6-CAT.9 | `crm` | `salesforce`, `hubspot` | **Done** | Read-only CRM context (accounts, contacts, tickets) for support harness agents |

**Tier-3 named presets (deliver with H-INT-10 closeout or M-P6-PRE.1):**

| Preset function | Slugs (primary) | Harness use |
|-----------------|-----------------|-------------|
| `harness_security_stack()` | `trivy` + `semgrep` + optional `snyk` | STABLE promote gate + V-SEC repo policy |
| `harness_sandbox_stack()` | `e2b` + optional `modal` | Cloud `sandbox.exec` for lab/product hosts |
| `harness_identity_stack()` | `keycloak` (lab) or `auth0` (prod) | Multi-tenant debug API / host auth |
| `harness_gitops_stack()` | `argocd` + `github_actions` | Agent host deploy after eval gate |

##### M.6 P6 — Master register (32 slugs)

| Wave | ID | Slug | Category | Priority | Status | Harness ROI | Acceptance |
|------|-----|------|----------|----------|--------|-------------|------------|
| H-INT-10 | M-P6.1 | `trivy` | security_scanner | **P0** | **Done** | Image/SBOM scan before STABLE promote | **M-P6-CAT.1**; migrates M-P5.30 |
| H-INT-10 | M-P6.2 | `snyk` | security_scanner | **P0** | **Done** | SAST/SCA in agent pack promotion pipeline | **M-P6-CAT.1** |
| H-INT-10 | M-P6.3 | `semgrep` | security_scanner | **P0** | **Done** | Policy-as-code on agents/skills repos | **M-P6-CAT.1** |
| H-INT-10 | M-P6.4 | `infisical` | secrets_store | **P0** | **Done** | Dev-friendly secrets sync (lab + prod) | Health probe; pairs with `harness_production_stack` |
| H-INT-11 | M-P6.5 | `e2b` | sandbox_host | **P0** | **Done** | Cloud isolation for `sandbox.exec` | **M-P6-CAT.2**; sandbox tool bridge |
| H-INT-11 | M-P6.6 | `modal` | sandbox_host | **P1** | **Done** | Serverless agent/compute workloads | **M-P6-CAT.2** |
| H-INT-11 | M-P6.7 | `daytona` | sandbox_host | **P1** | **Done** | Dev environment sandbox alternative | **M-P6-CAT.2** |
| H-INT-12 | M-P6.8 | `auth0` | identity_provider | **P0** | **Done** | SaaS OIDC for multi-tenant harness hosts | **M-P6-CAT.3** |
| H-INT-12 | M-P6.9 | `keycloak` | identity_provider | **P0** | **Done** | Self-hosted OIDC (VPC customers) | **M-P6-CAT.3**; infra optional |
| H-INT-12 | M-P6.10 | `workos` | identity_provider | **P1** | **Done** | Enterprise SSO + directory sync | **M-P6-CAT.3** |
| H-INT-13 | M-P6.11 | `argocd` | ci_cd | **P0** | **Done** | GitOps deploy Tier-3 hosts after eval gate | Read API; `harness_gitops_stack` |
| H-INT-13 | M-P6.12 | `buildkite` | ci_cd | **P1** | **Done** | Eval-before-merge pipelines | Extends `CiCdBackend` |
| H-INT-13 | M-P6.13 | `jenkins` | ci_cd | **P1** | **Done** | Enterprise CI parity | Extends `CiCdBackend` |
| H-INT-14 | M-P6.14 | `elevenlabs` | speech_provider | **P0** | **Done** | TTS catalog slug; bridges `speech_adapters/` | **M-P6-CAT.4**; `speech.synthesize` tool |
| H-INT-14 | M-P6.15 | `deepgram` | speech_provider | **P0** | **Done** | STT for HITL voice + audio RAG ingest | **M-P6-CAT.4**; `speech.transcribe` tool |
| H-INT-15 | M-P6.16 | `newrelic` | observability_backend | **P1** | **Done** | APM gap beside Datadog/Honeycomb | Health + query API |
| H-INT-15 | M-P6.17 | `splunk` | observability_backend | **P1** | **Done** | Enterprise log search (RuntimeEvents export) | Search adapter |
| H-INT-15 | M-P6.18 | `zendesk` | issue_tracker | **P1** | **Done** | Support tickets → agent tasks / HITL | Read/create ticket API |
| H-INT-15 | M-P6.19 | `statsig` | feature_flag | **P1** | **Done** | Agent experiment gates beside Unleash/LD | Adaptive canary smoke |
| H-INT-16 | M-P6.20 | `prefect` | workflow_orchestrator | **P1** | **Done** | Batch eval / dataset refresh orchestration | **M-P6-CAT.5** |
| H-INT-16 | M-P6.21 | `airflow` | workflow_orchestrator | **P1** | **Done** | Data-eng standard for RAG reindex jobs | **M-P6-CAT.5** |
| H-INT-16 | M-P6.22 | `typesense` | vector_store | **P1** | **Done** | Fast hybrid search lab backend | Thin RAG bridge + health |
| H-INT-16 | M-P6.23 | `neon` | relational_store | **P1** | **Done** | Serverless Postgres for trace/eval lab | Extends `postgresql` patterns |
| H-INT-16 | M-P6.24 | `pulsar` | message_bus | **P1** | **Done** | Multi-tenant streaming bus | Infra optional |
| H-INT-17 | M-P6.25 | `algolia` | search_provider | **P2** | **Done** | SaaS search for product agents | Search API adapter |
| H-INT-17 | M-P6.26 | `confluent` | message_bus | **P2** | **Done** | Managed Kafka for enterprise event bus | Pairs with `kafka` slug |
| H-INT-17 | M-P6.27 | `backblaze_b2` | object_storage | **P2** | **Done** | Low-cost eval/shadow-workspace artifacts | S3-compat API |
| H-INT-17 | M-P6.28 | `triton` | vision_serving | **P2** | **Done** | Remote CV inference (W-ML.4) | **M-P6-CAT.6** |
| H-INT-17 | M-P6.29 | `replicate` | ml_inference_host | **P2** | **Done** | Hosted models without lab GPU | **M-P6-CAT.7** |
| H-INT-17 | M-P6.30 | `stripe` | billing_meter | **P2** | **Done** | Usage metering for future harness SaaS | **M-P6-CAT.8**; read-only meter events |
| H-INT-17 | M-P6.31 | `salesforce` | crm | **P2** | **Done** | Enterprise CRM context (support agents) | **M-P6-CAT.9**; read-only |
| H-INT-17 | M-P6.32 | `hubspot` | crm | **P2** | **Done** | SMB CRM context (support agents) | **M-P6-CAT.9**; read-only |

**Explicitly excluded from M.6 P6:** LLM vendor slugs; blockchain; duplicate thin observability without tool surface; `pinecone`/`milvus` until explicitly requested; Band 3 business agent implementations inside provider packages.

**Per-slug checklist (greenfield):** category CAT gate (if new) → contract → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → bootstrap register → optional preset/probe → gate green → paydown log row.

##### M.6 P6 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-02 | M-P6.0 | Register **32** harness-expansion slugs from integration gap audit; §6.1y + §6.2ag + Band **2ac** |
| 2026-06-02 | M-P6-WIRE | Post-catalog closeout: Tier-1 tools (`security.scan`, `workflow.*`), `HostedSandboxSession` bridge, `IntegrationSpeechAdapter`, `wire_application_identity`, V-SEC promote gate script, infra `p6` profile, CI hook |

##### M.6 P6 — Post-catalog wiring closeout (Done — 2026-06-02)

| ID | Deliverable | Status |
|----|-------------|--------|
| M-P6-WIRE.1 | `security.scan` tool + `ToolWiringContext.security_scanner` | **Done** |
| M-P6-WIRE.2 | `workflow.trigger` / `workflow.poll` / `workflow.fetch_logs` + `workflow_orchestrator` wiring | **Done** |
| M-P6-WIRE.3 | `sandbox.exec` → `SandboxHostBackend` via `HostedSandboxSession` | **Done** |
| M-P6-WIRE.4 | Speech catalog → speech tools via `IntegrationSpeechAdapter` | **Done** |
| M-P6-WIRE.5 | Harness OIDC auth via `wire_application_identity()` (lab + generic FastAPI hosts) | **Done** |
| M-P6-WIRE.6 | `check_harness_security_promote_gate.py` (wiring default; optional live scan) | **Done** |
| M-P6-WIRE.7 | Docker profile `p6` (keycloak, typesense, airflow) | **Done** |
| M-P6-WIRE.8 | `extend_tool_profile_for_integration()` + lab MCP P6 wiring + product host identity | **Done** |
| M-P6-OPS.1 | Release CLI security scan + P6 infra E2E script + `harness.reliability_smoke` P6 tools | **Done** |

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
| **Done (beta)** | `weaviate`, `milvus`, `snowflake`, `vault` | vector_store / relational_store / secrets | `integrations/providers/vector_store/weaviate/`, `vector_store/milvus/`, `relational_store/snowflake/`, `secrets/vault/` |

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

Documented in [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) Appendix E:

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
| N.8 | `guides/AGENT_CREATION_GUIDE.md` Step 4E (dedicated application) | **Done** | — | Step 4E + Appendix F cross-links; gate doc test |
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

**Catalog reference:** [`architecture/TOOLS.md`](architecture/TOOLS.md)

**Delivery rule:** One domain or migration slice per iteration — implement → gate → update `architecture/TOOLS.md` → next step.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| O.0 | Architecture & catalog documented | **Done** | §7.1.6–§7.1.7, §22 | Runtime canon + `architecture/TOOLS.md` + this section (2026-05-30) |
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
[ ] 8. Update architecture/TOOLS.md status + this plan tracker
```

#### T-EXPAND — Integration bridge catalog expansion (2026-06-07) — **Done**

**Goal:** Close the integration→tool coverage gap (~78% integrations without LLM tools) by shipping provider-agnostic bundles that compose existing `IntegrationCategory` contracts.

| Wave | Bundles | Tools | Status |
|------|---------|------:|--------|
| T1 (DX / runtime-bound) | `workspace`, `memory`, `knowledge`, `document`, `browser`, `storage` (get) | 12 | **Done** |
| T2 (prod harness) | `storage` (+put/presigned/delete), `issues`, `platform` | 10 | **Done** |
| T3 (async / graph / collab / cache) | `message_bus`, `graph`, `collaboration`, `cache` | 8 | **Done** |

**Delivered:**

- **67** catalog `tool_id` values · **28** shipped bundles (`shipped_plugins.py`)
- Typed `ToolWiringContext` slots for all new integration categories
- `TaskMemoryViewBinding` protocol (avoids Tier-0 ↔ UAEP import cycle)
- UAEP `runtime_bound_catalog.py` for `workspace.*` / `memory.*` (mirrors `sandbox.exec`)
- `extend_tool_profile_for_integration()` P6 auto-enable (excludes ingest-only `document_parser`)
- Gate: **909** passed (`uv run pytest -m gate -q`)

**Follow-up (2026-06-07) — Done:**

- `IssueCreator` protocol + `issues.create_issue` (no `getattr` in GitLab tool path)
- `harness.integration_bridge_smoke` skill pack + resolver test fix (skills vs tools `build_registry_from_profile`)
- Lab harness `wire_lab_tools(harness=True)` enables runtime-bound + bridge tools
- PoC template `extend_tool_profile_for_integration()` wiring
- MCP full-catalog export smoke (130 tools)

#### T-EXPAND T4 — Agent Builder Essentials (2026-06-07) — **Done**

**Goal:** Close highest-ROI integration→tool gaps for agent/environment builders (SQL, document JSON, RAG lifecycle, workspace DX, collaboration read path, auto-enable wiring).

| Bundle | Tools | Status |
|--------|------:|--------|
| `database` | `database.query`, `database.execute` | **Done** |
| `records` | `records.get`, `records.put`, `records.delete`, `records.query` | **Done** |
| `rag` (+2) | `rag.delete_documents`, `rag.describe_collection` | **Done** |
| `workspace` (+2) | `workspace.delete_file`, `workspace.search` | **Done** |
| `collaboration` (+4) | `collaboration.list_messages`, `get_message`, `list_calendar`, `get_user` | **Done** |
| wiring | `relational_store` / `document_store` ctx slots; auto-enable notify/obs/database/records/collaboration | **Done** |

**Delivered:** **81** catalog `tool_id` values · **30** shipped bundles.

#### T-EXPAND T5 — Production Harness Ops (2026-06-07) — **Done**

**Goal:** Production harness operations for identity, persisted run trace read, integration health probes, online evaluation registry, and platform/security extensions.

| Bundle | Tools | Status |
|--------|------:|--------|
| `identity` | `identity.verify_token`, `identity.get_user`, `identity.list_tenants` | **Done** |
| `harness` | `harness.get_run`, `harness.list_runs`, `harness.get_run_cost`, `harness.get_run_events` | **Done** |
| `health` | `health.check_integration`, `health.check_profile` | **Done** |
| `eval` | `eval.record_observation`, `eval.list_observations`, `eval.summarize_release` | **Done** |
| `security` (+1) | `security.summarize_findings` | **Done** |
| `platform` (+1) | `platform.put_secret` | **Done** |
| wiring | `trace_reader` / `evaluation_registry` / `integration_profile` ctx slots; runtime-bound `harness.*`; observability bundle promoted STABLE | **Done** |

**Delivered:** **95** catalog `tool_id` values · **34** shipped bundles.

#### T-EXPAND T6 — LKW Filesystem + Harness Economics (2026-06-07) — **Done**

**Goal:** LKW read-only filesystem browse (LKW.3), V-COST/billing tool surface, rerank/cache/CRM/platform extensions.

| Bundle | Tools | Status |
|--------|------:|--------|
| `filesystem` | `filesystem.list`, `filesystem.glob`, `filesystem.read_text`, `filesystem.stat` | **Done** |
| `billing` | `billing.record_usage`, `billing.list_usage` | **Done** |
| `cost` | `cost.get_run_budget`, `cost.check_quota` | **Done** |
| `crm` | `crm.get_account`, `crm.list_contacts`, `crm.list_tickets` | **Done** |
| `platform` (+1) | `platform.delete_secret` | **Done** |
| `rag` (+1) | `rag.rerank` | **Done** |
| `cache` (+2) | `cache.delete`, `cache.list_keys` | **Done** |
| wiring | `read_allowlist_roots` ctx slot; runtime-bound `cost.*`; LKW auto-enable filesystem | **Done** |

**Delivered:** **110** catalog `tool_id` values · **38** shipped bundles.

#### T-EXPAND T7 — Index Lifecycle + Async Queue (2026-06-07) — **Done**

**Goal:** RAG index inspection, async task queue ops, observability range/tail, eval release compare, cost forecast.

| Bundle | Tools | Status |
|--------|------:|--------|
| `message_bus` (+2) | `message_bus.list_tasks`, `message_bus.cancel` | **Done** |
| `rag` (+3) | `rag.list_documents`, `rag.get_document`, `rag.check_index_status` | **Done** |
| `document` (+1) | `document.parse_preview` | **Done** |
| `observability` (+2) | `metrics.query_range`, `logs.tail` | **Done** |
| `eval` (+1) | `eval.compare_releases` | **Done** |
| `cost` (+1) | `cost.forecast_spend` | **Done** |
| contracts | `TaskQueue.cancel` / `list_tasks`; `VectorStoreDocumentListerBinding` | **Done** |
| wiring | auto-enable message_bus + observability extensions; runtime-bound `cost.forecast_spend` | **Done** |

**Delivered:** **120** catalog `tool_id` values · **38** shipped bundles.

#### T-EXPAND T8 — Governance + Agent Safety + LKW write (2026-06-07) — **Done**

**Goal:** Read-only HITL ops, allowlisted filesystem write, RAG metadata search/purge, schema introspection, CI/CD workflow ops.

| Bundle | Tools | Status |
|--------|------:|--------|
| `hitl` (+3, new) | `hitl.list_pending`, `hitl.get_decision`, `hitl.summarize_queue` | **Done** |
| `filesystem` (+1) | `filesystem.write_text` | **Done** |
| `rag` (+2) | `rag.search_by_metadata`, `rag.purge_collection` | **Done** |
| `database` (+1) | `database.describe_schema` | **Done** |
| `records` (+1) | `records.describe_collection` | **Done** |
| `platform` (+2) | `platform.list_workflow_runs`, `platform.cancel_workflow_run` | **Done** |
| contracts | `HumanDecisionStoreBinding`; `CiCdBackend.list/cancel`; `VectorstoreIndexLifecycleBinding.search/purge` | **Done** |
| wiring | LKW auto-enable write + RAG maintenance; integration profile CI/CD + schema tools | **Done** |

**Delivered:** **130** catalog `tool_id` values · **39** shipped bundles.

#### T-EXPAND T9 — Async orchestration + interaction (2026-06-07) — **Done**

**Goal:** Workflow run ops, notify batch, collaboration write-back, websearch cache invalidation, harness run diff/export, interaction session reads.

| Bundle | Tools | Status |
|--------|------:|--------|
| `workflow` (+2) | `workflow.list_runs`, `workflow.cancel_run` | **Done** |
| `notify` (+1) | `notify.send_batch` | **Done** |
| `collaboration` (+2) | `collaboration.reply_message`, `collaboration.create_event` | **Done** |
| `websearch` (+1) | `websearch.invalidate_cache` | **Done** |
| `harness` (+2) | `harness.compare_runs`, `harness.export_run_bundle` | **Done** |
| `interaction` (+2, new) | `interaction.list_sessions`, `interaction.get_last_input` | **Done** |
| contracts | `WorkflowOrchestratorBackend.list/cancel`; `CollaborationSuite.reply/create`; `WebSearchCacheBinding` | **Done** |
| wiring | integration profile workflow/collaboration/notify extensions; `session_storage` via `session_tool_wiring.py` + `SessionStorageToolBinding` | **Done** |

**Delivered:** **140** catalog `tool_id` values · **40** shipped bundles.

**Verification:** `152 passed` (`tests/unit/tools/providers/` + exporters) · `check_harness_no_getattr.py` OK · MCP full-catalog export smoke (**140** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{workflow,notify,collaboration,websearch,harness,interaction}/`

#### T-EXPAND T10 — LKW storage bridge + deferred scheduling (2026-06-07) — **Done**

**Goal:** Close T8/T9 deferred tools (`workspace.export_artifact`, `notify.schedule`) and extend builder/LKW ops without new bundles.

| Bundle | Tools | Status |
|--------|------:|--------|
| `workspace` (+2) | `workspace.export_artifact`, `workspace.import_artifact` | **Done** |
| `notify` (+1) | `notify.schedule` | **Done** |
| `interaction` (+1) | `interaction.get_session_history` | **Done** |
| `eval` (+1) | `eval.export_observations` | **Done** |
| `storage` (+1) | `storage.exists` | **Done** |
| `memory` (+1) | `memory.delete_key` | **Done** |
| `pagerduty` (+1) | `pagerduty.acknowledge_incident` | **Done** |
| `message_bus` (+1) | `message_bus.purge_completed` | **Done** |
| `records` (+1) | `records.count` | **Done** |
| contracts | `ScheduledNotificationBinding`; `SessionStorageBinding.get_session_history`; `TaskMemoryViewBinding.delete`; `TaskQueue.purge_completed` | **Done** |
| wiring | `notify_tool_wiring.py` + `PolicyScopedMemoryView.delete` | **Done** |

**Delivered:** **150** catalog `tool_id` values · **40** shipped bundles.

**Verification:** `164 passed` (`tests/unit/tools/providers/` + exporters) · `check_harness_no_getattr.py` OK · MCP full-catalog export smoke (**150** tools)

**Closeout notes (accepted platform limits):**

| Area | Platform behavior | Product follow-up |
|------|-------------------|-------------------|
| `notify.schedule` | Records deferred delivery in `ScheduledNotificationBinding` (in-memory default via Tier-3 wiring) | Production dispatcher/cron in application host |
| `message_bus.purge_completed` | **Done** — KV task index on broker queues (`rabbitmq`, `kafka`); Celery unchanged | Residual: Celery result-backend purge |
| `pagerduty.acknowledge_incident` | **Done** — `PagerDutyEventsClient.acknowledge_incident` + adapter + typed `PagerDutyIncidentChannel` | — |

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{workspace,notify,interaction,eval,storage,memory,pagerduty,message_bus,records}/`

#### T-EXPAND T11 — HITL write path + cloud/vector store ops (2026-06-07) — **Done**

**Goal:** Close T8/T10 deferred governance and integration-bridge gaps without product scope.

| Bundle | Tools | Status |
|--------|------:|--------|
| `hitl` (+2) | `hitl.submit_response`, `hitl.list_for_task` | **Done** |
| `notify` (+2) | `notify.list_scheduled`, `notify.cancel_scheduled` | **Done** |
| `cloud_platform` (new) | `cloud_platform.health`, `cloud_platform.resolve` | **Done** |
| `vector_store` (new) | `vector_store.count`, `vector_store.delete`, `vector_store.list_collections`, `vector_store.health` | **Done** |
| contracts | `HumanDecisionStoreBinding.record` / `list_for_task`; `ScheduledNotificationBinding.cancel_scheduled` | **Done** |
| wiring | `ToolWiringContext.cloud_platform`; `IntegrationProfile` cloud platform resolution | **Done** |

**Delivered:** **160** catalog `tool_id` values · **42** shipped bundles.

**Verification:** provider unit tests + MCP full-catalog export smoke (**160** tools) · `check_harness_no_getattr.py` OK

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{hitl,notify,cloud_platform,vector_store,health}/`

#### T-EXPAND T12 — Integration slot health + notify dispatcher (2026-06-07) — **Done**

**Goal:** Close post-T11 harness ops gaps (category health probes, scheduled notify dispatch, Celery purge index).

| Bundle | Tools | Status |
|--------|------:|--------|
| `health` (+9) | `health.check_object_storage`, `health.check_key_value_cache`, `health.check_message_bus`, `health.check_graph_store`, `health.check_identity_provider`, `health.check_relational_store`, `health.check_wiki_knowledge`, `health.check_search_provider`, `health.check_notification_channel` | **Done** |
| `notify` (+1) | `notify.dispatch_due` | **Done** |
| queue | Celery optional KV task index + `purge_completed` | **Done** |
| contracts | `ScheduledNotificationBinding.mark_delivered` | **Done** |
| planner | LEG-DEPTH — remove `use_rag`/`use_websearch` from LLM schema; deprecation trace | **Done** |
| observability | OBS-DEPTH.2 trace bridge phase gate; live emit via `runtime_event_bus` | **Done** |

**Delivered:** **170** catalog `tool_id` values · **42** shipped bundles.

#### T-EXPAND T13 — CRIT-V eval tools (2026-06-07) — **Done**

**Goal:** Ship semantic verification tools for Phase CRIT-V (PEV verify depth) without Nexus orchestrator wiring.

| Bundle | Tools | Status |
|--------|------:|--------|
| `eval` (+2) | `eval.judge`, `eval.trajectory` | **Done** |

**Delivered:** **172** catalog `tool_id` values · **42** shipped bundles.

**Verification:** `test_eval_critic_tools.py` · `test_catalog_expansion.py` (172) · MCP export smoke (**172** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md)


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
| Q.0.1 | Appendix C traceability matrix (audit → Q ID) | **Done** | Docs | C-all | Appendix C below; each row has owner phase |
| Q.0.2 | Phase Q execution order + PR sizing guide | **Done** | Docs | — | §4 + subsection **Q execution order** below |
| Q.0.3 | Gate policy: no Q PR without `pytest -m gate` | **Done** | CI | — | Documented in Q DoD; CI unchanged paths |

---

#### Phase Q-N — Nexus, loops, orchestration, error handling

**Components:** `intergrax/runtime/nexus/`, `intergrax/runtime/execution/`, `intergrax/runtime/hooks/`, `intergrax/runtime/interrupts/`, `intergrax/runtime/policy/`, `intergrax/runtime/nexus/retry/`, `intergrax/agents/agent_engine.py`, `intergrax/agents/uaep.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-N.1 | **Decompose `NexusLoop`** — extract HITL runner, long-running coordinator calls, event publisher, shadow/sandbox cleanup into dedicated modules; `NexusLoop` orchestrates only | **Done** | High | `nexus/orchestration/` (`graph_runner`, `task_events`, `lifecycle_bridge`, …) | `nexus_loop.py` ~586 lines; gate green |
| Q-N.2 | **Fix duplicate `_normalize_human_response`** — single call in `_handle_task_impl` | **Done** | High | `nexus_loop.py` L229–231 | Duplicate call removed (2026-06-01) |
| Q-N.3 | **Retry semantics document + facade** — one doc section: `RetryEngine` (graph/validation/alternate agent) vs `RuntimeConfig.max_run_retries` (LLM/tool in `RuntimeEngine`); optional `RetryCoordinator` delegating both | **Done** | High | `nexus/retry/`, `nexus/config.py`, architecture §31.1 | Doc merged; no duplicate retry without trace event |
| Q-N.4 | **Unify policy injection** — `PolicyEngine` only in public Nexus/UAEP APIs; remove `RuntimePolicyEngine` union from external signatures; `coerce_policy_engine` internal | **Done** | Medium | `nexus_loop.py`, `uaep.py`, factories | Type check / mypy clean on factories; gate green |
| Q-N.5 | **§42 hook parity — decision / interrupt / retry** — wire `BEFORE/AFTER_DECISION`, `BEFORE/AFTER_INTERRUPT`, `BEFORE/AFTER_RETRY` in NexusLoop + UAEP + `RetryEngine`; update `hooks/parity.py` to **WIRED** or **Won't fix** with canon amendment | **Done** | Medium | `hooks/`, `nexus_loop.py`, `uaep.py`, `retry_engine.py` | `parity.py` no NOT_WIRED for these six OR canon §42.20 amended + tests |
| Q-N.6 | **§42 hook parity — trace persist** — `BEFORE/AFTER_TRACE_PERSIST` **WIRED** at trace finalize path; `parity.py` → **WIRED** | **Done** | Medium | `hooks/`, `task_trace.py`, trace emitter | Parity test; hook invoked in integration test |
| Q-N.7 | **Rename Nexus context helpers module** — `runtime_steps/tools.py` → `runtime_steps/tool_context_helpers.py` (or merge into `tools_step.py`); update imports | **Done** | Low | `tool_context_helpers.py` + shim `tools.py` | Backward-compatible re-export (2026-06-01) |
| Q-N.8 | **Split `RuntimeConfig`** — `ModelRuntimeConfig`, `RetrievalRuntimeConfig`, `ToolsRuntimeConfig`, `PlanningRuntimeConfig`, `TraceRuntimeConfig`; composed `RuntimeConfig`; `validate()` cross-field | **Done** | High | `nexus/config.py` | Backward-compatible properties or migration shim one release; all factories updated |
| Q-N.9 | **Type `integration_profile`** — `IntegrationProfile` from `intergrax.integrations` on `RuntimeConfig` / wiring contexts | **Done** | Medium | `nexus/config.py`, `engine/runtime_context.py` | No `Optional[object]` for profile in public config |
| Q-N.10 | **`production_mode` lab default** — `lab_application` / scaffold sets `production_mode=False`; document in Step 4E | **Done** | Low | Tier-3 factories, `guides/AGENT_CREATION_GUIDE.md` | `harness_production_mode()` in `applications/_shared/runtime_defaults.py` |
| Q-N.11 | **Graph callback typing** — `ExecutionNode` instead of `object` in `GraphExecutor` / NexusLoop node callbacks | **Done** | Low | `execution/graph_executor.py`, `nexus_loop.py` | Mypy/ruff on execution package |
| Q-N.12 | **Interrupt handler hygiene** — remove duplicate `InterruptType` import; add unit test for interrupt → policy path | **Done** | Low | `interrupts/handler.py` | Duplicate import removed (2026-06-01) |
| Q-N.13 | **`AgentEngine` static UAEP** — document or inject `event_bus` for `AgentEngine.run` static path; no silent missing events | **Done** | Low | `agents/agent_engine.py` | `_resolve_static_executor`; `tests/unit/agents/test_agent_engine_event_bus.py` |
| Q-N.14 | **Unit tests for `NexusLoop` helpers** — `_finish_task`, lifecycle transitions, HITL branch stubs (mock deps) | **Done** | High | `tests/unit/runtime/nexus/test_nexus_loop.py` | New file; ≥15 focused tests; marker `gate` |
| Q-N.15 | **`GraphExecutor` unit coverage** — failure recovery, skip completed, handoff edge (beyond stub integration) | **Done** | Medium | `tests/unit/runtime/execution/` | `test_graph_executor_coverage.py` + checkpoint skip in `test_runtime_checkpoint.py` |

---

#### Phase Q-L — LLM adapters

**Components:** `intergrax/llm_adapters/`, `docs/architecture/LLM_ADAPTERS.md`, governance plugin.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-L.1 | **Remove or complete `tracked_llm_call`** — if kept: `finally` calls `usage.end_call`; if removed: delete `tracked_call.py` + references | **Done** | Medium | `_shared/tracked_call.py` | File removed (unused) (2026-06-01) |
| Q-L.2 | **Public API surface** — re-export `LLMAdapter`, `LLMProvider`, `LLMAdapterRegistry`, `LLMProfile` from `llm_adapters/__init__.py` | **Done** | Low | `llm_adapters/__init__.py` | Public re-exports (2026-06-01) |
| Q-L.3 | **Provider catalog table in docs** — 19 rows: slug, adapter class, env vars, tools/stream/structured, native vs compat | **Done** | High | `docs/architecture/LLM_ADAPTERS.md` | Table matches `LLMProvider` enum + conformance list |
| Q-L.4 | **Fix `LLMProfile` docstring** — `max_retries` only via `options={}`; align examples in guide | **Done** | Low | `registry/profile.py`, tests | Example fixed (2026-06-01) |
| Q-L.5 | **Per-provider `supports_streaming()` / `supports_structured_output()`** — override defaults (`False` base default for streaming); table in Q-L.3 | **Done** | Medium | Each `providers/*.py`, ABC defaults | Conformance reads flags; no false positives |
| Q-L.6 | **`PolicyEngine` + `llm_cost_evaluation`** — rule hook on `TASK_COMPLETED` or policy replay; or remove “next step” from docs until done | **Done** | Medium | `governance/`, `observability_bridge.py`, `policy_engine.py` | Test: over-quota/warn triggers policy decision or structured log contract |
| Q-L.7 | **Usage tracking doc** — distinguish adapter `LLMAdapterUsageLog` vs runtime `LLMUsageTracker` | **Done** | Low | `docs/architecture/LLM_ADAPTERS.md` § Observability | Two-layer table |
| Q-L.8 | **Conformance: structured output** — parametrize providers with `supports_structured_output`; mock SDK | **Done** | Medium | `tests/unit/llm_adapters/` | Added to gate subset in `llm-adapters-guard.yml` |
| Q-L.9 | **Bedrock `context_window_tokens`** — lookup table or model metadata for common `model_id` | **Done** | Low | `providers/aws_bedrock_adapter.py` | `_CONTEXT_WINDOWS` + prefix fallback; `test_bedrock_context_window.py` |
| Q-L.10 | **OpenAI-compat adapter init** — replace `__dict__.update` with explicit delegation or composition wrapper | **Done** | Low | `openai_compat_providers.py`, factory | `_delegate` + `__getattr__` composition |
| Q-L.11 | **Central env appendix** — single table: `INTERGRAX_LLM_*`, secrets map, per-provider overrides | **Done** | Medium | `architecture/LLM_ADAPTERS.md` appendix | Cross-links from each `providers/*/USAGE.md` |

---

#### Phase Q-R — RAG pipeline & Nexus RAG integration

**Components:** `intergrax/rag/`, `runtime/nexus/context/context_builder.py`, `runtime_steps/rag_step.py`, `history_step.py`, `pipelines/no_planner_pipeline.py`, `tools/providers/rag/`, `agents/legal/*` plan flags.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-R.1 | **Delete dead code in `ContextBuilder`** — `_build_backend_where`, `_map_hits_to_chunks`, unused `VectorStoreHit` import | **Done** | High | `context_builder.py` | Dead helpers removed (2026-06-01) |
| Q-R.2 | **Single retrieval per turn (design)** — ADR in plan: either (A) retrieval only in `RagStep`/`rag.retrieve`, or (B) only in `HistoryStep`; remove duplicate vector calls | **Done** | High | `history_step.py`, `context_builder.py` | `HistoryStep` uses `perform_retrieval=False` (2026-06-01) |
| Q-R.3 | **`ContextBuilder` respects plan `use_rag`** — `_should_use_rag` checks plan/engine `use_rag` when present, not only `enable_rag` | **Done** | High | `context_builder.py` | `request.metadata["use_rag"]`; unit test (2026-06-01) |
| Q-R.4 | **`NoPlannerPipeline` conditional `RagStep`** — include `RagStep` only when plan/tool_ids require RAG | **Done** | High | `no_planner_pipeline.py`, `pipeline_factory.py` | Pipeline test matrix |
| Q-R.5 | **Prefetch vs final `top_k`** — `RetrievalRequest.prefetch_k` optional; Nexus passes `max_docs_per_query` as `final_k` only; service uses profile `prefetch_top_k` when unset | **Done** | High | `retrieval_request.py`, `retrieval_service.py` | `test_retrieval_request_prefetch.py` (2026-06-01) |
| Q-R.6 | **Unify RAG config surface** — map `RuntimeConfig.max_docs_per_query` / threshold → `RagProfile` at factory wire time; deprecate duplicate fields with shim + trace | **Done** | High | `nexus/config.py`, `RetrievalRuntimeConfig`, `rag_profile.py` | One source of truth documented |
| Q-R.7 | **`RagProfile.extras`** — use for vendor knobs or remove field | **Done** | Low | `rag_profile.py` | No unused field in frozen profile |
| Q-R.8 | **`INTERGRAX_RAG_METRICS_ENABLED` in `rag_profile_from_env`** or documented exclusion | **Done** | Low | `rag_profile.py`, architecture §7.1.2 | `extras.metrics_enabled` from env (2026-06-01) |
| Q-R.9 | **`rag/answers/` deprecation path** — mark package deprecated; redirect doc to `RetrievalService`; no new imports from Nexus | **Done** | Medium | `rag/answers/`, `chat_agent` removal (Q-X.1) | Grep: zero imports from `runtime/` and `agents/` except tests |
| Q-R.10 | **`UserProfileManager` LTM via `RetrievalService`** — same metadata scope / `RagProfile` chunking policy | **Done** | Medium | `memory/user_profile_manager.py` | Unit test with fake `RetrievalService` |
| Q-R.11 | **Naming guide — three “context builders”** — table in `AGENT_CREATION_GUIDE` or `intergrax/rag/README.md`: Nexus `ContextBuilder`, `ContextManager`, `DefaultContextBuilder` | **Done** | Low | Docs | Linked from architecture §28 pointer |
| Q-R.12 | **Legacy `use_rag` plan flags** — migrate Legal/Nexus plans to `tool_ids` including `rag.retrieve`; emit deprecation `RuntimeEvent` on boolean | **Done** | Medium | `engine_plan_models.py`, `legal/*`, `tool_runtime.py` | Legal tests use `tool_ids`; booleans shim one release |

---

#### Phase Q-M — Memory

**Components:** `intergrax/memory/`, `runtime/task_memory/`, `runtime/nexus/context/`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-M.1 | **Memory architecture one-pager** — four stores: session history, user LTM, task KV (`TaskMemory`), shared graph context; diagram + when to enable SQLite | **Done** | High | `docs/` section in plan §0 or `AGENT_CREATION_GUIDE` Appendix | Linked from §0.3 execution path |
| Q-M.2 | **Task memory visibility in scaffold** — `wire_task_memory` in lab/product templates; env `INTERGRAX_TASK_MEMORY_DB` in `.env.example`; Step 4E paragraph | **Done** | Medium | `applications/*`, scaffold, guide | Scaffold acceptance asserts task memory path optional |
| Q-M.3 | **`resolve_task_memory_persistence` defaults** — log warning when None in lab; debug API hint | **Done** | Low | `task_memory/store.py`, `lab_application` factory | Doc + single integration test |

---

#### Phase Q-O — Observability & metrics

**Components:** `runtime/events/`, `runtime/nexus/tracing/`, `runtime/metrics/`, `debug/`, `llm_adapters/tracking/`, `rag/tracking/`, `applications/_shared/platform_wiring.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-O.1 | **Register RAG observability plugin in default bootstrap** — `register_rag_observability_plugin(plugins)` alongside LLM in `platform_wiring.py` | **Done** | **Critical** | `platform_wiring.py` | `test_platform_wiring_observability.py` (2026-06-01) |
| Q-O.2 | **RAG observability bridge tests** — mirror `test_observability_bridge.py` (LLM) | **Done** | High | `tests/unit/rag/tracking/` | `test_rag_observability_bridge.py` (2026-06-01) |
| Q-O.3 | **Parser trace export strategy** — route `parser_trace_flush` through `ObservabilityBackend` **or** document intentional bypass + single env table | **Done** | Medium | `parser_trace_flush.py`, `parser_trace_exporter.py`, integrations | Documented in architecture §7.1.2 RAG observability |
| Q-O.4 | **`metrics/export.py` typed trace summary** — use `DiagnosticPayload` / `trace_models` schema ids instead of substring heuristics | **Done** | Medium | `runtime/metrics/export.py` | Unit test with synthetic trace events |
| Q-O.5 | **Lint `metrics/export.py`** — remove duplicate `ExecutionMetrics` import | **Done** | Low | `metrics/export.py` | Ruff clean (2026-06-01) |
| Q-O.6 | **`export_run_metrics` behavioral field** — populate from governance/replay or remove from DTO | **Done** | Low | `metrics/export.py` | `ExecutionMetrics` from trace events in `export_run_metrics` |
| Q-O.7 | **Mount LLM metrics routes on lab** — `register_llm_metrics_routes(app)` when `INTERGRAX_LLM_METRICS_ENABLED` | **Done** | Medium | `lab_application/host/factory.py` | Routes registered at factory (2026-06-01) |
| Q-O.8 | **Observability env profile doc** — one table: trace DB, runtime events DB, LLM/RAG metrics, parser trace, integration observability slug | **Done** | High | New subsection §0 or `infra/README` cross-link | All Tier-3 `.env.example` reference same names |
| Q-O.9 | **RAG metrics parity decision** — implement log-only parity **or** `register_rag_metrics_routes` + optional Pushgateway | **Done** | Medium | `rag/tracking/`, architecture §7.1.2 | Matches documented behavior |
| Q-O.10 | **Unify phase mapping** — `trace_bridge` delegates phase to `phase_coverage.py`; single source | **Done** | Medium | `events/trace_bridge.py`, `phase_coverage.py` | Unit test: same `ExecutionPhase` for sample events |
| Q-O.11 | **Debug router type imports** — explicit imports for `DebugHitlResumeService`, `AgentRegistry` in annotations | **Done** | Low | `debug/router.py`, `debug/app.py` | Explicit imports in `debug/router.py` |
| Q-O.12 | **`trace_bridge` unit tests** | **Done** | Medium | `tests/unit/runtime/events/test_trace_bridge.py` | Gate marker |
| Q-O.13 | **Clarify dual Prometheus** — in-process scrape vs `integrations` PromQL backend | **Done** | Low | `docs/architecture/LLM_ADAPTERS.md` § Observability | Prevents operator confusion |
| Q-O.14 | **Event/trace store adoption** — SQLite-first default; scale-out criteria for `cassandra` / `elasticsearch` | **Done** | Low | Architecture §33.1 + `cassandra/USAGE.md` | No separate ADR file |

---

#### Phase Q-X — Legacy removal & code hygiene

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-X.1 | **`ChatAgent` removal** — migrate remaining tests to `RuntimeEngine`/`NexusLoop`; delete `intergrax/chat_agent.py`; keep import guard script as negative test | **Done** | High | `chat_agent.py`, `tests/unit/chat_agent/` | Grep zero production imports; gate green |
| Q-X.2 | **`task_metadata_bridge` shrink** — migrate callers to typed `Task` metadata; deprecate flat bridge with warning event | **Done** | Medium | `task_metadata_bridge.py`, `uaep.py` | `execution_options_for_request`; legacy warnings; Task hydrates typed fields |
| Q-X.3 | **Copyright / naming consistency** — `Intergrax` header; fix `Integrax` typo in `chat_agent` (or file deleted in Q-X.1) | **Done** | Low | Affected files from audit | Spot-check script or ruff rule |
| Q-X.4 | **`tools_base` deprecation timeline** — document removal after Q-R.12; no new imports | **Done** | Low | `tools/tools_base.py`, governance script | Module docstring + `DeprecationWarning` on import |
| Q-X.5 | **Sync M.6 “Future” slugs table** — weaviate, milvus, snowflake, vault → **Done (beta)** with paths | **Done** | Low | This plan M.6 P3 section | Table matches repo `integrations/providers/` |

---

#### Phase Q-T — Test harness gaps

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-T.1 | NexusLoop unit suite | **Done** | High | See Q-N.14 | — |
| Q-T.2 | `test_rag_profile_from_env` | **Done** | Medium | `tests/unit/rag/profiles/` | Gate (2026-06-01) |
| Q-T.3 | `test_context_builder_retrieval` | **Done** | High | `tests/unit/runtime/nexus/context/` | `test_context_builder.py` (2026-06-01) |
| Q-T.4 | `test_user_profile_manager` | **Done** | Medium | `tests/unit/memory/` | Index + search |
| Q-T.5 | **Catalog vs legacy RAG path** — integration test one pipeline run, retrieval call count ≤1 | **Done** | High | `tests/integration/runtime/` | Implements Q-R.2 acceptance |
| Q-T.6 | **Observability wiring E2E** — lab factory bootstraps LLM+RAG plugins | **Done** | High | `tests/integration/runtime/test_platform_wiring_observability.py` | Q-O.1 (2026-06-01) |

---

#### Phase Q-D — Documentation & plan sync

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-D.1 | Update `docs/README.md` current focus → Phase Q | **Done** | High | `docs/README.md` | — |
| Q-D.2 | Canon §52 Phase L status → **Done** (pointer to Phase Q) | **Done** | Low | `intergrax_runtime_architecture.md` §52 | — |
| Q-D.3 | §2 architecture map — §42 row points to Phase Q-N.5–Q-N.6 | **Done** | Low | This file §2 | — |
| Q-D.4 | `AGENT_CREATION_GUIDE` — Q-M.1 memory diagram + Q-R.11 naming | **Done** | Medium | Guide appendices | — |
| Q-D.5 | **§5.2 reuse enforcement** — document existing gates (`check_agents_vendor_imports`, `check_integration_vendor_imports`, `check_production_chat_agent_imports`) in AGENT_CREATION_GUIDE anti-patterns | **Done** | Low | Guide + `scripts/` | New agent authors see one list |

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

### Phase Q+ — Harness Hardening (post-audit 2026-06-01)

**Source:** Technical debt audit after Phase Q — architecture compliance, typing, observability gaps, legacy parallel stacks, Nexus/planning monoliths.  
**Goal:** Intergrax as a **strong, typed, observable harness** comparable in discipline to Cursor / Claude Code / Google ADK-style agent labs — not merely “gate green”.  
**Principle:** evolve, not rewrite · explicit `Protocol` / Pydantic at boundaries · **zero new `getattr` in `runtime/nexus` and `agents/`** (integrations/LLM SDK edges exempt) · one Q+.* ID per PR · gate green.

**Relationship to Phase Q:** Phase Q closed the **first** audit (Appendix C). Phase Q+ closes the **second** audit (Appendix D). Do not reopen Q.* rows unless a regression is found.

**Out of scope for Phase Q+:**

- Phase K.1/K.2 product agents (unless explicitly prioritized — record in Appendix D)
- K.6 / B.15 Legal live LLM E2E
- New integration catalog slugs (Phase M on-demand)
- Rewriting all LLM provider adapters (only isolate SDK reflection — Q+-I.*)
- Mandatory Cassandra / multi-tenant scale-out (architecture §33.1 criteria only)

**Phase Q+ complete when:** All Q+ rows **Done** or **Won't fix** (canon amendment); Appendix D 100%; §0.5 Harness hardening **Done**; gate unchanged or increased; grep gate: no new `getattr` in `runtime/nexus/` + `agents/` (CI script Q+.0.3).

---

#### Q+.0 — Program governance

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+.0.1 | **Appendix D** — audit topic → Q+ ID matrix (P0–P3) | **Done** | High | This file Appendix D | Every audit section mapped |
| Q+.0.2 | **Q+ execution order** — Waves 1–5 below | **Done** | High | §4 Priority Order | Team follows wave sequence |
| Q+.0.3 | **CI grep gate** — fail on new `getattr`/`setattr` in `intergrax/runtime/nexus/`, `intergrax/agents/` | **Done** | High | `scripts/check_harness_no_getattr.py` + gate workflow | Zero grandfathered harness paths (2026-06-01) |

---

#### Q+-T — Typing & explicit contracts (P0)

**Audit:** loose coupling, `getattr`, `Any` on harness paths, classes not implementing Protocols.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-T.1 | **`UAEPAgent` Protocol** — `get_steps`, `run_step`, optional `resume_step`, `decide_after_step`; replace `supports_uaep()` duck typing | **Done** | **Critical** | `agents/uaep_protocol.py`, `agents/uaep.py` | Standalone `@runtime_checkable` Protocol; no `getattr` in UAEP |
| Q+-T.2 | **`ToolInvokerProtocol`** — explicit `registry`; remove `catalog_context` invoker chain `getattr` | **Done** | **Critical** | `runtime/nexus/tools/`, `catalog_context.py` | Typed invoker only |
| Q+-T.3 | **`RuntimeState` trace hook** — `trace_event: Optional[TraceEmitterFn]`; remove `getattr(state, "trace_event")` | **Done** | High | `tool_access_policy.py` | `TraceEmittingRuntimeState` Protocol |
| Q+-T.4 | **`Agent.can_handle(TaskContext)`** — replace `task_context: Any` on `Agent` ABC | **Done** | High | `agents/agent_contract.py`, product agents | Production agents use `TaskContext` |
| Q+-T.5 | **`EnginePlan` / tool plan union** — `tool_runtime` reads `tool_ids` without `getattr(source, …)` | **Done** | High | `tool_runtime.py`, `engine_plan_models.py` | `ToolPlanLike` + `EnginePlan.resolved_tool_ids()` |
| Q+-T.6 | **`long_running_bridge`** — `RuntimeEventPublisher` accepts `RuntimeEvent` only (not `object`) | **Done** | Medium | `orchestration/long_running_bridge.py` | Align with `NexusRuntimeEventPublisher` |
| Q+-T.7 | **`context_builder` session snapshot** — typed session view; no `getattr(session, attr)` loop | **Done** | Medium | `context/context_builder.py` | `ChatSession` fields directly |
| Q+-T.8 | **`rag_step_policy`** — use `NexusPlan` / `EnginePlan` fields only | **Done** | Low | `pipelines/rag_step_policy.py` | `isinstance(plan, EnginePlan)` |

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
| Q+-L.2 | **Legal agent → catalog `ToolRuntime`** — remove runtime dependency on `ToolsAgent` / `ToolsStep` planner loop | **Done** | **Critical** | `agents/legal/`, `catalog_tool_planner.py` | Legal uses `CatalogToolPlanner` + `tool_planner` |
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
| Q+-P.1 | **Split `engine_planner`** — parse / validate / LLM call modules; each &lt; ~300 lines | **Done** | Medium | `engine_planner_parse.py`, `engine_planner_messages.py`, `engine_planner_diagnostics.py`, `engine_planner_orchestrator.py` | Orchestration + traces extracted |
| Q+-P.2 | **Split `step_planner`** — strategy registry vs executor | **Done** | Medium | `planning/step_planner/` (`config`, `step_factory`, `assembly`, `strategies`, `planner`) | Package import stable; gate tests |
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
| Q+-I.1 | **SDK reflection quarantine** — document per-provider `*_sdk_bridge.py`; no new getattr in `runtime/` | **Done** | Low | Architecture §5.2.2 | Vendor SDK bridges quarantined to provider modules |

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

### Phase R — Harness AI Alignment (post-audit 2026-06-01)

**Source:** Harness AI philosophy audit (scaffold, harness, LLM, tool vs skill, context engineering, subagents, policy) — traceability in **Appendix E**.  
**Status:** **Done (MVP)** (2026-06-01). **Prerequisite met:** Phase **Q+ Done**.  
**Goal:** Intergrax vocabulary and Tier-0 modules align with industry harness terminology **without** breaking Integration → Tool → Agent stack; add **Skill Library** for reuse and external compatibility.  
**Principle:** evolve, not rewrite · skills **compose** tools (never replace `ToolRuntime`) · one R.* ID per PR · gate green.

**Out of scope for Phase R:**

- Nested full harness per child (Cursor 1:1 subagent OS) — use graph delegation first (R-Delegate)
- Auto-discovery of skills from filesystem without validation
- Mandatory migration of all Tier-2 agents to skills in one release

**Phase R (MVP) complete:** Appendix E 100% **Done** or **Won't fix**; §0 Phase R row **Done**; gate **450 passed** (2026-06-01). Further skill catalog expansion is product work, not a harness gate.

---

#### R.0 — Canon, ADR, terminology (do first)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R.0.1 | **ADR: Skill layer Option 2** — reject “skills = tools only”; document four-layer model | **Done** | **Critical** | Architecture §7.1.8, §5.3 | Option 1 listed as rejected with rationale |
| R.0.2 | **Canon sections** — §5.3 Harness mapping, §7.1.8 Skills, §28.1 Context engineering, §42.14.3 Delegation, §42.11.4 Policy bundle | **Done** | **Critical** | `intergrax_runtime_architecture.md` | Cross-linked from plan §0 |
| R.0.3 | **Remove tool/skill conflation** in code docstrings | **Done** | High | `tools/core/contracts.py` | `ToolContract` describes **tool** only |
| R.0.4 | **README navigation** — Phase R, skills layer in root + docs README | **Done** | Medium | `/README.md`, `docs/README.md` | GitHub landing + docs index mention skills |

**Delivery rule:** Same as §6.1 — one R.* ID → PR → update Appendix E status → gate.

---

#### R-Skill — Skill Library (Tier-0)

**Problem:** Integrations and tools are production-grade; **skills are not**. Agents duplicate prompts, tool allow-lists, and policy fragments. External harness ecosystems (Cursor skills, internal markdown packs) cannot plug in without a **validated manifest**.

**Target layout:**

```text
intergrax/skills/
├── core/                   # SkillContract, SkillManifest, SkillProvider protocol
├── registry/               # SkillCatalog, SkillProfile, register_default_skills()
├── importers/              # cursor_skill_md.py, … (validate → SkillManifest)
├── _shared/
└── providers/
    └── <domain>/           # e.g. legal/, research/
        ├── manifest.py     # SkillManifest instance(s)
        ├── prompts.yaml    # or Prompt Registry refs
        └── USAGE.md
```

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Skill.1 | **`SkillManifest` + `SkillContract`** — frozen manifest: `skill_id`, `version`, `description`, `tool_ids`, `prompt_instruction_ids`, `policy_fragment_id`, `risk_tier`, `tags` | **Done** | **Critical** | `intergrax/skills/core/contracts.py` | Pydantic/jsonschema round-trip test |
| R-Skill.2 | **`SkillRegistry` + `SkillProfile` + `SkillCatalog`** — mirror Tool registry pattern | **Done** | **Critical** | `intergrax/skills/registry/` | `build_registry_from_profile()` |
| R-Skill.3 | **`SkillResolver`** — given `skill_ids`, produce resolved `allowed_tools` ∪, prompt pack refs, policy fragments; **no LLM execution** in resolver | **Done** | **Critical** | `intergrax/skills/resolver.py` | Unit: two skills merge tool lists with conflict rules |
| R-Skill.4 | **Tier-3 wiring** — skill profile in `ApplicationBuildContext`, `skill_wiring.py`, legal host | **Done** | High | `applications/_shared/skill_wiring.py` | Legal registry resolves skills |
| R-Skill.5 | **`AgentContract.skill_ids`** + validation against registry at register time | **Done** | High | `intergrax/contracts/`, `AgentRegistry` | Unknown skill_id → register error |
| R-Skill.6 | **`docs/architecture/SKILLS.md`** — catalog, layering diagram, import rules | **Done** | Medium | `docs/architecture/SKILLS.md`, `docs/README.md` index row | Approved index entry |
| R-Skill.7 | **Scaffold `new-skill`** | **Done** | Medium | `intergrax/scaffold/new_skill.py` | `python -m intergrax.scaffold new-skill <id>` |
| R-Skill.8 | **`CursorSkillImporter`** — parse `SKILL.md` + frontmatter → `SkillManifest` (best-effort; reject on schema fail) | **Done** | High | `intergrax/skills/importers/cursor_skill_md.py` | Fixture test with sample SKILL.md |
| R-Skill.9 | **Pilot skill pack** — `legal.contract_review` (tool_ids + prompt refs + policy fragment) | **Done** | High | `intergrax/skills/providers/legal/` | Legal agent lists `skill_ids`; gate green |
| R-Skill.10 | **Nexus trace events** — `SKILL_RESOLVED`, `SKILL_IMPORT_FAILED` | **Done** | Low | `runtime/events/context_skill_recording.py` | `record()` on register + import service |

**Skill vs tool enforcement:**

| Rule | Enforcement |
|------|-------------|
| Skill MUST NOT be a `ToolContract` | CI: no `ToolHandler` named `skill.*` without ADR |
| Skill MAY reference only registered `tool_id`s | `SkillResolver` validates against `ToolRegistry` |
| LLM tool-calling surface = **tools only** | Skills expand allow-list before run, not at invoke time |
| External skill without manifest validation | **Rejected** at import — no silent attach |

---

#### R-Context — Context engineering (Tier-1)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Context.1 | **`ContextBudgetPolicy`** — `max_chars`, `max_tokens_estimate`, `summary_tier` defaults; applied in `ContextManager.build_agent_context()` | **Done** | **Critical** | `runtime/nexus/context/context_budget.py` | Test: over-budget input trimmed |
| R-Context.2 | **Trace events** — `CONTEXT_ASSEMBLED`, `CONTEXT_TRIMMED` with before/after sizes | **Done** | High | `ContextManager` + `context_skill_recording` | Emitted when `event_bus` wired |
| R-Context.3 | **AGENT_CREATION_GUIDE** — “Context engineering” subsection links canon §28.1 | **Done** | Medium | `guides/AGENT_CREATION_GUIDE.md` Appendix G | No duplicate truth |
| R-Context.4 | **Finish unified tool path** — residual `use_rag` / `RagStep` callers → `rag.retrieve` | **Done** | High | `tool_gateway.py`, legal bridge, `context_builder.py` | Bridge uses `tool_ids`; LLM booleans sync in `LegalToolPlan` only |

---

#### R-Delegate — Graph-native delegation (subagent equivalent)

Intergrax does **not** implement Cursor-style nested harness in Phase R. **Delegation** = Nexus graph node with isolated memory namespace and bounded context assembly.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Delegate.1 | **`DelegationSpec` on `ExecutionNode`** — `child_agent_id`, `isolated_memory_namespace`, `context_assembly_override` | **Done** | High | `contracts/delegation.py`, `execution_graph.py` | Schema + validation |
| R-Delegate.2 | **Memory namespace isolation** — child reads/writes under `task_id/delegation/{node_id}/` via `MemoryView` | **Done** | High | `delegation_memory.py`, UAEP | Unit test |
| R-Delegate.3 | **Trace linkage** — `parent_run_id`, `parent_node_id` on child run metadata | **Done** | Medium | `graph_executor.py` | Request metadata on child node |
| R-Delegate.4 | **Integration tests** — two-agent graph with delegation node | **Done** | Medium | `test_graph_executor_delegation.py` | Gate |

---

#### R-Policy — Unified policy bundle (Tier-1 + Tier-3)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Policy.1 | **`RuntimePolicyBundle`** — aggregates tool, memory, budget, HITL, plan-loop; optional `domain_fragments: dict[str, Any]` | **Done** | High | `runtime/policy/policy_bundle.py` | Import via `policy_bundle` module (not `policy.__init__`) |
| R-Policy.2 | **Tier-3 composition** — lab/product factories build bundle once per app | **Done** | High | `policy_wiring.py`, lab/legal `wiring.py` | `ApplicationBuildContext.policy_bundle` |
| R-Policy.3 | **Canon §42.11.5** — “how to read policy for a run” operator section | **Done** | Medium | Architecture §42.11.5 | Operator runbook table |

---

#### Phase R — Definition of done

1. R row **Done** with date in Appendix E paydown log.
2. **Gate:** `uv run pytest -m gate -q` green.
3. **Skills:** at least one first-party skill pack + one importer test (R-Skill.8 or Won't fix with reason).
4. **No** new `ToolContract` entries that represent multi-step business workflows without ADR.
5. Update Appendix E status.

---

#### Phase R — Recommended execution order

```text
Wave R0 (canon):           R.0.1 → R.0.2 → R.0.3 → R.0.4
Wave R1 (skill core):      R-Skill.1 → R-Skill.2 → R-Skill.3 → R-Skill.5 → R-Skill.4
Wave R2 (skill ecosystem): R-Skill.8 → R-Skill.7 → R-Skill.9 → R-Skill.6 → R-Skill.10
Wave R3 (context):         R-Context.1 → R-Context.2 → R-Context.4 → R-Context.3
Wave R4 (delegate):        R-Delegate.1 → R-Delegate.2 → R-Delegate.3 → R-Delegate.4
Wave R5 (policy):          R-Policy.1 → R-Policy.2 → R-Policy.3
```

**Gate before Phase K.1/K.2 scale:** **Met** — Q+ **Done**, R-Skill.1–R-Skill.5 and R-Context.1 **Done**.

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
| S-H.1 | **Platform skill bundle `harness`** — ≥3 skills (e.g. `harness.tool_smoke`, `harness.context_demo`, `harness.trace_read`) | **Done** | **Critical** | `intergrax/skills/providers/harness/`, `architecture/SKILLS.md`, bootstrap | `test_harness_skill_bundle.py` |
| S-H.2 | **Lab wiring** — `SkillProfile` + `ToolProfile` + policy bundle documented as canonical harness preset | **Done** | High | `skill_wiring.py`, `guides/HARNESS_ENVIRONMENT.md` | lab enables `harness` bundle |
| S-H.3 | **Cursor SKILL.md importer** in gate | **Done** | Medium | `tests/unit/skills/importers/test_cursor_skill_md.py` | `pytest.mark.gate` |
| S-H.4 | **`rag.answers` test migration** — no deprecation warnings in gate | **Done** | Low | `tests/integration/rag/answers/` | `RetrievalService` only |
| S-H.5 | **Echo/signoff path** — lab run proves skills + trace + policy bundle (existing agents) | **Done** | High | `tests/acceptance/agent_os/test_lab_application.py` | gate + harness wiring tests |

#### S-Doc — Operator & author surfaces

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-Doc.1 | **`guides/HARNESS_ENVIRONMENT.md`** — lab stack, env vars, stable integrations, OTLP, policy bundle read order | **Done** | **Critical** | `docs/guides/HARNESS_ENVIRONMENT.md`, `docs/README.md` index | Linked from plan §6 |
| S-Doc.2 | **Context / trace operator section** — `CONTEXT_*` events, debug API, metrics routes | **Done** | Medium | `guides/HARNESS_ENVIRONMENT.md` | Pointers to canon §28.1 |

#### Phase S — Definition of done

1. **Stable** integration list for lab harness stack published and tested (S-Ops.1).
2. **OTLP path** documented and wired for lab when env configured (S-Ops.2).
3. **≥ 3** `harness.*` platform skills + legal/research bundles registered (S-H.1).
4. **`guides/HARNESS_ENVIRONMENT.md`** complete; lab wiring matches doc (S-H.2, S-Doc.1).
5. Gate: `uv run pytest -m gate -q` green; `python scripts/check_harness_no_getattr.py` OK.
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

### Phase T — Harness Cleanliness (post-S 2026-06-01)

**Status:** **Done** (2026-06-01). **Prerequisites:** Phase S **Done**.  
**Goal:** Close harness technical debt — unified lab preset, typed Tier-2 agents, native catalog planner, expanded stable stack, gate hygiene — without new business agents.

| # | Deliverable | Status | Location | Acceptance |
|---|-------------|--------|----------|------------|
| T-Ops.1 | **`lab_harness_preset()`** — default lab profile (sqlite + log + lab_json + OTEL; optional redis/qdrant) | **Done** | `IntegrationProfile`, `integration_wiring.py`, `settings.py` | `test_lab_harness_preset.py` |
| T-H.1 | **Echo/signoff `skill_ids`** — `harness.tool_smoke` on `AgentContract` | **Done** | `agents/echo`, `agents/signoff_probe` | `test_harness_reference_agent_skills.py` |
| T-H.2 | **`rag.answers` gate hygiene** — gate uses `RetrievalService` only; legacy tests marked `legacy_rag_answers` | **Done** | `tests/integration/rag/answers/` | No `rag.answers` in `-m gate` |
| T-H.3 | **Typed `TaskContext` in Tier-2 agents** — no `getattr` on capability/message content in `agents/` | **Done** | echo, research, signoff, org worker, lab mocks | `check_harness_no_getattr.py` scans `agents/` |
| T-Ops.5 | **`CatalogToolPlanner`** without `ToolsAgent` wrapper | **Done** | `tool_planning_service.py`, `catalog_tool_planner.py` | `test_catalog_tool_planner.py` |
| T-Ops.6 | **Tier-2 stable stack** — `postgresql` + `sentry` in `HARNESS_LAB_STABLE_SLUGS` | **Done** | `harness_lab_stack.py`, postgresql `register.py` | `test_harness_lab_stable_stack.py` |

#### Phase T — Definition of done

1. Lab default wiring uses `lab_harness_preset()` (OTEL on unless env disables).
2. Echo and signoff_probe declare `harness.tool_smoke` via `skill_ids`.
3. Gate RAG path is `RetrievalService`-only; legacy `rag.answers` tests excluded from gate.
4. `python scripts/check_harness_no_getattr.py` passes with `agents/` in scan roots.
5. `CatalogToolPlanner` does not import `ToolsAgent`.
6. `postgresql` stable in catalog and harness stack list.

**After Phase T Done (historical):** Harness cleanliness complete. **Scheduling (2026-06-02):** product milestone K.1/K.2 is **deferred** (§6.3), not the default next step.

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
| U-Typ.2 | **`ToolPlanningConfig` in Tier-1** — planner prompts/config in `runtime/nexus/tools/`; `ToolPlanningService` does not import `tools.tools_agent` | **Done** | High | `runtime/nexus/tools/` | `test_catalog_tool_planner.py` |
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
| U-Leg.2 | **`rag.answers` removal or archive** — migrate remaining `legacy_rag_answers` tests to `RetrievalService`; delete or move module under `intergrax/legacy/` | **Done** | Medium | `intergrax/legacy/rag_answers/` | `test_rag_answers_removed.py` |
| U-Leg.3 | **Legacy tool plan booleans** — document sunset for `from_legacy` / `uses_legacy_booleans_only`; gate new usage | **Done** | Low | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | Deprecation warnings |

#### U-Doc — Operator & architecture alignment

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Doc.1 | **`guides/HARNESS_ENVIRONMENT.md`** — security (auth, MCP), strict profile, policy bundle wiring truth | **Done** | High | `docs/guides/HARNESS_ENVIRONMENT.md` | Phase U security section |
| U-Doc.2 | **Canon §52 / strategy** — lab vs production harness checklist references Phase U | **Won't fix** | Medium | — | Deferred; plan + HARNESS_ENVIRONMENT sufficient |
| U-Doc.3 | **Fix Phase K footer** in `guides/HARNESS_ENVIRONMENT.md` (post-T, gated on U) | **Done** | Low | `guides/HARNESS_ENVIRONMENT.md` | Gated on Phase U |

#### U-CI — Verification & smoke

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-CI.1 | **harness-smoke includes Phase U tests** — auth, strict harness, lab settings | **Done** | High | `.github/workflows/unit-tests.yml` | harness-smoke extended |
| U-CI.2 | **Acceptance: production harness path** — one gate test: strict lab + sqlite trace + policy bundle + skill-resolved tools | **Done** | **Critical** | `tests/acceptance/agent_os/`, unit strict harness | `pytest -m gate` **479 passed** |
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
| V-SEC.3 | Retrieval poisoning defense (trust score/quarantine flow) | **Done** | High | `retrieval_security_wiring.py` filters chunks in `RagStep` when profile enabled |
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
| 2026-06-02 | V-CG.1, V-AM.1, V-ALG.1 | Typed baseline contracts added (`intergrax/runtime/architecture/`) + report-only artifacts script (`scripts/phase_v_foundations_report.py`) + unit tests |
| 2026-06-02 | V-CG.2, V-CG.3, V-CG.4 | Lineage/impact/compatibility modules + capability graph guard script (`scripts/phase_v_capability_graph_guard.py`) + enforce switch + unit tests |
| 2026-06-02 | V-AM.2, V-ALG.2, V-EVAL.1 | Metrics pipeline contracts + promotion flow evaluator + unified evaluation mode contracts + governance artifacts script (`scripts/phase_v_governance_report.py`) + unit tests |
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

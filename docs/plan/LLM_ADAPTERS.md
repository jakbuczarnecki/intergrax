# Llm Adapters — Implementation Plan

**Architecture (1:1):** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-6.1 | §6 LLM | Structured output validation on 100% reference + certified agent paths | P1 | **Done** |
| AUDIT-IDEAL-6.2 | §6 LLM | Live cost/latency/quality model routing (AHI prod path) | P2 | Planned |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

### 6.1v Harness implementation queue — LLM completion response envelope (closed)

**Purpose:** Single ordered list for **Phase M-LLM-R** (Band 2z). **Closed 2026-06-06** — **39/39 Done**. Runs **in parallel** with W-ADAPT waves 5–7 (Tier-0 LLM contract; independent of L4 runtime loop).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **M-LLM-R.0.2–0.3** | Docs | **Done** | ADR-LLM-001 + canon §5.2.2 addendum | Linked from plan |
| 2 | **M-LLM-R.1.1–1.8** | Code | **Done** | Contract types + builders + public exports | Import smoke; no dict returns |
| 3 | **M-LLM-R.2.1–2.6** | Code | **Done** | `LLMAdapter` ABC typed signatures | ABC compiles; stubs updated |
| 4 | **M-LLM-R.3.1–3.7** | Code | **Done** | All provider adapters return envelope | Conformance per provider family |
| 5 | **M-LLM-R.4.1–4.6** | Code | **Done** | Nexus runtime consumers | `test_core_llm_step` + tool planner |
| 6 | **M-LLM-R.5.1–5.3** | Code | **Done** | RAG + websearch + legacy | RAG unit tests green |
| 7 | **M-LLM-R.6.1–6.4** | Code | **Done** | Agents + scaffold + CI lint | `check_llm_adapter_typed_returns.py` + `check_agents_llm_adapter_response.py` |
| 8 | **M-LLM-R.7.1–7.5** | Code | **Done** | Usage alignment + replay/trace bridge | `test_replay_engine` + diagnostics |
| 9 | **M-LLM-R.8.1–8.4** | Docs/CI | **Done** | Docs + conformance + closeout | M-LLM.14 Done; Appendix L complete |

**Suggested PR order:** See [Phase M-LLM-R — Suggested PR order](plan/LLM_ADAPTERS.md).

**Explicitly excluded:** K.1, K.2, product HTTP API DTOs, provider SDK rewrites — [§6.3a](#63a-business-backlog-register-consolidated).### 6.1w Harness implementation queue — Integration expansion (M.6 P4 closed)

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

---

### 6.2ad Phase M-LLM-R execution order (Band 2z — closed 2026-06-06)

**Status:** **Done** · register: [Phase M-LLM-R](plan/LLM_ADAPTERS.md) · queue: [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

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

**Closeout gate:** `scripts/check_llm_adapter_typed_returns.py` + `scripts/check_agents_llm_adapter_response.py` + full `tests/unit/llm_adapters/` gate green (M-LLM-R.8.3, M-LLM-R.6.4).### 6.2ac Phase W-ADAPT execution order (Band 2y — closed)

**Status:** **Done** (2026-06-02) · register: [Phase W-ADAPT](plan/CRITIC_VERIFICATION.md) · queue: [§6.1t](#61t-harness-implementation-queue--adaptive-harness-intelligence-closed)

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

---

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

---

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

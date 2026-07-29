# Intergrax LLM Adapters

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/LLM_ADAPTERS.md`](../plan/LLM_ADAPTERS.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5  
**Audit layers:** 6 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Audit instruction:** [`audit/LLM_ADAPTERS.md`](../audit/LLM_ADAPTERS.md)  
**Developer guide:** [`intergrax/llm_adapters/USAGE.md`](../../intergrax/llm_adapters/USAGE.md)  
**ADR:** [ADR-LLM-001](../adr/entries/2026-06-06/ADR-LLM-001.md) (envelope) · [ADR-LLM-002](../adr/entries/2026-06-14/ADR-LLM-002.md) (ModelCatalog) · [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md) (routing rules)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (LLM_ADAPTERS canon).

- **Implement / audit default:** adapter envelope + routing hub. Failover: [`satellites/LLM_ADAPTERS_routing_failover.md`](satellites/LLM_ADAPTERS_routing_failover.md). Providers: [`satellites/LLM_ADAPTERS_providers_catalog.md`](satellites/LLM_ADAPTERS_providers_catalog.md). Audit register: [`satellites/LLM_ADAPTERS_audit_register.md`](satellites/LLM_ADAPTERS_audit_register.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/LLM_ADAPTERS.md`](../plan/LLM_ADAPTERS.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/LLM_ADAPTERS.md`](../guides/audit_slices/LLM_ADAPTERS.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/LLM_ADAPTERS_audit_register.md`](satellites/LLM_ADAPTERS_audit_register.md) | audit register |
| [`satellites/LLM_ADAPTERS_providers_catalog.md`](satellites/LLM_ADAPTERS_providers_catalog.md) | providers catalog |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.
## Layer map

```text
Tier-3  ApplicationEnvironmentProfile.llm_profile  →  LLMProfile.create_adapter()
        LLMRoutingProfile.rules (M-LLM-X.9)          →  custom/built-in LLMRoutingRule classes
        ReasoningProfile.planner_llm_profile         →  separate planner adapter (COG-PROD)
        resolve_llm_adapter() precedence             →  agent > env > INTERGRAX_LLM_*

Tier-1  RuntimeConfig.llm_adapter                    →  single primary adapter per Nexus run
        LLMRoutingEvaluator (M-LLM-X.9)              →  first-match rule → ModelRouter
        context_preflight / engine_history_layer     →  adapter.context_window_tokens
        ModelRouter                                  →  selects profile before adapter create

Tier-0  LLMAdapterRegistry                           →  lazy provider factories
        ModelCatalog (target)                        →  model_id → context, capabilities
        providers/*                                  →  vendor SDK + envelope mapping
        tracking/ governance/ _shared/               →  metrics, quota, resilience, conformance
```

---

## Response envelope (M-LLM-R)

All adapter completion methods return typed envelopes — **not** bare `str` or untyped dicts.

| Method | Return type |
|--------|-------------|
| `generate_messages` | `LLMAdapterResponse` |
| `generate_with_tools` | `LLMAdapterResponse` |
| `stream_messages` / `stream_with_tools` | `Iterable[LLMStreamEvent]` |
| `generate_structured` | `LLMStructuredResult[T]` |

### `LLMAdapterResponse`

| Field | Type | Notes |
|-------|------|-------|
| `content` | `str` | Assistant text (alias: `.text`) |
| `finish_reason` | `LLMFinishReason` | Includes `CONTENT_FILTER`, `REFUSAL`, `LENGTH`, `TOOL_CALLS` — normalized via `parse_finish_reason()` |
| `usage` | `LLMTokenUsage` | Per-call token accounting |
| `model` / `provider` | `str` | Identity metadata |
| `response_id` | `str \| None` | Provider correlation id |
| `refusal` | `str \| None` | Provider-native safety/refusal text when present |
| *(post-adapter)* | `GuardrailScanResult` | Optional Tier-3 `llm_guardrail` via middleware (`AFTER_LLM_OUTPUT`) — [`INTEGRATIONS.md`](INTEGRATIONS.md) §47 |
| `tool_calls` | `tuple[LLMToolCall, ...]` | Native tool calls |
| `provider_extensions` | `LLMProviderExtensions` | Tagged optional slices (usage source, vendor fields) |

### Structured output (AUDIT-IDEAL-6.1 — Done)

`generate_structured(..., output_model: type[T])` returns `LLMStructuredResult[T]`. Adapters parse provider JSON and validate with Pydantic via `_validate_with_model()` (see `openai_responses_adapter.py`, OpenAI-compat delegate). Reference agents and certified paths MUST use this method — not manual `json.loads` on bare strings. Gate: `check_agents_llm_adapter_response.py` + conformance tests under `tests/unit/llm_adapters/`.

**Ollama generation schema projection:** the canonical Pydantic `model_json_schema()` remains the final validation contract. `LangChainOllamaAdapter` passes a provider-compatible generation projection to `with_structured_output(..., method="json_schema")` (see `intergrax/llm_adapters/providers/_ollama_schema.py`). The projection is generic, provider-specific, and may relax generation-only constraints (for example `maxLength` values Ollama cannot compile). Returned payloads are always revalidated with the original `output_model`; no field rewriting or aliasing occurs. The projection does not guarantee semantic correctness — only grammar-safe constrained generation.

**Ollama model-aware tool capabilities (TOKEN-9 / TOKEN-9-R1):** `LangChainOllamaAdapter.supports_tools()` reflects the installed model's `/api/show` capability list via `intergrax/llm_adapters/providers/ollama_capabilities.py`. There is no static model-name allowlist. Capability resolution is lazy, cached per adapter instance, and fail-closed. Valid resolved states include `capabilities=["completion","tools"]`, `["completion"]`, or `[]`; missing or malformed capability payloads are **unresolved** (`resolved=False`) and must not be treated as ordinary no-tools models. Unresolved capability state never enters structured-output fallback — the Token Optimization router returns `CAPABILITY_RESOLUTION_FAILED`. Structured output remains fallback only for **resolved** models that genuinely lack `tools`. `generate_with_tools()` uses `ChatOllama.bind_tools()` and maps LangChain `AIMessage.tool_calls` to `LLMToolCall`. Native tool calling is preferred by the Token Optimization router. Ollama does not enforce `tool_choice`; the router still requires exactly one valid tool call. Not every Ollama model declares `tools` — there is no universal Ollama-tools claim.

Example:

```python
from intergrax.llm_adapters import LLMAdapter, LLMAdapterResponse

completion: LLMAdapterResponse = adapter.generate_messages(messages, run_id=run_id)
answer = completion.content
if completion.usage:
    print(completion.usage.total_tokens)
for tc in completion.tool_calls:
    plan_args = tc.arguments_json
```

Build helpers: `intergrax/llm_adapters/_shared/adapter_response_builders.py`.  
Call lifecycle: `LLMCallLifecycle` in `intergrax/llm_adapters/_shared/call_lifecycle.py`.

### Trace and replay bridge (M-LLM-R.7.2)

- `intergrax/runtime/replay/trace_replay_bridge.py` — `serialized_trace_events_to_replay_dtos`
- `intergrax/runtime/replay/llm_call_mapper.py` — `llm_call_info_from_adapter_response`

### Adaptive harness hook (M-LLM-R.7.5)

Optional `LLMCallSummary` on `SignalAssemblyInput.last_llm_call` → `HarnessOutcomeSignal.last_llm_*`.

### CI guards

| Script | Purpose |
|--------|---------|
| `scripts/maintenance/check_llm_adapter_typed_returns.py` | ABC public methods must not return bare `str` / dict |
| `scripts/maintenance/check_agents_llm_adapter_response.py` | Tier-2 agents must not annotate adapter returns as `str` |
| `scripts/maintenance/check_agents_vendor_imports.py` | Tier-2 agents must not import vendor LLM SDKs directly |

### M-LLM-R as-built conformance (audit dimensions)

Re-validate per [`audit/LLM_ADAPTERS.md`](../audit/LLM_ADAPTERS.md) §3:

| # | Dimension | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Typed envelope (`LLMAdapterResponse` / `LLMStructuredResult`) | **Yes** | §Response envelope; `check_llm_adapter_typed_returns.py` |
| 2 | Agents do not treat LLM returns as bare `str` | **Yes** | `check_agents_llm_adapter_response.py` |
| 3 | Vendor SDK only inside `providers/*` | **Yes** | `check_agents_vendor_imports.py`; tier boundary §Design principles |
| 4 | Refusal / content filter surfaced | **Yes** | `refusal` field + `LLMFinishReason.CONTENT_FILTER` / `REFUSAL` |
| 5 | Streaming parity with non-stream paths | **Partial** | `LLMStreamEvent` contract; provider-specific tool-call streaming gaps |
| 6 | `LLMProfile` drives model selection | **Yes** | §Model selection; `resolve_llm_adapter()` |
| 7 | Token/cost on envelope + run aggregation | **Yes** | §Token accounting; Prometheus + `LLMUsageTracker` |
| 8 | Retries, timeout, circuit breaker | **Yes** | `LLMCallConfig`; §Resilience |
| 9 | Structured output schema validation | **Yes** | `generate_structured` + Pydantic; AUDIT-IDEAL-6.1 |
| 10 | Guardrail middleware when profile set | **Yes** | `AFTER_LLM_OUTPUT`; INTEGRATIONS §47 |
| 11 | Tenant scope + hard quota | **Yes** | `llm_tenant_scope`; `INTERGRAX_LLM_TENANT_MAX_TOKENS` |
| 12 | Metrics export on task complete | **Yes** | `runtime.llm_metrics_export` plugin; `/metrics/llm` |
| 13 | Attachments respect `ModalityProfile.max_media_bytes` | **Yes** | §Modality attachments |
| 14 | Capability flags from catalog when `ModelRecord` known | **Yes** | `CatalogCapabilityAdapter` + registry wire · M-LLM-X.14.1 |
| 15 | Secrets via `llm/<provider>/api_key` | **Yes** | §Secrets; `create_adapter_from_secrets_store()` |
| 16 | Replay bridge maps trace → adapter calls | **Yes** | §Trace and replay bridge |

**Planner ≠ producer (COG-PROD):** **Done** — `ReasoningProfile.planner_llm_profile` → `resolve_planner_llm_adapter()` (not an LLM-AUDIT open gap).

---

## Provider selection

### Mechanism

| Step | Component | Behavior |
|------|-----------|----------|
| 1 | `LLMProvider` enum | Canonical slug (`openai`, `claude`, `openrouter`, …) — 19 built-ins |
| 2 | `LLMAdapterRegistry` | Lazy import of adapter class; `register()` for extensions |
| 3 | `LLMProfile(provider=..., model=..., options=...)` | Declarative Tier-3 selection |
| 4 | Env defaults | `INTERGRAX_LLM_PROVIDER`, `INTERGRAX_LLM_MODEL` via `llm_profile_from_env()` |
| 5 | Host resolver | `resolve_llm_adapter(env, agent_override)` — agent factory > env profile > platform env |

### OpenAI-compatible cluster

Ten slugs share `OpenAIChatCompletionsAdapter` via `create_openai_compat_adapter()`:

`groq`, `vllm`, `together`, `fireworks`, `openrouter`, `deepseek`, `xai`, `llama_cpp`, `cohere`, `azure_ai_inference`

Each declares `OpenAICompatProviderConfig` (API key env, base URL, default model, optional per-model context map).

### Extension without forking core

```python
from intergrax.llm_adapters import LLMAdapterRegistry

LLMAdapterRegistry.register("my_gateway", my_factory, override=False)
profile = LLMProfile(provider="my_gateway", model="vendor/model-id")
```

Custom provider slugs validate against `LLMAdapterRegistry.registered_providers()` — no enum edit required (**M-LLM-X.14.3**).

### Provider plugin layer (planned — LLM-PROVIDER-PLUGIN-1)

Today's `LLMAdapterRegistry.register()` factory hook is sufficient for runtime extension but is **not** a full provider plugin system (no deterministic metadata snapshot, config/health/security posture contract, or package discovery parity with runtime integrations registry v2).

**Planned (Backlog, P2):** add a thin **`LLMProviderRegistration` / metadata contract** layer above `LLMAdapter` that registers provider packages, exposes safe public metadata, and factories `LLMAdapter` instances. **`LLMAdapter` remains the execution contract**; `LLMProvider` enum stays for stable built-ins. See plan [`LLM-PROVIDER-PLUGIN-1`](../plan/LLM_ADAPTERS.md#phase-llm-provider-plugin--provider-plugin-registration-layer-backlog).

### When to use `openrouter`

`openrouter` is the **multi-vendor escape hatch**: one provider slug, arbitrary upstream model strings (`anthropic/claude-opus-4`, …). Context windows resolve via bundled **`ModelCatalog`**, optional **`fetch_gateway_metadata`** merge, or profile override. When no **exact** catalog entry matches, **`ModelCatalogMissDiagV1`** is recorded (including `provider_default` for unknown OpenRouter ids) on Plane A trace (`llm_catalog_miss`), runtime bus (`LLM_CALL`), and Prometheus (`intergrax_llm_catalog_miss_total` when metrics enabled).

---

## Model selection

### Rules

- **`LLMProfile.model`** is a **free string** — no platform model enum.
- New vendor models work **immediately** for API calls; **context budgeting** depends on catalog resolution (§Model catalog).
- Per-step hints (ACP): `StepLLMRouter.resolve_model(model_hint)` against host allowlist — target: backed by same catalog + `LLMAdapter` (M-LLM-X.5).
- Planner ≠ producer: `ReasoningProfile.planner_llm_profile` → `resolve_planner_llm_adapter()` in `nexus_factory.py`.

### Precedence (single Nexus run)

```text
1. RuntimeConfig.llm_adapter          — primary producer (one instance today)
2. resolve_planner_llm_adapter()      — optional separate planner adapter
3. CriticProfile / EvaluationProfile  — separate LLMProfile for judge paths (CRITIC_VERIFICATION)
4. ModelRouter + fallback_profiles    — target: runtime selection before adapter create (M-LLM-X.4)
```

---

## Model catalog and context window

### Problem (current)

Context limits are **scattered** in per-adapter dicts with inconsistent fallbacks:

| Adapter family | Unknown-model fallback | Risk |
|----------------|------------------------|------|
| OpenAI Responses | 128 000 | Optimistic |
| Claude, Gemini, Mistral, OpenAI-compat | **32 000** | **Under-budgeting** → aggressive history trim |
| Ollama | 8 192 | Under-budgeting unless override |
| Bedrock | 32 000 + **prefix heuristics** | Best current pattern |

Only **Ollama** accepts constructor `context_window_tokens=` today; other adapters ignore profile override.

### Target architecture — `ModelCatalog`

Central Tier-0 registry (Phase M-LLM-X.1):

```text
intergrax/llm_adapters/registry/model_catalog.py   — resolve API (Done)
intergrax/llm_adapters/registry/model_catalog.yaml — bundled defaults (Done)
intergrax/llm_adapters/registry/context_window.py  — resolve_context_window_tokens (Done)
```

#### `ModelRecord` (frozen)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `model_id` | `str` | yes | Exact or normalized id |
| `context_window_tokens` | `int` | yes | Input + output budget for Nexus |
| `supports_vision` | `bool` | no | Default false |
| `supports_tools` | `bool` | no | Default true for chat models |
| `supports_structured_output` | `bool` | no | |
| `provider_hints` | `tuple[str, ...]` | no | Slugs where this id is valid |
| `family_prefix` | `str \| None` | no | For prefix rules |

`CatalogCapabilityAdapter` overlays catalog capability flags on the concrete adapter returned by `LLMAdapterRegistry.create()` but does not erase provider-specific `model_capabilities` on the inner adapter. Consumers that need concrete capability state (for example Token Optimization router preflight) call `unwrap_catalog_capability_adapter()` for inspection only and keep using the outer wrapper for generation, tool calling, structured output, and usage accounting.

#### Resolution order (deterministic)

```text
1. LLMProfile.options["context_window_tokens"]     — operator override (authoritative)
2. ModelCatalog exact match on model_id
3. ModelCatalog prefix rules (family: claude-*, gpt-*, gemini-*, anthropic.*, meta-llama/*)
4. Provider adapter legacy dict (deprecated; shrink over time)
5. Provider family default from catalog (e.g. claude_default: 200_000)
6. Safe conservative default (documented per provider; emit diagnostic once)
```

All adapters call **`resolve_context_window_tokens(provider, model, profile_options)`** at construction — **one code path**.

#### OpenRouter / dynamic metadata (M-LLM-X.2)

Optional fetch from OpenRouter `/api/v1/models` (or compatible gateway) with TTL cache; merge into catalog for session. Fail closed to prefix rules + profile override.

#### Operator override (required for self-hosted)

```python
LLMProfile(
    provider=LLMProvider.VLLM,
    model="my-custom-70b",
    options={"context_window_tokens": 131_072},
)
```

Must propagate to **all** adapter constructors (not Ollama-only).

### Nexus consumers (unchanged contract, better input)

These read `adapter.context_window_tokens` — they **automatically benefit** from catalog accuracy:

- `resolve_input_budget_tokens()` — `context_budget.py`
- `verify_context_preflight()` — `context_preflight.py`
- `engine_history_layer` — history compression budget

**Context path rule:** Messages passed to `LLMAdapter.generate_messages` / `stream_messages` in production **SHOULD** originate from `ContextCompiler` / `ContextEngine` (or an explicitly approved equivalent) — not ad-hoc agent concatenation. See [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §12 Context Path Unification.

---

## Token accounting

### Per-call (source of truth)

- `LLMAdapterResponse.usage` → `LLMTokenUsage` (prefer SDK counts; `provider_extensions.usage_source` when estimated)
- `LLMAdapter.count_messages_tokens()` → tiktoken with `model_name_for_token_estimation` hint

### Run / tenant aggregation

| Layer | Type | Owner |
|-------|------|-------|
| Adapter | `LLMAdapterUsageLog` | Per adapter instance |
| Runtime | `LLMUsageTracker` on `RuntimeState` | Nexus finalize |
| Metrics | Prometheus counters | `tracking/metrics.py` |
| Quota | `check_llm_tenant_quota` | `governance/quota.py` |

### Target consistency (M-LLM-X.3)

| Path | Current | Target |
|------|---------|--------|
| `verify_context_preflight` | **`adapter.count_messages_tokens`** (default) | Maintain; optional custom counter |
| `ContextBudgetPolicy` defaults | **`from_adapter()`** factory available | Nexus compile paths adopt factory (X.3.3 rollout) |
| Billing / SLO | SDK usage on envelope | Unchanged |

**Rule:** Budgeting and preflight MUST use the same tokenizer path as the adapter when an `LLMAdapter` is in scope.

---

## `LLMProfile` and Tier-3 wiring

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

profile = LLMProfile(
    provider=LLMProvider.GROQ,
    model="llama-3.3-70b-versatile",
    options={
        "max_retries": 2,
        "context_window_tokens": 128_000,  # target M-LLM-X.1 — universal override
    },
)
llm = profile.create_adapter(secrets={"api_key": key})
```

### Target fields (M-LLM-X.4)

| Field | Purpose |
|-------|---------|
| `provider` | `LLMProvider` or string slug (post X.6) |
| `model` | Vendor model id |
| `options` | Passed to adapter ctor + `LLMCallConfig` |
| `fallback_profiles` | Ordered list for failover chain (target) |
| `routing_policy_hint` | `balanced` \| `cheapest` \| `fastest` \| `quality` (target) |

### Secrets

- Env: per-provider keys (`OPENAI_API_KEY`, …) — see §Providers
- Vault: `llm/<provider>/api_key` via `SecretsStore` — `create_adapter_from_secrets_store()`

---

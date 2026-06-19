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

## Purpose and maturity

Tier-0 **LLM adapter layer** is the Harness cognition entry point: one `LLMAdapter` contract, multi-vendor providers, typed completion envelopes, tenant metering, and Nexus context budgeting.

| Dimension | Current (post X-13) | Enterprise target (X-8 + X-14) | Evidence |
|-----------|---------------------|----------------------------------|----------|
| Typed completion envelope | **L3** | L3 (maintain) | M-LLM-R closed; CI guards |
| Provider abstraction (19 slugs) | **L3** | **L3+** (enum-free plugin) | `LLMAdapterRegistry`; X-6.1 → **M-LLM-X.14.3** |
| Model ID as free string | **L3** | L3 (maintain) | `LLMProfile.model: str` |
| Model metadata / context window | **L3** | **L3+** (live gateway merge) | `ModelCatalog` **Done**; dynamic fetch → **M-LLM-X.14.2** |
| Multi-model routing / failover | **L5** (strict core + UAEP + ACP) | L5 (maintain) | X-10…X-13 **Done** · LLM-AUDIT-17…20 **Done** |
| Secondary LLM surfaces (planner, websearch, critic) | **L4** (snapshot sync) | **L4+** (opt-in evaluating wrap) | X-13.4–13.6 **Done**; full mid-run swap → **M-LLM-X.14.5** |
| Token accounting consistency | **L3** | **L3+** (vendor tokenizer opt-in) | LC-2 **Done**; tiktoken caveat → **M-LLM-X.14.7** |
| Developer experience | **L3** | **L3+** | USAGE + doctor **Done**; scaffold → **M-LLM-X.14.8** |
| Observability & governance | **L3** | L3 (maintain) | Prometheus, quota, replay bridge |
| **Domain closeout** (audit register + journal) | **L4 partial** | **L4 enterprise** | **M-LLM-X.8** mandatory · **LLM-AUDIT-21** |

**Maturity labels (honest, 2026-06-19):**

- **Routing hot path:** **L5 strict enterprise-ready** (core `llm_adapter`, UAEP, ACP dynamic router).
- **Whole LLM_ADAPTERS domain:** **L4+** — routing and LC baseline are production-grade; **enterprise domain closeout** requires **M-LLM-X.8** then **M-LLM-X.14**.

**Strategic rule:** The Harness owns provider plumbing; agents and applications declare **profiles**, never vendor SDKs.

Deep production audit (2026-06-14): foundation is **production-grade L3** on contract and ops. **Full Harness LC (2026-06-17):** no open P0/P1 on LC scope. **Post X-13:** open enterprise gaps tracked as **LLM-AUDIT-21…26** → [Wave M-LLM-X-14](../plan/LLM_ADAPTERS.md#phase-m-llm-x-14--enterprise-domain-maturity-2026-06-19).

---

## Design principles

1. **One contract** — all completions return `LLMAdapterResponse` (or stream/structured variants); no bare `str`.
2. **Provider slug + adapter class** — commercial and self-hosted vendors map through `LLMAdapterRegistry`; OpenAI-compatible HTTP endpoints share `openai_compat_factory.py`.
3. **Model ID is opaque string** — vendors may ship new models (`claude-opus-4`, `gpt-5.2`, `fable`, …) without platform enum changes; **metadata** resolves separately via `ModelCatalog`.
4. **Context window is authoritative for budgeting** — Nexus context engine, preflight, and history compression read `adapter.context_window_tokens`; wrong values cause silent quality loss or API errors.
5. **Two usage layers preserved** — per-call `LLMTokenUsage` on the envelope; run-level `LLMAdapterUsageLog` + `LLMUsageTracker` (do not merge without explicit bridge).
6. **Tier boundaries** — `intergrax/llm_adapters/` MUST NOT import from `agents/` or `applications/`; Tier-2 uses injected `LLMAdapter` or `StepLLMRouter` port only.

---

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
| `scripts/check_llm_adapter_typed_returns.py` | ABC public methods must not return bare `str` / dict |
| `scripts/check_agents_llm_adapter_response.py` | Tier-2 agents must not annotate adapter returns as `str` |
| `scripts/check_agents_vendor_imports.py` | Tier-2 agents must not import vendor LLM SDKs directly |

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
| 14 | Capability flags default false (W-ML.1) | **Partial** | Per-provider overrides; catalog-driven flags: M-LLM-X.1.7 |
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
profile = LLMProfile(provider="my_gateway", model="vendor/model-id")  # str coercion target: M-LLM-X.6
```

Built-in enum extension still requires a harness PR for `_BUILTIN_ADAPTERS` + `LLMProvider` — **plugin entry point** (M-LLM-X.6) will allow string slugs without enum edits for external gateways.

### When to use `openrouter`

`openrouter` is the **multi-vendor escape hatch**: one provider slug, arbitrary upstream model strings (`anthropic/claude-opus-4`, …). Requires **ModelCatalog** or profile override for correct context windows (today: 32k fallback — incorrect for most models).

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

## Model routing and failover

### Current state

- `ModelRouter` + `FailoverLLMAdapter` wired via `resolve_llm_adapter()` and `LLMProfile.create_adapter_with_failover()` (**Done** LC-3).
- `resolve_live_model_routing_wiring()` applies routing hints on product hosts; decision drives adapter creation (**Done** LC-3).
- `LLMCallConfig` — retry, circuit breaker, rate limit; cross-provider failover via profile chain (**Done**).

### Target state (M-LLM-X.4 / X.5)

```text
LLMProfileChain
  primary: LLMProfile
  fallbacks: tuple[LLMProfile, ...]
  policy: RoutingPolicyHint

LLMAdapterFactory.create_with_failover(chain) → LLMAdapter
  on retriable error (429, 502, 503, timeout): next profile in chain
  emit LLMRoutingDiagV1 on trace (provider, model, reason, attempt)
```

AHI `RoutingTuningEngine` **recommends** profile order; policy engine **approves**; runtime executes chain — aligns IDEAL §3.5 model selection by cost/latency/quality.

**Explicit non-goal:** Central LLM gateway microservice (§5.2.4) — separate ADR if pursued; M-LLM-X stays in-process Tier-0.

### LLM routing rules (M-LLM-X.9 — Done)

**ADR:** [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md)

Authors configure dynamic model selection through a **Protocol contract** — not a rigid enum DSL. Built-in parametric rules and **custom classes** from Tier-3 share the same interface.

#### Contracts (target location: `intergrax/llm_adapters/routing/`)

| Type | Role |
|------|------|
| `RoutingContext` | Immutable snapshot: `task_class`, `budget_remaining_ratio`, `tokens_used`, `step_index`, `model_hint`, `tenant_id`, `agent_id`, … |
| `RoutingTarget` | Rule output: `LLMProfile \| None`, `RoutingHint \| None`, `reason: str` (for trace) |
| `LLMRoutingRule` | **Protocol** — `rule_id`, `priority`, `matches(ctx)`, `resolve(ctx)` |
| `LLMRoutingRuleBase` | Optional ABC with helper methods (`budget_below`, `task_is`, …) |
| `LLMRoutingProfile` | Tier-3: `default_profile`, `allowed_profiles`, `rules: tuple[LLMRoutingRule, ...]` |
| `LLMRoutingEvaluator` | Sort by priority → first `matches()` → validate allowlist → `ModelRouter` |

#### Evaluation flow

```text
RoutingContext (snapshot from Nexus / budget meter / classifier)
    → sort rules by priority (descending)
    → first rule where matches(ctx) is True
    → target = rule.resolve(ctx)
    → guard: target.profile ∈ allowed_profiles (reject + trace on violation)
    → ModelRouter + FailoverLLMAdapter
    → trace: rule_id, reason, profile_id (`LLMRoutingRuleDiagV1` + failover `LLMRoutingAttemptDiagV1`)
```

#### Built-in rules (same Protocol)

**Done (X-9 + X-10):** full parametric **platform catalog** (Tier-0) — constructor params replace former enum DSL. Tier-3 authors may always add **custom** `LLMRoutingRule` subclasses; see custom example below and [plan M-LLM-X.10](../plan/LLM_ADAPTERS.md#wave-m-llm-x-10--llm-routing-enterprise-closeout--predefined-rule-catalog).

| Class | Status | Covers |
|-------|--------|--------|
| `BudgetBelowRule(threshold, profile\|hint)` | **Done** | `budget_remaining_ratio < threshold` |
| `BudgetAboveRule(threshold, profile\|hint)` | **Done** | `budget_remaining_ratio > threshold` |
| `BudgetExceededDegradeRule()` | **Done** | `budget_degrade_active` → `CHEAPEST` |
| `TaskClassInRule(classes, profile\|hint)` | **Done** (alias `TaskClassRule`) | `task_class in classes` |
| `TaskClassNotInRule(classes, …)` | **Done** | negated task class |
| `TokenUsedAboveRule(threshold, hint)` | **Done** (alias `TokenThresholdRule`) | `tokens_used > threshold` |
| `TokenUsedBelowRule(threshold, …)` | **Done** | low token usage |
| `StepIndexAtLeastRule` / `StepIndexBelowRule` | **Done** | per-step routing |
| `AgentIdInRule` / `TenantIdInRule` | **Done** | identity-based routing |
| `ModelHintPresentRule` | **Done** | honour agent `model_hint` |
| `PolicyHintRule(hint)` | **Done** | force `RoutingHint` |
| `CompositeAllRule` / `CompositeAnyRule` | **Done** | AND / OR composition |
| `AlwaysRule(profile\|hint)` | **Done** | explicit catch-all fallback |

#### Enterprise routing (M-LLM-X.10 — Done · scope)

**Delivered (2026-06-19):** predefined catalog, Protocol + custom rules, auto context on **materialize** path, reference lab host, CI gate, ACP `DynamicLLMRouter`, initial `LLMRoutingRuleDiagV1`.

| Capability | Task ID | Status | Scope note |
|------------|---------|--------|------------|
| Predefined catalog (Tier-0) + custom rules (Tier-3) | M-LLM-X.10.1 | **Done** | Builtin + `LLMRoutingRule` Protocol |
| Auto `RoutingContext` on materialize / ACP start | M-LLM-X.10.2 | **Done** | `runtime_config_bridge`, `acp_run` — not all call sites |
| Trace `rule_id` + `routing_reason` | M-LLM-X.10.3 | **Done** | Start-of-run; per-eval on UAEP/Nexus via X-11.4/X-12.7; ACP Plane A parity via **X-13.2** |
| Reference lab host (predefined demo; products may use custom) | M-LLM-X.10.4 | **Done** | CI gate lab-only |
| Acceptance: budget rule switches profile | M-LLM-X.10.5 | **Done** | Resolver + materialize — not full Nexus run |
| `DynamicLLMRouter` on ACP hosts | M-LLM-X.10.6 | **Done** | ACP only |
| USAGE + architecture checklist | M-LLM-X.10.7 | **Done** | |
| CI `check_llm_routing_rules.py` | M-LLM-X.10.8 | **Done** | |
| AHI persistent bandit + ProfileVersion read | AHI-MAINT-06 | **Done** | Hint path; no full canary apply |

**Maturity label:** **L4+ enterprise-ready** — start-of-run + ACP + UAEP mid-run evaluating adapter (M-LLM-X.11). **Strict L5** delivered in **M-LLM-X.12** · LLM-AUDIT-19 **Done**.

#### Enterprise routing hardening (M-LLM-X.11 — Done)

**Delivered (2026-06-19):** live re-eval on Nexus `llm_adapter`, `refresh_llm_routing_context()` in UAEP step loop, unified Tier-3 resolver call sites, per-evaluation trace + allowlist violation diag, mid-run acceptance (mocked adapter factory), harness host parity, CI `check_llm_routing_context_wiring.py`.

| Capability | Task ID | Status |
|------------|---------|--------|
| `RoutingEvaluatingLLMAdapter` — re-eval before each LLM call | M-LLM-X.11.1 | **Done** |
| `refresh_llm_routing_context()` + `llm_routing_snapshot` on `RuntimeConfig` | M-LLM-X.11.2 | **Done** (UAEP step boundary) |
| `resolve_environment_llm_adapter()` on all Tier-3 wiring modules | M-LLM-X.11.3 | **Done** |
| Per-eval `LLMRoutingRuleDiagV1` + allowlist violation diag | M-LLM-X.11.4 | **Done** (UAEP/Nexus state; ACP partial) |
| Mid-run budget threshold → profile swap (acceptance) | M-LLM-X.11.5 | **Done** (evaluating adapter; mocked factory) |
| Harness host + materialize evaluating adapter parity | M-LLM-X.11.6 | **Done** |
| USAGE mid-run section | M-LLM-X.11.7 | **Done** |
| CI `check_llm_routing_context_wiring.py` | M-LLM-X.11.8 | **Done** |

**Closes:** LLM-AUDIT-18 (declared X-11 scope). **Does not claim strict L5** — see post-audit register **LLM-AUDIT-19**.

#### Routing strict enterprise closeout (M-LLM-X.12 — Done)

**Delivered (2026-06-19):** Tier-3 `RoutingEvaluatingLLMAdapter` with injected factory; Tier-0 `metering.py` + `runtime_sync.py`; Nexus graph/CE sync hooks; per-run observability; ACP routing diagnostics; production metering E2E; CI `check_llm_routing_tier_boundary.py`.

| Deliverable | Task | Status |
|-------------|------|--------|
| Inner usage metering + tracker re-register on swap | M-LLM-X.12.1 | **Done** |
| Tier-clean evaluating wrapper (Tier-3) | M-LLM-X.12.2 | **Done** |
| Graph + CE routing snapshot sync | M-LLM-X.12.3 | **Done** |
| Per-call context refresh via provider | M-LLM-X.12.4 | **Done** |
| Live `RoutingContext` on adapter swap / AHI | M-LLM-X.12.5 | **Done** |
| `budget_degrade_active` in Nexus sync | M-LLM-X.12.6 | **Done** |
| Instance-bound observers (no globals) | M-LLM-X.12.7 | **Done** |
| ACP `DynamicLLMRouter` routing diagnostics | M-LLM-X.12.8 | **Done** |
| First-eval profile correction | M-LLM-X.12.9 | **Done** |
| Production metering E2E | M-LLM-X.12.10 | **Done** |
| Docs + audit register | M-LLM-X.12.11 | **Done** |
| Secondary LLM surfaces policy (documented) | M-LLM-X.12.12 | **Done** |

**Maturity label:** **L5 strict enterprise-ready** for declarative routing on core + UAEP + ACP paths.

**Secondary LLM policy (M-LLM-X.12.12 · X-13.4–13.6):** Tool planner, websearch map/reduce/rerank, and critic paths receive **routing snapshot sync** or explicit **critic routing metadata** when `llm_routing_profile` is enabled. They are not auto-wrapped in `RoutingEvaluatingLLMAdapter`; hosts may still opt into dedicated evaluating wraps per surface.

**Closes:** **LLM-AUDIT-19** (X-12), **LLM-AUDIT-20** (X-13).

#### Post-L5 follow-up register (M-LLM-X.13 — Done)

**Source:** Post X-12 enterprise audit (2026-06-19). All gaps below closed in **M-LLM-X.13** (2026-06-19).

| Gap (post-audit) | Severity | Task ID | Status |
|------------------|----------|---------|--------|
| `runtime_state.py` imports Tier-3 `RoutingEvaluatingLLMAdapter` for `isinstance` wiring | **P2** | M-LLM-X.13.1 | **Done** — `evaluating_hooks.py` Protocol |
| ACP records routing in `step.diagnostics` only — no Plane A `llm_routing_rule` trace step | **P2** | M-LLM-X.13.2 | **Done** — `acp_routing_trace_bridge.py` |
| No dedicated concurrent-run isolation test for per-run observers | **P2** | M-LLM-X.13.3 | **Done** |
| `tool_planning_service` LLM bypasses evaluating wrap | **P3** | M-LLM-X.13.4 | **Done** — snapshot sync |
| Websearch map/reduce/rerank LLM bypass evaluating wrap | **P3** | M-LLM-X.13.5 | **Done** — snapshot sync |
| Critic evaluator LLM bypass evaluating wrap | **P3** | M-LLM-X.13.6 | **Done** — routing policy metadata |
| `nexus_plan_bridge` / `llm_task_classifier` skip routing snapshot sync | **P2** | M-LLM-X.13.7 | **Done** |

**Plan:** [Wave M-LLM-X-13](../plan/LLM_ADAPTERS.md#phase-m-llm-x-13--post-l5-routing-polish-2026-06-19)

#### Enterprise domain maturity register (M-LLM-X-14 — Planned)

**Source:** Post X-13 maturity assessment (2026-06-19). Routing **L5** does **not** imply whole-domain enterprise closeout.

| Gap | Severity | Task ID | Audit ID |
|-----|----------|---------|----------|
| Domain audit register + journal not formally closed | **P1** | M-LLM-X.8.1–8.3 | **LLM-AUDIT-21** |
| Capability flags not catalog-driven (`supports_vision`, tools, structured) | **P2** | M-LLM-X.14.1 | **LLM-AUDIT-22** |
| OpenRouter / gateway live metadata not merged into catalog session | **P1** | M-LLM-X.14.2 | **LLM-AUDIT-23** |
| ACP `make_acp_routing_context_provider` budget token bridge incomplete | **P2** | M-LLM-X.14.4 | **LLM-AUDIT-24** |
| Secondary LLM surfaces: sync only — no opt-in evaluating wrap | **P2** | M-LLM-X.14.5 | **LLM-AUDIT-25** |
| Plugin `LLMProfile.provider` still enum-coupled for extension authors | **P2** | M-LLM-X.14.3 | **LLM-AUDIT-26** |
| Production multi-step routing soak (budget burn, no factory mocks) | **P2** | M-LLM-X.14.6 | — |
| Vendor-native tokenizer plugins (non-OpenAI budget accuracy) | **P3** | M-LLM-X.14.7 | — |
| Scaffold DX — agent template points to USAGE + catalog | **P3** | M-LLM-X.14.8 | — |

**Suggested wave order:** **X-8** (closeout) → **14.2** → **14.1** → **14.4** → **14.3** → **14.5** → **14.6** → **14.7** → **14.8**.

**Enterprise-grade domain DoD:** X-8 **Done** + X-14 **Done** + all **LLM-AUDIT-21…26** → **Done** + `tests/unit/llm_adapters/` + LLM CI gates green.

#### Routing strict enterprise closeout — audit register (historical gaps, closed)

| Gap (audit) | Severity | Task ID |
|-------------|----------|---------|
| `LLMUsageTracker` reads wrapper; inner adapter accumulates tokens → `BudgetBelowRule` may not fire mid-run | **P0** | M-LLM-X.12.1 |
| `evaluating_adapter.py` imports `applications/_shared/llm_resolver` — Tier-0 → Tier-3 violation | **P1** | M-LLM-X.12.2 |
| `sync_llm_routing_snapshot` only in UAEP — Nexus graph / CE paths skip refresh | **P1** | M-LLM-X.12.3 |
| Context stale between multiple LLM calls within one step | **P1** | M-LLM-X.12.4 |
| `create_adapter_for_routing_evaluation` passes empty `RoutingContext()` to AHI wiring | **P1** | M-LLM-X.12.5 |
| `budget_degrade_active` not mapped in Nexus sync | **P1** | M-LLM-X.12.6 |
| Global `set_routing_evaluation_observer` — concurrent run risk | **P2** | M-LLM-X.12.7 |
| ACP `DynamicLLMRouter` without `on_evaluated` trace in `acp_run` | **P2** | M-LLM-X.12.8 |
| First eval trusts resolver profile even when rules disagree | **P2** | M-LLM-X.12.9 |
| E2E is mock-based — no production meter + trace proof | **P2** | M-LLM-X.12.10 |
| Docs claimed L5 prematurely | **P2** | M-LLM-X.12.11 |
| Planner / critic / websearch LLM bypass evaluating wrapper | **P3** | M-LLM-X.12.12 |

**Strict L5 criteria (checklist — all must pass before L5 label):**

1. `budget_remaining_ratio` in `RoutingContext` reflects **actual** run token usage on core adapter path.
2. Context sync runs on **UAEP + Nexus graph + context-engine** paths before routing eval.
3. No Tier-0 import from `applications/` for routing hot path.
4. Per-eval trace on **ACP and Nexus** with correlated `run_id` (no process-global observers).
5. Acceptance test: budget threshold → model swap **without** mocking `create_adapter_for_routing_evaluation`.
6. Audit register **LLM-AUDIT-19** → **Done**.

#### Custom rule example (Tier-3)

```python
class LowBudgetForceLocalRule(LLMRoutingRuleBase):
    rule_id = "my_app.low_budget_vllm"
    priority = 10

    def matches(self, context: RoutingContext) -> bool:
        return (
            context.budget_remaining_ratio is not None
            and context.budget_remaining_ratio < 0.2
            and context.task_class in {"contract_review", "due_diligence"}
        )

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(
            profile=LLMProfile(provider="vllm", model="meta-llama/Llama-3.1-8B"),
            reason=f"budget_low for {context.task_class}",
        )
```

Wired in manifest: `LLMRoutingProfile(rules=(LowBudgetForceLocalRule(), BudgetBelowRule(...), ...))`.

#### Three-layer model

```text
Layer 1 — Author rules (LLMRoutingProfile on Tier-3)     → explicit logic; always wins over L4
Layer 2 — LLMRoutingEvaluator + ModelRouter (Tier-0)    → hot path (SYS-INV-10 single router)
Layer 3 — AHI ROUTING_TUNING (AdaptiveProfile, optional) → bandit proposes ProfileVersion;
                                                           does not execute arbitrary author code
```

#### Provider coverage

| Model source | Routing path |
|--------------|--------------|
| Cloud API (OpenAI, Claude, Groq, …) | `LLMProfile(provider=…)` |
| Local (Ollama, vLLM, llama.cpp) | Existing `LLMProvider` slugs |
| HF self-hosted weights | Model id on `vllm` / `llama_cpp` profile |
| HF Inference API (chat) | Optional provider plugin (M-LLM-X.9.8) or gateway via `openrouter` |

**Anti-patterns:** string `eval` for rules; provider selection inside Tier-2 agents; routing target outside `allowed_profiles`; parallel router subsystem.

---

## Developer surfaces

### Nexus path (primary)

Inject `LLMAdapter` via `RuntimeConfig`; call `generate_messages` / tools / stream; read `.content` and `.usage`.

### ACP path (convergence target)

Today: `StepLLMRouter` in `agents/authoring/llm_router.py` — separate stub port when `llm_port` is None.

**Target (M-LLM-X.5):** `StepLLMRouter` delegates to `LLMAdapter.generate_messages` via `LLMAdapterCompletePort` — **Done** (LC-3).

### Documentation

| Artifact | Status |
|----------|--------|
| `docs/architecture/LLM_ADAPTERS.md` | This file |
| `docs/plan/LLM_ADAPTERS.md` | Phase M-LLM-X register |
| `intergrax/llm_adapters/USAGE.md` | **Done** — quickstart, env matrix, overrides, failover, extension |
| `docs/guides/AGENT_CREATION_GUIDE.md` § LLM | Cross-link only |

### Startup validation (target M-LLM-X.7)

`LLMProfile.validate_runtime()` — optional lightweight check: catalog hit, context window > 0, API key present, optional `adapter.validate()` ping.

---

## Modality plane A — generative multimodal (LLM)

LLM adapters own **Plane A** ([`MODALITY.md`](MODALITY.md) §7.1.9). Plane C (YOLO, ONNX, …) stays in `model_inference/`.

| Concern | Owner |
|---------|-------|
| Chat reasoning | `llm_adapters/` |
| Native vision/audio in dialog | `llm_adapters/` — capability flags (W-ML.1) |
| Deterministic CV / TTS tools | `model_inference/` + `speech_adapters/` |

### Capability flags

| Method | Meaning |
|--------|---------|
| `supports_vision()` | Image (optional video frame) input |
| `supports_audio_input()` | Audio in chat |
| `supports_audio_output()` | Spoken response |

Defaults **false** until mapping + conformance tests pass. **Target:** flags populated from `ModelCatalog` when known.

### Attachments

`intergrax/llm/messages.py` — `AttachmentRef`. Adapters map to vendor parts; `ModalityProfile.max_media_bytes` caps volume.

---

## Providers (19)

OpenAI-compatible slugs share `openai_compat_factory.py`. ABC defaults: streaming **false**, structured **false** unless overridden.

| Slug | Adapter module | Primary env | Stream | Structured | Notes |
|------|----------------|-------------|--------|------------|-------|
| `openai` | `openai_responses_adapter` | `OPENAI_API_KEY` | yes | yes | Native Responses API |
| `gemini` | `gemini_adapter` | `GOOGLE_API_KEY` | yes | yes | |
| `ollama` | `ollama_adapter` | `OLLAMA_BASE_URL` | yes | partial | Local; context override today |
| `mistral` | `mistral_adapter` | `MISTRAL_API_KEY` | yes | yes | |
| `claude` | `claude_adapter` | `ANTHROPIC_API_KEY` | yes | yes | |
| `azure_openai` | `azure_openai_adapter` | `AZURE_OPENAI_*` | yes | yes | |
| `aws_bedrock` | `aws_bedrock_adapter` | `AWS_*` | yes | partial | Prefix context heuristics |
| `groq` | `openai_compat` | `GROQ_API_KEY` | compat | compat | |
| `vllm` | `openai_compat` | `INTERGRAX_DEFAULT_VLLM_BASE_URL` | compat | compat | Self-hosted; Intergrax Docker host **8100** → container 8000 |
| `together` | `openai_compat` | `TOGETHER_API_KEY` | compat | compat | |
| `fireworks` | `openai_compat` | `FIREWORKS_API_KEY` | compat | compat | |
| `openrouter` | `openai_compat` | `OPENROUTER_API_KEY` | compat | compat | Multi-vendor model strings |
| `deepseek` | `openai_compat` | `DEEPSEEK_API_KEY` | compat | compat | |
| `xai` | `openai_compat` | `XAI_API_KEY` | compat | compat | |
| `llama_cpp` | `openai_compat` | `INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL` | compat | compat | Self-hosted CPU-friendly; Intergrax Docker host **8102** |
| `cohere` | `openai_compat` | `COHERE_API_KEY` | compat | compat | Chat Completions shim |
| `cohere_native` | `cohere_native_adapter` | `COHERE_API_KEY` | yes | partial | Prefer when native tools needed |
| `vertex_gemini` | `vertex_gemini_adapter` | `GOOGLE_APPLICATION_CREDENTIALS` | yes | yes | |
| `azure_ai_inference` | `azure_ai_inference_adapter` | `AZURE_AI_*` | yes | partial | |

Per-provider model env vars: `INTERGRAX_DEFAULT_<PROVIDER>_MODEL` (see [`USAGE.md`](../../intergrax/llm_adapters/USAGE.md)).

### Self-hosted inference (Ollama vs vLLM vs llama.cpp)

| Concern | Ollama | vLLM | llama.cpp |
|---------|--------|------|-----------|
| Adapter module | `ollama_adapter.py` (LangChain) | `openai_compat_providers.VllmChatAdapter` | `openai_compat_providers.LlamaCppChatAdapter` |
| API shape | Ollama native HTTP | OpenAI Chat Completions (`/v1`) | OpenAI Chat Completions (`/v1`) |
| Tier-0 slug | `LLMProvider.OLLAMA` | `LLMProvider.VLLM` | `LLMProvider.LLAMA_CPP` |
| Local Docker | `infra/docker/ollama` · profile `rag` · port **11434** | `infra/docker/vllm` · profile **`vllm`** (opt-in) · host **8100** | `infra/docker/llama-cpp` · profile **`llama-cpp`** (opt-in) · host **8102** |
| GPU | Optional (CPU OK for dev) | **NVIDIA GPU required** for practical use | **CPU-first** (optional CUDA in compose) |
| P5 integration | `interaction_surface/ollama` (health probe) | Not registered — adapter + Docker health (`/v1/models`) | Not registered — same as vLLM |

**Do not** add a LangChain-style duplicate adapter for vLLM or llama.cpp — OpenAI-compat factory is the canonical path (M-LLM.3, M-LLM.7).

**Intergrax Docker wiring (vLLM):**

```bash
cd infra/integration && ./manage.sh start vllm
export INTERGRAX_DEFAULT_VLLM_BASE_URL=http://127.0.0.1:8100/v1
export INTERGRAX_LLM_PROVIDER=vllm
```

**Intergrax Docker wiring (llama.cpp):**

```bash
cd infra/integration && ./manage.sh start llama-cpp
export INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL=http://127.0.0.1:8102/v1
export INTERGRAX_LLM_PROVIDER=llama_cpp
export INTERGRAX_LLM_MODEL=default
```

Port **8100** (vLLM) and **8102** (llama.cpp) avoid conflict with Chroma (**8000**) and Weaviate (**8080**) — see [`infra/PORTS.md`](../../infra/PORTS.md).

**Live smoke (vLLM only):** `test_vllm_live_one_shot` in `tests/unit/llm_adapters/test_network_smoke.py` (marker `network`; weekly GitHub workflow).

**llama.cpp verification (local only, not GitHub CI):** [`infra/docker/llama-cpp/VERIFY_RUNBOOK.md`](../../infra/docker/llama-cpp/VERIFY_RUNBOOK.md) · `tests/e2e/llama_cpp/` (`e2e`, `no_ci`, `network`).

---

## Nexus runtime (automatic)

| Feature | Mechanism |
|---------|-----------|
| Tenant scope | `UnifiedTaskRunner` → `llm_tenant_scope` |
| Task-complete export | `bootstrap_nexus_platform()` → plugin `runtime.llm_metrics_export` |
| Hard quota | `INTERGRAX_LLM_TENANT_MAX_TOKENS` |
| Soft governance warn | `INTERGRAX_LLM_GOVERNANCE_WARN_TOKENS` |
| Pushgateway | `INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL` |
| Distributed rate limit | `set_llm_distributed_rate_limiter` + `use_distributed_rate_limit` |
| Context preflight | `verify_context_preflight()` — default **`adapter.count_messages_tokens`** |

---

## Observability (Prometheus & governance)

Tier-0 metrics: `intergrax/llm_adapters/tracking/`.

### Scrape (recommended)

```python
from intergrax.llm_adapters.tracking.exposition import register_llm_metrics_routes

register_llm_metrics_routes(app)  # GET /metrics/llm
```

### Example PromQL

```promql
sum by (tenant_id) (rate(intergrax_llm_calls_total[5m]))
sum by (provider) (rate(intergrax_llm_errors_total[5m]))
  / sum by (provider) (rate(intergrax_llm_calls_total[5m]))
sum by (tenant_id, model) (rate(intergrax_llm_output_tokens_total[5m]))
```

### Usage tracking: two layers

| Layer | Type | When |
|-------|------|------|
| **Adapter** | `LLMAdapterUsageLog` | Per SDK call |
| **Runtime** | `LLMUsageTracker` | Nexus run finalize |

Do not merge counters without explicit bridge code.

---

## Resilience & secrets

- **`LLMCallConfig`:** retries, timeout, in-process rate limit, circuit breaker, optional Redis distributed limit.
- **Failover (target):** profile chain — M-LLM-X.4.
- **Secrets:** `registry/secrets.py` — env + `llm/<provider>/api_key`.

---

## Environment appendix

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_LLM_PROVIDER` | Default provider slug for `llm_profile_from_env()` |
| `INTERGRAX_LLM_MODEL` | Default model id |
| `INTERGRAX_LLM_METRICS_ENABLED` | Enable metrics plugin + `/metrics/llm` |
| `INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL` | Optional push on `TASK_COMPLETED` |
| `INTERGRAX_LLM_TENANT_MAX_TOKENS` | Hard per-tenant quota |
| `INTERGRAX_LLM_GOVERNANCE_WARN_TOKENS` | Soft warn on task complete |
| `INTERGRAX_BEDROCK_USE_CONVERSE` | Bedrock Converse API toggle |
| `INTERGRAX_LLM_MODEL_CATALOG_PATH` | **Target** M-LLM-X.1 — optional override YAML |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … | Per-provider secrets |

---

## Audit register (2026-06-14)

### AUDIT-IDEAL §6 (LLM layer — master register cross-ref)

| ID | Gap | Priority | Status | Phase / owner |
|----|-----|----------|--------|---------------|
| AUDIT-IDEAL-6.1 | Structured output validation on reference + certified paths | P1 | **Done** | M-LLM-R — §Structured output |
| AUDIT-IDEAL-6.2 | Live cost/latency/quality routing (AHI prod path) | P2 | **Done** | M-LLM-X.5 — LC-3 hot-path wiring |
| AUDIT-IDEAL-6.3 | Central `ModelCatalog` + context window resolution | P0 | **Done** | M-LLM-X.1 — LC-1 |
| AUDIT-IDEAL-6.4 | Tokenizer-consistent context preflight | P0 | **Done** | M-LLM-X.3 — LC-2/LC-2b |
| AUDIT-IDEAL-6.5 | Profile failover chain | P1 | **Done** | M-LLM-X.4 — LC-3 |
| AUDIT-IDEAL-6.6 | ACP `StepLLMRouter` backed by `LLMAdapter` | P1 | **Done** | M-LLM-X.5 — LC-3 |
| AUDIT-IDEAL-6.7 | Developer `USAGE.md` + startup validation | P2 | **Done** | M-LLM-X.7 + LLM-MAINT-01 — `validate_runtime()` + `intergrax doctor` LLM checks |

### Production audit gaps (LLM-AUDIT-*)

| ID | Gap | Severity | Phase | Status |
|----|-----|----------|-------|--------|
| LLM-AUDIT-1 | No central `ModelCatalog`; per-adapter context dicts stale | **P0** | M-LLM-X.1 | **Done** |
| LLM-AUDIT-2 | `context_window_tokens` override only on Ollama | **P0** | M-LLM-X.1 | **Done** |
| LLM-AUDIT-3 | Preflight token estimate uses chars/4 not adapter tokenizer | **P0** | M-LLM-X.3.1–3.4 | **Done** |
| LLM-AUDIT-4 | `ModelRouter` not on Nexus hot path | **P1** | M-LLM-X.4–5 | **Done** |
| LLM-AUDIT-5 | No provider failover chain | **P1** | M-LLM-X.4 | **Done** |
| LLM-AUDIT-6 | ACP `StepLLMRouter` disconnected from `LLMAdapter` | **P1** | M-LLM-X.5 | **Done** |
| LLM-AUDIT-7 | OpenRouter / gateway models default 32k context | **P1** | M-LLM-X.2 | **Done** — catalog `provider_defaults.openrouter: 128000`; dynamic fetch → backlog |
| LLM-AUDIT-8 | No `intergrax/llm_adapters/USAGE.md` | **P2** | M-LLM-X.7 | **Done** — USAGE + doctor hook (LLM-MAINT-01) |
| LLM-AUDIT-9 | AUDIT-IDEAL-6.2 wiring ceremonial — no runtime swap | **P1** | M-LLM-X.5 | **Done** |
| LLM-AUDIT-10 | Plugin provider story undocumented | **P2** | M-LLM-X.6 | **Partial** — USAGE §Extension; enum-free profile pending X.6.1 |
| LLM-AUDIT-11 | `ContextBudgetPolicy` default 4k decoupled from adapter window | **P0** | M-LLM-X.3.3 | **Done** |
| LLM-AUDIT-12 | Prefix context heuristics only on Bedrock (not Claude/OpenAI/Gemini) | **P0** | M-LLM-X.1.2–1.3 | **Done** |
| LLM-AUDIT-13 | Cohere dual slug (`cohere` vs `cohere_native`) confuses developers | **P2** | M-LLM-X.7.5 | **Done** |
| LLM-AUDIT-14 | Capability flags not catalog-driven (`supports_vision`, tools, structured) | **P2** | M-LLM-X.14.1 | **Planned** |
| LLM-AUDIT-15 | `engine_history_layer` token count inconsistent with preflight (chars/4) | **P0** | M-LLM-X.3.5 | **Done** — history already used adapter; preflight aligned in LC-2 |
| LLM-AUDIT-16 | No unified LLM routing rule contract — static hints only; no custom author logic | **P1** | M-LLM-X.9 | **Done** — ADR-LLM-003 |
| LLM-AUDIT-17 | Routing enterprise E2E — start-of-run + ACP (auto context, trace, reference host) | **P1** | M-LLM-X.10 | **Done** |
| LLM-AUDIT-18 | Routing mid-run Nexus — live re-eval, context refresh, full trace loop, true E2E run | **P1** | M-LLM-X.11 | **Done** (X-11 scope) |
| LLM-AUDIT-19 | Routing strict L5 — budget meter accuracy, all Nexus paths, tier boundary, production E2E, ACP trace parity | **P1** | M-LLM-X.12 | **Done** |
| LLM-AUDIT-20 | Post-L5 polish — ACP Plane A trace, tier bridge, concurrent test, secondary LLM + auxiliary Nexus paths | **P2** | M-LLM-X.13 | **Done** |
| LLM-AUDIT-21 | Domain closeout — audit register, AUDIT_IDEAL sync, implementation journal | **P1** | M-LLM-X.8 | **Planned** |
| LLM-AUDIT-22 | Capability flags not catalog-driven | **P2** | M-LLM-X.14.1 | **Planned** |
| LLM-AUDIT-23 | Dynamic gateway metadata (OpenRouter `/models`) not on catalog hot path | **P1** | M-LLM-X.14.2 | **Planned** |
| LLM-AUDIT-24 | ACP mid-run budget routing — `AcpInvocationUsageView` not mapped to `RoutingContext` | **P2** | M-LLM-X.14.4 | **Planned** |
| LLM-AUDIT-25 | Secondary LLM surfaces lack opt-in evaluating wrap (planner / websearch / critic) | **P2** | M-LLM-X.14.5 | **Planned** |
| LLM-AUDIT-26 | Plugin provider story — `LLMProfile.provider` enum coupling | **P2** | M-LLM-X.14.3 | **Planned** |

**Deferred (documented, no blocking X-phase task):** tiktoken OpenAI-centric token estimate for non-OpenAI models — **M-LLM-X.14.7** documents limitation and optional vendor tokenizer plugins; not blocking L5 routing.

**By design:** two-layer usage model (`LLMAdapterUsageLog` + `LLMUsageTracker`) — do not merge without explicit bridge (ADR-LLM-001).

**Ops (host responsibility):** distributed Redis rate limit requires `set_llm_distributed_rate_limiter` at Tier-3 bootstrap — not a Tier-0 code gap.

**Single adapter per Nexus run:** `RuntimeConfig.llm_adapter` holds one primary instance today; multi-model via profile chain + routing is M-LLM-X.4–5 (not a separate LLM-AUDIT ID).

**Closed baselines:** M-LLM (13/13), M-LLM-R (39/39), M-LLM-X LC-1…LC-3 **Done**; routing X-9…X-13 **Done**.

**Audit revalidation (2026-06-19, post X-13):** routing **L5** (LLM-AUDIT-17…20 **Done**) · LC + MAINT queues closed · **enterprise domain** open: **LLM-AUDIT-21…26** → [M-LLM-X-8 + X-14](../plan/LLM_ADAPTERS.md#phase-m-llm-x-14--enterprise-domain-maturity-2026-06-19).

---

## Anti-patterns

| Anti-pattern | Why forbidden | Correct approach |
|--------------|---------------|------------------|
| Adapter returns bare `str` | Breaks metering, guardrails, replay | `LLMAdapterResponse` / `LLMStructuredResult[T]` |
| Hardcoded model in Tier-2 agent | Bypasses profile, catalog, routing | `LLMProfile` + host resolver + `LLMRoutingProfile` |
| Custom routing logic without Protocol | Untraceable, bypasses allowlist | `LLMRoutingRule` subclass in Tier-3 |
| String eval / dynamic rule paths | Security + audit failure | Importable typed rule classes only |
| Direct OpenAI/Anthropic SDK in agents | Tier violation | Injected `LLMAdapter` via Nexus |
| Manual JSON parse for structured output | No schema validation | `generate_structured(output_model=...)` |
| Per-adapter context dict without catalog entry | Stale windows for new models | `ModelCatalog` + ADR-LLM-002 resolution |
| Silent 32k fallback on OpenRouter ids | Wrong history trim | Profile override or M-LLM-X.2 metadata |
| Merging adapter + run usage counters | Double-count or lost attribution | Two-layer model per ADR-LLM-001 |

---

## CI

```bash
uv run pytest tests/unit/llm_adapters/ -m gate -q
python scripts/check_llm_adapter_typed_returns.py
python scripts/check_agents_llm_adapter_response.py
python scripts/check_agents_vendor_imports.py
```

**Target gates (M-LLM-X):** `check_model_catalog_coverage.py`, `check_context_preflight_uses_adapter_tokens.py`, `check_llm_routing_rules.py`, `check_llm_routing_tier_boundary.py`, `check_llm_routing_context_wiring.py`.

Workflows: `unit-tests.yml`, `llm-adapters-guard.yml`, optional `llm-network-smoke.yml`.

---

## Related work (other domains)

| Item | Owner domain |
|------|----------------|
| Central LLM gateway service (single egress) | Platform ADR — not M-LLM-X |
| Multimodal attachment mapping (W-ML.1) | LLM_ADAPTERS + MODALITY |
| Cost envelopes (V-COST.*) | UNIFIED_EXECUTION_RUNTIME |
| AHI routing tuning production loop | ADAPTIVE_HARNESS_INTELLIGENCE + M-LLM-X.5 · **M-LLM-X.9** |
| LLM routing rules (author + custom classes) | LLM_ADAPTERS M-LLM-X.9 · ADR-LLM-003 |
| LLM routing enterprise closeout (start-of-run + ACP) | M-LLM-X.10 · LLM-AUDIT-17 |
| LLM routing enterprise hardening (mid-run Nexus) | M-LLM-X.11 · LLM-AUDIT-18 |
| LLM routing strict enterprise closeout (L5 honest) | M-LLM-X.12 · LLM-AUDIT-19 **Done** |
| LLM routing post-L5 polish (secondary surfaces, ACP Plane A) | M-LLM-X.13 · LLM-AUDIT-20 **Done** |
| LLM enterprise domain maturity (catalog caps, gateway meta, ACP budget, plugin DX) | M-LLM-X.14 · LLM-AUDIT-22…26 |
| LLM domain closeout (register + journal) | M-LLM-X.8 · LLM-AUDIT-21 |
| `BudgetReactionProfile.degrade_model` unification | AGENT_CONTRACTS + M-LLM-X.9.6 |
| AHI `ProfileVersion` llm_routing persistence | AHI-MAINT-06 |
| Product HTTP API DTOs | Tier-3 applications |

**Out of scope:** per-business-agent adapter code in `llm_adapters/`, YOLO/ONNX engines, Phase K business agents.

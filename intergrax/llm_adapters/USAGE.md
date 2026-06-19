# LLM Adapters — Developer Guide

**Canon:** [`docs/architecture/LLM_ADAPTERS.md`](../../docs/architecture/LLM_ADAPTERS.md) · **Plan:** [`docs/plan/LLM_ADAPTERS.md`](../../docs/plan/LLM_ADAPTERS.md) · **ADR:** [`docs/adr/entries/2026-06-14/ADR-LLM-002.md`](../../docs/adr/entries/2026-06-14/ADR-LLM-002.md)

Tier-0 module for multi-vendor LLM access. Agents and applications use **`LLMProfile`** + **`LLMAdapter`** — never vendor SDKs directly.

---

## Quickstart

```python
from intergrax.llm_adapters import LLMProfile, LLMProvider, LLMAdapterResponse
from intergrax.llm.messages import ChatMessage

profile = LLMProfile(
    provider=LLMProvider.GROQ,
    model="llama-3.3-70b-versatile",
    options={"max_retries": 2},
)
adapter = profile.create_adapter()  # reads GROQ_API_KEY from env

messages = [ChatMessage(role="user", content="Say OK")]
completion: LLMAdapterResponse = adapter.generate_messages(messages, run_id="demo")
print(completion.content)
if completion.usage:
    print(completion.usage.total_tokens)
```

**Tier-3 hosts:** prefer `resolve_llm_adapter(env)` from `intergrax.applications._shared.llm_resolver` (agent override → env profile → `INTERGRAX_LLM_*`).

---

## Provider selection

| Mechanism | When |
|-----------|------|
| `LLMProfile(provider=LLMProvider.…)` | Explicit in application manifest / factory |
| `llm_profile_from_env(prefix="INTERGRAX_LLM")` | Lab, K8s, Docker |
| `LLMAdapterRegistry.register("my_gateway", factory)` | Custom gateway (see §Extension) |
| `openrouter` slug | Multi-vendor model strings (`anthropic/claude-opus-4`, …) |

**Not Integration Library** — LLM slugs live in `intergrax/llm_adapters/`, not `intergrax/integrations/`.

### Built-in providers (19)

| Slug | Primary secret env | Default model env |
|------|-------------------|-------------------|
| `openai` | `OPENAI_API_KEY` | `INTERGRAX_DEFAULT_OPENAI_MODEL` |
| `claude` | `ANTHROPIC_API_KEY` | `INTERGRAX_DEFAULT_CLAUDE_MODEL` |
| `gemini` | `GOOGLE_API_KEY` | `INTERGRAX_DEFAULT_GEMINI_MODEL` |
| `mistral` | `MISTRAL_API_KEY` | `INTERGRAX_DEFAULT_MISTRAL_MODEL` |
| `azure_openai` | `AZURE_OPENAI_*` | deployment-specific |
| `aws_bedrock` | `AWS_*` | `INTERGRAX_DEFAULT_BEDROCK_MODEL_ID` |
| `ollama` | — | `INTERGRAX_DEFAULT_OLLAMA_MODEL` |
| `groq` | `GROQ_API_KEY` | `INTERGRAX_DEFAULT_GROQ_MODEL` |
| `vllm` | `VLLM_API_KEY` (optional) | `INTERGRAX_DEFAULT_VLLM_MODEL` |
| `together` | `TOGETHER_API_KEY` | `INTERGRAX_DEFAULT_TOGETHER_MODEL` |
| `fireworks` | `FIREWORKS_API_KEY` | `INTERGRAX_DEFAULT_FIREWORKS_MODEL` |
| `openrouter` | `OPENROUTER_API_KEY` | `INTERGRAX_DEFAULT_OPENROUTER_MODEL` |
| `deepseek` | `DEEPSEEK_API_KEY` | `INTERGRAX_DEFAULT_DEEPSEEK_MODEL` |
| `xai` | `XAI_API_KEY` | `INTERGRAX_DEFAULT_XAI_MODEL` |
| `llama_cpp` | optional | `INTERGRAX_DEFAULT_LLAMA_CPP_MODEL` |
| `cohere` | `COHERE_API_KEY` | `INTERGRAX_DEFAULT_COHERE_MODEL` |
| `cohere_native` | `COHERE_API_KEY` | `INTERGRAX_DEFAULT_COHERE_MODEL` |
| `vertex_gemini` | `GOOGLE_APPLICATION_CREDENTIALS` | `INTERGRAX_DEFAULT_GEMINI_MODEL` |
| `azure_ai_inference` | `AZURE_AI_INFERENCE_API_KEY` | `INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_MODEL` |

Platform defaults: `INTERGRAX_LLM_PROVIDER`, `INTERGRAX_LLM_MODEL`.

### Cohere: `cohere` vs `cohere_native`

| Slug | Use when |
|------|----------|
| `cohere` | OpenAI-compatible Chat Completions shim; simplest migration |
| `cohere_native` | Native Cohere SDK — prefer for native tools / streaming parity |

Same `COHERE_API_KEY` for both.

---

## Model selection

**Model id is a free string** — no platform enum. New vendor models work immediately for API calls.

```python
LLMProfile(provider=LLMProvider.CLAUDE, model="claude-opus-4-20250514")
LLMProfile(provider=LLMProvider.OPENROUTER, model="openai/gpt-4.1")
```

**Context budgeting** depends on `ModelCatalog` (see §Context window). Until catalog is wired for your model, set an explicit override.

---

## Context window and ModelCatalog

Nexus context engine uses `adapter.context_window_tokens` for history trim and preflight. Wrong values → aggressive trim or API `context_length_exceeded`.

### Operator override (always works — use for self-hosted / new models)

```python
LLMProfile(
    provider=LLMProvider.VLLM,
    model="my-custom-70b",
    options={"context_window_tokens": 131_072},
)
```

Resolution order (ADR-LLM-002): **profile override → catalog exact → prefix rules → family default → conservative fallback**.

Optional operator catalog overlay:

```bash
INTERGRAX_LLM_MODEL_CATALOG_PATH=/etc/intergrax/model_catalog.yaml
```

Bundled catalog (implementation): `intergrax/llm_adapters/registry/model_catalog.yaml`.

### OpenRouter / gateways

Enable optional metadata fetch (when implemented):

```python
options={"fetch_gateway_metadata": True}
```

---

## Response envelope

All completions return **`LLMAdapterResponse`** — use `.content`, `.usage`, `.tool_calls`, `.finish_reason` (ADR-LLM-001).

```python
if completion.tool_calls:
    for tc in completion.tool_calls:
        args = tc.arguments_json
```

Streaming: `Iterable[LLMStreamEvent]` — final event carries full `LLMAdapterResponse`.

---

## Resilience and failover

Pass via `LLMProfile.options` or adapter ctor kwargs:

| Option | Purpose |
|--------|---------|
| `max_retries` | Retry count |
| `timeout_sec` | Per-call timeout |
| `calls_per_minute` | In-process rate limit |
| `circuit_breaker_threshold` | Open circuit after N failures |
| `use_distributed_rate_limit` | Redis limiter (requires host wiring) |

**Failover chain (target M-LLM-X.4):** `fallback_profiles` on `LLMProfile` — primary then alternates on 429/5xx.

**Distributed rate limit:** host must call `set_llm_distributed_rate_limiter(...)` at bootstrap — not automatic.

---

## Secrets

```python
from intergrax.llm_adapters.registry import LLMProfile

profile.create_adapter(secrets={"api_key": key})
profile.create_adapter_from_secrets_store(vault)  # path: llm/<provider>/api_key
```

Never commit `.env` keys.

---

## Nexus vs ACP

| Path | API |
|------|-----|
| **Nexus / tools / RAG** | Inject `LLMAdapter`; call `generate_messages` |
| **ACP agents** | `StepLLMRouter` with `model_hint` — **target:** thin wrapper over same adapter (M-LLM-X.5) |

Planner ≠ producer: `ReasoningProfile.planner_llm_profile` → separate adapter via `resolve_planner_llm_adapter()`.

---

## Extension — custom provider

```python
from intergrax.llm_adapters import LLMAdapterRegistry, LLMAdapter

def my_factory(**kwargs) -> LLMAdapter:
    ...

LLMAdapterRegistry.register("my_gateway", my_factory)
profile = LLMProfile(provider="my_gateway", model="vendor/model")  # M-LLM-X.6
```

Built-in enum extension still requires a harness PR for `_BUILTIN_ADAPTERS`.

---

## Validation and ops

```python
# Target M-LLM-X.7.2
profile.validate_runtime()  # catalog hit, key present, context > 0
```

Metrics:

```bash
INTERGRAX_LLM_METRICS_ENABLED=true
INTERGRAX_LLM_TENANT_MAX_TOKENS=500000   # optional hard quota
```

Scrape: `register_llm_metrics_routes(app)` → `GET /metrics/llm`.

### Distributed rate limiting (LLM-MAINT-04)

For multi-replica Tier-3 hosts, wire Redis-backed token buckets at process startup:

```python
from intergrax.integrations.providers.key_value_cache.redis import create_redis_rate_limiter
from intergrax.llm_adapters._shared.resilience import set_llm_distributed_rate_limiter

limiter = create_redis_rate_limiter(env.integration_profile.resolve_key_value_cache())
set_llm_distributed_rate_limiter(limiter)
```

Requires `integration_profile.key_value_cache` slug `redis`. Cross-ref: [`docs/plan/ELASTIC_CAPACITY_AND_SCALING.md`](../../docs/plan/ELASTIC_CAPACITY_AND_SCALING.md) (platform scaling) · [`docs/plan/TIER3_APPLICATION_ENVIRONMENT.md`](../../docs/plan/TIER3_APPLICATION_ENVIRONMENT.md) (host wiring).

**Failover profiles (LLM-MAINT-03):** set `LLMProfile.fallback_profiles` on `ApplicationEnvironmentProfile.capabilities.llm` — `resolve_llm_adapter(env)` builds `FailoverLLMAdapter` automatically when fallbacks or routing hints are present.

---

## LLM routing rules (M-LLM-X.9 · M-LLM-X.10)

**ADR:** [`ADR-LLM-003`](../../docs/adr/entries/2026-06-19/ADR-LLM-003.md) · **Canon:** [`LLM_ADAPTERS.md`](../../docs/architecture/LLM_ADAPTERS.md) § LLM routing rules

Tier-3 hosts configure dynamic model selection with **`LLMRoutingProfile`** on `ApplicationEnvironmentProfile`. Each rule implements **`LLMRoutingRule`** (`matches` + `resolve`).

**Two authoring paths (both supported):**

1. **Predefined catalog (Tier-0)** — platform ships production-ready parametric classes in `intergrax.llm_adapters.routing` (preferred for common cases; no boilerplate).
2. **Custom rules (Tier-3)** — subclass **`LLMRoutingRuleBase`** or implement **`LLMRoutingRule`** directly when domain logic does not fit the catalog. Mix builtin and custom rules in the same `LLMRoutingProfile`.

**Auto context (M-LLM-X.10.2):** Nexus / harness paths call `build_routing_context_from_runtime()` — authors do not pass `routing_context=` manually on default host wiring.

### Predefined rule catalog

| Class | Params | `matches` |
|-------|--------|-----------|
| `BudgetBelowRule` | `threshold`, `profile` or `hint` | `budget_remaining_ratio < threshold` |
| `BudgetAboveRule` | `threshold`, `profile` or `hint` | `budget_remaining_ratio > threshold` |
| `BudgetExceededDegradeRule` | — | `budget_degrade_active` → `CHEAPEST` |
| `TaskClassInRule` | `classes`, `profile` or `hint` | `task_class in classes` |
| `TaskClassNotInRule` | `classes`, `profile` or `hint` | `task_class not in classes` |
| `TokenUsedAboveRule` | `threshold`, `hint` | `tokens_used > threshold` |
| `TokenUsedBelowRule` | `threshold`, `profile` or `hint` | `tokens_used < threshold` |
| `StepIndexAtLeastRule` | `min_step`, `profile` or `hint` | `step_index >= min_step` |
| `StepIndexBelowRule` | `max_step`, `profile` or `hint` | `step_index < max_step` |
| `AgentIdInRule` | `agent_ids`, `profile` or `hint` | `agent_id in agent_ids` |
| `TenantIdInRule` | `tenant_ids`, `profile` or `hint` | `tenant_id in tenant_ids` |
| `ModelHintPresentRule` | `profile` or `hint` | non-empty `model_hint` |
| `PolicyHintRule` | `hint` | always (use low priority) |
| `CompositeAllRule` | `rules`, `profile` or `hint` | all nested rules match |
| `CompositeAnyRule` | `rules`, `profile` or `hint` | any nested rule matches |
| `AlwaysRule` | `profile` or `hint`, `priority=-100` | unconditional fallback |

Export: `BUILTIN_ROUTING_RULE_TYPES` from `intergrax.llm_adapters.routing`.

```python
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    CompositeAllRule,
    LLMRoutingProfile,
    TaskClassInRule,
)

primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")

env = ApplicationEnvironmentProfile.lab_defaults()
env.llm_profile = primary
env.llm_routing_profile = LLMRoutingProfile(
    default_profile=primary,
    allowed_profiles=(primary, local),
    rules=(
        CompositeAllRule(
            rules=(
                TaskClassInRule(classes=("contract_review",)),
                BudgetBelowRule(threshold=0.2, profile=local),
            ),
            profile=local,
            priority=12,
        ),
        BudgetBelowRule(threshold=0.15, profile=local),
    ),
)

adapter = resolve_llm_adapter(env)  # routing context auto-filled on Nexus path
```

### Custom rules (Tier-3)

When catalog classes are insufficient, add app-specific rules alongside builtins:

```python
from intergrax.llm_adapters.routing import LLMRoutingRuleBase, RoutingContext, RoutingTarget

class LegalLowBudgetRule(LLMRoutingRuleBase):
    rule_id = "legal.low_budget"
    priority = 10

    def matches(self, context: RoutingContext) -> bool:
        return self.budget_below(context, 0.2) and self.task_is(context, "contract_review")

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(profile=local, reason="legal_budget")

env.llm_routing_profile = LLMRoutingProfile(
    default_profile=primary,
    allowed_profiles=(primary, local),
    rules=(LegalLowBudgetRule(), BudgetBelowRule(threshold=0.15, profile=local)),
)
```

**Per-step routing (agents):** ACP hosts with `llm_routing_profile` auto-wrap `StepLLMRouter` via `wrap_dynamic_llm_router()`.

**Observability:** rule evaluations emit `LLMRoutingRuleDiagV1` (`intergrax.diag.engine.core_llm.routing_rule`) with `matched_rule_id`, `routing_reason`, `profile_id`.

**Reference host (lab only):** `build_lab_environment_profile()` demonstrates the **predefined catalog** for CI (`check_llm_routing_rules.py`). Product hosts are not limited to builtins.

**Enterprise checklist (M-LLM-X.10 — Done · start-of-run + ACP):**

- [x] 12+ predefined parametric rule classes (Tier-0 catalog)
- [x] Custom `LLMRoutingRule` subclasses supported (Tier-3)
- [x] `build_routing_context_from_runtime()` on materialize path
- [x] Trace `LLMRoutingRuleDiagV1` (initial evaluation)
- [x] Lab reference host manifest
- [x] Acceptance: budget rule switches profile (resolver-level)
- [x] `DynamicLLMRouter` on ACP when profile set
- [x] CI gate `python scripts/check_llm_routing_rules.py`

**Enterprise hardening (M-LLM-X.11 — Done · mid-run Nexus):**

- [x] `RoutingEvaluatingLLMAdapter` — live re-eval on each LLM call
- [x] `refresh_llm_routing_context()` in Nexus / UAEP step loop
- [x] All Tier-3 wiring uses `resolve_environment_llm_adapter()` or context provider
- [x] Per-evaluation trace + `LLMRoutingAllowlistViolationDiagV1`
- [x] Mid-run acceptance: budget threshold → model swap via evaluating adapter
- [x] Harness host + `materialize_runtime_config` evaluating adapter parity
- [x] CI gate `check_llm_routing_context_wiring.py`

**Mid-run routing:** when `llm_routing_profile` is set, `resolve_llm_adapter()` wraps the core adapter in `RoutingEvaluatingLLMAdapter`. `RuntimeConfig.llm_routing_snapshot` is refreshed via `sync_llm_routing_snapshot_for_state()` before each UAEP step.

**Known limitations (post X-11 audit — address in M-LLM-X.12):**

- Budget-driven rules may not see accurate `tokens_used` until tracker reads inner adapter usage (12.1).
- Context sync is strongest on **UAEP**; classic Nexus graph paths may lag (12.3).
- **ACP** per-eval trace not fully wired on `DynamicLLMRouter` (12.8).
- Evaluating adapter currently lives in Tier-0 with Tier-3 factory import — refactor planned (12.2).

**Strict L5 closeout (M-LLM-X.12 — Done):**

- [x] Budget meter ↔ routing context accuracy
- [x] Tier-clean evaluating factory (no `applications/` import from Tier-0 routing)
- [x] Nexus-wide context sync (graph + CE paths)
- [x] Per-call context refresh within a step
- [x] ACP trace parity + production E2E without factory mocks on core path
- [x] Closes **LLM-AUDIT-19**

**Current maturity label:** **L5** (strict mid-run routing on core UAEP/Nexus/ACP paths).

**Testing:** unit-test `rule.matches(fake_context)` and `rule.resolve(...)` without Nexus. CI gate: `python scripts/check_llm_routing_rules.py`.

**HF models:** serve weights via **vLLM** or **llama.cpp** — use the model id on the local profile; HF Hub remains object storage only.

---

## Self-hosted Docker (Ollama / vLLM / llama.cpp)

| Backend | Start | Base URL env |
|---------|-------|--------------|
| Ollama (dev / embeddings) | `cd infra/integration && ./manage.sh start rag` | `OLLAMA_HOST=http://127.0.0.1:11434` |
| vLLM (production GPU) | `cd infra/integration && ./manage.sh start vllm` | `INTERGRAX_DEFAULT_VLLM_BASE_URL=http://127.0.0.1:8100/v1` |
| llama.cpp (CPU-friendly) | `cd infra/integration && ./manage.sh start llama-cpp` | `INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL=http://127.0.0.1:8102/v1` |

vLLM requires **NVIDIA GPU** + `nvidia-container-toolkit`. llama.cpp is **CPU-first** (optional CUDA). Host ports **8100** (vLLM) and **8102** (llama.cpp) avoid Chroma **8000** and Weaviate **8080** — see [`infra/PORTS.md`](../../infra/PORTS.md).

On **WSL2**, set `VLLM_USE_V1=0` (default in compose) if the v1 engine fails to initialize.

**RAG embeddings:** use `VllmEmbeddingProvider` (`provider_id=vllm`) or `LlamaCppEmbeddingProvider` (`provider_id=llama_cpp`) with a **separate** embed server — see `infra/docker/vllm-embed` (host **8101**) or `infra/docker/llama-cpp-embed` (host **8103**).

```bash
export INTERGRAX_LLM_PROVIDER=vllm
export INTERGRAX_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export INTERGRAX_DEFAULT_VLLM_BASE_URL=http://127.0.0.1:8100/v1
```

```bash
export INTERGRAX_LLM_PROVIDER=llama_cpp
export INTERGRAX_LLM_MODEL=default
export INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL=http://127.0.0.1:8102/v1
```

---

## Testing

```python
from testing_support.builder import FakeLLMAdapter

adapter = FakeLLMAdapter(fixed_text="ok")
```

Optional live smoke (not PR gate) — **vLLM only** in GitHub `llm-network-smoke.yml`:

```bash
cd infra/integration && ./manage.sh start vllm
export INTERGRAX_DEFAULT_VLLM_BASE_URL=http://127.0.0.1:8100/v1
export INTERGRAX_DEFAULT_VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
uv run pytest tests/unit/llm_adapters/test_network_smoke.py::test_vllm_live_one_shot -m network -q
```

**llama.cpp — local E2E only (never GitHub CI):**

```bash
infra/docker/llama-cpp/verify.ps1   # Windows
# infra/docker/llama-cpp/verify.sh  # Linux/macOS/Git Bash
```

See [`infra/docker/llama-cpp/VERIFY_RUNBOOK.md`](../../infra/docker/llama-cpp/VERIFY_RUNBOOK.md).

Skips automatically when vLLM is unreachable or env is unset. Workflow: `.github/workflows/llm-network-smoke.yml`.

Conformance helpers: `intergrax/llm_adapters/_shared/conformance.py`.

---

## Token estimation note

Budgeting uses `tiktoken` with `model_name_for_token_estimation` when available. Non-OpenAI models may use approximate counts — prefer SDK `usage` on `LLMAdapterResponse` for billing. Vendor-specific tokenizer plugins are deferred post-M-LLM-X.

---

## Related

- Tier-3 wiring: [`applications/USAGE.md`](../../applications/USAGE.md)
- Agent authoring: [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md)
- Context preflight: [`docs/architecture/CONTEXT_ENGINEERING.md`](../../docs/architecture/CONTEXT_ENGINEERING.md)

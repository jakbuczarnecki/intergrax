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

# LLM_ADAPTERS - providers catalog

**Parent hub:** [`LLM_ADAPTERS.md`](../LLM_ADAPTERS.md)

## Providers (19)

OpenAI-compatible slugs share `openai_compat_factory.py`. ABC defaults: streaming **false**, structured **false** unless overridden.

| Slug | Adapter module | Primary env | Stream | Structured | Notes |
|------|----------------|-------------|--------|------------|-------|
| `openai` | `openai_responses_adapter` | `OPENAI_API_KEY` | yes | yes | Native Responses API |
| `gemini` | `gemini_adapter` | `GOOGLE_API_KEY` | yes | yes | |
| `ollama` | `native_ollama_adapter` (default); `ollama_adapter` (LangChain optional) | `OLLAMA_BASE_URL` | yes | partial | Local; context override today |
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

Per-provider model env vars: `INTERGRAX_DEFAULT_<PROVIDER>_MODEL` (see [`USAGE.md`](../../../../intergrax/llm_adapters/USAGE.md)).

### Self-hosted inference (Ollama vs vLLM vs llama.cpp)

| Concern | Ollama | vLLM | llama.cpp |
|---------|--------|------|-----------|
| Adapter module | `native_ollama_adapter.py` (`NativeOllamaAdapter`, default); `ollama_adapter.py` (`LangChainOllamaAdapter`, optional extra) | `openai_compat_providers.VllmChatAdapter` | `openai_compat_providers.LlamaCppChatAdapter` |
| API shape | Ollama native HTTP | OpenAI Chat Completions (`/v1`) | OpenAI Chat Completions (`/v1`) |
| Tier-0 slug | `LLMProvider.OLLAMA` | `LLMProvider.VLLM` | `LLMProvider.LLAMA_CPP` |
| Local Docker | `infra/docker/ollama` · profile `rag` · port **11434** | `infra/docker/vllm` · profile **`vllm`** (opt-in) · host **8100** | `infra/docker/llama-cpp` · profile **`llama-cpp`** (opt-in) · host **8102** |
| GPU | Optional (CPU OK for dev) | **NVIDIA GPU required** for practical use | **CPU-first** (optional CUDA in compose) |
| P5 integration | `interaction_surface/ollama` (health probe) | Not registered - adapter + Docker health (`/v1/models`) | Not registered - same as vLLM |

**Do not** add a LangChain-style duplicate adapter for vLLM or llama.cpp - OpenAI-compat factory is the canonical path (M-LLM.3, M-LLM.7).

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

Port **8100** (vLLM) and **8102** (llama.cpp) avoid conflict with Chroma (**8000**) and Weaviate (**8080**) - see [`infra/PORTS.md`](../../infra/PORTS.md).

**Live smoke (vLLM only):** `test_vllm_live_one_shot` in `tests/unit/llm_adapters/test_network_smoke.py` (marker `network`; weekly GitHub workflow).

**llama.cpp verification (local only, not GitHub CI):** [`infra/docker/llama-cpp/VERIFY_RUNBOOK.md`](../../infra/docker/llama-cpp/VERIFY_RUNBOOK.md) · `tests/e2e/llama_cpp` (`e2e`, `no_ci`, `network`).

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
| Context preflight | `verify_context_preflight()` - default **`adapter.count_messages_tokens`** |

---

## Observability (Prometheus & governance)

Tier-0 metrics: `intergrax/llm_adapters/tracking`.

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
- **Failover:** profile chain via `fallback_profiles` / `FailoverLLMAdapter` - Done.
- **Secrets:** `registry/secrets.py` - env + `llm/<provider>/api_key`.

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
| `INTERGRAX_LLM_MODEL_CATALOG_PATH` | Optional `ModelCatalog` override YAML |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … | Per-provider secrets |

---

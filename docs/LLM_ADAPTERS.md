# Intergrax LLM Adapters

**Last updated:** 2026-06-01

Tier-0 LLM layer — `LLMAdapter`, registry, `LLMProfile`. Outside Integration Library (§5.2.2).

**Related:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) · [LLM_OBSERVABILITY.md](LLM_OBSERVABILITY.md) · [applications/USAGE.md](../applications/USAGE.md) · Phase **M-LLM**

---

## Providers (19)

OpenAI-compatible slugs share `openai_compat_factory.py`. Override `supports_streaming()` / `supports_structured_output()` per adapter (ABC defaults: streaming **false**, structured **false**).

| Slug | Adapter module | Primary env | Stream | Structured | Notes |
|------|----------------|-------------|--------|------------|-------|
| `openai` | `openai_responses_adapter` | `OPENAI_API_KEY` | yes | yes | Native Responses API |
| `gemini` | `gemini_adapter` | `GEMINI_API_KEY` | yes | yes | |
| `ollama` | `ollama_adapter` | `OLLAMA_BASE_URL` | yes | partial | Local |
| `mistral` | `mistral_adapter` | `MISTRAL_API_KEY` | yes | yes | |
| `claude` | `claude_adapter` | `ANTHROPIC_API_KEY` | yes | yes | |
| `azure_openai` | `azure_openai_adapter` | `AZURE_OPENAI_*` | yes | yes | |
| `aws_bedrock` | `aws_bedrock_adapter` | `AWS_*` | yes | partial | |
| `groq` | `openai_compat` | `GROQ_API_KEY` | compat | compat | |
| `vllm` | `openai_compat` | `VLLM_BASE_URL` | compat | compat | |
| `together` | `openai_compat` | `TOGETHER_API_KEY` | compat | compat | |
| `fireworks` | `openai_compat` | `FIREWORKS_API_KEY` | compat | compat | |
| `openrouter` | `openai_compat` | `OPENROUTER_API_KEY` | compat | compat | |
| `deepseek` | `openai_compat` | `DEEPSEEK_API_KEY` | compat | compat | |
| `xai` | `openai_compat` | `XAI_API_KEY` | compat | compat | |
| `llama_cpp` | `openai_compat` | `LLAMA_CPP_BASE_URL` | compat | compat | |
| `cohere` | `openai_compat` | `COHERE_API_KEY` | compat | compat | Chat Completions shim |
| `cohere_native` | `cohere_native_adapter` | `COHERE_API_KEY` | yes | partial | |
| `vertex_gemini` | `vertex_gemini_adapter` | `GOOGLE_APPLICATION_CREDENTIALS` | yes | yes | |
| `azure_ai_inference` | `azure_ai_inference_adapter` | `AZURE_AI_*` | yes | partial | |

Central env appendix: `INTERGRAX_LLM_PROVIDER`, `INTERGRAX_LLM_MODEL`, `INTERGRAX_LLM_TENANT_MAX_TOKENS`, `INTERGRAX_LLM_METRICS_ENABLED`, `INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL`. Per-provider secrets: `llm/<provider>/api_key` via `SecretsStore`.

---

## Tier-3 wiring

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
llm = profile.create_adapter(secrets={"api_key": key})  # or create_adapter_from_secrets_store(vault)
```

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

Details: [LLM_OBSERVABILITY.md](LLM_OBSERVABILITY.md).

---

## Resilience & secrets

- `LLMCallConfig`: retries, in-process rate limit, circuit breaker, optional Redis rate limit.
- `registry/secrets.py`: env + `SecretsStore` paths (`llm/<provider>/api_key`).

---

## CI (unit gate only — no product E2E)

```bash
uv run pytest tests/unit/llm_adapters/ -m gate -q
```

Workflows: `unit-tests.yml`, `llm-adapters-guard.yml`, optional `llm-network-smoke.yml`.

---

## Harness-aligned next steps

| Item | Canon / goal |
|------|----------------|
| Wire Prometheus scrape in Tier-3 host Helm/K8s | §7.1 observability |
| PolicyEngine rules consuming `llm_cost_evaluation` logs | §governance replay |
| Central LLM gateway service (single egress) | §5.2.4 — needs architecture approval |
| Model routing / fallback chains in `LLMProfile` | Agent harness flexibility |

**Out of scope:** product E2E gates, per-business-agent adapter code in `llm_adapters/`.

# Intergrax LLM Adapters

**Last updated:** 2026-05-30

Tier-0 module for LLM calls — single contract (`LLMAdapter`), registry, Tier-3 profiles. **Not** part of the Integration Library (architecture §5.2.2).

**Related:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) · [applications/USAGE.md](../applications/USAGE.md) · [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase M-LLM

---

## Providers (19)

| Key | Class |
|-----|-------|
| `openai` | `OpenAIChatResponsesAdapter` |
| `claude` | `ClaudeChatAdapter` |
| `azure_openai` | `AzureOpenAIChatAdapter` |
| `azure_ai_inference` | `AzureAiInferenceChatAdapter` |
| `gemini` / `vertex_gemini` | `GeminiChatAdapter` / `VertexGeminiChatAdapter` |
| `mistral` | `MistralChatAdapter` |
| `aws_bedrock` | `BedrockChatAdapter` (Converse + stream + tools stream) |
| `ollama` | `LangChainOllamaAdapter` |
| `groq`, `vllm`, `together`, `fireworks`, `openrouter`, `deepseek`, `xai`, `llama_cpp` | `openai_compat_providers.*` |
| `cohere` / `cohere_native` | OpenAI-compat / Cohere SDK v2 |

---

## Tier-3 wiring

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env, set_llm_tenant_id
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

set_llm_tenant_id(tenant_id)  # billing aggregates in metrics
profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile", options={"max_retries": 2})
llm = profile.create_adapter(secrets={"api_key": vault_key})  # or env via resolve_api_key
```

**Legal:** `LEGAL_LLM_PROVIDER`, `LEGAL_LLM_MODEL` → `LLMProfile` in `legal_application/host/wiring.py`.

---

## Resilience (`LLMCallConfig`)

| Field | Purpose |
|-------|---------|
| `max_retries` / `retry_backoff_sec` | Transient API errors |
| `calls_per_minute` | Per-provider rate limit |
| `circuit_breaker_threshold` / `circuit_breaker_cooldown_sec` | Fail-fast after repeated errors |

Applied in `LLMAdapter._execute()` via `_shared/resilience.py`.

---

## Observability

| Mechanism | Usage |
|-----------|--------|
| `INTERGRAX_LLM_METRICS_ENABLED=true` | Enable counters |
| `set_llm_tenant_id()` | Tenant label on metrics (multi-tenant billing) |
| `render_prometheus_text()` | Prometheus scrape body |
| `render_otlp_json()` | OTLP-style JSON snapshot |
| `register_llm_metrics_routes(app)` | FastAPI `GET /metrics/llm`, `/metrics/llm/otlp` |

Labels: `tenant_id`, `provider`, `model`.

---

## Secrets (Tier-3)

`registry/secrets.py` — resolve API keys from env or `secrets` map passed to `LLMProfile.create_adapter(secrets=...)`.  
Production: load keys from Integration `secrets_store` in host startup, then `profile.with_secrets({...})`.

---

## CI

| Workflow | Role |
|----------|------|
| `unit-tests.yml` | Full regression gate (`-m gate`) |
| `llm-adapters-guard.yml` | PR path guard — builtin conformance + resilience + metrics |
| `llm-network-smoke.yml` | Weekly live smoke (Groq, OpenAI, Claude, Bedrock, Vertex) |

```bash
uv run pytest tests/unit/llm_adapters/ -m gate -q
```

---

## Environment (selected)

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_LLM_METRICS_ENABLED` | Metrics on |
| `INTERGRAX_LLM_PROVIDER` / `INTERGRAX_LLM_MODEL` | Default profile |
| `INTERGRAX_BEDROCK_USE_CONVERSE` | Bedrock Converse API |
| `INTERGRAX_VERTEX_PROJECT` | Vertex Gemini |
| Provider `*_API_KEY` | See `registry/secrets.py` |

---

## Architecture-aligned next steps

| Item | Canon |
|------|--------|
| Wire `set_llm_tenant_id` in Nexus `UnifiedTaskRunner` | §5.2 tracing + governance |
| Push OTLP to `observability_backend` Integration | §7.1 catalog |
| Per-tenant quotas in governance layer | §governance |
| Adapter conformance required in provider PRs | **Done** — `llm-adapters-guard.yml` |

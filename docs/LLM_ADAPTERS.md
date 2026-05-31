# Intergrax LLM Adapters

**Last updated:** 2026-05-30

Tier-0 LLM layer — single `LLMAdapter` contract, registry, profiles. **Outside** Integration Library (§5.2.2).

**Related:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) · [applications/USAGE.md](../applications/USAGE.md) · Phase **M-LLM** in [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Providers (19)

| Key | Class |
|-----|-------|
| `openai` | `OpenAIChatResponsesAdapter` |
| `claude` | `ClaudeChatAdapter` |
| `azure_openai` / `azure_ai_inference` | Azure adapters |
| `gemini` / `vertex_gemini` | Google GenAI |
| `mistral` | `MistralChatAdapter` |
| `aws_bedrock` | Converse + stream + tools stream |
| `ollama` | `LangChainOllamaAdapter` |
| OpenAI-compatible slugs | `openai_compat_providers.*` |
| `cohere` / `cohere_native` | Compat / SDK v2 |

---

## Tier-3 wiring

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile", options={"max_retries": 2})
llm = profile.create_adapter(secrets={"api_key": vault_key})
```

Env: `INTERGRAX_LLM_PROVIDER`, `INTERGRAX_LLM_MODEL`, or app-prefixed vars in `host/wiring.py`.

---

## Nexus runtime integration (Tier-0)

| Mechanism | Location |
|-----------|----------|
| Tenant scope | `UnifiedTaskRunner` → `llm_tenant_scope(task.tenant_id)` |
| Metrics export plugin | `bootstrap_nexus_platform()` → `runtime.llm_metrics_export` on `TASK_COMPLETED` |
| Per-tenant quota | `INTERGRAX_LLM_TENANT_MAX_TOKENS` → `check_llm_tenant_quota()` in `_execute()` |

Manual tenant binding (non-Nexus scripts): `set_llm_tenant_id(tenant_id)` or `llm_tenant_scope(...)`.

---

## Resilience (`LLMCallConfig`)

`max_retries`, `calls_per_minute`, `circuit_breaker_threshold` — via `_shared/resilience.py` in `_execute()`.

---

## Observability

| Mechanism | Usage |
|-----------|--------|
| `INTERGRAX_LLM_METRICS_ENABLED=true` | Counters on |
| `register_llm_metrics_routes(app)` | `GET /metrics/llm`, `/metrics/llm/otlp` |
| Prometheus scrape | `render_prometheus_text()` |
| OTLP JSON | `render_otlp_json()` |
| Integration catalog | Scrape HTTP or query `intergrax_llm_*` after remote-write (Tier-3) |

Labels: `tenant_id`, `provider`, `model`.

---

## Secrets

`registry/secrets.py` — `LLMProfile.create_adapter(secrets=...)`, `with_secrets()`.  
Tier-3 loads keys from Integration `secrets_store` at host startup.

---

## CI (no product E2E)

| Workflow | Role |
|----------|------|
| `unit-tests.yml` | Unit + runtime gate (no application E2E paths) |
| `llm-adapters-guard.yml` | PR guard for `llm_adapters/` changes |
| `llm-network-smoke.yml` | Optional live API smoke |

```bash
uv run pytest tests/unit/llm_adapters/ -m gate -q
```

---

## Environment

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_LLM_METRICS_ENABLED` | Metrics |
| `INTERGRAX_LLM_TENANT_MAX_TOKENS` | Per-tenant token budget (0=off) |
| `INTERGRAX_LLM_PROVIDER` / `INTERGRAX_LLM_MODEL` | Default profile |
| Provider `*_API_KEY` | See `registry/secrets.py` |

---

## Production roadmap (Tier-0, platform)

| Item | Status |
|------|--------|
| Nexus `llm_tenant_scope` in `UnifiedTaskRunner` | **Done** |
| Runtime plugin OTLP/log export on task complete | **Done** |
| Per-tenant token quota guard | **Done** |
| PR conformance guard | **Done** |
| Remote-write to Prometheus Pushgateway | Backlog |
| Governance policy engine rules for LLM cost | Backlog |
| Provider-specific SLA dashboards | Backlog |

**Out of scope:** product E2E gates, per-SKU business agent adapters in this module.

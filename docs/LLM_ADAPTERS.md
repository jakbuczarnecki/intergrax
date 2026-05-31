# Intergrax LLM Adapters

**Last updated:** 2026-05-30

Tier-0 LLM layer — `LLMAdapter`, registry, `LLMProfile`. Outside Integration Library (§5.2.2).

**Related:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) · [LLM_OBSERVABILITY.md](LLM_OBSERVABILITY.md) · [applications/USAGE.md](../applications/USAGE.md) · Phase **M-LLM**

---

## Providers (19)

See provider table in git history / `LLMProvider` enum. OpenAI-compatible slugs share `openai_compat_factory.py`.

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

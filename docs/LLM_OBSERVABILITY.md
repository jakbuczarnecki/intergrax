# LLM observability — Prometheus & governance

**Tier-0** metrics from `intergrax/llm_adapters/`. Complements [LLM_ADAPTERS.md](LLM_ADAPTERS.md).

---

## Scrape (recommended)

Enable metrics and register HTTP routes on the Tier-3 FastAPI host:

```python
from intergrax.llm_adapters.tracking.exposition import register_llm_metrics_routes

register_llm_metrics_routes(app)  # GET /metrics/llm
```

Configure Prometheus to scrape `http://<host>/metrics/llm`.

---

## Pushgateway (optional)

```bash
INTERGRAX_LLM_METRICS_ENABLED=true
INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL=http://pushgateway:9091
```

Pushes on each `TASK_COMPLETED` via runtime plugin (grouping `tenant/<tenant_id>`).

---

## Example PromQL (SLO)

```promql
# Calls per tenant (5m rate)
sum by (tenant_id) (rate(intergrax_llm_calls_total[5m]))

# Error ratio per provider
sum by (provider) (rate(intergrax_llm_errors_total[5m]))
  / sum by (provider) (rate(intergrax_llm_calls_total[5m]))

# Token throughput per model
sum by (tenant_id, model) (rate(intergrax_llm_output_tokens_total[5m]))
```

Query via Integration `observability_backend` = `prometheus` (`create_prometheus_observability_backend`).

---

## Governance signals

| Env | Purpose |
|-----|---------|
| `INTERGRAX_LLM_TENANT_MAX_TOKENS` | Hard cap (raises `LLMQuotaExceeded`) |
| `INTERGRAX_LLM_GOVERNANCE_WARN_TOKENS` | Soft warn on task complete (logs only) |

Correlate logs with Nexus trace using `run_id` / `task_id` in `llm_metrics_export` structured fields.

---

## Distributed rate limit (multi-replica)

```python
from intergrax.llm_adapters._shared.resilience import set_llm_distributed_rate_limiter
from intergrax.integrations.providers.key_value_cache.redis.bundle import create_redis_rate_limiter

set_llm_distributed_rate_limiter(create_redis_rate_limiter(url="redis://..."))

profile = LLMProfile(provider=..., options={"use_distributed_rate_limit": True, "calls_per_minute": 120})
```

Falls back to in-process limiter when Redis limiter is not set.

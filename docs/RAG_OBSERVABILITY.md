# RAG observability

Tier-0 RAG metrics mirror the LLM metrics pattern (`docs/LLM_OBSERVABILITY.md`).

## Enable

```bash
export INTERGRAX_RAG_METRICS_ENABLED=true
```

**Lab/product bootstrap (default):** `bootstrap_nexus_platform()` in `applications/_shared/platform_wiring.py` registers both LLM and RAG metrics plugins (Phase Q-O.1).

Manual registration:

```python
from intergrax.rag.tracking.observability_bridge import register_rag_observability_plugin

plugins = register_rag_observability_plugin(plugins)
```

## Recorded fields

Per `(tenant_id, retriever_id, route_tier)`:

| Field | Source |
|-------|--------|
| `calls` | Each successful `RetrievalService.retrieve()` |
| `retrieval_latency_ms` | Sum of `RetrievalTrace.retrieval_latency_ms` |
| `rerank_latency_ms` | Sum of rerank stage latency |
| `hybrid_calls` | When `trace.hybrid_used` |
| `agentic_iterations` | Deep-tier agentic loop iterations |
| `recall_at_k_avg` | When golden harness or eval sets `trace.recall_at_k` |

## Trace diagnostics

`RetrievalResult.trace` includes:

- `hybrid_used`, `agentic_total_latency_ms`, `recall_at_k`
- Existing: `route_tier`, `retriever_id`, latencies, agentic stop reason

Exported on `TASK_COMPLETED` via structured log field `rag_metrics` (same bus hook as LLM metrics).

## Golden regression

CI gate: `tests/fixtures/rag_golden/retrieval_cases.json` — scenarios `retrieval`, `graph_rag`, `multi_hop`, `agentic`.

Workflow: `.github/workflows/rag-guard.yml`

## Local backends (infra)

| Backend | Profile | Port |
|---------|---------|------|
| Prometheus | `observability` | 9090 |
| Phoenix | `observability` | 6006 |
| Langfuse | `observability` | 3000 (needs Postgres from `core`) |

Start: `cd infra/integration && ./manage.sh start observability`

## Metrics HTTP routes (parity decision)

RAG metrics follow the **same log + optional Pushgateway pattern as LLM** (Phase Q-O.9). There is no separate `GET /metrics/rag` unless you add `register_rag_metrics_routes` in Tier-3; default harness exports `rag_metrics` on `TASK_COMPLETED` via `register_rag_observability_plugin`.

## Parser trace export (ADR)

`parser_trace_flush` / `parser_trace_exporter` may write directly to configured backends (Phoenix, Langfuse) **without** going through `ObservabilityBackend`. This is intentional for document-ingest latency. Env knobs: `INTERGRAX_PARSER_TRACE_*` (see `infra/README.md` observability table). Nexus run traces remain on SQLite / configured trace store.

## Observability env profile (harness)

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_TRACE_DB` | Nexus run trace SQLite |
| `INTERGRAX_RUNTIME_EVENTS_DB` | Canonical runtime events |
| `INTERGRAX_LLM_METRICS_ENABLED` | LLM metrics plugin + `/metrics/llm` |
| `INTERGRAX_RAG_METRICS_ENABLED` | RAG metrics plugin (`rag_profile.extras`) |
| `INTERGRAX_PARSER_TRACE_ENABLED` | Document parser span export |
| Integration `observability_backend` slug | PromQL / Sentry / etc. |

Tier-3 `.env.example` files should list the same names (Phase Q-O.8).

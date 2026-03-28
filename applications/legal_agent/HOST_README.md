# Legal backend host (product shell)

**Tier-2** FastAPI application built on `intergrax.fastapi_core` and **Legal Agent** routes (`mount_legal_agent_routes`). Entrypoint: `legal_agent.host`.

## Local run

From the **Intergrax repository root** (with `.env` loaded as needed). Set `PYTHONPATH=applications` so the `legal_agent` package (root `applications/legal_agent/`) is importable:

```bash
uv sync
export PYTHONPATH=applications   # PowerShell: $env:PYTHONPATH="applications"
uv run python -m legal_agent.host.main
```

Alternatively:

```bash
export PYTHONPATH=applications
uv run uvicorn legal_agent.host.main:app --reload --host 0.0.0.0 --port 8000
```

In **dev**, OpenAPI docs: `http://127.0.0.1:8000/docs`. In **prod** (`LEGAL_BACKEND_ENV=prod`), `/docs` is off by default—enable with `LEGAL_BACKEND_OPENAPI=true` (e.g. internal network only).

## Environment variables (summary)

Full list in `.env.example` (Legal backend host section).

| Variable | Meaning |
|----------|---------|
| `LEGAL_BACKEND_ENV` | `dev` \| `stage` \| `prod` |
| `LEGAL_PRODUCT_PROFILE` | `strict_legal` \| `safe` \| `research` \| `fast` |
| `LEGAL_LLM_PROVIDER` | `ollama`, `openai`, `claude`, … |
| `LEGAL_DEFAULT_AGENT_ID` | Key in the agent map (default `legal-default`) |
| `LEGAL_ROUTE_PREFIX` | Router prefix (default `/v1/legal`) |
| `LEGAL_IDENTITY_SOURCE` | `body_or_context` (dev) / `context_only` (enforced in prod) |
| `LEGAL_BACKEND_BOOTSTRAP_API_KEY` | Single API key + tenant/user (MVP) |
| `LEGAL_BACKEND_API_KEYS_JSON` | JSON key map (prod) |
| `LEGAL_SESSION_SQLITE_PATH` | Optional persistent sessions (SQLite); otherwise in-process only |
| `LEGAL_BACKEND_CORS_ORIGINS` | Comma-separated origins, e.g. `https://app.example.com` |

## Production

- Set **`LEGAL_BACKEND_ENV=prod`** and **`LEGAL_IDENTITY_SOURCE=context_only`** (or omit—defaults apply).
- **API keys** required (`LEGAL_BACKEND_BOOTSTRAP_API_KEY` + tenant/user **or** `LEGAL_BACKEND_API_KEYS_JSON`). Emergency escape hatch: `LEGAL_BACKEND_ALLOW_UNAUTHENTICATED=true` (never on the public internet).
- The Tier-2 agent starts with **`production_mode=False`** until the host wires full governance and `trace_db_path` for Nexus requirements when `production_mode=True`.

## Roadmap

File uploads, RAG ingest jobs, REST history, billing—extensions of this host without forking the Nexus engine.

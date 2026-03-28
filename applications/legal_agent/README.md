# Legal product (`legal_agent`)

Self-contained **Tier-2** Legal domain (pipeline, serving, compliance) plus a **Tier-3** deployable host in `legal_agent.host`—FastAPI entrypoint that mounts `intergrax.fastapi_core` (`/health`, `/runs`, …) and **Legal** HTTP routes (`mount_legal_agent_routes`). **`intergrax` does not import this tree.**

| Layer | Location |
|-------|----------|
| Tier‑3 product host | [`host/`](host/README.md) — `main.py`, `factory.py`, `settings.py`, `wiring.py` |
| Tier‑2 domain & API | `serving/`, `pipeline/`, `config/`, … |

- **Strategy / phases:** [ROADMAP.md](ROADMAP.md)  
- **Implementation checklist (ordered steps, components, SaaS):** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)  

The former `intergrax.apps.legal_backend` layout lives here as `legal_agent.host`.

## Local run

From the **Intergrax repository root** (with `.env` loaded as needed). The importable package is `legal_agent` under `applications/legal_agent/`, so **`applications`** must be on `PYTHONPATH`:

```bash
uv sync
export PYTHONPATH=applications   # Linux / macOS; PowerShell: $env:PYTHONPATH="applications"
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

See [ROADMAP.md](ROADMAP.md) and the numbered tasks in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

# Legal product host (Tier‑3)

FastAPI application shell that **composes** Intergrax `fastapi_core` with the Tier‑2 Legal agent (`mount_legal_agent_routes`). `intergrax` never imports this module; only `legal_agent` imports `intergrax`.

## Modules

| File | Responsibility |
|------|----------------|
| `main.py` | ASGI `app`, loads `.env`, CLI `run()` for uvicorn |
| `factory.py` | `create_legal_backend_app()` — `create_app`, CORS/OpenAPI overrides, agent build, route mount |
| `settings.py` | `LegalBackendSettings.from_env()` — prod auth guardrails |
| `wiring.py` | `build_legal_agent()` — profile + LLM from settings |

## Run

From repository root with `PYTHONPATH=applications`:

```bash
uv run python -m legal_agent.host.main
# or
uv run uvicorn legal_agent.host.main:app --host 0.0.0.0 --port 8000
```

## Next steps (roadmap)

See `../ROADMAP.md` for goals and **`../IMPLEMENTATION_PLAN.md`** for ordered tasks (sessions, attachments, RAG, jobs, SaaS packaging).

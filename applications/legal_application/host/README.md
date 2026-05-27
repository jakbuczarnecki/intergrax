# Legal product host (Tier‑3)

FastAPI application shell that **composes** Intergrax `fastapi_core` with the Tier‑2 Legal capability (`legal`) via `mount_legal_agent_routes`. `intergrax` never imports this module; only `legal_application` and `legal` import `intergrax`.

## Modules

| File | Responsibility |
|------|----------------|
| `main.py` | ASGI `app`, loads `.env`, CLI `run()` for uvicorn |
| `factory.py` | `create_legal_backend_app()` — `create_app`, CORS/OpenAPI overrides, agent build, route mount |
| `settings.py` | `LegalBackendSettings.from_env()` — prod auth guardrails |
| `wiring.py` | `build_legal_agent()` — profile + LLM from settings |

## Run

From repository root (pytest and `uv run` add `applications/` + `agents/` via repo `conftest.py`):

```bash
uv run python -m legal_application.host.main
# or
uv run uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
```

## Next steps (roadmap)

See [`agents/legal/ROADMAP.md`](../../../agents/legal/ROADMAP.md) and [`agents/legal/IMPLEMENTATION_PLAN.md`](../../../agents/legal/IMPLEMENTATION_PLAN.md).

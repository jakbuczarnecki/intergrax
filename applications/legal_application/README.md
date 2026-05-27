# Legal application (Tier-3)

Deployable **execution environment** for the Legal capability (`agents/legal/`).

| Path | Role |
|------|------|
| `host/` | FastAPI ASGI app — `legal_application.host.main:app` |
| `serving/` | HTTP routes, runtime bridge, chat API |
| `legal_tests/` | Host/serving integration tests |

**Run:**

```bash
uv run python -m legal_application.host.main
uv run uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
```

**Imports:** `legal_application` (this package) + `legal` (capability). Do **not** use legacy `legal_agent` package paths.

See also: [`host/README.md`](host/README.md), [`agents/legal/README.md`](../../agents/legal/README.md).

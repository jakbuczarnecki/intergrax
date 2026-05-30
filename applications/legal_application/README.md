# Legal application (Tier-3)

Deployable **execution environment** for the Legal capability (`agents/legal/`).

**Build & deploy:** [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md)

| Path | Role |
|------|------|
| `host/` | FastAPI ASGI app — `legal_application.host.main:app` |
| `serving/` | HTTP routes, runtime bridge, chat API |
| `legal_tests/` | Host/serving integration tests |

**Run:**

```bash
cp applications/legal_application/.env.example applications/legal_application/.env
uv run python -m legal_application.host.main
uv run uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
```

**MCP:** FastMCP at `/mcp` (default) — `list_agents`, `run_agent` (`LEGAL_INCLUDE_MCP`, `LEGAL_MCP_MOUNT_PATH`).

**Docker:**

```bash
applications/legal_application/docker/build-docker.sh
# Windows: applications\legal_application\docker\build-docker.bat
```

**Imports:** `legal_application` (this package) + `legal` (capability). Do **not** use legacy `legal_agent` package paths.

See also: [`host/README.md`](host/README.md), [`agents/legal/README.md`](../../agents/legal/README.md).

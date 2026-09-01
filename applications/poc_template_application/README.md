# POC Template Application (Tier-3)

Living reference for Phase N - generated from `intergrax.scaffold new-application` (lab profile + echo agent).

**Build & deploy:** [`BUILD_AND_DEPLOY.md`](docs/BUILD_AND_DEPLOY.md) · **Architecture:** [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

## Three-command quickstart

From **repository root**:

```bash
# 1. Verify
uv run pytest applications/poc_template_application/tests -q

# 2. Run locally
cp applications/poc_template_application/.env.example applications/poc_template_application/.env
uv run uvicorn poc_template_application.host.main:app --host 127.0.0.1 --port 8095

# 3. Container
applications/poc_template_application/docker/build-docker.sh
# Windows: applications\poc_template_application\docker\build-docker.bat
```

## Agents

`EchoAgent` - minimal roster for CI and onboarding.

## HTTP

```bash
curl -s http://127.0.0.1:8095/v1/poc_template/agents
curl -s -X POST http://127.0.0.1:8095/v1/poc_template/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"echo.basic"}'
```

## MCP (FastMCP)

Same process as FastAPI - default `http://127.0.0.1:8095/mcp`. Tools: `list_agents`, `run_agent`.

## Docs

- [`applications/USAGE.md`](../USAGE.md) - Tier-3 layout
- [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md) - composition engine

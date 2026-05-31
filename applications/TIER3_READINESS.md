# Tier-3 application layer — readiness checklist

**Status:** Ready to generate new deployable applications (Phase N complete).

## Generate a new application

```bash
# Agent + application (recommended for greenfield)
python -m intergrax.scaffold new-stack my_feature --profile lab --capability my_feature.basic

# Application only (agent already exists)
python -m intergrax.scaffold new-application my_feature --profile lab --agents my_feature

# Product-style host (FastAPI Core + /health)
python -m intergrax.scaffold new-application my_product --profile product --agents echo --port 8000
```

See [`docs/AGENT_CREATION_GUIDE.md`](../docs/AGENT_CREATION_GUIDE.md) Step **4E**.

## What you get

| Piece | Location |
|-------|----------|
| Composition contract | `manifest.py`, `AgentBinding.mount` |
| Registry wiring | `host/wiring.py` → `build_application_registry` |
| Tool catalog | `host/tool_wiring.py` → `ToolProfile` |
| HTTP + MCP | `host/factory.py`, `mcp/server.py` |
| Env + deploy | `.env.example`, `BUILD_AND_DEPLOY.md`, `docker/build-docker.*` |
| Smoke tests | `<pkg>_tests/host/` |

## Verify locally

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest applications/poc_template_application/poc_template_application_tests -q
```

Gate (CI): `uv run pytest -m gate -q`

## Docker (optional)

Default CI **does not** build images (slow). Scripts are validated in gate; optional integration test:

```bash
uv run pytest tests/integration/applications/test_poc_template_docker_build.py -m integration
```

Requires Docker CLI. See [`USAGE.md`](USAGE.md#docker-and-ci).

## Reference hosts

| Application | Profile | Notes |
|-------------|---------|--------|
| `poc_template_application` | lab | Committed scaffold reference |
| `lab_application` | lab | Debug API + integrations lab profile |
| `legal_application` | product | Mature chat/legal serving (extend scaffold product) |
| `research_application` | product | Multi-agent pipeline |

## Engine package

Composition API: [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md)

Plan tracker: [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](../docs/INTERGRAX_IMPLEMENTATION_PLAN.md) — Phase N + Tier-3 readiness table.

# Governed Contractor API (Tier-3)

Scaffolded **product** profile - FastAPI Core (`/health`, `/v1/*`) + ``POST /v1/governed_contractor/run``.

**Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

**Build & deploy:** [`docs/BUILD_AND_DEPLOY.md`](docs/BUILD_AND_DEPLOY.md)

## Three-command quickstart

From **repository root**:

```bash
uv run pytest applications/governed_contractor_application/tests -q
cp applications/governed_contractor_application/.env.example applications/governed_contractor_application/.env
uv run uvicorn governed_contractor_application.host.main:app --host 127.0.0.1 --port 8000
applications/governed_contractor_application/docker/build-docker.sh
```

## Agents

ExternalContractorAdapterAgent

## HTTP

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/governed_contractor/agents
curl -s -X POST http://127.0.0.1:8000/v1/governed_contractor/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"external_contractor.adapt"}'
```

## MCP

Default ``/mcp`` - ``list_agents``, ``run_agent``. Configure ``GOVERNED_CONTRACTOR_INCLUDE_MCP``, ``GOVERNED_CONTRACTOR_MCP_MOUNT_PATH``.

## Extending beyond the generic product skeleton

This host uses ``POST /v1/governed_contractor/run`` and ``/agents``. For chat-style routes,
API-key auth, and domain-specific serving (like Legal), copy patterns from
``applications/legal_application/serving/`` after the scaffold - do not put agent logic here.

## Docs

- Engine: `intergrax/applications/USAGE.md`
- Layout: `applications/USAGE.md`
- Full stack (agent + app): `python -m intergrax.scaffold new-stack <slug>`

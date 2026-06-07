# Dispute Sim API (Tier-3)

Scaffolded **product** profile — FastAPI Core (`/health`, `/v1/*`) + ``POST /v1/dispute_sim/run``.

**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · **Plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

**Build & deploy:** [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md)

## Three-command quickstart

From **repository root**:

```bash
uv run pytest applications/dispute_sim_application/dispute_sim_application_tests -q
cp applications/dispute_sim_application/.env.example applications/dispute_sim_application/.env
uv run uvicorn dispute_sim_application.host.main:app --host 127.0.0.1 --port 8020
applications/dispute_sim_application/docker/build-docker.sh
```

## Agents

DisputeIntakeAgent, DisputeAnalystAgent, DisputeStrategistAgent, DisputeScenarioAgent

## HTTP

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/dispute_sim/agents
curl -s -X POST http://127.0.0.1:8020/v1/dispute_sim/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"dispute_intake.basic"}'
```

## MCP

Default ``/mcp`` — ``list_agents``, ``run_agent``. Configure ``DISPUTE_SIM_INCLUDE_MCP``, ``DISPUTE_SIM_MCP_MOUNT_PATH``.

## Extending beyond the generic product skeleton

This host uses ``POST /v1/dispute_sim/run`` and ``/agents``. For chat-style routes,
API-key auth, and domain-specific serving (like Legal), copy patterns from
``applications/legal_application/serving/`` after the scaffold — do not put agent logic here.

## Docs

- Engine: `intergrax/applications/USAGE.md`
- Layout: `applications/USAGE.md`
- Full stack (agent + app): `python -m intergrax.scaffold new-stack <slug>`

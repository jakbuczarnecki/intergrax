# Dispute Simulation Workspace API (Tier-3)

**DSW** - multi-agent product host for dispute material intake, argument analysis, strategy, and court-process simulation.

**Architecture:** [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · **Agents:** [`agents/README.md`](../../agents/README.md)

**Build & deploy:** [`BUILD_AND_DEPLOY.md`](docs/BUILD_AND_DEPLOY.md)

## Agents

| Agent | Capability |
|-------|------------|
| DisputeIntakeAgent | `dispute.intake` |
| DisputeAnalystAgent | `dispute.analyze` |
| DisputeStrategistAgent | `dispute.strategy` |
| DisputeScenarioAgent | `dispute.scenario` |

## Quickstart

From **repository root**:

```bash
uv run pytest applications/dispute_sim_application/tests -q
cp applications/dispute_sim_application/.env.example applications/dispute_sim_application/.env
uv run uvicorn dispute_sim_application.host.main:app --host 127.0.0.1 --port 8025
```

## HTTP

```bash
curl -s http://127.0.0.1:8025/health
curl -s http://127.0.0.1:8025/v1/dispute_sim/agents
curl -s -X POST http://127.0.0.1:8025/v1/dispute_sim/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"dispute.intake"}'
```

## MCP

Default `/mcp` - `list_agents`, `run_agent`. Configure `DISPUTE_SIM_INCLUDE_MCP`, `DISPUTE_SIM_MCP_MOUNT_PATH`.

## Docs

- Engine: `intergrax/applications/USAGE.md`
- Layout: `applications/USAGE.md`
- All environments: `applications/README.md`

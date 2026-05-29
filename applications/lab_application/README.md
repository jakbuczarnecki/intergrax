# Intergrax Lab Application (Tier-3)

Universal experimentation environment for the Agent Operating System (Phase L.3).

## Purpose

Run arbitrary registered agents without building a dedicated product application.
Inspect traces, checkpoints, runtime events, partial results, and experiments through the debug API.

## Start

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
```

Or factory mode:

```bash
uv run uvicorn lab_application.host.factory:create_lab_application --factory --host 127.0.0.1 --port 8090
```

## Execute an agent

```bash
curl -s -X POST http://127.0.0.1:8090/v1/lab/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello lab","capability":"echo.basic"}'
```

## Inspect

| Endpoint | Purpose |
|----------|---------|
| `GET /v1/lab/agents` | Registered agents and capabilities |
| `GET /debug/tasks/{task_id}` | Run metadata and trace |
| `GET /debug/tasks/{task_id}/events` | Runtime events |
| `GET /debug/tasks/{task_id}/checkpoints` | Checkpoint history |
| `GET /debug/tasks/{task_id}/progress` | Partial results (long-running) |
| `POST /debug/human-response` | HITL approve / reject / escalate |
| `POST /debug/interactions/intake?execute=true` | Interaction → task → Nexus |
| `GET/POST /debug/experiments` | Experiment registry |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LAB_INCLUDE_ECHO` | `true` | Register EchoAgent |
| `LAB_INCLUDE_MOCK_AGENTS` | `true` | Register lab mock agents |
| `LAB_INCLUDE_RESEARCH` | `false` | Register Research + Summary |
| `LAB_ROUTE_PREFIX` | `/v1/lab` | Lab run API prefix |

## Architecture

- **Tier-2 agents** live in `agents/` (Echo, lab mocks, future business agents)
- **Tier-3 lab application** composes registry + Nexus + debug surface
- Agent logic never belongs in this application — only wiring and routes

See [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) — single canonical guide (Step 4C for lab registration).

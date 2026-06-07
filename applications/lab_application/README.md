# Intergrax Lab Application (Tier-3)

Universal experimentation environment for the Agent Operating System (Phase L.3).

**Build & deploy:** [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) · **Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · **Plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

## Purpose

Run arbitrary registered agents without building a dedicated product application.
Inspect traces, checkpoints, runtime events, partial results, and experiments through the debug API.

## Start

```bash
cp applications/lab_application/.env.example applications/lab_application/.env
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
```

Or factory mode:

```bash
uv run uvicorn lab_application.host.factory:create_lab_application --factory --host 127.0.0.1 --port 8090
```

## MCP (FastMCP)

FastMCP is mounted on the same uvicorn process as FastAPI (default `/mcp`).
Tools: `list_agents`, `run_agent` — same Nexus loop as HTTP. Configure `LAB_INCLUDE_MCP`, `LAB_MCP_MOUNT_PATH`.

## Harness stack (Phase M.9 + M.10)

Enable the agent harness integration profile and tools:

```bash
# factory / uvicorn — set LAB_HARNESS=true in .env
LAB_HARNESS=true
```

Or programmatically:

```python
from lab_application.host.integration_wiring import wire_lab_integrations
from lab_application.host.tool_wiring import wire_lab_tools

wiring = wire_lab_integrations(settings=settings, harness=True)
tools = wire_lab_tools(integration_profile=wiring.profile, harness=True)
```

Profile: `IntegrationProfile.harness_lab()` — **composite observability** (Sentry `errors.capture` + LangSmith `observability.query_traces`), **PagerDuty** notification adapter for HITL escalation (`notify_channel="pagerduty"` on long-running tasks), SQLite persistence.

HITL escalation path: runtime `NexusLoop` → `LongRunningCoordinator.notify_escalation()` → profile-resolved PagerDuty adapter (no `INTERGRAX_NOTIFICATION_BACKEND` override required when `harness=True`).

**M.11 default notify channel:** harness hosts inject `default_long_running_notify_channel` (`pagerduty`) into long-running tasks via `make_lab_harness_task_enricher()` — lab run API (`long_running: true`) and interaction intake; no per-task `notify_channel` required.

---

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
| `LAB_INTERACTION_SURFACE` | `auto` | `auto`, `lab_json`, `slack`, `teams` — via `IntegrationProfile` + interaction factory |
| `LAB_HARNESS` | `false` | `true` — `IntegrationProfile.harness_lab()` (Sentry + LangSmith + PagerDuty tools) |

## Integrations (Phase M.8)

Lab composes Tier-0 backends through ``IntegrationProfile.lab()``:

- **sqlite** — trace, events, checkpoints, experiments (``wire_lab_integrations()``)
- **log** — outbound notifications (no network)
- **lab_json** — interaction surface when ``LAB_INTERACTION_SURFACE=lab_json`` (default intake uses ``auto`` for Slack/Teams parity tests)

See ``applications/lab_application/host/integration_wiring.py``.

## Docker

```bash
applications/lab_application/docker/build-docker.sh
# Windows: applications\lab_application\docker\build-docker.bat
```

See [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md).

## Tests

```bash
uv run pytest applications/lab_application/lab_application_tests -q
```

## Architecture

- **Tier-2 agents** live in `agents/` (Echo, lab mocks, future business agents)
- **Tier-3 lab application** composes registry + Nexus + debug surface
- Agent logic never belongs in this application — only wiring and routes

See [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) — single canonical guide (Step 4C for lab registration).

**Tier-3 wiring:** [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md) (engine) · [`applications/USAGE.md`](../USAGE.md) (application layout).

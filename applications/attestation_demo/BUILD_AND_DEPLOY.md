# Build & deploy — Attestation Demo

Tier-3 application package: `applications/attestation_demo/`. Operational runbook for local development, verification, and container deployment.

> Quick overview: [`README.md`](README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Python deps from repo root `pyproject.toml` / `uv.lock` |
| Repo clone | Monorepo; **build context is always repository root** |
| Docker (optional) | Image build via `docker/` |
| Docker Buildx (recommended) | Per-app `.dockerignore` via `--ignorefile` |

Tier-2 agents used by this host: **boundary_demo** (under `agents/` on `PYTHONPATH`).

---

## 1. Configuration

```bash
cp applications/attestation_demo/.env.example applications/attestation_demo/.env
```

Edit `.env` (gitignored). Variables use the application prefix **`ATTESTATION_DEMO_`**.

| Variable | Default | Role |
|----------|---------|------|
| `INTERGRAX_ENV` | `dev` | `prod` for production-like runs |
| `ATTESTATION_DEMO_BACKEND_HOST` | `127.0.0.1` | Bind address |
| `ATTESTATION_DEMO_BACKEND_PORT` | `8097` | HTTP port |

Agent roster and wiring: `manifest.py`, `host/wiring.py`, `host/integration_wiring.py`, `host/tool_wiring.py`.

---

## 2. Local run (development)

From **repository root**:

```bash
uv run uvicorn attestation_demo.host.main:app --host 127.0.0.1 --port 8097
```

Or use the module CLI:

```bash
uv run python -m attestation_demo.host.main
```

### Smoke check

```bash
curl -s http://127.0.0.1:8097/v1/attestation_demo/agents
```

### Partner PoC trigger (primary)

```bash
curl -s -X POST http://127.0.0.1:8097/v1/attestation_demo/poc/run \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Partner PoC sample",
    "capability": "attestation.demo",
    "partition_key": "attestation_demo",
    "row_key": "poc-001",
    "record_data": { "title": "PoC report", "version": 1 }
  }'
```

Response includes `boundary_events[]` (unsigned) and `trust_model`.

MCP is **disabled** by default for this host (`ATTESTATION_DEMO_INCLUDE_MCP=false`).

---

## 7. Partner deploy (AgentReceipt / Cullen)

Handoff package: [`partner_handoff/README.md`](partner_handoff/README.md)

| Deliverable | Path |
|-------------|------|
| Sample request | `partner_handoff/poc_run_request.v1.json` |
| Sample response shape | `partner_handoff/poc_run_response.v1.json` |
| Integration guide | `partner_handoff/README.md` |

### Recommended production posture

1. Set `INTERGRAX_HARNESS_API_KEY` in `.env` (or container env).
2. Expose port **8097** (or reverse-proxy to `/v1/attestation_demo/poc/run`).
3. Share base URL + API key with partner adapter repo.
4. Partner calls `POST /v1/attestation_demo/poc/run` → maps `boundary_events[]` → `createSignedReceipt` (`client_observed`).

```bash
curl -s -X POST "https://<host>/v1/attestation_demo/poc/run" \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: $INTERGRAX_HARNESS_API_KEY" \
  -d @applications/attestation_demo/partner_handoff/poc_run_request.v1.json
```

Journal comparison (same host): `GET /debug/tasks/{run_id}/trace`

---

## 3. Verify before deploy

```bash
uv run pytest applications/attestation_demo/attestation_demo_tests -q
uv run pytest tests/unit/runtime/attestation/ -q
```

Gate (repo CI):

```bash
uv run pytest -m gate -q
```

---

## 4. Container image

Build context = **monorepo root** (`.`).

### Build scripts (recommended)

From **repository root**:

```bash
applications/attestation_demo/docker/build-docker.sh
```

Windows:

```bat
applications\attestation_demo\docker\build-docker.bat
```

### Compose

```bash
docker compose -f applications/attestation_demo/docker/docker-compose.yml up --build
```

### Manual run

```bash
docker run --rm --env-file applications/attestation_demo/.env -p 8097:8097 attestation-demo
```

---

## 5. Dependencies (pyproject.toml)

Base install from repo root (`uv sync`). No application-specific extras beyond the monorepo harness stack.

| Extra | When needed |
|-------|-------------|
| `dev-ci` | Local gate tests before deploy |
| `harness-author` | Scaffold / expand only |

---

## 6. Architecture decisions

Application ADRs: [`adr/README.md`](adr/README.md)  
Platform EBE canon: [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

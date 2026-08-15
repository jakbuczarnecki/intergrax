# Build & deploy — Attestation Demo

Tier-3 application package: `applications/attestation_demo`. Operational runbook for local development, verification, and container deployment.

> Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../../intergrax/applications/USAGE.md)

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Workspace lock + application project `applications/attestation_demo/pyproject.toml` |
| Repo clone | Monorepo; **build context is always repository root** |
| Docker (optional) | Image build via `docker` |
| Docker Buildx (recommended) | Per-app `.dockerignore` via `--ignorefile` |

Tier-2 agents used by this host: **boundary_demo** (under `agents` on `PYTHONPATH`).

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

Response includes `boundary_events[]` (host-signed by default, EBE-9) and `trust_model`.

MCP is **disabled** by default for this host (`ATTESTATION_DEMO_INCLUDE_MCP=false`).

---

## 7. Partner deploy (BoundaryAttest)

Handoff package: [`partner_handoff/README.md`](../partner_handoff/README.md)

| Deliverable | Path |
|-------------|------|
| Sample request | `partner_handoff/poc_run_request.v1.json` |
| Sample response shape | `partner_handoff/poc_run_response.v2.json` (unsigned ref) · `partner_handoff/ebe9_golden_vector.v1.json` (signed) |
| Integration guide | `partner_handoff/README.md` · `partner_handoff/EBE-9_HOST_SIGNING.md` |

### Recommended production posture

1. Set `INTERGRAX_HARNESS_API_KEY` in `.env` (or container env).
2. Expose port **8097** (or reverse-proxy to `/v1/attestation_demo/poc/run`).
3. Share base URL + API key with partner adapter repo.
4. Partner calls `POST /v1/attestation_demo/poc/run` → verifies `host_attestation` per event → maps `boundary_events[]` → `createSignedReceipt` (`client_observed` wrapper).

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
uv run pytest applications/attestation_demo/tests -q
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

## Application dependency project

Canonical packaging: [docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../../docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md).

```bash
uv sync --project applications/attestation_demo
uv run --project applications/attestation_demo python -m attestation_demo.host.main
```

The application `pyproject.toml` selects Intergrax platform extras. Docker uses the same application project (`uv sync --frozen --no-dev --project applications/attestation_demo`); do not pass root `--extra` flags in the Dockerfile.

## Application runtime graph (isolated images)

Canonical packaging and image isolation: [docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md](../../../docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md).

```bash
uv sync --project applications/attestation_demo
uv run python scripts/build/build_application_image.py --application attestation_demo --tag attestation-demo-application:local
```

Compose uses docker/runtime-context/ produced by the same builder (--context-dir ... --keep-context). Do not build with repository-root context.

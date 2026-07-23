# Build & deploy — Governed Contractor

Tier-3 application package: `applications/governed_contractor_application/`. This document is the **operational runbook** for local development, verification, and container deployment.

> Vertical: Governed External Contractor (GEC) — architecture [`ARCHITECTURE.md`](ARCHITECTURE.md) · plan [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · partner [`PARTNER_HANDOFF.md`](PARTNER_HANDOFF.md)  
> Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

**Note:** GEC-0 is scaffold + documentation. Domain proof APIs arrive in later phases. Local `/health` and scaffold `/run` are smoke surfaces only — not a production or partner contract claim.

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Workspace lock + application project `applications/governed_contractor_application/pyproject.toml` |
| Repo clone | Monorepo; **build context is always repository root** |
| Docker (optional) | Image build via `docker/` |
| Docker Buildx (recommended) | Per-app `.dockerignore` via `--ignorefile` |

Tier-2 agents used by this host: **external_contractor_adapter** (under `agents/` on `PYTHONPATH`).

---

## 1. Configuration

```bash
cp applications/governed_contractor_application/.env.example applications/governed_contractor_application/.env
```

Edit `.env` (gitignored). Variables use the application prefix **`GOVERNED_CONTRACTOR_`** — do not put app secrets only in the repository-root `.env`.

| Variable | Default | Role |
|----------|---------|------|
| `INTERGRAX_ENV` | `dev` | `prod` for production-like runs |
| `GOVERNED_CONTRACTOR_BACKEND_HOST` | see `.env.example` | Bind address |
| `GOVERNED_CONTRACTOR_BACKEND_PORT` | `8000` | HTTP port |

Agent roster and integrations: `manifest.py`, `host/wiring.py`, `host/integration_wiring.py`.

Partner endpoint credentials (future GEC-10) must stay in env/config — never in `intergrax/` core.

---

## 2. Local run (development)

From **repository root**:

```bash
uv run uvicorn governed_contractor_application.host.main:app --host 127.0.0.1 --port 8000
```

Or use the module CLI (reads `GOVERNED_CONTRACTOR_BACKEND_*` from `.env`):

```bash
uv run python -m governed_contractor_application.host.main
```

### Smoke check

```bash
curl -s http://127.0.0.1:8000/health
```

### Product API

Routes are mounted under `/v1/governed_contractor`. Default capability: `external_contractor.adapt`. See `serving/` and application README.

---

## 3. Verify before deploy

```bash
uv run pytest agents/external_contractor_adapter/tests -q
uv run pytest applications/governed_contractor_application/tests -q
uv run pytest tests/unit/applications/test_application_deploy_triad.py -q -k governed_contractor
```

Gate (repo CI):

```bash
uv run pytest -m gate -q
```

---

## 4. Container image

Build context = **monorepo root** (`.`). Dockerfile lives under this application only as a path reference.

### Build scripts (recommended)

```bash
# Linux / macOS / Git Bash
applications/governed_contractor_application/docker/build-docker.sh

# Windows (cmd)
applications\governed_contractor_application\docker\build-docker.bat
```

Override image tag: `IMAGE_TAG=my-registry/governed_contractor:1.0.0` (sh) or `build-docker.bat my-registry/governed_contractor:1.0.0` (bat).

### Manual — BuildKit

```bash
docker buildx build -f applications/governed_contractor_application/docker/Dockerfile \
  --ignorefile applications/governed_contractor_application/docker/.dockerignore \
  -t governed_contractor-application .
```

### Manual — classic Docker

```bash
cp applications/governed_contractor_application/docker/.dockerignore .dockerignore
docker build -f applications/governed_contractor_application/docker/Dockerfile -t governed_contractor-application .
```

**Notes:**

- First build can take several minutes (full `uv sync` inside the image).
- The image adjusts `tool.uv.environments` from `win32` to `linux` during build (dev lockfile targets Windows).
- Image `HEALTHCHECK` probes `/health`.
- Scripts use BuildKit when `docker buildx` is available; otherwise they fall back to `docker build`.
- Dockerfile runs a **build-time factory smoke** (MCP/scheduler/interactions disabled) before the runtime stage.
- `docker-compose.yml` sets an explicit Compose project `name:` so Docker Desktop does not label the stack `docker`.

---

## 5. Run container

```bash
docker run --rm \
  --env-file applications/governed_contractor_application/.env \
  -e INTERGRAX_ENV=prod \
  -e GOVERNED_CONTRACTOR_BACKEND_HOST=0.0.0.0 \
  -e GOVERNED_CONTRACTOR_BACKEND_PORT=8000 \
  -p 8000:8000 \
  governed_contractor-application
```

### Docker Compose

From **repository root**:

```bash
docker compose -f applications/governed_contractor_application/docker/docker-compose.yml up --build
```

Ensure `applications/governed_contractor_application/.env` exists (compose uses `env_file: ../.env`).

When the application ships Ollama bootstrap helpers, prefer:

```bash
applications/governed_contractor_application/scripts/build-local-docker.sh
# Windows: applications\governed_contractor_application\scripts\build-local-docker.bat
```

---

## Platform scaffolding principles

- **Roster isolation:** product application startup must not depend on unrelated reference/demo agents.
- **Environment-scoped capability graph:** default runtime builds the graph from manifest + environment registry snapshot.
- **Optional MCP:** HTTP-only startup must not import MCP stacks unless explicitly enabled via `GOVERNED_CONTRACTOR_INCLUDE_MCP=true`.
- **Minimal Docker closure:** copy only agent packages required by the application roster.

---

## 6. Production checklist

- [ ] `INTERGRAX_ENV=prod` and application-prefixed secrets in orchestrator / `.env`, not committed
- [ ] `GOVERNED_CONTRACTOR_*` reviewed against `host/settings.py`
- [ ] Image tagged and pushed to your registry
- [ ] Health check wired to `GET /health`
- [ ] Agent roster in `manifest.py` matches agents copied in `docker/Dockerfile` / `.dockerignore`
- [ ] No unsupported production-readiness claims for GEC until GEC-11 evidence exists

---

## 7. Troubleshooting

| Issue | What to try |
|-------|-------------|
| `unknown flag: --ignorefile` | Use **Buildx** or copy `docker/.dockerignore` to repo root |
| Import errors for agents | Confirm `agents/external_contractor_adapter/` is listed in `docker/.dockerignore` exceptions |
| Slow rebuild | Use BuildKit cache; avoid copying whole repo without per-app `.dockerignore` |
| Wrong capability on `/run` | Use `external_contractor.adapt` (not `<slug>.basic`) |

---

*Tier-3 product scaffold runbook — GEC vertical.*


## Application dependency project

Canonical packaging: [docs/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../../docs/architecture/APPLICATION_DEPENDENCY_MODEL.md).

`ash
uv sync --project applications/governed_contractor_application
uv run --project applications/governed_contractor_application python -m governed_contractor_application.host.main
`

The application pyproject.toml selects Intergrax platform extras. Docker uses the same project (uv sync --frozen --no-dev --project applications/governed_contractor_application); do not pass root --extra flags in the Dockerfile.

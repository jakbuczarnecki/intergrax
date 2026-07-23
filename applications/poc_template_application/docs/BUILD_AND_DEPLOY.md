        # Build & deploy — POC Template Application

        Tier-3 application package: ``applications/poc_template_application/``. This document is the **operational runbook** for local development, verification, and container deployment.

        > Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

        ---

        ## Prerequisites

        | Tool | Purpose |
        |------|---------|
        | [uv](https://docs.astral.sh/uv/) | Python deps from repo root ``pyproject.toml`` / ``uv.lock`` |
        | Repo clone | Monorepo; **build context is always repository root** |
        | Docker (optional) | Image build via ``docker/`` |
        | Docker Buildx (recommended) | Per-app ``.dockerignore`` via ``--ignorefile`` |

        Tier-2 agents used by this host: **echo** (under ``agents/`` on ``PYTHONPATH``).

        ---

        ## 1. Configuration

        ```bash
        cp applications/poc_template_application/.env.example applications/poc_template_application/.env
        ```

        Edit ``.env`` (gitignored). Variables use the application prefix **`POC_TEMPLATE_`** — do not put app secrets only in the repository-root ``.env``.

        | Variable | Default | Role |
        |----------|---------|------|
        | ``INTERGRAX_ENV`` | ``dev`` | ``prod`` for production-like runs |
        | ``POC_TEMPLATE_BACKEND_HOST`` | see ``.env.example`` | Bind address |
        | ``POC_TEMPLATE_BACKEND_PORT`` | ``8095`` | HTTP port |

        Agent roster and integrations: ``manifest.py``, ``host/wiring.py``, ``host/integration_wiring.py``.

        ---

        ## 2. Local run (development)

        From **repository root**:

        ```bash
        uv run uvicorn poc_template_application.host.main:app --host 127.0.0.1 --port 8095
        ```

        Or use the module CLI (reads ``POC_TEMPLATE_BACKEND_*`` from ``.env``):

        ```bash
        uv run python -m poc_template_application.host.main
        ```

        ### Smoke check

        ```bash
        curl -s http://127.0.0.1:8095/v1/poc_template/agents
        ```

        ### Execute an agent

```bash
curl -s -X POST http://127.0.0.1:8095/v1/poc_template/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"echo.basic"}'
```

### MCP (FastMCP + FastAPI)

When ``POC_TEMPLATE_INCLUDE_MCP=true`` (default), FastMCP is mounted at
``POC_TEMPLATE_MCP_MOUNT_PATH`` (default ``/mcp``) on the **same** uvicorn process.
Tools ``list_agents`` and ``run_agent`` use the same Nexus loop as HTTP.

MCP endpoint: ``http://127.0.0.1:8095/mcp`` (streamable HTTP transport).

        ---

        ## 3. Verify before deploy

        ```bash
        uv run pytest applications/poc_template_application/tests -q
        uv run pytest tests/unit/applications/ -q -k "poc_template" --ignore-glob="*" 2>/dev/null || true
        ```

        Gate (repo CI):

        ```bash
        uv run pytest -m gate -q
        ```

        ---

        ## 4. Container image

        Build context = **monorepo root** (``.``). Dockerfile lives under this application only as a path reference.

        ### Build scripts (recommended)

        Run from **repository root** or from ``applications/poc_template_application/docker/`` (scripts ``cd`` to repo root):

        ```bash
        # Linux / macOS / Git Bash
        applications/poc_template_application/docker/build-docker.sh

        # Windows (cmd)
        applications\poc_template_application\docker\build-docker.bat
        ```

        Override image tag: ``IMAGE_TAG=my-registry/poc_template:1.0.0`` (sh) or ``build-docker.bat my-registry/poc_template:1.0.0`` (bat).

        ### Manual — BuildKit

        ```bash
        docker buildx build -f applications/poc_template_application/docker/Dockerfile \
          --ignorefile applications/poc_template_application/docker/.dockerignore \
          -t poc_template-application .
        ```

        ### Manual — classic Docker

        ```bash
        cp applications/poc_template_application/docker/.dockerignore .dockerignore
        docker build -f applications/poc_template_application/docker/Dockerfile -t poc_template-application .
        ```

        **Notes:**

        - First build can take several minutes (full ``uv sync`` inside the image).
        - The image adjusts ``tool.uv.environments`` from ``win32`` to ``linux`` during build (dev lockfile targets Windows).
        - Image ``HEALTHCHECK`` probes ``/v1/poc_template/agents``.
        - Scripts use BuildKit when ``docker buildx`` is available; otherwise they fall back to ``docker build``.

        ---

        ## 5. Run container

        ```bash
        docker run --rm \
          --env-file applications/poc_template_application/.env \
          -e INTERGRAX_ENV=prod \
          -e POC_TEMPLATE_BACKEND_HOST=0.0.0.0 \
          -e POC_TEMPLATE_BACKEND_PORT=8095 \
          -p 8095:8095 \
          poc_template-application
        ```

        ### Docker Compose

        From **repository root**:

        ```bash
        docker compose -f applications/poc_template_application/docker/docker-compose.yml up --build
        ```

        Ensure ``applications/poc_template_application/.env`` exists (compose uses ``env_file: ../.env``).

        ---

        ## 6. Production checklist

        - [ ] ``INTERGRAX_ENV=prod`` and application-prefixed secrets in orchestrator / ``.env``, not committed
        - [ ] ``POC_TEMPLATE_*`` reviewed against ``host/settings.py``
        - [ ] Image tagged and pushed to your registry: ``docker tag poc_template-application <registry>/poc_template-application:<version>``
        - [ ] Health check wired to ``GET /v1/poc_template/agents`` (or orchestrator equivalent)
        - [ ] Agent roster in ``manifest.py`` matches agents copied in ``docker/Dockerfile`` / ``.dockerignore``

        ---

        ## 7. Troubleshooting

        | Issue | What to try |
        |-------|-------------|
        | ``unknown flag: --ignorefile`` | Use **Buildx** or copy ``docker/.dockerignore`` to repo root |
        | Import errors for agents | Confirm ``agents/<slug>/`` is listed in ``docker/.dockerignore`` exceptions |
        | Slow rebuild | Use BuildKit cache; avoid copying whole repo without per-app ``.dockerignore`` |
        | Wrong agents in registry | Check ``manifest.py`` flags / ``host/wiring.py`` and ``LAB_INCLUDE_*`` (lab) |

        ---

        *Generated for Intergrax Tier-3 scaffold (profile: lab).*


## Application dependency project

Canonical packaging: [docs/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../../docs/architecture/APPLICATION_DEPENDENCY_MODEL.md).

`ash
uv sync --project applications/poc_template_application
uv run --project applications/poc_template_application python -m poc_template_application.host.main
`

The application pyproject.toml selects Intergrax platform extras. Docker uses the same project (uv sync --frozen --no-dev --project applications/poc_template_application); do not pass root --extra flags in the Dockerfile.

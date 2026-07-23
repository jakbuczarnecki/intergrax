        # Build & deploy — Intergrax Assistant

        Tier-3 application package: ``applications/intergrax_assistant_application/``. This document is the **operational runbook** for local development, verification, and container deployment.

        > Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

        ---

        ## Prerequisites

        | Tool | Purpose |
        |------|---------|
        | [uv](https://docs.astral.sh/uv/) | Python deps from repo root ``pyproject.toml`` / ``uv.lock`` |
        | Repo clone | Monorepo; **build context is always repository root** |
        | Docker (optional) | Image build via ``docker/`` |
        | Docker Buildx (recommended) | Per-app ``.dockerignore`` via ``--ignorefile`` |

        Tier-2 agents used by this host: **intergrax_assistant** (under ``agents/`` on ``PYTHONPATH``).

        ---

        ## 1. Configuration

        ```bash
        cp applications/intergrax_assistant_application/.env.example applications/intergrax_assistant_application/.env
        ```

        Edit ``.env`` (gitignored). Variables use the application prefix **`INTERGRAX_ASSISTANT_`** — do not put app secrets only in the repository-root ``.env``.

        | Variable | Default | Role |
        |----------|---------|------|
        | ``INTERGRAX_ENV`` | ``dev`` | ``prod`` for production-like runs |
        | ``INTERGRAX_ASSISTANT_BACKEND_HOST`` | see ``.env.example`` | Bind address |
        | ``INTERGRAX_ASSISTANT_BACKEND_PORT`` | ``8096`` | HTTP port |

        Agent roster and integrations: ``manifest.py``, ``host/wiring.py``, ``host/integration_wiring.py``.

        ---

        ## 2. Local run (development)

        From **repository root**:

        ```bash
        uv run uvicorn intergrax_assistant_application.host.main:app --host 127.0.0.1 --port 8096
        ```

        Or use the module CLI (reads ``INTERGRAX_ASSISTANT_BACKEND_*`` from ``.env``):

        ```bash
        uv run python -m intergrax_assistant_application.host.main
        ```

        ### Smoke check

        ```bash
        curl -s http://127.0.0.1:8096/v1/intergrax_assistant/agents
        ```

        ### Execute an agent

```bash
curl -s -X POST http://127.0.0.1:8096/v1/intergrax_assistant/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"platform.assist"}'
```

### MCP (FastMCP + FastAPI)

When ``INTERGRAX_ASSISTANT_INCLUDE_MCP=true`` (default), FastMCP is mounted at
``INTERGRAX_ASSISTANT_MCP_MOUNT_PATH`` (default ``/mcp``) on the **same** uvicorn process.
Tools ``list_agents`` and ``run_agent`` use the same Nexus loop as HTTP.

MCP endpoint: ``http://127.0.0.1:8096/mcp`` (streamable HTTP transport).

        ---

        ## 3. Verify before deploy

        ```bash
        uv run pytest applications/intergrax_assistant_application/tests -q
        uv run pytest tests/unit/applications/ -q -k "intergrax_assistant" --ignore-glob="*" 2>/dev/null || true
        ```

        Gate (repo CI):

        ```bash
        uv run pytest -m gate -q
        ```

        ---

        ## 4. Container image

        Build context = **monorepo root** (``.``). Dockerfile lives under this application only as a path reference.

        ### Build scripts (recommended)

        Run from **repository root** or from ``applications/intergrax_assistant_application/docker/`` (scripts ``cd`` to repo root):

        ```bash
        # Linux / macOS / Git Bash
        applications/intergrax_assistant_application/docker/build-docker.sh

        # Windows (cmd)
        applications\intergrax_assistant_application\docker\build-docker.bat
        ```

        Override image tag: ``IMAGE_TAG=my-registry/intergrax_assistant:1.0.0`` (sh) or ``build-docker.bat my-registry/intergrax_assistant:1.0.0`` (bat).

        ### Manual — BuildKit

        ```bash
        docker buildx build -f applications/intergrax_assistant_application/docker/Dockerfile \
          --ignorefile applications/intergrax_assistant_application/docker/.dockerignore \
          -t intergrax_assistant-application .
        ```

        ### Manual — classic Docker

        ```bash
        cp applications/intergrax_assistant_application/docker/.dockerignore .dockerignore
        docker build -f applications/intergrax_assistant_application/docker/Dockerfile -t intergrax_assistant-application .
        ```

        **Notes:**

        - First build can take several minutes (full ``uv sync`` inside the image).
        - The image adjusts ``tool.uv.environments`` from ``win32`` to ``linux`` during build (dev lockfile targets Windows).
        - Image ``HEALTHCHECK`` probes ``/v1/intergrax_assistant/agents``.
        - Scripts use BuildKit when ``docker buildx`` is available; otherwise they fall back to ``docker build``.

        ---

        ## 5. Run container

        ```bash
        docker run --rm \
          --env-file applications/intergrax_assistant_application/.env \
          -e INTERGRAX_ENV=prod \
          -e INTERGRAX_ASSISTANT_BACKEND_HOST=0.0.0.0 \
          -e INTERGRAX_ASSISTANT_BACKEND_PORT=8096 \
          -p 8096:8096 \
          intergrax_assistant-application
        ```

        ### Docker Compose

        From **repository root**:

        ```bash
        docker compose -f applications/intergrax_assistant_application/docker/docker-compose.yml up --build
        ```

        Ensure ``applications/intergrax_assistant_application/.env`` exists (compose uses ``env_file: ../.env``).

        ---

        ## 6. Production checklist

        - [ ] ``INTERGRAX_ENV=prod`` and application-prefixed secrets in orchestrator / ``.env``, not committed
        - [ ] ``INTERGRAX_ASSISTANT_*`` reviewed against ``host/settings.py``
        - [ ] Image tagged and pushed to your registry: ``docker tag intergrax_assistant-application <registry>/intergrax_assistant-application:<version>``
        - [ ] Health check wired to ``GET /v1/intergrax_assistant/agents`` (or orchestrator equivalent)
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
uv sync --project applications/intergrax_assistant_application
uv run --project applications/intergrax_assistant_application python -m intergrax_assistant_application.host.main
`

The application pyproject.toml selects Intergrax platform extras. Docker uses the same project (uv sync --frozen --no-dev --project applications/intergrax_assistant_application); do not pass root --extra flags in the Dockerfile.

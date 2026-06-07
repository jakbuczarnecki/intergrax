        # Build & deploy — Local Workspace

        Tier-3 application package: ``applications/local_workspace_application/``. This document is the **operational runbook** for local development, verification, and container deployment.

        > Quick overview: [`README.md`](README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

        ---

        ## Prerequisites

        | Tool | Purpose |
        |------|---------|
        | [uv](https://docs.astral.sh/uv/) | Python deps from repo root ``pyproject.toml`` / ``uv.lock`` |
        | Repo clone | Monorepo; **build context is always repository root** |
        | Docker (optional) | Image build via ``docker/`` |
        | Docker Buildx (recommended) | Per-app ``.dockerignore`` via ``--ignorefile`` |

        Tier-2 agents used by this host: **local_indexer, local_search, local_synthesizer** (under ``agents/`` on ``PYTHONPATH``).

        ---

        ## 1. Configuration

        ```bash
        cp applications/local_workspace_application/.env.example applications/local_workspace_application/.env
        ```

        Edit ``.env`` (gitignored). Variables use the application prefix **`LOCAL_WORKSPACE_`** — do not put app secrets only in the repository-root ``.env``.

        | Variable | Default | Role |
        |----------|---------|------|
        | ``INTERGRAX_ENV`` | ``dev`` | ``prod`` for production-like runs |
        | ``LOCAL_WORKSPACE_BACKEND_HOST`` | see ``.env.example`` | Bind address |
        | ``LOCAL_WORKSPACE_BACKEND_PORT`` | ``8020`` | HTTP port |

        Agent roster and integrations: ``manifest.py``, ``host/wiring.py``, ``host/integration_wiring.py``.

        ---

        ## 2. Local run (development)

        From **repository root**:

        ```bash
        uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
        ```

        Or use the module CLI (reads ``LOCAL_WORKSPACE_BACKEND_*`` from ``.env``):

        ```bash
        uv run python -m local_workspace_application.host.main
        ```

        ### Smoke check

        ```bash
        curl -s http://127.0.0.1:8020/health
        ```

        ### Product API

Routes are mounted under ``/v1/local_workspace``. See ``serving/`` and application README for contract details.

        ---

        ## 3. Verify before deploy

        ```bash
        uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
        uv run pytest tests/unit/applications/ -q -k "local_workspace" --ignore-glob="*" 2>/dev/null || true
        ```

        Gate (repo CI):

        ```bash
        uv run pytest -m gate -q
        ```

        ---

        ## 4. Container image

        Build context = **monorepo root** (``.``). Dockerfile lives under this application only as a path reference.

        ### Build scripts (recommended)

        Run from **repository root** or from ``applications/local_workspace_application/docker/`` (scripts ``cd`` to repo root):

        ```bash
        # Linux / macOS / Git Bash
        applications/local_workspace_application/docker/build-docker.sh

        # Windows (cmd)
        applications\local_workspace_application\docker\build-docker.bat
        ```

        Override image tag: ``IMAGE_TAG=my-registry/local_workspace:1.0.0`` (sh) or ``build-docker.bat my-registry/local_workspace:1.0.0`` (bat).

        ### Manual — BuildKit

        ```bash
        docker buildx build -f applications/local_workspace_application/docker/Dockerfile \
          --ignorefile applications/local_workspace_application/docker/.dockerignore \
          -t local_workspace-application .
        ```

        ### Manual — classic Docker

        ```bash
        cp applications/local_workspace_application/docker/.dockerignore .dockerignore
        docker build -f applications/local_workspace_application/docker/Dockerfile -t local_workspace-application .
        ```

        **Notes:**

        - First build can take several minutes (full ``uv sync`` inside the image).
        - The image adjusts ``tool.uv.environments`` from ``win32`` to ``linux`` during build (dev lockfile targets Windows).
        - Image ``HEALTHCHECK`` probes ``/health``.
        - Scripts use BuildKit when ``docker buildx`` is available; otherwise they fall back to ``docker build``.

        ---

        ## 5. Run container

        ```bash
        docker run --rm \
          --env-file applications/local_workspace_application/.env \
          -e INTERGRAX_ENV=prod \
          -e LOCAL_WORKSPACE_BACKEND_HOST=0.0.0.0 \
          -e LOCAL_WORKSPACE_BACKEND_PORT=8020 \
          -p 8020:8020 \
          local_workspace-application
        ```

        ### Docker Compose

        From **repository root**:

        ```bash
        docker compose -f applications/local_workspace_application/docker/docker-compose.yml up --build
        ```

        Ensure ``applications/local_workspace_application/.env`` exists (compose uses ``env_file: ../.env``).

        ---

        ## 6. Production checklist

        - [ ] ``INTERGRAX_ENV=prod`` and application-prefixed secrets in orchestrator / ``.env``, not committed
        - [ ] ``LOCAL_WORKSPACE_*`` reviewed against ``host/settings.py``
        - [ ] Image tagged and pushed to your registry: ``docker tag local_workspace-application <registry>/local_workspace-application:<version>``
        - [ ] Health check wired to ``GET /health`` (or orchestrator equivalent)
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

        *Generated for Intergrax Tier-3 scaffold (profile: product).*

        # Build & deploy — Intergrax Lab

        Tier-3 application package: ``applications/lab_application/``. This document is the **operational runbook** for local development, verification, and container deployment.

        > Quick overview: [`README.md`](README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

        ---

        ## Prerequisites

        | Tool | Purpose |
        |------|---------|
        | [uv](https://docs.astral.sh/uv/) | Python deps from repo root ``pyproject.toml`` / ``uv.lock`` |
        | Repo clone | Monorepo; **build context is always repository root** |
        | Docker (optional) | Image build via ``docker/`` |
        | Docker Buildx (recommended) | Per-app ``.dockerignore`` via ``--ignorefile`` |

        Tier-2 agents used by this host: **echo, lab, signoff_probe, research** (under ``agents/`` on ``PYTHONPATH``).

        ---

        ## 1. Configuration

        ```bash
        cp applications/lab_application/.env.example applications/lab_application/.env
        ```

        Edit ``.env`` (gitignored). Variables use the application prefix **`LAB_`** — do not put app secrets only in the repository-root ``.env``.

        | Variable | Default | Role |
        |----------|---------|------|
        | ``INTERGRAX_ENV`` | ``dev`` | ``prod`` for production-like runs |
        | ``LAB_BACKEND_HOST`` | see ``.env.example`` | Bind address |
        | ``LAB_BACKEND_PORT`` | ``8090`` | HTTP port |

        Agent roster and integrations: ``manifest.py``, ``host/wiring.py``, ``host/integration_wiring.py``, ``host/tool_wiring.py``.

        ### Tool catalog (lab defaults)

        Lab enables ``rag.retrieve``, ``websearch.query``, and ``sandbox.exec`` via ``host/tool_wiring.py``. MCP exposes ``list_catalog_tools`` / ``describe_catalog_tool``. See [`intergrax/tools/USAGE.md`](../../intergrax/tools/USAGE.md).

        ---

        ## 2. Local run (development)

        From **repository root**:

        ```bash
        uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
        ```

        Or use the module CLI (reads ``LAB_BACKEND_*`` from ``.env``):

        ```bash
        uv run python -m lab_application.host.main
        ```

        ### Smoke check

        ```bash
        curl -s http://127.0.0.1:8090/v1/lab/agents
        ```

        ### Execute an agent

```bash
curl -s -X POST http://127.0.0.1:8090/v1/lab/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"echo.basic"}'
```

### MCP (FastMCP + FastAPI)

When ``LAB_INCLUDE_MCP=true`` (default), FastMCP is mounted at
``LAB_MCP_MOUNT_PATH`` (default ``/mcp``) on the **same** uvicorn process.
Tools ``list_agents`` and ``run_agent`` use the same Nexus loop as HTTP.

MCP endpoint: ``http://127.0.0.1:8090/mcp`` (streamable HTTP transport).

### Debug / observability API

Mounted on the same process (Phase L.3 + infra paydown):

| Endpoint | Purpose |
|----------|---------|
| `GET /debug/tasks/{run_id}/trace` | Trace timeline + optional runtime events |
| `GET /debug/tasks/{run_id}/metrics` | Unified metrics export |
| `GET /debug/tasks/{task_id}/events` | Canonical RuntimeEvent stream |
| `GET /debug/notifications/receipts` | Successful outbound delivery receipts |
| `GET /debug/notifications/dead-letters` | Failed notification DLQ rows |

        ---

        ## 3. Verify before deploy

        ```bash
        uv run pytest applications/lab_application/lab_application_tests -q
        uv run pytest tests/unit/applications/ -q -k "lab" --ignore-glob="*" 2>/dev/null || true
        ```

        Gate (repo CI):

        ```bash
        uv run pytest -m gate -q
        ```

        ---

        ## 4. Container image

        Build context = **monorepo root** (``.``). Dockerfile lives under this application only as a path reference.

        ### Build scripts (recommended)

        Run from **repository root** or from ``applications/lab_application/docker/`` (scripts ``cd`` to repo root):

        ```bash
        # Linux / macOS / Git Bash
        applications/lab_application/docker/build-docker.sh

        # Windows (cmd)
        applications\lab_application\docker\build-docker.bat
        ```

        Override image tag: ``IMAGE_TAG=my-registry/lab:1.0.0`` (sh) or ``build-docker.bat my-registry/lab:1.0.0`` (bat).

        ### Manual — BuildKit

        ```bash
        docker buildx build -f applications/lab_application/docker/Dockerfile \
          --ignorefile applications/lab_application/docker/.dockerignore \
          -t lab-application .
        ```

        ### Manual — classic Docker

        ```bash
        cp applications/lab_application/docker/.dockerignore .dockerignore
        docker build -f applications/lab_application/docker/Dockerfile -t lab-application .
        ```

        **Notes:**

        - First build can take several minutes (full ``uv sync`` inside the image).
        - The image adjusts ``tool.uv.environments`` from ``win32`` to ``linux`` during build (dev lockfile targets Windows).
        - Image ``HEALTHCHECK`` probes ``/v1/lab/agents``.
        - Scripts use BuildKit when ``docker buildx`` is available; otherwise they fall back to ``docker build``.

        ---

        ## 5. Run container

        ```bash
        docker run --rm \
          --env-file applications/lab_application/.env \
          -e INTERGRAX_ENV=prod \
          -e LAB_BACKEND_HOST=0.0.0.0 \
          -e LAB_BACKEND_PORT=8090 \
          -p 8090:8090 \
          lab-application
        ```

        ### Docker Compose

        From **repository root**:

        ```bash
        docker compose -f applications/lab_application/docker/docker-compose.yml up --build
        ```

        Ensure ``applications/lab_application/.env`` exists (compose uses ``env_file: ../.env``).

        ---

        ## 6. Production checklist

        - [ ] ``INTERGRAX_ENV=prod`` and application-prefixed secrets in orchestrator / ``.env``, not committed
        - [ ] ``LAB_*`` reviewed against ``host/settings.py``
        - [ ] Image tagged and pushed to your registry: ``docker tag lab-application <registry>/lab-application:<version>``
        - [ ] Health check wired to ``GET /v1/lab/agents`` (or orchestrator equivalent)
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

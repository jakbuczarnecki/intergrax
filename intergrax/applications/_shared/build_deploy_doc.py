# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""``BUILD_AND_DEPLOY.md`` template for Tier-3 applications (Phase N)."""

from __future__ import annotations

from textwrap import dedent

def _display_name(short: str) -> str:
    return " ".join(part.capitalize() for part in short.split("_"))


def render_build_deploy_doc(
    *,
    pkg: str,
    short: str,
    port: int,
    env_prefix: str,
    route_prefix: str,
    profile: str,
    agent_dirs: list[str],
    example_capability: str,
    health_path: str | None = None,
    tests_pkg: str | None = None,
    display: str | None = None,
) -> str:
    """
    Render application-local build & deploy guide.

    *profile* is ``lab`` or ``product`` (affects smoke checks and run API notes).
    """
    agents_csv = ", ".join(agent_dirs)
    tests_dir = tests_pkg or f"{pkg}_tests"
    display_title = display or _display_name(short)
    health = health_path or (
        f"{route_prefix}/agents" if profile == "lab" else "/health"
    )
    run_section = ""
    if profile == "lab":
        run_section = dedent(
            f"""\
            ### Execute an agent

            ```bash
            curl -s -X POST http://127.0.0.1:{port}{route_prefix}/run \\
              -H "Content-Type: application/json" \\
              -d '{{"message":"hello","capability":"{example_capability}"}}'
            ```

            ### MCP (FastMCP + FastAPI)

            When ``{env_prefix}INCLUDE_MCP=true`` (opt-in; default **false**), FastMCP is mounted at
            ``{env_prefix}MCP_MOUNT_PATH`` (default ``/mcp``) on the **same** uvicorn process.
            Tools ``list_agents`` and ``run_agent`` use the same Nexus loop as HTTP.

            MCP endpoint: ``http://127.0.0.1:{port}/mcp`` (streamable HTTP transport).
            """
        )
    elif route_prefix:
        run_section = dedent(
            f"""\
            ### Product API

            Routes are mounted under ``{route_prefix}``. See ``serving/`` and application README for contract details.
            """
        )

    return dedent(
        f"""\
        # Build & deploy — {display_title}

        Tier-3 application package: ``applications/{pkg}/``. This document is the **operational runbook** for local development, verification, and container deployment.

        > Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

        ---

        ## Prerequisites

        | Tool | Purpose |
        |------|---------|
        | [uv](https://docs.astral.sh/uv/) | Python deps from repo root ``pyproject.toml`` / ``uv.lock`` |
        | Repo clone | Monorepo; **build context is always repository root** |
        | Docker (optional) | Image build via ``docker/`` |
        | Docker Buildx (recommended) | Per-app ``.dockerignore`` via ``--ignorefile`` |

        Tier-2 agents used by this host: **{agents_csv}** (under ``agents/`` on ``PYTHONPATH``).

        ---

        ## 1. Configuration

        ```bash
        cp applications/{pkg}/.env.example applications/{pkg}/.env
        ```

        Edit ``.env`` (gitignored). Variables use the application prefix **`{env_prefix}`** — do not put app secrets only in the repository-root ``.env``.

        | Variable | Default | Role |
        |----------|---------|------|
        | ``INTERGRAX_ENV`` | ``dev`` | ``prod`` for production-like runs |
        | ``{env_prefix}BACKEND_HOST`` | see ``.env.example`` | Bind address |
        | ``{env_prefix}BACKEND_PORT`` | ``{port}`` | HTTP port |

        Agent roster and integrations: ``manifest.py``, ``host/wiring.py``, ``host/integration_wiring.py``.

        ---

        ## 2. Local run (development)

        From **repository root**:

        ```bash
        uv run uvicorn {pkg}.host.main:app --host 127.0.0.1 --port {port}
        ```

        Or use the module CLI (reads ``{env_prefix}BACKEND_*`` from ``.env``):

        ```bash
        uv run python -m {pkg}.host.main
        ```

        ### Smoke check

        ```bash
        curl -s http://127.0.0.1:{port}{health}
        ```

        {run_section}
        ---

        ## 3. Verify before deploy

        ```bash
        uv run pytest applications/{pkg}/{tests_dir} -q
        uv run pytest tests/unit/applications/ -q -k "{short}" --ignore-glob="*" 2>/dev/null || true
        ```

        Gate (repo CI):

        ```bash
        uv run pytest -m gate -q
        ```

        ---

        ## 4. Container image

        Build context = **monorepo root** (``.``). Dockerfile lives under this application only as a path reference.

        ### Build scripts (recommended)

        Run from **repository root** or from ``applications/{pkg}/docker/`` (scripts ``cd`` to repo root):

        ```bash
        # Linux / macOS / Git Bash
        applications/{pkg}/docker/build-docker.sh

        # Windows (cmd)
        applications\\{pkg}\\docker\\build-docker.bat
        ```

        Override image tag: ``IMAGE_TAG=my-registry/{short}:1.0.0`` (sh) or ``build-docker.bat my-registry/{short}:1.0.0`` (bat).

        ### Manual — BuildKit

        ```bash
        docker buildx build -f applications/{pkg}/docker/Dockerfile \\
          --ignorefile applications/{pkg}/docker/.dockerignore \\
          -t {short}-application .
        ```

        ### Manual — classic Docker

        ```bash
        cp applications/{pkg}/docker/.dockerignore .dockerignore
        docker build -f applications/{pkg}/docker/Dockerfile -t {short}-application .
        ```

        **Notes:**

        - First build can take several minutes (full ``uv sync`` inside the image).
        - The image adjusts ``tool.uv.environments`` from ``win32`` to ``linux`` during build (dev lockfile targets Windows).
        - Image ``HEALTHCHECK`` probes ``{health}``.
        - Scripts use BuildKit when ``docker buildx`` is available; otherwise they fall back to ``docker build``.
        - Dockerfile runs a **build-time factory smoke** (MCP/scheduler/interactions disabled) before the runtime stage.
        - ``docker-compose.yml`` sets an explicit Compose project ``name:`` so Docker Desktop does not label the stack ``docker``.

        ---

        ## 5. Run container

        ```bash
        docker run --rm \\
          --env-file applications/{pkg}/.env \\
          -e INTERGRAX_ENV=prod \\
          -e {env_prefix}BACKEND_HOST=0.0.0.0 \\
          -e {env_prefix}BACKEND_PORT={port} \\
          -p {port}:{port} \\
          {short}-application
        ```

        ### Docker Compose

        From **repository root**:

        ```bash
        docker compose -f applications/{pkg}/docker/docker-compose.yml up --build
        ```

        Ensure ``applications/{pkg}/.env`` exists (compose uses ``env_file: ../.env``).

        When the application ships Ollama bootstrap helpers, prefer:

        ```bash
        applications/{pkg}/scripts/build-local-docker.sh
        # Windows: applications\\{pkg}\\scripts\\build-local-docker.bat
        ```

        Those scripts create ``.env`` from ``.env.example`` when missing, build the image, start Ollama,
        pull the configured model (with retries on Windows via ``docker compose exec -T``), and bring up the stack.

        ---

        ## Platform scaffolding principles (LKW feedback)

        - **Roster isolation:** product application startup must not depend on unrelated reference/demo agents.
        - **Environment-scoped capability graph:** default runtime builds the graph from manifest + environment
          registry snapshot; global catalog graphs are opt-in via explicit ``catalog=...``.
        - **Optional MCP:** HTTP-only startup (``INCLUDE_MCP=false`` default) must not import ``fastmcp``,
          ``mcp``, or ``fastapi_mcp``; enable MCP explicitly via ``{env_prefix}INCLUDE_MCP=true``.
        - **Minimal Docker closure:** copy only agent packages required by the application roster (plus shared packages), never ``COPY agents/ ./agents/`` by default.
        - **Feedback loop:** every LKW runtime/deployment issue should be evaluated as a platform/scaffold feedback signal, not only as an application-local patch.

        ---

        ## 6. Production checklist

        - [ ] ``INTERGRAX_ENV=prod`` and application-prefixed secrets in orchestrator / ``.env``, not committed
        - [ ] ``{env_prefix}*`` reviewed against ``host/settings.py``
        - [ ] Image tagged and pushed to your registry: ``docker tag {short}-application <registry>/{short}-application:<version>``
        - [ ] Health check wired to ``GET {health}`` (or orchestrator equivalent)
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

        *Generated for Intergrax Tier-3 scaffold (profile: {profile}).*
        """
    )

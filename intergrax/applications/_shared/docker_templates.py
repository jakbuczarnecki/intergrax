# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Per-application Docker file templates (Phase N.5)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def _default_smoke_env(*, env_prefix: str) -> tuple[str, ...]:
    return (
        "INTERGRAX_ENV=dev",
        f"{env_prefix}BACKEND_ENV=dev",
        f"{env_prefix}INCLUDE_MCP=false",
        f"{env_prefix}INCLUDE_SCHEDULER=false",
        f"{env_prefix}INCLUDE_INTERACTIONS=false",
        f"{env_prefix}INCLUDE_TASK_CONTROL=false",
        "INTERGRAX_SHADOW_ROOT=/tmp/intergrax_app_shadow",
        "INTERGRAX_SQLITE_DATA_DIR=/tmp/intergrax_app_sqlite",
    )


def render_runtime_graph_dockerfile(
    *,
    pkg: str,
    port: int,
    env_prefix: str,
    health_path: str = "/health",
    uvicorn_module: str | None = None,
    factory_import: str | None = None,
    factory_call: str | None = None,
    smoke_env: tuple[str, ...] | None = None,
    short: str | None = None,
) -> str:
    """Dockerfile for a *materialized* minimal runtime-graph build context."""
    short_name = short or pkg.replace("_application", "").replace("_", "-")
    target = uvicorn_module or f"{pkg}.host.main:app"
    health_url = f"http://127.0.0.1:{port}{health_path}"
    import_stmt = factory_import or f"from {pkg}.host.factory import create_{short_name.replace('-', '_')}_backend_app"
    # Prefer conventional factory names derived from package short id.
    if factory_call is None and factory_import is None:
        # Keep smoke optional-friendly: import package host module only.
        import_stmt = f"import {pkg}"
        factory_expr = "True"
    else:
        factory_expr = factory_call or "True"
    smoke_lines = smoke_env if smoke_env is not None else _default_smoke_env(env_prefix=env_prefix)
    # Keep continuation indent aligned with the template so textwrap.dedent works.
    smoke_env_block = " \\\n            ".join(smoke_lines)
    return dedent(
        f"""\
        # © Artur Czarnecki. All rights reserved.
        # syntax=docker/dockerfile:1
        #
        # Build ONLY from a materialized runtime-graph context produced by:
        #   uv run python scripts/build/build_application_image.py --application {pkg}
        # Do not use the monorepo root as Docker context.
        # Source of truth: applications/{pkg}/pyproject.toml

        FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
        WORKDIR /app

        COPY pyproject.toml uv.lock README.md ./
        COPY .intergrax-runtime-graph.json ./
        COPY intergrax/ ./intergrax/
        COPY applications/ ./applications/
        COPY agents/ ./agents/

        # Install the selected application graph only (agents via workspace deps).
        RUN uv sync --frozen --no-dev --project applications/{pkg} \\
            && rm -rf /root/.cache/uv

        RUN PYTHONPATH=/app:/app/agents:/app/applications \\
            {smoke_env_block} \\
            .venv/bin/python -c "{import_stmt}; assert {factory_expr}"

        FROM python:3.12-slim-bookworm AS runtime
        WORKDIR /app

        ENV PYTHONUNBUFFERED=1 \\
            PYTHONDONTWRITEBYTECODE=1 \\
            PYTHONPATH=/app:/app/agents:/app/applications \\
            INTERGRAX_ENV=prod \\
            {env_prefix}BACKEND_HOST=0.0.0.0 \\
            {env_prefix}BACKEND_PORT={port}

        RUN apt-get update && apt-get install -y --no-install-recommends \\
            libsqlite3-0 \\
            && rm -rf /var/lib/apt/lists/*

        COPY --from=builder /app/.venv /app/.venv
        COPY --from=builder /app/intergrax /app/intergrax
        COPY --from=builder /app/applications /app/applications
        COPY --from=builder /app/agents /app/agents
        COPY --from=builder /app/pyproject.toml /app/uv.lock /app/README.md /app/
        COPY --from=builder /app/.intergrax-runtime-graph.json /app/.intergrax-runtime-graph.json

        ENV PATH="/app/.venv/bin:$PATH"

        EXPOSE {port}
        WORKDIR /app/applications/{pkg}

        HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \\
          CMD python -c "import urllib.request; urllib.request.urlopen('{health_url}')" || exit 1

        CMD ["uvicorn", "{target}", "--host", "0.0.0.0", "--port", "{port}"]
        """
    )


def render_dockerfile(
    *,
    pkg: str,
    short: str,
    port: int,
    env_prefix: str,
    agent_dirs: list[str],
    health_path: str,
    uvicorn_module: str | None = None,
    factory_import: str | None = None,
    factory_call: str | None = None,
    smoke_env: tuple[str, ...] | None = None,
    copy_readme: bool = True,
) -> str:
    """Render application ``docker/Dockerfile`` for materialized runtime-context builds."""
    del agent_dirs, copy_readme  # agent selection comes from application pyproject
    body = render_runtime_graph_dockerfile(
        pkg=pkg,
        port=port,
        env_prefix=env_prefix,
        health_path=health_path,
        uvicorn_module=uvicorn_module,
        factory_import=factory_import,
        factory_call=factory_call,
        smoke_env=smoke_env,
        short=short,
    )
    # Emphasize that repo-root context is forbidden; Compose uses runtime-context/.
    marker = (
        f"# Compose / local builds must use applications/{pkg}/docker/runtime-context/\n"
        f"# prepared by scripts/build/build_application_image.py (or --keep-context).\n"
    )
    return body.replace(
        "# Do not use the monorepo root as Docker context.\n",
        "# Do not use the monorepo root as Docker context.\n" + marker,
        1,
    )


def render_dockerignore(*, pkg: str, short: str, agent_dirs: list[str]) -> str:
    """Defensive ignore for accidental root-context builds — not the dependency graph."""
    del agent_dirs  # selection is declared in application pyproject.toml
    return dedent(
        f"""\
        # Defensive ignore only. Canonical builds use materialized runtime-context/.
        # Application: {pkg} ({short})
        # Source of truth: applications/{pkg}/pyproject.toml

        .git
        .venv
        **/.venv
        .env
        **/.env
        build/
        **/__pycache__/
        *.pyc
        .pytest_cache
        .mypy_cache
        .ruff_cache
        notebooks/
        docs/
        tests/
        infra/
        **/proof
        **/proof_artifacts
        coverage/
        htmlcov/
        """
    )


def render_docker_compose(
    *,
    pkg: str,
    short: str,
    port: int,
    env_prefix: str,
    project_name: str | None = None,
    include_ollama: bool = False,
    include_qdrant: bool = False,
) -> str:
    compose_name = project_name or f"intergrax_{short}"
    service_lines = [
        f"  {short}:",
        "    build:",
        "      context: ./runtime-context",
        "      dockerfile: Dockerfile",
        f"    image: {short}-application:latest",
        "    ports:",
        f'      - "{port}:{port}"',
        "    env_file:",
        "      - ../.env",
        "    environment:",
        "      INTERGRAX_ENV: dev",
        f"      {env_prefix}BACKEND_ENV: dev",
        f'      {env_prefix}BACKEND_HOST: "0.0.0.0"',
        f'      {env_prefix}BACKEND_PORT: "{port}"',
        f'      {env_prefix}INCLUDE_MCP: "false"',
    ]
    depends: list[str] = []
    named_volumes: list[str] = []

    if include_qdrant:
        service_lines.extend(
            [
                f"      {env_prefix}VECTOR_STORE: qdrant",
                "      INTERGRAX_QDRANT_URL: http://qdrant:6333",
            ]
        )
        depends.append("qdrant")

    if include_ollama:
        service_lines.extend(
            [
                "      INTERGRAX_LLM_PROVIDER: ollama",
                "      OLLAMA_HOST: http://ollama:11434",
            ]
        )
        depends.append("ollama")

    if depends:
        service_lines.append("    depends_on:")
        service_lines.extend(f"      - {dep}" for dep in depends)

    blocks = [
        "# © Artur Czarnecki. All rights reserved.",
        f"# Prepare minimal context, then compose:",
        f"#   uv run python scripts/build/build_application_image.py --application {pkg} \\",
        f"#     --context-dir applications/{pkg}/docker/runtime-context --keep-context --manifest-only",
        f"#   uv run python scripts/build/build_application_image.py --application {pkg} \\",
        f"#     --tag {short}-application:latest --context-dir applications/{pkg}/docker/runtime-context --keep-context",
        f"# Or: docker compose -f applications/{pkg}/docker/docker-compose.yml up --build",
        "# (requires runtime-context/ already materialized; same isolation contract as CLI).",
        "",
        f"name: {compose_name}",
        "",
        "services:",
        *service_lines,
    ]

    if include_qdrant:
        blocks.extend(
            [
                "",
                "  qdrant:",
                "    image: qdrant/qdrant:latest",
                "    expose:",
                '      - "6333"',
                "    volumes:",
                "      - qdrant_data:/qdrant/storage",
            ]
        )
        named_volumes.append("  qdrant_data:")

    if include_ollama:
        blocks.extend(
            [
                "",
                "  ollama:",
                "    image: ollama/ollama:latest",
                "    expose:",
                '      - "11434"',
                "    volumes:",
                "      - ollama_data:/root/.ollama",
            ]
        )
        named_volumes.append("  ollama_data:")

    if named_volumes:
        blocks.extend(["", "volumes:", *named_volumes])

    return "\n".join(blocks) + "\n"


def render_build_docker_sh(*, pkg: str, short: str, port: int) -> str:
    """Render ``docker/build-docker.sh`` — canonical minimal-context image build."""
    image = f"{short}-application"
    return dedent(
        f"""\
        #!/usr/bin/env bash
        # Build Tier-3 application image via materialized runtime graph.
        set -euo pipefail

        SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
        REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../../.." && pwd)"
        PKG="{pkg}"
        IMAGE_TAG="${{IMAGE_TAG:-{image}}}"
        PORT="{port}"

        cd "${{REPO_ROOT}}"
        uv run python scripts/build/build_application_image.py \\
          --application "${{PKG}}" \\
          --tag "${{IMAGE_TAG}}" \\
          --context-dir "applications/${{PKG}}/docker/runtime-context" \\
          --keep-context

        echo ""
        echo "Built: ${{IMAGE_TAG}}"
        echo "Run:   docker run --rm --env-file applications/${{PKG}}/.env -p ${{PORT}}:${{PORT}} ${{IMAGE_TAG}}"
        """
    )


def render_build_docker_bat(*, pkg: str, short: str, port: int) -> str:
    """Render ``docker/build-docker.bat`` for Windows."""
    image = f"{short}-application"
    return dedent(
        f"""\
        @echo off
        REM Build Tier-3 application image via materialized runtime graph.
        setlocal EnableExtensions

        set "PKG={pkg}"
        set "IMAGE_TAG={image}"
        if not "%~1"=="" set "IMAGE_TAG=%~1"
        set "PORT={port}"

        cd /d "%~dp0\\..\\..\\.."
        if errorlevel 1 (
          echo Failed to locate repository root.
          exit /b 1
        )

        uv run python scripts/build/build_application_image.py --application %PKG% --tag %IMAGE_TAG% --context-dir applications/%PKG%/docker/runtime-context --keep-context
        if errorlevel 1 exit /b 1

        echo.
        echo Built: %IMAGE_TAG%
        echo Run:   docker run --rm --env-file applications/%PKG%/.env -p %PORT%:%PORT% %IMAGE_TAG%
        endlocal
        """
    )


def render_local_docker_compose_sh(*, pkg: str, short: str, port: int) -> str:
    return dedent(
        f"""\
        #!/usr/bin/env sh
        # © Artur Czarnecki. All rights reserved.
        set -eu

        SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
        APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
        REPO_ROOT=$(CDPATH= cd -- "$APP_DIR/../.." && pwd)
        COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"

        cd "$REPO_ROOT"

        echo "Materializing minimal runtime context for {pkg}..."
        uv run python scripts/build/build_application_image.py \
          --application "{pkg}" \
          --context-dir "applications/{pkg}/docker/runtime-context" \
          --materialize-only

        echo "Building and starting {short.replace('_', ' ')} via Docker Compose..."
        docker compose -f "$COMPOSE_FILE" up --build -d

        echo "Stack is starting. Verify with:"
        echo "  curl http://127.0.0.1:{port}/health"
        """
    )


def render_local_docker_compose_bat(*, pkg: str, short: str, port: int) -> str:
    return dedent(
        f"""\
        @echo off
        setlocal

        set "SCRIPT_DIR=%~dp0"
        set "APP_DIR=%SCRIPT_DIR%.."
        set "COMPOSE_FILE=%APP_DIR%\\docker\\docker-compose.yml"

        pushd "%APP_DIR%\\..\\.." >nul
        if errorlevel 1 (
            echo Failed to locate repository root.
            exit /b 1
        )

        echo Materializing minimal runtime context for {pkg}...
        uv run python scripts/build/build_application_image.py --application {pkg} --context-dir applications/{pkg}/docker/runtime-context --materialize-only
        if errorlevel 1 (
            popd >nul
            exit /b 1
        )

        echo Building and starting {short.replace('_', ' ')} via Docker Compose...
        docker compose -f "%COMPOSE_FILE%" up --build -d
        if errorlevel 1 (
            popd >nul
            exit /b 1
        )

        echo Stack is starting. Verify with:
        echo   curl http://127.0.0.1:{port}/health

        popd >nul
        endlocal
        """
    )


def render_local_docker_bootstrap_sh(*, pkg: str, short: str, port: int, route_prefix: str) -> str:
    return dedent(
        f"""\
        #!/usr/bin/env sh
        # © Artur Czarnecki. All rights reserved.
        set -eu

        SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
        APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
        REPO_ROOT=$(CDPATH= cd -- "$APP_DIR/../.." && pwd)
        COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"
        ENV_FILE="$APP_DIR/.env"
        ENV_EXAMPLE="$APP_DIR/.env.example"

        if [ ! -f "$ENV_FILE" ]; then
          if [ ! -f "$ENV_EXAMPLE" ]; then
            echo "Missing .env and .env.example in $SCRIPT_DIR" >&2
            exit 1
          fi
          cp "$ENV_EXAMPLE" "$ENV_FILE"
          echo "Created $ENV_FILE from .env.example"
        fi

        read_env_value() {{
          key="$1"
          grep -E "^[[:space:]]*$key[[:space:]]*=" "$ENV_FILE" 2>/dev/null \\
            | tail -n 1 \\
            | sed -E "s/^[[:space:]]*$key[[:space:]]*=[[:space:]]*//" \\
            | sed -E 's/^"(.*)"$/\\1/' \\
            | sed -E "s/^'(.*)'$/\\1/" || true
        }}

        MODEL=$(read_env_value "INTERGRAX_DEFAULT_OLLAMA_MODEL")
        if [ -z "${{MODEL:-}}" ]; then
          MODEL=$(read_env_value "INTERGRAX_LLM_MODEL")
        fi
        if [ -z "${{MODEL:-}}" ]; then
          MODEL="llama3.1:latest"
        fi

        cd "$REPO_ROOT"

        echo "Building {short.replace('_', ' ')} Docker image..."
        docker compose -f "$COMPOSE_FILE" build

        echo "Starting Ollama service..."
        docker compose -f "$COMPOSE_FILE" up -d ollama

        echo "Pulling Ollama model: $MODEL"
        for attempt in 1 2 3; do
          if docker compose -f "$COMPOSE_FILE" exec -T ollama ollama pull "$MODEL"; then
            break
          fi
          if [ "$attempt" -eq 3 ]; then
            echo "Ollama pull failed after 3 attempts" >&2
            exit 1
          fi
          echo "Retrying ollama pull ($attempt/3)..."
          sleep 5
        done

        echo "Starting local stack..."
        docker compose -f "$COMPOSE_FILE" up -d

        echo "Stack is starting. Verify with:"
        echo "  curl http://127.0.0.1:{port}/health"
        echo "  curl http://127.0.0.1:{port}{route_prefix}/agents"
        """
    )


def render_local_docker_bootstrap_bat(*, pkg: str, short: str, port: int, route_prefix: str) -> str:
    return dedent(
        f"""\
        @echo off
        setlocal enabledelayedexpansion

        set "SCRIPT_DIR=%~dp0"
        set "APP_DIR=%SCRIPT_DIR%.."
        set "COMPOSE_FILE=%APP_DIR%\\docker\\docker-compose.yml"
        set "ENV_FILE=%APP_DIR%\\.env"
        set "ENV_EXAMPLE=%APP_DIR%\\.env.example"

        pushd "%APP_DIR%\\..\\.." >nul

        if not exist "%ENV_FILE%" (
            if not exist "%ENV_EXAMPLE%" (
                echo Missing .env and .env.example in %SCRIPT_DIR%
                popd >nul
                exit /b 1
            )
            copy "%ENV_EXAMPLE%" "%ENV_FILE%" >nul
            echo Created %ENV_FILE% from .env.example
        )

        set "MODEL="
        for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
            set "KEY=%%A"
            set "VALUE=%%B"
            set "KEY=!KEY: =!"
            if /i "!KEY!"=="INTERGRAX_DEFAULT_OLLAMA_MODEL" set "MODEL=!VALUE!"
        )
        if "!MODEL!"=="" (
            for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
                set "KEY=%%A"
                set "VALUE=%%B"
                set "KEY=!KEY: =!"
                if /i "!KEY!"=="INTERGRAX_LLM_MODEL" set "MODEL=!VALUE!"
            )
        )
        if "!MODEL!"=="" set "MODEL=llama3.1:latest"
        set "MODEL=!MODEL:"=!"
        set "MODEL=!MODEL:'=!"

        echo Building Docker image...
        docker compose -f "%COMPOSE_FILE%" build
        if errorlevel 1 goto fail

        echo Starting Ollama service...
        docker compose -f "%COMPOSE_FILE%" up -d ollama
        if errorlevel 1 goto fail

        echo Pulling Ollama model: !MODEL!
        call :pull_model
        if errorlevel 1 goto fail

        echo Starting local stack...
        docker compose -f "%COMPOSE_FILE%" up -d
        if errorlevel 1 goto fail

        echo Stack is starting. Verify with:
        echo   curl http://127.0.0.1:{port}/health
        echo   curl http://127.0.0.1:{port}{route_prefix}/agents

        popd >nul
        exit /b 0

        :pull_model
        for /l %%I in (1,1,3) do (
            echo Ollama pull attempt %%I/3...
            docker compose -f "%COMPOSE_FILE%" exec -T ollama ollama pull "!MODEL!"
            if not errorlevel 1 exit /b 0
            timeout /t 5 /nobreak >nul
        )
        exit /b 1

        :fail
        echo Local Docker setup failed.
        popd >nul
        exit /b 1
        """
    )


def write_application_docker(
    target: Path,
    *,
    pkg: str,
    short: str,
    port: int,
    env_prefix: str,
    agent_dirs: list[str],
    health_path: str,
    uvicorn_module: str | None = None,
    factory_import: str | None = None,
    factory_call: str | None = None,
    smoke_env: tuple[str, ...] | None = None,
    project_name: str | None = None,
    include_ollama_bootstrap: bool = False,
    include_ollama_compose: bool = False,
    include_qdrant_compose: bool = False,
    route_prefix: str = "",
    force: bool = True,
) -> None:
    """Write ``docker/`` files: Dockerfile, ignore, compose, and build scripts."""
    docker_dir = target / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    files: dict[Path, str] = {
        docker_dir / "Dockerfile": render_dockerfile(
            pkg=pkg,
            short=short,
            port=port,
            env_prefix=env_prefix,
            agent_dirs=agent_dirs,
            health_path=health_path,
            uvicorn_module=uvicorn_module,
            factory_import=factory_import,
            factory_call=factory_call,
            smoke_env=smoke_env,
        ),
        docker_dir / ".dockerignore": render_dockerignore(
            pkg=pkg,
            short=short,
            agent_dirs=agent_dirs,
        ),
        docker_dir / "docker-compose.yml": render_docker_compose(
            pkg=pkg,
            short=short,
            port=port,
            env_prefix=env_prefix,
            project_name=project_name,
            include_ollama=include_ollama_compose,
            include_qdrant=include_qdrant_compose,
        ),
        docker_dir / "build-docker.sh": render_build_docker_sh(pkg=pkg, short=short, port=port),
        docker_dir / "build-docker.bat": render_build_docker_bat(pkg=pkg, short=short, port=port),
    }
    scripts_dir = target / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if include_ollama_bootstrap:
        files[scripts_dir / "build-local-docker.sh"] = render_local_docker_bootstrap_sh(
            pkg=pkg,
            short=short,
            port=port,
            route_prefix=route_prefix,
        )
        files[scripts_dir / "build-local-docker.bat"] = render_local_docker_bootstrap_bat(
            pkg=pkg,
            short=short,
            port=port,
            route_prefix=route_prefix,
        )
    else:
        files[scripts_dir / "build-local-docker.sh"] = render_local_docker_compose_sh(
            pkg=pkg,
            short=short,
            port=port,
        )
        files[scripts_dir / "build-local-docker.bat"] = render_local_docker_compose_bat(
            pkg=pkg,
            short=short,
            port=port,
        )
    for path, content in files.items():
        if path.exists() and not force:
            raise FileExistsError(f"File already exists: {path}")
        path.write_text(content, encoding="utf-8")

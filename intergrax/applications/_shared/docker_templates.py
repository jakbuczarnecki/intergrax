# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Per-application Docker file templates (Phase N.5)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def render_dockerfile(
    *,
    pkg: str,
    short: str,
    port: int,
    env_prefix: str,
    agent_dirs: list[str],
    health_path: str,
    uvicorn_module: str | None = None,
) -> str:
    """Render ``applications/<pkg>/docker/Dockerfile`` content."""
    target = uvicorn_module or f"{pkg}.host.main:app"
    copy_agents = "\n".join(f"COPY agents/{d}/ ./agents/{d}/" for d in agent_dirs)
    health_url = f"http://127.0.0.1:{port}{health_path}"
    return dedent(
        f"""\
        # © Artur Czarnecki. All rights reserved.
        # syntax=docker/dockerfile:1
        #
        # Build from monorepo root (first install may take several minutes).
        # BuildKit: docker buildx build -f applications/{pkg}/docker/Dockerfile \\
        #   --ignorefile applications/{pkg}/docker/.dockerignore -t {short}-application .
        # Classic: cp applications/{pkg}/docker/.dockerignore .dockerignore \\
        #   && docker build -f applications/{pkg}/docker/Dockerfile -t {short}-application .
        # Run: docker run --env-file applications/{pkg}/.env -p {port}:{port} {short}-application

        FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
        WORKDIR /app

        COPY pyproject.toml uv.lock ./
        COPY intergrax/ ./intergrax/
        COPY applications/__init__.py ./applications/__init__.py
        COPY applications/{pkg}/ ./applications/{pkg}/
        {copy_agents}

        # Dev lock targets win32; resolve dependencies for Linux inside the image.
        RUN sed -i "s/sys_platform == 'win32'/sys_platform == 'linux'/" pyproject.toml \\
            && uv sync --no-dev

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

        COPY --from=builder /app /app
        ENV PATH="/app/.venv/bin:$PATH"

        EXPOSE {port}
        WORKDIR /app/applications/{pkg}

        HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \\
          CMD python -c "import urllib.request; urllib.request.urlopen('{health_url}')" || exit 1

        CMD ["uvicorn", "{target}", "--host", "0.0.0.0", "--port", "{port}"]
        """
    )


def render_dockerignore(*, pkg: str, short: str, agent_dirs: list[str]) -> str:
    agent_exceptions = "\n".join(f"!agents/{d}/" for d in agent_dirs)
    return dedent(
        f"""\
        # Per-application ignore file — monorepo root as context.
        # BuildKit: docker buildx build -f applications/{pkg}/docker/Dockerfile \\
        #   --ignorefile applications/{pkg}/docker/.dockerignore -t {short}-application .
        # Classic: cp this file to repo-root .dockerignore, then docker build -f ...

        .git
        .venv
        build/
        **/__pycache__/
        *.pyc
        .pytest_cache
        notebooks/
        docs/
        tests/
        infra/
        applications/*
        !applications/{pkg}/
        agents/*
        {agent_exceptions}
        """
    )


def render_docker_compose(
    *,
    pkg: str,
    short: str,
    port: int,
    env_prefix: str,
) -> str:
    return dedent(
        f"""\
        # © Artur Czarnecki. All rights reserved.
        # Run from repository root: docker compose -f applications/{pkg}/docker/docker-compose.yml up --build

        services:
          {short}:
            build:
              context: ../../..
              dockerfile: applications/{pkg}/docker/Dockerfile
            image: {short}-application:latest
            ports:
              - "{port}:{port}"
            env_file:
              - ../.env
            environment:
              {env_prefix}BACKEND_HOST: "0.0.0.0"
              {env_prefix}BACKEND_PORT: "{port}"
              INTERGRAX_ENV: prod
        """
    )


def render_build_docker_sh(*, pkg: str, short: str, port: int) -> str:
    """Render ``docker/build-docker.sh`` — run from anywhere; uses monorepo root as context."""
    image = f"{short}-application"
    return dedent(
        f"""\
        #!/usr/bin/env bash
        # Build Tier-3 application image from monorepo root (Phase N).
        set -euo pipefail

        SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
        REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../../.." && pwd)"
        PKG="{pkg}"
        IMAGE_TAG="${{IMAGE_TAG:-{image}}}"
        PORT="{port}"

        cd "${{REPO_ROOT}}"

        if docker buildx version >/dev/null 2>&1; then
          echo "Building ${{IMAGE_TAG}} (BuildKit)..."
          docker buildx build \\
            -f "applications/${{PKG}}/docker/Dockerfile" \\
            --ignorefile "applications/${{PKG}}/docker/.dockerignore" \\
            -t "${{IMAGE_TAG}}" \\
            .
        else
          echo "BuildKit not found — using docker build (consider: docker buildx install)"
          docker build \\
            -f "applications/${{PKG}}/docker/Dockerfile" \\
            -t "${{IMAGE_TAG}}" \\
            .
        fi

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
        REM Build Tier-3 application image from monorepo root (Phase N).
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

        docker buildx version >nul 2>&1
        if %ERRORLEVEL% equ 0 (
          echo Building %IMAGE_TAG% ^(BuildKit^)...
          docker buildx build -f applications/%PKG%/docker/Dockerfile --ignorefile applications/%PKG%/docker/.dockerignore -t %IMAGE_TAG% .
        ) else (
          echo BuildKit not found — using docker build
          docker build -f applications/%PKG%/docker/Dockerfile -t %IMAGE_TAG% .
        )
        if errorlevel 1 exit /b 1

        echo.
        echo Built: %IMAGE_TAG%
        echo Run:   docker run --rm --env-file applications/%PKG%/.env -p %PORT%:%PORT% %IMAGE_TAG%
        endlocal
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
    force: bool = True,
) -> None:
    """Write ``docker/`` files: Dockerfile, ignore, compose, and build scripts."""
    docker_dir = target / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    files = {
        docker_dir / "Dockerfile": render_dockerfile(
            pkg=pkg,
            short=short,
            port=port,
            env_prefix=env_prefix,
            agent_dirs=agent_dirs,
            health_path=health_path,
            uvicorn_module=uvicorn_module,
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
        ),
        docker_dir / "build-docker.sh": render_build_docker_sh(pkg=pkg, short=short, port=port),
        docker_dir / "build-docker.bat": render_build_docker_bat(pkg=pkg, short=short, port=port),
    }
    for path, content in files.items():
        if path.exists() and not force:
            raise FileExistsError(f"File already exists: {path}")
        path.write_text(content, encoding="utf-8")

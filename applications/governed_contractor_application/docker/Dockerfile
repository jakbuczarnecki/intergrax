        # © Artur Czarnecki. All rights reserved.
        # syntax=docker/dockerfile:1
        #
        # Build from monorepo root (first install may take several minutes).
        # BuildKit: docker buildx build -f applications/governed_contractor_application/docker/Dockerfile \
        #   --ignorefile applications/governed_contractor_application/docker/.dockerignore -t governed_contractor-application .
        # Classic: cp applications/governed_contractor_application/docker/.dockerignore .dockerignore \
        #   && docker build -f applications/governed_contractor_application/docker/Dockerfile -t governed_contractor-application .
        # Run: docker run --env-file applications/governed_contractor_application/.env -p 8000:8000 governed_contractor-application

        FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
        WORKDIR /app

        COPY pyproject.toml uv.lock ./
        COPY README.md ./
COPY intergrax/ ./intergrax/
        COPY applications/__init__.py ./applications/__init__.py
        COPY applications/governed_contractor_application/ ./applications/governed_contractor_application/
        COPY agents/external_contractor_adapter/ ./agents/external_contractor_adapter/

        # pyproject.toml already declares disjoint Linux and Windows uv environments.
        # Do not rewrite platform markers inside Docker; resolving on Linux selects the Linux environment.
        RUN uv sync --no-dev

        # Build-time start-path smoke catches missing copied packages and eager optional
        # integrations before docker compose starts the runtime container.
        RUN PYTHONPATH=/app:/app/agents:/app/applications \
            INTERGRAX_ENV=dev \
    GOVERNED_CONTRACTOR_BACKEND_ENV=dev \
    GOVERNED_CONTRACTOR_INCLUDE_MCP=false \
    GOVERNED_CONTRACTOR_INCLUDE_SCHEDULER=false \
    GOVERNED_CONTRACTOR_INCLUDE_INTERACTIONS=false \
    GOVERNED_CONTRACTOR_INCLUDE_TASK_CONTROL=false \
    INTERGRAX_SHADOW_ROOT=/tmp/intergrax_app_shadow \
    INTERGRAX_SQLITE_DATA_DIR=/tmp/intergrax_app_sqlite \
            .venv/bin/python -c "from governed_contractor_application.host.factory import create_governed_contractor_backend_app; app = create_governed_contractor_backend_app(); assert app.title"

        FROM python:3.12-slim-bookworm AS runtime
        WORKDIR /app

        ENV PYTHONUNBUFFERED=1 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH=/app:/app/agents:/app/applications \
            INTERGRAX_ENV=prod \
            GOVERNED_CONTRACTOR_BACKEND_HOST=0.0.0.0 \
            GOVERNED_CONTRACTOR_BACKEND_PORT=8000

        RUN apt-get update && apt-get install -y --no-install-recommends \
            libsqlite3-0 \
            && rm -rf /var/lib/apt/lists/*

        COPY --from=builder /app /app
        ENV PATH="/app/.venv/bin:$PATH"

        EXPOSE 8000
        WORKDIR /app/applications/governed_contractor_application

        HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
          CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

        CMD ["uvicorn", "governed_contractor_application.host.main:app", "--host", "0.0.0.0", "--port", "8000"]

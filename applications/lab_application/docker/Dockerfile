        # © Artur Czarnecki. All rights reserved.
        # syntax=docker/dockerfile:1
        #
        # Build from monorepo root (first install may take several minutes).
        # BuildKit: docker buildx build -f applications/lab_application/docker/Dockerfile \
        #   --ignorefile applications/lab_application/docker/.dockerignore -t lab-application .
        # Classic: cp applications/lab_application/docker/.dockerignore .dockerignore \
        #   && docker build -f applications/lab_application/docker/Dockerfile -t lab-application .
        # Run: docker run --env-file applications/lab_application/.env -p 8090:8090 lab-application

        FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
        WORKDIR /app

        COPY pyproject.toml uv.lock ./
        COPY intergrax/ ./intergrax/
        COPY applications/__init__.py ./applications/__init__.py
        COPY applications/lab_application/ ./applications/lab_application/
        COPY agents/echo/ ./agents/echo/
COPY agents/lab/ ./agents/lab/
COPY agents/signoff_probe/ ./agents/signoff_probe/
COPY agents/research/ ./agents/research/

        # Dev lock targets win32; resolve dependencies for Linux inside the image.
        RUN sed -i "s/sys_platform == 'win32'/sys_platform == 'linux'/" pyproject.toml \
            && uv sync --no-dev

        FROM python:3.12-slim-bookworm AS runtime
        WORKDIR /app

        ENV PYTHONUNBUFFERED=1 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH=/app:/app/agents:/app/applications \
            INTERGRAX_ENV=prod \
            LAB_BACKEND_HOST=0.0.0.0 \
            LAB_BACKEND_PORT=8090

        RUN apt-get update && apt-get install -y --no-install-recommends \
            libsqlite3-0 \
            && rm -rf /var/lib/apt/lists/*

        COPY --from=builder /app /app
        ENV PATH="/app/.venv/bin:$PATH"

        EXPOSE 8090
        WORKDIR /app/applications/lab_application

        HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
          CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/v1/lab/agents')" || exit 1

        CMD ["uvicorn", "lab_application.host.main:app", "--host", "0.0.0.0", "--port", "8090"]

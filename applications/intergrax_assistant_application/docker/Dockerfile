# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# Build from monorepo root (first install may take several minutes).
# BuildKit: docker buildx build -f applications/intergrax_assistant_application/docker/Dockerfile \
#   --ignorefile applications/intergrax_assistant_application/docker/.dockerignore -t intergrax_assistant-application .
# Classic: cp applications/intergrax_assistant_application/docker/.dockerignore .dockerignore \
#   && docker build -f applications/intergrax_assistant_application/docker/Dockerfile -t intergrax_assistant-application .
# Run: docker run --env-file applications/intergrax_assistant_application/.env -p 8096:8096 intergrax_assistant-application

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY intergrax/ ./intergrax/
COPY applications/__init__.py ./applications/__init__.py
COPY applications/intergrax_assistant_application/ ./applications/intergrax_assistant_application/
COPY agents/intergrax_assistant/ ./agents/intergrax_assistant/

# Dev lock targets win32; resolve dependencies for Linux inside the image.
RUN sed -i "s/sys_platform == 'win32'/sys_platform == 'linux'/" pyproject.toml \
    && uv sync --no-dev

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/agents:/app/applications \
    INTERGRAX_ENV=prod \
    INTERGRAX_ASSISTANT_BACKEND_HOST=0.0.0.0 \
    INTERGRAX_ASSISTANT_BACKEND_PORT=8096

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8096
WORKDIR /app/applications/intergrax_assistant_application

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8096/v1/intergrax_assistant/agents')" || exit 1

CMD ["uvicorn", "intergrax_assistant_application.host.main:app", "--host", "0.0.0.0", "--port", "8096"]

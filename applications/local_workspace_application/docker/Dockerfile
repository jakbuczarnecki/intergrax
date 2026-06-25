# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# Build from monorepo root (first install may take several minutes).
# BuildKit: docker buildx build -f applications/local_workspace_application/docker/Dockerfile \
#   --ignorefile applications/local_workspace_application/docker/.dockerignore -t local_workspace-application .
# Classic: cp applications/local_workspace_application/docker/.dockerignore .dockerignore \
#   && docker build -f applications/local_workspace_application/docker/Dockerfile -t local_workspace-application .
# Run: docker run --env-file applications/local_workspace_application/.env -p 8020:8020 local_workspace-application

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY intergrax/ ./intergrax/
COPY applications/__init__.py ./applications/__init__.py
COPY applications/local_workspace_application/ ./applications/local_workspace_application/
COPY agents/local_indexer/ ./agents/local_indexer/
COPY agents/local_search/ ./agents/local_search/
COPY agents/local_synthesizer/ ./agents/local_synthesizer/

# pyproject.toml already declares disjoint Linux and Windows uv environments.
# Do not rewrite platform markers inside Docker; resolving on Linux selects the Linux environment.
RUN uv sync --no-dev

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/agents:/app/applications \
    INTERGRAX_ENV=dev \
    LOCAL_WORKSPACE_BACKEND_ENV=dev \
    LOCAL_WORKSPACE_BACKEND_HOST=0.0.0.0 \
    LOCAL_WORKSPACE_BACKEND_PORT=8020

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8020
WORKDIR /app/applications/local_workspace_application

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8020/health')" || exit 1

CMD ["uvicorn", "local_workspace_application.host.main:app", "--host", "0.0.0.0", "--port", "8020"]

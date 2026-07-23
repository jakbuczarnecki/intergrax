# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# Build from monorepo root (first install may take several minutes).
# BuildKit: docker buildx build -f applications/poc_template_application/docker/Dockerfile \
#   --ignorefile applications/poc_template_application/docker/.dockerignore -t poc_template-application .
# Classic: cp applications/poc_template_application/docker/.dockerignore .dockerignore \
#   && docker build -f applications/poc_template_application/docker/Dockerfile -t poc_template-application .
# Dependency source of truth: applications/poc_template_application/pyproject.toml
# (selects Intergrax workspace package + platform extras).
#
# Run: docker run --env-file applications/poc_template_application/.env -p 8095:8095 poc_template-application

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY intergrax/ ./intergrax/
COPY applications/ ./applications/
COPY agents/echo/ ./agents/echo/
# Application project selects Intergrax extras; do not pass --extra here.
# Root pyproject already declares disjoint Linux/Windows uv environments.
RUN uv sync --frozen --no-dev --project applications/poc_template_application

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/agents:/app/applications \
    INTERGRAX_ENV=prod \
    POC_TEMPLATE_BACKEND_HOST=0.0.0.0 \
    POC_TEMPLATE_BACKEND_PORT=8095

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8095
WORKDIR /app/applications/poc_template_application

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8095/v1/poc_template/agents')" || exit 1

CMD ["uvicorn", "poc_template_application.host.main:app", "--host", "0.0.0.0", "--port", "8095"]

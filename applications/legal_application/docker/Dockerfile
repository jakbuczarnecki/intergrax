# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# Build from monorepo root (first install may take several minutes).
# BuildKit: docker buildx build -f applications/legal_application/docker/Dockerfile \
#   --ignorefile applications/legal_application/docker/.dockerignore -t legal-application .
# Classic: cp applications/legal_application/docker/.dockerignore .dockerignore \
#   && docker build -f applications/legal_application/docker/Dockerfile -t legal-application .
# Dependency source of truth: applications/legal_application/pyproject.toml
# (selects Intergrax workspace package + platform extras).
#
# Run: docker run --env-file applications/legal_application/.env -p 8000:8000 legal-application

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY intergrax/ ./intergrax/
COPY applications/ ./applications/
COPY agents/legal/ ./agents/legal/
# Application project selects Intergrax extras; do not pass --extra here.
# Root pyproject already declares disjoint Linux/Windows uv environments.
RUN uv sync --frozen --no-dev --project applications/legal_application

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/agents:/app/applications \
    INTERGRAX_ENV=prod \
    LEGAL_BACKEND_HOST=0.0.0.0 \
    LEGAL_BACKEND_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
WORKDIR /app/applications/legal_application

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "legal_application.host.main:app", "--host", "0.0.0.0", "--port", "8000"]

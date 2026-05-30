# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# Build from monorepo root (first install may take several minutes).
# BuildKit: docker buildx build -f applications/research_application/docker/Dockerfile \
#   --ignorefile applications/research_application/docker/.dockerignore -t research-application .
# Classic: cp applications/research_application/docker/.dockerignore .dockerignore \
#   && docker build -f applications/research_application/docker/Dockerfile -t research-application .
# Run: docker run --env-file applications/research_application/.env -p 8010:8010 research-application

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY intergrax/ ./intergrax/
COPY applications/__init__.py ./applications/__init__.py
COPY applications/research_application/ ./applications/research_application/
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
    RESEARCH_BACKEND_HOST=0.0.0.0 \
    RESEARCH_BACKEND_PORT=8010

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8010
WORKDIR /app/applications/research_application

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health')" || exit 1

CMD ["uvicorn", "research_application.host.main:app", "--host", "0.0.0.0", "--port", "8010"]

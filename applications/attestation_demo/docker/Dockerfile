# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# Build from monorepo root (first install may take several minutes).
# BuildKit: docker buildx build -f applications/attestation_demo/docker/Dockerfile \
#   --ignorefile applications/attestation_demo/docker/.dockerignore -t attestation-demo .
# Classic: cp applications/attestation_demo/docker/.dockerignore .dockerignore \
#   && docker build -f applications/attestation_demo/docker/Dockerfile -t attestation-demo .
# Run: docker run --env-file applications/attestation_demo/.env -p 8097:8097 attestation-demo

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY intergrax/ ./intergrax/
COPY applications/__init__.py ./applications/__init__.py
COPY applications/attestation_demo/ ./applications/attestation_demo/
COPY agents/ ./agents/
COPY prompts/tools_agent_context/ ./prompts/tools_agent_context/
COPY prompts/tools_agent_planner/ ./prompts/tools_agent_planner/
COPY prompts/tools_agent_system/ ./prompts/tools_agent_system/

RUN uv sync --no-dev

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/agents:/app/applications \
    INTERGRAX_ENV=prod \
    ATTESTATION_DEMO_BACKEND_HOST=0.0.0.0 \
    ATTESTATION_DEMO_BACKEND_PORT=8097

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8097
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8097/v1/attestation_demo/agents')" || exit 1

CMD ["uvicorn", "attestation_demo.host.main:app", "--host", "0.0.0.0", "--port", "8097"]

# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# Build ONLY from a materialized runtime-graph context produced by:
#   uv run python scripts/build/build_application_image.py --application attestation_demo
# Do not use the monorepo root as Docker context.
# Compose / local builds must use applications/attestation_demo/docker/runtime-context/
# prepared by scripts/build/build_application_image.py (or --keep-context).
# Source of truth: applications/attestation_demo/pyproject.toml

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY .intergrax-runtime-graph.json ./
COPY intergrax/ ./intergrax/
COPY applications/ ./applications/
COPY agents/ ./agents/

# Install the selected application graph only (agents via workspace deps).
RUN uv sync --frozen --no-dev --project applications/attestation_demo \
    && rm -rf /root/.cache/uv

RUN PYTHONPATH=/app:/app/agents:/app/applications \
    INTERGRAX_ENV=dev \
    ATTESTATION_DEMO_BACKEND_ENV=dev \
    ATTESTATION_DEMO_INCLUDE_MCP=false \
    ATTESTATION_DEMO_INCLUDE_SCHEDULER=false \
    ATTESTATION_DEMO_INCLUDE_INTERACTIONS=false \
    ATTESTATION_DEMO_INCLUDE_TASK_CONTROL=false \
    INTERGRAX_SHADOW_ROOT=/tmp/intergrax_app_shadow \
    INTERGRAX_SQLITE_DATA_DIR=/tmp/intergrax_app_sqlite \
    .venv/bin/python -c "import attestation_demo; assert True"

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

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/intergrax /app/intergrax
COPY --from=builder /app/applications /app/applications
COPY --from=builder /app/agents /app/agents
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/README.md /app/
COPY --from=builder /app/.intergrax-runtime-graph.json /app/.intergrax-runtime-graph.json

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8097
WORKDIR /app/applications/attestation_demo

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8097/health')" || exit 1

CMD ["uvicorn", "attestation_demo.host.main:app", "--host", "0.0.0.0", "--port", "8097"]

        # © Artur Czarnecki. All rights reserved.
        # syntax=docker/dockerfile:1
        #
        # Build from monorepo root (first install may take several minutes).
        # BuildKit: docker buildx build -f applications/dispute_sim_application/docker/Dockerfile \
        #   --ignorefile applications/dispute_sim_application/docker/.dockerignore -t dispute_sim-application .
        # Classic: cp applications/dispute_sim_application/docker/.dockerignore .dockerignore \
        #   && docker build -f applications/dispute_sim_application/docker/Dockerfile -t dispute_sim-application .
        # Dependency source of truth: applications/dispute_sim_application/pyproject.toml
# (selects Intergrax workspace package + platform extras).
#
# Run: docker run --env-file applications/dispute_sim_application/.env -p 8020:8020 dispute_sim-application

        FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
        WORKDIR /app

        COPY pyproject.toml uv.lock README.md ./
        COPY intergrax/ ./intergrax/
COPY applications/ ./applications/
        COPY agents/dispute_analyst/ ./agents/dispute_analyst/
COPY agents/dispute_intake/ ./agents/dispute_intake/
COPY agents/dispute_scenario/ ./agents/dispute_scenario/
COPY agents/dispute_strategist/ ./agents/dispute_strategist/
# Application project selects Intergrax extras; do not pass --extra here.
# Root pyproject already declares disjoint Linux/Windows uv environments.
RUN uv sync --frozen --no-dev --project applications/dispute_sim_application

        FROM python:3.12-slim-bookworm AS runtime
        WORKDIR /app

        ENV PYTHONUNBUFFERED=1 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH=/app:/app/agents:/app/applications \
            INTERGRAX_ENV=prod \
            DISPUTE_SIM_BACKEND_HOST=0.0.0.0 \
            DISPUTE_SIM_BACKEND_PORT=8020

        RUN apt-get update && apt-get install -y --no-install-recommends \
            libsqlite3-0 \
            && rm -rf /var/lib/apt/lists/*

        COPY --from=builder /app /app
        ENV PATH="/app/.venv/bin:$PATH"

        EXPOSE 8020
        WORKDIR /app/applications/dispute_sim_application

        HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
          CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8020/health')" || exit 1

        CMD ["uvicorn", "dispute_sim_application.host.main:app", "--host", "0.0.0.0", "--port", "8020"]

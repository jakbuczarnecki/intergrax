FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir uv httpx pydantic

CMD ["bash", "-lc", "cd /workspace && uv venv /opt/venv-proof-runner && UV_PROJECT_ENVIRONMENT=/opt/venv-proof-runner uv sync --frozen --no-dev && UV_PROJECT_ENVIRONMENT=/opt/venv-proof-runner uv run python tests/system/tools_side_effect_safety/proof_runner/run_proof.py"]

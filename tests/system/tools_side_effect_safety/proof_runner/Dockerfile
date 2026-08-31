FROM python:3.12-slim

ARG DOCKER_VERSION=27.5.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" \
    | tar -xz -C /usr/local/bin --strip-components=1 docker/docker \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir uv httpx pydantic

CMD ["bash", "-lc", "cd /workspace && uv venv --clear /opt/venv-proof-runner && UV_PROJECT_ENVIRONMENT=/opt/venv-proof-runner uv sync --frozen --no-dev && UV_PROJECT_ENVIRONMENT=/opt/venv-proof-runner uv run python tests/system/tools_side_effect_safety/proof_runner/run_proof.py"]

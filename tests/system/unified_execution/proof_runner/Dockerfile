FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir uv httpx pydantic

CMD ["bash", "-lc", "cd /workspace && uv venv --clear /opt/venv-proof-runner && UV_PROJECT_ENVIRONMENT=/opt/venv-proof-runner uv sync --frozen --no-dev && UV_PROJECT_ENVIRONMENT=/opt/venv-proof-runner uv run python tests/system/unified_execution/proof_runner/run_proof.py"]

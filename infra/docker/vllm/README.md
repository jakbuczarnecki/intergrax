# Local vLLM path for TOKEN-10F

This directory implements a reproducible local OpenAI-compatible vLLM path.
The live vLLM proof was not executed as part of TOKEN-10F.

## Prerequisites

- Docker Engine with Docker Compose v2.
- NVIDIA Container Toolkit and a supported NVIDIA GPU.
- Sufficient VRAM for the selected model; the default 3B model still requires
  model download space and GPU memory.
- Network access for the first Hugging Face model download.

The compose file uses the pinned `vllm/vllm-openai:v0.23.0` image and binds
host `127.0.0.1:8100` to container port `8000` (`127.0.0.1:8100:8000`).
It uses `Qwen/Qwen2.5-3B-Instruct` by default. Override the
model with `VLLM_MODEL` only when the machine has sufficient resources.

## Start and check readiness

```powershell
$env:VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
$env:VLLM_API_KEY = "local-token"
docker compose -f infra/docker/vllm/docker-compose.yml up -d vllm
Invoke-WebRequest http://127.0.0.1:8100/health
```

```bash
export VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct"
export VLLM_API_KEY="local-token"
docker compose -f infra/docker/vllm/docker-compose.yml up -d vllm
curl --fail http://127.0.0.1:8100/health
```

`VLLM_API_KEY` is only an operator-side environment value. It is not embedded
in the compose file or proof artifacts.

## Run the universal harness

```bash
uv run python scripts/token_optimization/run_universal_proof.py \
  --config configs/token_optimization/proof_vllm.toml
```

Artifacts are written under:

```text
.artifacts/token_optimization/proof/token-optimization-vllm-smoke/<run-id>/
```

Use `--validate-only` to validate the TOML contract without making a provider
call. The offline unit smoke is network-free; `live_adapter` is explicit and
does not silently fall back to an offline adapter.

## Stop

```bash
docker compose -f infra/docker/vllm/docker-compose.yml down
```

The Hugging Face cache volume is persistent and can be removed separately
when reclaiming downloaded model storage.

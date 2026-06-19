# llama.cpp stack — Docker verify runbook

**Audience:** Operators confirming Intergrax llama.cpp infra (chat **8102**, embed **8103**) before production wiring.

**CI policy:** E2E tests live under `tests/e2e/llama_cpp/` with markers `e2e`, `no_ci`, `network` — **never** run in GitHub PR/unit gate.

Related: [`infra/PORTS.md`](../../PORTS.md) · [`intergrax/llm_adapters/USAGE.md`](../../../intergrax/llm_adapters/USAGE.md)

---

## What you are verifying

| # | Capability | Check |
|---|------------|-------|
| 1 | Chat OpenAI API (`/v1/models`, `/v1/chat/completions`) | E2E chat + profile tests |
| 2 | `LlamaCppChatAdapter` via `LLMAdapterRegistry` | Adapter completion |
| 3 | Embed OpenAI API (`/v1/embeddings`) | E2E embedding pipeline |
| 4 | `LlamaCppEmbeddingProvider` in RAG stack | Document embed dimensions |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker Desktop / Engine running | `docker version` succeeds |
| **~4 GB free disk** | First pull downloads GGUF from Hugging Face |
| [uv](https://docs.astral.sh/uv/) | Repo dev environment |
| Optional `HUGGING_FACE_HUB_TOKEN` | For gated models in `.env` |

---

## Step 1 — Start stack

**Standalone (recommended for verify):**

```bash
cd infra/docker
./manage.sh llama-cpp start
./manage.sh llama-cpp-embed start
```

**Integration profile (chat + embed together):**

```bash
cd infra/integration
./manage.sh start llama-cpp
```

Windows (integration):

```powershell
cd infra\integration
.\manage.ps1 start llama-cpp
```

**First start** may take **5–15 minutes** while the container downloads the default GGUF model.

---

## Step 2 — Health probes (manual)

```bash
curl -s http://127.0.0.1:8102/v1/models
curl -s http://127.0.0.1:8103/v1/models
```

Both should return JSON (HTTP 200). Chat and embed are **separate** containers.

---

## Step 3 — Environment

```bash
export INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL=http://127.0.0.1:8102/v1
export INTERGRAX_DEFAULT_LLAMA_CPP_MODEL=default
export INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL=http://127.0.0.1:8103/v1
export INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL=default
export INTERGRAX_LLAMA_CPP_VERIFY=1
```

PowerShell:

```powershell
$env:INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8102/v1"
$env:INTERGRAX_DEFAULT_LLAMA_CPP_MODEL = "default"
$env:INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL = "http://127.0.0.1:8103/v1"
$env:INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL = "default"
$env:INTERGRAX_LLAMA_CPP_VERIFY = "1"
```

---

## Step 4 — Automated E2E (recommended)

From **repository root**:

```bash
infra/docker/llama-cpp/verify.sh
```

Windows:

```powershell
infra\docker\llama-cpp\verify.ps1
```

The script waits for `/v1/models`, sets `INTERGRAX_LLAMA_CPP_VERIFY=1`, and runs:

```bash
uv run pytest tests/e2e/llama_cpp/ -m "e2e and no_ci" -q
```

**Expected:** all tests pass (typically 5). Failures indicate stack or adapter wiring issues — not skipped.

---

## Step 5 — Manual pytest (alternative)

```bash
export INTERGRAX_LLAMA_CPP_VERIFY=1
uv run pytest tests/e2e/llama_cpp/ -m "e2e and no_ci" -q
```

Without `INTERGRAX_LLAMA_CPP_VERIFY=1`, tests **skip** when the server is down (safe for optional local runs).

---

## Step 6 — Wire platform LLM

```bash
export INTERGRAX_LLM_PROVIDER=llama_cpp
export INTERGRAX_LLM_MODEL=default
export INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL=http://127.0.0.1:8102/v1
```

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Connection refused on 8102 | `docker compose --profile llama-cpp ps` — wait for model download |
| Embed tests fail, chat OK | Start embed service: `./manage.sh llama-cpp` includes both in integration profile |
| Slow first response | Normal — GGUF load on CPU |
| Port conflict with Weaviate | Use host **8102**, not container default 8080 on host |

---

## Stop stack

```bash
cd infra/integration && ./manage.sh stop llama-cpp
```

---

## Explicit non-goals

- Not part of `uv run pytest -m "gate and not no_ci"` (PR CI)
- Not part of `.github/workflows/llm-network-smoke.yml`
- P5 `interaction_surface/llama_cpp` — deferred (same as vLLM)

# Build & deploy — Local Workspace

Tier-3 application package: `applications/local_workspace_application/`. This document is the operational runbook for local development, verification, and container deployment.

> Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| `uv` | Python deps from repo root `pyproject.toml` / `uv.lock` |
| Repo clone | Monorepo; build context is always repository root |
| Docker | Local stack and image build |
| Docker Compose | LKW backend + Qdrant + Ollama |

Tier-2 agents used by this host: **local_indexer, local_search, local_synthesizer** under `agents/` on `PYTHONPATH`.

---

## 1. Configuration

```bash
cp applications/local_workspace_application/.env.example applications/local_workspace_application/.env
```

Edit `.env` if needed. Variables use the application prefix `LOCAL_WORKSPACE_` plus Intergrax runtime variables such as `INTERGRAX_QDRANT_URL`, `INTERGRAX_SHADOW_ROOT`, and `INTERGRAX_LLM_MODEL`.

Minimum local stack variables are documented in `.env.example`:

| Variable | Role |
|----------|------|
| `LOCAL_WORKSPACE_BACKEND_PORT` | LKW HTTP port, default `8020` |
| `INTERGRAX_ALLOWED_READ_ROOTS` | Host paths that LKW may read when indexing local files |
| `INTERGRAX_SQLITE_DATA_DIR` | Local SQLite/runtime data directory |
| `INTERGRAX_SHADOW_ROOT` | Shadow workspace root for generated artifacts |
| `LOCAL_WORKSPACE_VECTOR_STORE` | `qdrant` by default; `inmemory` only for test/dev fallback |
| `INTERGRAX_QDRANT_URL` | Qdrant endpoint |
| `INTERGRAX_QDRANT_COLLECTION` | Default local RAG collection |
| `INTERGRAX_LLM_PROVIDER` | `ollama` by default |
| `INTERGRAX_LLM_MODEL` / `INTERGRAX_DEFAULT_OLLAMA_MODEL` | Ollama model pulled by local bootstrap scripts |
| `LOCAL_WORKSPACE_ENABLE_REDIS` | Optional; keep false until background ingest / queue work requires Redis |

Agent roster and integrations: `manifest.py`, `host/environment_profile.py`, `host/tool_wiring.py`.

---

## 2. Recommended local Docker bootstrap

From `applications/local_workspace_application/`:

Windows:

```bat
scripts/build-local-docker.bat
```

Linux/macOS:

```bash
chmod +x scripts/build-local-docker.sh
./scripts/build-local-docker.sh
```

The scripts perform the local bootstrap path:

```text
.env.example -> .env if missing
Docker image build
Ollama service start
ollama pull <model from .env>
LKW stack start
```

Model resolution order:

```text
INTERGRAX_DEFAULT_OLLAMA_MODEL
INTERGRAX_LLM_MODEL
llama3.1:latest fallback
```

After startup:

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/local_workspace/agents
```

---

## 3. Local run without Docker

From repository root:

```bash
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Or use the module CLI, which reads `LOCAL_WORKSPACE_BACKEND_*` from `.env`:

```bash
uv run python -m local_workspace_application.host.main
```

Smoke check:

```bash
curl -s http://127.0.0.1:8020/health
```

Routes are mounted under `/v1/local_workspace`. See `serving/` and application README for contract details.

---

## 4. Verify before deploy

```bash
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
```

Focused agent smoke:

```bash
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q
```

---

## 5. Manual container image build

Build context = monorepo root. Dockerfile lives under the application as a path reference.

```bash
docker buildx build -f applications/local_workspace_application/docker/Dockerfile \
  --ignorefile applications/local_workspace_application/docker/.dockerignore \
  -t local_workspace-application .
```

Classic Docker fallback:

```bash
cp applications/local_workspace_application/docker/.dockerignore .dockerignore
docker build -f applications/local_workspace_application/docker/Dockerfile -t local_workspace-application .
```

Notes:

- First build can take several minutes because `uv sync --no-dev` runs inside the image.
- `pyproject.toml` already declares disjoint Linux and Windows uv environments; Dockerfile must not rewrite platform markers.
- Image healthcheck probes `/health`.

---

## 6. Manual Docker Compose run

From repository root:

```bash
docker compose -f applications/local_workspace_application/docker/docker-compose.yml up --build
```

Compose starts:

```text
local_workspace
qdrant
ollama
```

Only the LKW API is exposed to the host on port `8020`. Qdrant and Ollama are internal compose services used by `local_workspace` via:

```text
http://qdrant:6333
http://ollama:11434
```

Ensure `applications/local_workspace_application/.env` exists. The bootstrap scripts create it automatically when missing.

---

## 7. Production checklist

- [ ] `INTERGRAX_ENV=prod` and application-prefixed secrets in orchestrator / `.env`, not committed.
- [ ] `LOCAL_WORKSPACE_*` reviewed against `host/settings.py` and `host/environment_profile.py`.
- [ ] Image tagged and pushed to your registry.
- [ ] Health check wired to `GET /health` or orchestrator equivalent.
- [ ] Agent roster in `manifest.py` matches agents copied in `docker/Dockerfile` / `.dockerignore`.
- [ ] Qdrant persistence volume configured for the target environment.
- [ ] Ollama/vLLM model availability is validated before serving real requests.

---

## 8. Troubleshooting

| Issue | What to try |
|-------|-------------|
| `unknown flag: --ignorefile` | Use Buildx or copy `docker/.dockerignore` to repo root |
| `Readme file does not exist: README.md` during image build | Dockerfile must copy root `README.md` before `uv sync` |
| uv environment marker overlap | Dockerfile must not rewrite `tool.uv.environments` markers |
| Port `6333` already allocated | Qdrant should remain internal in compose; only expose if debugging manually |
| Ollama model missing | Run the bootstrap script or `docker compose exec ollama ollama pull <model>` |
| Import errors for agents | Confirm agents are copied in `docker/Dockerfile` |
| Slow rebuild | Expected on first build; avoid copying the whole repo without `.dockerignore` |
| Wrong agents in registry | Check `manifest.py`, `host/environment_profile.py`, and `host/tool_wiring.py` |

---

*Generated for Intergrax Tier-3 scaffold (profile: product).*

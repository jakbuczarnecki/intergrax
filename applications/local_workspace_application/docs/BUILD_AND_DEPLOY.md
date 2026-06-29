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
uv run pytest applications/local_workspace_application/tests -q
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
otel-collector
```

Only the LKW API is exposed to the host on port `8020`. Port `4318` is also exposed for optional local OTLP HTTP debugging. Qdrant and Ollama remain internal compose services used by `local_workspace` via:

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

---

## 9. Optional OTLP observability export

LKW supports exporting policy-sanitized Intergrax observability envelopes as **OTLP logs** to an OpenTelemetry Collector. This is the first external persistence proof for policy-sanitized Intergrax observability records exported as OTLP logs. It does **not** add full trace browsing.

Local Docker Compose starts `otel-collector` and enables OTLP export explicitly in `local_workspace.environment`. Manual or non-compose runs keep export disabled unless you opt in via `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED` | `false` | Enable observability export (disabled by default for manual/non-compose runs) |
| `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND` | `otlp` | Export backend (only `otlp` supported) |
| `LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT` | — | Required when enabled; e.g. `http://otel-collector:4318/v1/logs` |
| `LOCAL_WORKSPACE_OBSERVABILITY_SERVICE_NAME` | `intergrax-lkw` | OTLP resource `service.name` |
| `LOCAL_WORKSPACE_OBSERVABILITY_SERVICE_VERSION` | — | OTLP resource `service.version` |
| `LOCAL_WORKSPACE_OBSERVABILITY_ENVIRONMENT` | — | OTLP resource `deployment.environment` |
| `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_CONTENT` | `false` | Forced to `false` by policy; raw content is never exported |
| `LOCAL_WORKSPACE_OBSERVABILITY_OTLP_TIMEOUT_SECONDS` | `30` | HTTP transport timeout |

### Run with persisted OTLP logs

From repository root:

```bash
docker compose -f applications/local_workspace_application/docker/docker-compose.yml up --build
```

After a run via Swagger or curl (`POST /v1/local_workspace/run`), inspect persisted OTLP log records on the host:

```text
applications/local_workspace_application/.observability/otel/lkw-otlp-logs.jsonl
```

Linux/macOS:

```bash
tail -n 20 applications/local_workspace_application/.observability/otel/lkw-otlp-logs.jsonl
```

Windows PowerShell:

```powershell
Get-Content applications\local_workspace_application\.observability\otel\lkw-otlp-logs.jsonl -Tail 20
```

### Inspect persisted OTLP logs

For a readable local timeline and duplicate-export check, use the lightweight inspector from repository root:

```powershell
applications\local_workspace_application\scripts\inspect-otlp-logs.bat
applications\local_workspace_application\scripts\inspect-otlp-logs.bat --list-runs
applications\local_workspace_application\scripts\inspect-otlp-logs.bat --run-id run_... --check-duplicates
applications\local_workspace_application\scripts\inspect-otlp-logs.bat --tool-id rag.retrieve
```

The default invocation reads `.observability/otel/lkw-otlp-logs.jsonl`, selects the latest run, prints a compact event timeline, and reports duplicate status. The duplicate check groups by `intergrax.event_id` plus run, event type, agent, tool, and capability metadata.

### What to look for

Exported OTLP logs should include Intergrax attributes such as:

```text
intergrax.run_id
intergrax.task_id
intergrax.capability
intergrax.tool_id
intergrax.status
intergrax.tenant_id
intergrax.workspace_id
```

To verify no duplicate export for the same runtime event, group persisted log records by `intergrax.event_id` (and `intergrax.run_id`, `intergrax.event_type`, `intergrax.agent_id`, `intergrax.tool_id`, `intergrax.capability`). Each `event_id` should appear at most once per run.

**Safety boundaries:**

- Export is **disabled by default** in `.env.example`; no remote observability export occurs without explicit configuration.
- OTLP endpoint is **required** when export is enabled.
- `export_content=false` — raw documents, chunks, prompts, tool args, secrets, and full local paths are not exported by default.
- Export failure must not fail product runs.
- Only the `otlp` backend is supported; any other backend fails fast.
- No Grafana, Loki, Elasticsearch, Langfuse, Arize, Phoenix, Jaeger, Tempo, or vendor SDK is included.



*Generated for Intergrax Tier-3 scaffold (profile: product).*

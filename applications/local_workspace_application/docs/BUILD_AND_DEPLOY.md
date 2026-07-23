# Build & deploy — Local Workspace

Tier-3 application package: `applications/local_workspace_application/`. This document is the operational runbook for local development, verification, and container deployment.

> Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../applications/USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| `uv` | Workspace lock + application project `applications/local_workspace_application/pyproject.toml` |
| Repo clone | Monorepo; build context is always repository root |
| Docker | Local stack and image build |
| Docker Compose | LKW backend + Qdrant + Ollama + optional observability backends |

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
| `LOCAL_WORKSPACE_DATA_HOME` | Canonical LKW local data root (`build/local_workspace` in repo dev); alias `LKW_DATA_HOME` |
| `INTERGRAX_ALLOWED_READ_ROOTS` | Host paths that LKW may read when indexing local files |
| `INTERGRAX_SQLITE_DATA_DIR` | Current runtime SQLite/runtime data directory (compatibility; align with data-home layout) |
| `INTERGRAX_SHADOW_ROOT` | Current runtime shadow workspace root for generated artifacts (compatibility; align with data-home layout) |
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

Explicit `docker compose` commands from repository root are the **cross-platform reference path**. Windows `.bat` helpers below are convenience wrappers around the same stacks.

For the full external reviewer walkthrough (expected outputs, Kibana inspection, proof-helper PASS criteria), see [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md).

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

### Windows convenience: all Compose overlays

This helper starts the base stack plus every optional top-level overlay in `applications/local_workspace_application/docker/` without listing each `-f` file:

Windows:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat
```

Linux/macOS:

```bash
chmod +x applications/local_workspace_application/scripts/run-local-docker-all.sh
applications/local_workspace_application/scripts/run-local-docker-all.sh
```

These wrappers discover `docker-compose.*.yml` overlays (Elasticsearch, Kibana, Sentry, etc.). Internal fragments such as `sentry.services.yml` are included by their parent overlay and are not discovered directly. Pass any Docker Compose command after the script name when needed, for example:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat ps
applications\local_workspace_application\scripts\run-local-docker-all.bat down
```

```bash
applications/local_workspace_application/scripts/run-local-docker-all.sh ps
applications/local_workspace_application/scripts/run-local-docker-all.sh down -v
```

With no arguments it runs:

```text
docker compose -f docker-compose.yml -f docker-compose.<overlay>.yml ... up --build
```

### Run with Elasticsearch/OpenSearch-compatible observability backend

Use the optional Elasticsearch overlay when you want a self-contained local vendor backend instead of the default OTLP/JSONL proof. This is the stack used by the public platform proof — step-by-step evaluation: [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md).

```bash
docker compose \
  -f applications/local_workspace_application/docker/docker-compose.yml \
  -f applications/local_workspace_application/docker/docker-compose.elasticsearch.yml \
  up --build
```

The overlay starts an Elasticsearch single-node service and switches the LKW observability backend to:

```text
LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND=elasticsearch
LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL=http://elasticsearch:9200
LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX=intergrax-lkw-observability
```

Elasticsearch is exposed on host port `9200` for manual local readback:

```bash
curl -s http://127.0.0.1:9200/_cluster/health
curl -s "http://127.0.0.1:9200/intergrax-lkw-observability/_search?pretty"
```

This overlay is a local proof environment only. It does not add auth/TLS, batching, dead-letter, dashboards, or the formal OBS-VENDOR-7 readback/duplicate proof.

Elasticsearch observability retry/backoff is provider-owned and configurable per LKW deployment through `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_*` env variables.

### Run Elasticsearch observability proof helpers

Use the proof helper after starting the Elasticsearch overlay stack. Without a `run_id`, the helper checks LKW health, Elasticsearch health, and lists recent indexed observability runs.

Windows:

```bat
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat
```

Linux/macOS:

```bash
chmod +x applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh
applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh
```

After executing a real LKW run, pass the selected run id to run timeline, duplicate, safety, and combined proof checks:

```bat
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat run_...
```

```bash
applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh run_...
```

The helpers default to `http://127.0.0.1:8020/health`, `http://127.0.0.1:9200`, and `intergrax-lkw-observability`. Override them with environment variables when needed:

```text
LOCAL_WORKSPACE_OBSERVABILITY_PROOF_LKW_HEALTH_URL
LOCAL_WORKSPACE_OBSERVABILITY_PROOF_ES_URL
LOCAL_WORKSPACE_OBSERVABILITY_PROOF_ES_INDEX
```

A successful helper run prints a documentation summary with `run_id`, Elasticsearch URL/index, `duplicate_check=0`, and `safety_check=passed`. This helper prepares and validates proof evidence; full OBS-VENDOR-7 should be marked Done only after a real LKW run id and backend readback result are recorded.

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
| Port `9200` already allocated | Stop another Elasticsearch/OpenSearch instance or change the overlay host port |
| Ollama model missing | Run the bootstrap script or `docker compose exec ollama ollama pull <model>` |
| Import errors for agents | Confirm agents are copied in `docker/Dockerfile` |
| Slow rebuild | Expected on first build; avoid copying the whole repo without `.dockerignore` |
| Wrong agents in registry | Check `manifest.py`, `host/environment_profile.py`, and `host/tool_wiring.py` |

---

---

## 9. Optional observability export

LKW supports exporting policy-sanitized Intergrax observability envelopes to external backends. Local Docker Compose enables observability explicitly for proof environments; manual or non-compose runs keep export disabled unless you opt in via `.env`.

Supported local proof backends:

```text
otlp           → OpenTelemetry Collector → persisted JSONL file
elasticsearch  → Elasticsearch/OpenSearch-compatible HTTP index API
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED` | `false` | Enable observability export (disabled by default for manual/non-compose runs) |
| `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND` | `otlp` | Export backend: `otlp` or `elasticsearch` |
| `LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT` | — | Required for `backend_id=otlp`; e.g. `http://otel-collector:4318/v1/logs` |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL` | — | Required for `backend_id=elasticsearch`; e.g. `http://elasticsearch:9200` in compose |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX` | `intergrax-lkw-observability` | Elasticsearch/OpenSearch index for policy-safe documents |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_TIMEOUT_SECONDS` | `30` | Elasticsearch HTTP transport timeout |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_ENABLED` | `true` | Enable bounded retry for retriable Elasticsearch delivery failures |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_ATTEMPTS` | `3` | Total delivery attempts including the first one (`1` = no retry) |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_INITIAL_BACKOFF_SECONDS` | `0.25` | Initial sleep before the second attempt |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_BACKOFF_SECONDS` | `2.0` | Maximum sleep between attempts |
| `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH` | — | Optional JSONL file for safe Elasticsearch failed-delivery diagnostics; leave empty to disable. Point to a controlled runtime/app data directory (for example under `applications/local_workspace_application/.observability/`) |
| `LOCAL_WORKSPACE_OBSERVABILITY_SERVICE_NAME` | `intergrax-lkw` | OTLP resource `service.name` |
| `LOCAL_WORKSPACE_OBSERVABILITY_SERVICE_VERSION` | — | OTLP resource `service.version` |
| `LOCAL_WORKSPACE_OBSERVABILITY_ENVIRONMENT` | — | OTLP resource `deployment.environment` |
| `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_CONTENT` | `false` | Forced to `false` by policy; raw content is never exported |
| `LOCAL_WORKSPACE_OBSERVABILITY_OTLP_TIMEOUT_SECONDS` | `30` | OTLP HTTP transport timeout |

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

### Inspect persisted Elasticsearch/OpenSearch documents

For Elasticsearch/OpenSearch readback, duplicate-export checks, and safety-key scans, use the lightweight inspector from repository root:

```powershell
applications\local_workspace_application\scripts\inspect-elasticsearch-observability.bat --list-runs
applications\local_workspace_application\scripts\inspect-elasticsearch-observability.bat --run-id run_... --check-duplicates --check-safety
```

Defaults: `--url http://127.0.0.1:9200`, `--index intergrax-lkw-observability`. The inspector queries `/<index>/_search` read-only; it does not create indexes or modify documents. `--check-safety` validates document keys against canonical `FORBIDDEN_EXPORT_CONTENT_FIELDS` from the runtime export boundary; it is a readback guardrail and does not replace upstream export policy.

Use the proof helper for the repeatable live-proof workflow:

```powershell
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat run_...
```

### Elasticsearch failed-delivery JSONL

When Elasticsearch observability export is enabled and delivery ultimately fails, the provider-owned file sink can append one safe JSON diagnostic object per line. LKW only passes the deployment-owned path into runtime wiring; it does not write JSONL itself.

Recommended controlled runtime/app data path:

```text
applications/local_workspace_application/.observability/elasticsearch/failed-deliveries.jsonl
```

Set in `.env` or deployment environment:

```text
LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH=applications/local_workspace_application/.observability/elasticsearch/failed-deliveries.jsonl
```

Leave empty or whitespace-only to disable the sink (default).

Each JSONL line contains only these safe fields:

```text
provider_id
operation
index
status_code
reason
retriable
attempts
exhausted
```

Never written: raw documents, prompts, chunks, tool args, secrets, tokens, or absolute payload paths.

**Local proof (controlled failure):**

1. Enable observability export with `backend_id=elasticsearch`.
2. Set `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH` to the path above.
3. Trigger a safe failed delivery by pointing `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL` at an unreachable endpoint (for example `http://127.0.0.1:59200`) or by stopping Elasticsearch while export remains enabled. Optionally set `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_ATTEMPTS=1` for a faster proof.
4. Execute one LKW run (`POST /v1/local_workspace/run`).
5. Inspect the JSONL file:

```powershell
applications\local_workspace_application\scripts\inspect-elasticsearch-failed-deliveries.bat
applications\local_workspace_application\scripts\inspect-elasticsearch-failed-deliveries.bat --check-safety
```

Windows PowerShell (raw tail):

```powershell
Get-Content applications\local_workspace_application\.observability\elasticsearch\failed-deliveries.jsonl -Tail 5
```

The inspector is read-only. It validates that every JSON object contains exactly the safe failed-delivery fields and prints record counts plus basic status summaries.

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

Elasticsearch/OpenSearch documents use the same policy-safe `intergrax.*` metadata fields and are append-only by default. Query the configured index by fields such as `intergrax.run_id`, `intergrax.event_id`, `intergrax.event_type`, `intergrax.agent_id`, `intergrax.tool_id`, and `intergrax.capability`.

To verify no duplicate export for the same runtime event, group persisted records by `intergrax.event_id` (and `intergrax.run_id`, `intergrax.event_type`, `intergrax.agent_id`, `intergrax.tool_id`, `intergrax.capability`). Each `event_id` should appear at most once per run. Use `inspect-elasticsearch-observability.bat --check-duplicates` for the indexed backend; the formal live Docker Compose OBS-VENDOR-7 proof remains planned until a real `run_id` and backend query result are recorded.

**Safety boundaries:**

- Export is **disabled by default** in `.env.example`; no remote observability export occurs without explicit configuration.
- OTLP endpoint is **required** when `backend_id=otlp` and export is enabled.
- Elasticsearch URL and index are **required** when `backend_id=elasticsearch` and export is enabled.
- `export_content=false` — raw documents, chunks, prompts, tool args, secrets, and full local paths are not exported by default.
- Export failure must not fail product runs.
- Elasticsearch/OpenSearch export failures are classified inside the provider transport with safe diagnostics (`operation`, `index`, `status_code`, `reason`, `retriable`) and must not include raw exported document content, prompts, secrets, or full local paths in error messages.
- No Grafana, Loki, Langfuse, Arize, Phoenix, Jaeger, Tempo, or vendor SDK is included.



*Generated for Intergrax Tier-3 scaffold (profile: product).*


## Application dependency project

Canonical packaging: [docs/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../../docs/architecture/APPLICATION_DEPENDENCY_MODEL.md).

`ash
uv sync --project applications/local_workspace_application
uv run --project applications/local_workspace_application python -m local_workspace_application.host.main
`

The application pyproject.toml selects Intergrax platform extras. Docker uses the same project (uv sync --frozen --no-dev --project applications/local_workspace_application); do not pass root --extra flags in the Dockerfile.

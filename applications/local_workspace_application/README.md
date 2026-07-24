# Local Knowledge Workspace (LKW)

Tier-3 product host for private-by-default, tenant-scoped, deployment-neutral knowledge workspace capabilities: source-backed indexing, semantic search, and synthesis.

**“Local”** means user-controlled deployment and configuration (full self-hosted / fully local topology remains first-class). It does **not** mean storage must always reside on a single user device. Storage location is selected by configuration and provider wiring — see [ARCHITECTURE.md — Deployment, storage and tenancy model](docs/ARCHITECTURE.md#deployment-storage-and-tenancy-model).

**Architecture (canonical):** [ARCHITECTURE.md](docs/ARCHITECTURE.md) · **Plan:** [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)  
**Live verification:** [LKW_1_LIVE_VERIFICATION.md](docs/LKW_1_LIVE_VERIFICATION.md)  
**User journey:** [USER_JOURNEY.md](docs/USER_JOURNEY.md) · **Platform proof loop:** [PLATFORM_PROOF_LOOP.md](docs/PLATFORM_PROOF_LOOP.md)  
**Build & deploy:** [BUILD_AND_DEPLOY.md](docs/BUILD_AND_DEPLOY.md)

## Agents

| Agent | Capability |
|-------|------------|
| `LocalIndexerAgent` | `local.workspace.index` |
| `LocalSearchAgent` | `local.workspace.search` (default) |
| `LocalSynthesizerAgent` | `local.workspace.synthesize` |

## What LKW proves

For the fastest external evaluation path, start with the public LKW Platform Proof: [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../docs/public-adoption/LKW_PLATFORM_PROOF.md). It shows the Docker-based run path, Elasticsearch/Kibana inspection, expected outputs, and proof-helper PASS criteria.

LKW is not only a local document assistant. It is the first product proof that Intergrax can repeatedly create, configure, run, package, deploy, observe, and evolve agent applications.

LKW.1 product proof is now passed for the live path:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

**LKW.2** (graph pipeline + local workspace skills) is **closed — pipeline proof passed**. **LKW.2.1–LKW.2.4C** and closeout smoke verified direct capabilities (`local.workspace.index`, `local.workspace.search`, `local.workspace.synthesize`) and the pipeline capability (`local.workspace.pipeline`: index → search → synthesize → shadow artifact). **Next platform step:** **OBS-EXPORT-5** — remaining vendor adapters (Langfuse/Arize/Phoenix); LKW uses platform observability export wiring only (**INTEGRATIONS-1D**).

A new user should be able to follow [USER_JOURNEY.md](docs/USER_JOURNEY.md): clone the repository, configure LKW, start the local backend, index a document, search with evidence, synthesize a draft into the shadow workspace, and inspect the trace/evidence for the run.

Current LKW.2 status: [IMPLEMENTATION_PLAN.md §5](docs/IMPLEMENTATION_PLAN.md#5-lkw2-graph-pipeline--local-workspace-skills). [LKW_1_LIVE_VERIFICATION.md](docs/LKW_1_LIVE_VERIFICATION.md) is the historical LKW.1/H1 live proof record, not the current LKW.2 execution status.

## Reference local stack

The common self-hosted / developer reference topology (not the product definition) is:

- **LKW host:** FastAPI + Nexus + local agents;
- **Vector Store:** Qdrant (provider-selected; often local in Compose);
- **Document / runtime persistence:** as wired (SQLite paths under `INTERGRAX_SQLITE_DATA_DIR` and/or Document Store providers);
- **Shadow artifacts:** `INTERGRAX_SHADOW_ROOT`;
- **LLM:** Ollama by default, vLLM optionally;
- **Redis:** optional until background ingest / queue workflows require it.

In-memory vector storage is only for tests or temporary development. It is not the reference product default for durable RAG.

## Docker quickstart

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

The scripts copy `.env.example` to `.env` when needed, build the Docker image, start Ollama, pull the model configured in `.env`, and start the local stack.

## Quickstart

From repository root:

```bash
uv run pytest applications/local_workspace_application/tests -q
cp applications/local_workspace_application/.env.example applications/local_workspace_application/.env
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Before indexing real files, set `INTERGRAX_ALLOWED_READ_ROOTS` in `.env` to one or more absolute folders that LKW may read.

## HTTP

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/local_workspace/agents
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{"message":"find information about project X","capability":"local.workspace.search"}'
```

## Developer first run

This section defines the minimal first-run path for a new developer after LKW.2
(graph pipeline + local workspace skills are **closed — pipeline proof passed**).

> **Important:** this is a **local product proof / developer path**, not a production
> certification. Writes are **shadow writes only** — original source files are
> never modified.

### Prerequisites

| Tool     | Purpose                                    |
|----------|--------------------------------------------|
| `uv`     | Python deps from repo root `pyproject.toml` |
| Git      | Clone the repository                       |
| Docker   | Local stack (optional; in-memory mode works) |

> Full configuration reference: [`BUILD_AND_DEPLOY.md`](docs/BUILD_AND_DEPLOY.md).
> Conceptual user journey: [`USER_JOURNEY.md`](docs/USER_JOURNEY.md).

### 1. Start the LKW host

**Local (no Docker):**

```bash
cp applications/local_workspace_application/.env.example applications/local_workspace_application/.env
# Set INTERGRAX_ALLOWED_READ_ROOTS to one or more absolute folders LKW may read
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

**Docker:** see [`BUILD_AND_DEPLOY.md §2`](docs/BUILD_AND_DEPLOY.md#2-recommended-local-docker-bootstrap).

All examples below assume the host is running at `http://127.0.0.1:8020`.

### 2. Health check

```bash
curl -s http://127.0.0.1:8020/health
```

**Success:** the host responds with status `ok` (HTTP 200).

### 3. List available agents/capabilities

```bash
curl -s http://127.0.0.1:8020/v1/local_workspace/agents
```

**Success:** the response lists the expected capabilities:

- `local.workspace.index`
- `local.workspace.search`
- `local.workspace.synthesize`
- `local.workspace.pipeline`

### 4. Index a fixture or local document

Replace `<TENANT_ID>`, `<WORKSPACE_ID>`, and `<SOURCE_PATH>` with your values.

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<TENANT_ID>",
    "workspace_id": "<WORKSPACE_ID>",
    "message": "index documents",
    "capability": "local.workspace.index",
    "metadata": {
      "source_paths": ["<SOURCE_PATH>"],
      "collection_id": "<WORKSPACE_ID>"
    }
  }'
```

**Success:**

- `state` is `"completed"`
- metadata contains `application_run_summary.v1`
- accepted document count ≥ 1
- rejected count is 0 for the fixture path

### 5. Search the indexed content

Use the same `<TENANT_ID>` and `<WORKSPACE_ID>` from the index step.

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<TENANT_ID>",
    "workspace_id": "<WORKSPACE_ID>",
    "message": "<YOUR_QUERY>",
    "capability": "local.workspace.search",
    "metadata": {
      "collection_id": "<WORKSPACE_ID>",
      "query": "<YOUR_QUERY>",
      "top_k": 5
    }
  }'
```

**Success:**

- `state` is `"completed"`
- metadata contains `lkw_evidence.v1`
- evidence count ≥ 1
- source reference is present
- no raw fixture content is required in the response

### 6. Run the full pipeline

The pipeline capability (`local.workspace.pipeline`) runs **index → search → synthesize**
in a single request and produces a shadow artifact.

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<TENANT_ID>",
    "workspace_id": "<WORKSPACE_ID>",
    "message": "<YOUR_QUERY>",
    "capability": "local.workspace.pipeline",
    "metadata": {
      "source_paths": ["<SOURCE_PATH>"],
      "collection_id": "<WORKSPACE_ID>",
      "query": "<YOUR_QUERY>",
      "top_k": 5,
      "shadow_workspace": true,
      "output_name": "pipeline-synthesis-draft.md"
    }
  }'
```

**Success:**

- `state` is `"completed"`
- agent invocations in `application_run_summary.v1` include all three:
  `local_indexer`, `local_search`, `local_synthesizer`
- metadata contains `lkw_evidence.v1` with diagnostics key
  `lkw.synthesize_summary.v1` where `shadow_write` is `true`
- a shadow artifact was produced; check `run_artifact_bundle.v1` for artifact refs
- original source file is **not modified** (shadow writes only)

### Metadata keys that confirm success

| Key | Purpose |
|-----|---------|
| `application_run_summary.v1` | Agent invocations, tool calls, terminal status |
| `lkw_evidence.v1` | Search evidence and synthesis diagnostics |
| `runtime_event_summary.v1` | Redacted runtime events per run |
| `run_artifact_bundle.v1` | References to shadow artifacts produced |
| `lkw_proof_summary.v1` | Redacted reviewer-facing proof verdict; built from existing platform/application metadata above — not an observability exporter, not a vendor integration |

All four platform/application keys plus `lkw_proof_summary.v1` are present in a successful pipeline run.

### Troubleshooting

| Issue | Likely cause |
|-------|--------------|
| Host does not respond | Check host start command, port (`8020`), and environment profile |
| Agents endpoint does not list expected capabilities | Check `LOCAL_WORKSPACE_ENABLE_RAG=true` and `LOCAL_WORKSPACE_ENABLE_RAG_INGEST=true` in `.env`; verify skill bundles |
| Index succeeds but search returns no evidence | Use the same `tenant_id` and `workspace_id` for both steps; confirm `<SOURCE_PATH>` points to a readable fixture/test file |
| Pipeline returns `content_missing` | Verify index completed for the same tenant/workspace before running the pipeline; or let the pipeline index first via `source_paths` |
| No shadow artifact found | Check `run_artifact_bundle.v1` and `lkw_evidence.v1` → `lkw.synthesize_summary.v1.artifact_path` or `.artifact_ref` |

### Safety

- **Shadow writes only:** generated artifacts are written to `INTERGRAX_SHADOW_ROOT`,
  not to original source locations.
- **Read allowlist:** set `INTERGRAX_ALLOWED_READ_ROOTS` to limit which paths LKW may read.
- This is a **local product proof / developer path**, not a production certification.

## Running the LKW Slack Ask companion

Temporary slice **LKW-SLACK-WORKFLOW-1A**: approved Slack DM → configured tenant/workspace → Ask HTTP → threaded answer. Dynamic workspace selection arrives in **1B**.

### Configuration architecture

| Prefix | Role |
|--------|------|
| `INTERGRAX_SLACK_*` | Platform Slack **transport** (Socket Mode + Web API tokens) |
| `LOCAL_WORKSPACE_SLACK_*` | LKW **product** authorization and Ask routing |

Both blocks are documented in `applications/local_workspace_application/.env.example`.

### Where to get values

| Variable | Source / format |
|----------|-----------------|
| `LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID` | Slack workspace/team ID (`T…`). Not the workspace display name. Read from a real Slack event or Slack workspace metadata. |
| `LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID` | Slack human user ID (`U…`). Not email, display name, bot user ID, or channel ID. |
| `LOCAL_WORKSPACE_SLACK_TENANT_ID` | Same tenant ID used as `X-Tenant-Id` on Managed Workspace HTTP. Must already exist — do not invent it. |
| `LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID` | Workspace ID returned by Managed Workspace API for that tenant. Must already have synchronized/indexed sources. |
| `LOCAL_WORKSPACE_SLACK_ASK_BASE_URL` | Base URL of the running LKW host. Canonical local: `http://127.0.0.1:8020/`. |

### Canonical start

```bash
cp applications/local_workspace_application/.env.example applications/local_workspace_application/.env
# Fill INTERGRAX_SLACK_* and LOCAL_WORKSPACE_SLACK_* (never commit .env)
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

### Operator sequence

1. Copy `.env.example` to `.env`
2. Fill platform Slack tokens (`INTERGRAX_SLACK_*`)
3. Create or identify an LKW tenant and workspace
4. Synchronize at least one source
5. Fill `LOCAL_WORKSPACE_SLACK_*` values
6. Run configuration preflight (below)
7. Start LKW host
8. Run Ask HTTP preflight (`--question …`)
9. Send approved Slack DM

Configuration preflight (no secrets printed):

```bash
uv run python \
  applications/local_workspace_application/scripts/run-lkw-slack-ask-configuration-preflight.py

uv run python \
  applications/local_workspace_application/scripts/run-lkw-slack-ask-configuration-preflight.py \
  --question "What is the LKW live proof verification code?"
```

Expected outcomes: `PRECHECK=PASS` | `PARTIAL` | `BLOCKED` (exit `0` / `1` / `2`).

Roadmap sequence:

```text
1A workflow code
→ configuration closure
→ operator preflight
→ real live proof
→ 1B
```

Next task after this closure: `LKW-SLACK-WORKFLOW-1A-OPERATOR-PREFLIGHT`.

### Security

- Never commit `.env`.
- Do not paste tokens, API keys, or Ask answers into logs or proof docs.
- Do not record `source_path` or excerpts in proof documentation.
- Slack replies may include safe `file_name` labels only.

## MCP

`http://127.0.0.1:8020/mcp` — `list_agents`, `run_agent`, catalog tools.

## Runtime model

**Philosophy:** local **backend daemon** (Nexus + agents + index) + **thin frontends** (MCP, tray, optional Slack). See [ARCHITECTURE.md §3–§4](docs/ARCHITECTURE.md#3-product-philosophy).

**Install & data paths:** [ARCHITECTURE.md §7](docs/ARCHITECTURE.md#7-installation-lifecycle-and-on-disk-layout)

**Runtime:** [ARCHITECTURE.md §9](docs/ARCHITECTURE.md#9-local-os-runtime-and-interaction-model) — Slack is **optional** (§9.4).

**Implementation waves:** [ARCHITECTURE.md §15](docs/ARCHITECTURE.md#15-implementation-plan-derivation-canonical) · [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)

## Platform stack

LKW uses the canonical **Integration → Tool → Skill → Agent** model ([ARCHITECTURE.md §8](docs/ARCHITECTURE.md#8-integrations-tools-and-skills)):

- **Integrations:** LKW local product profile — SQLite, Qdrant, Docling, optional Redis, local LLM;
- **Tools:** `host/tool_wiring.py` — `rag.*`, `document.parse`, `workspace.*`, `memory.*`, `cache.*`;
- **Skills:** `harness` + `local` bundles (LKW.0, LKW.2.1–LKW.2.3); pipeline capability `local.workspace.pipeline` registered and live proof passed (LKW.2.4A–LKW.2.4C).

## Observability export (platform opt-in)

LKW uses **platform observability export mechanisms only** — there is no LKW-specific observability exporter, OTLP client, or vendor SDK in the application layer.

- Export is **disabled by default**; no remote observability export occurs unless explicitly configured.
- When enabled, pass **`ObservabilityExportOperatorConfig`** to `create_local_workspace_backend_app(observability_export=...)` or compose plugins via **`build_local_workspace_observability_plugins`** at product bootstrap.
- OTLP export is explicit opt-in through platform **`ObservabilityExportOperatorConfig`** + **`build_otlp_observability_export_runtime_plugin`** (INTEGRATIONS-1D); policy/redaction remains enforced upstream.
- Raw local documents, prompts, RAG chunks, synthesized content, and full local file paths are **not exported by default** (`export_content=false` enforced in platform wiring).

## Docs

See [docs/README.md](docs/README.md) for the full local documentation index.

- Final user journey: [USER_JOURNEY.md](docs/USER_JOURNEY.md)
- LKW architecture: [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- LKW live verification: [LKW_1_LIVE_VERIFICATION.md](docs/LKW_1_LIVE_VERIFICATION.md)
- LKW hardening: [ARCHITECTURE_HARDENING.md](docs/ARCHITECTURE_HARDENING.md)
- Platform proof loop: [PLATFORM_PROOF_LOOP.md](docs/PLATFORM_PROOF_LOOP.md)
- Implementation plan: [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- Agent workflow: [docs/guides/AGENT_CREATION_GUIDE.md](../../docs/guides/AGENT_CREATION_GUIDE.md)
- Application layout: [applications/USAGE.md](../USAGE.md)

# Intergrax Platform Proof — Local Knowledge Workspace

This walkthrough is the fastest practical way to see Intergrax working as a platform, not just as an architecture document.

It uses the **Local Knowledge Workspace (LKW)** as a real Tier-3 application built on the Intergrax harness. You will start the local stack, execute a real LKW run, inspect the exported runtime events in Elasticsearch, and open the same run in Kibana.

> This is a local proof path for technical evaluation. It is not a production readiness, security, compliance, SLA, or commercial-use claim. See [`COLLABORATION.md`](../../COLLABORATION.md) and [`LICENSE`](../../LICENSE).

---

## What this proves

This proof shows a real Intergrax application run flowing through multiple platform layers:

```text
User / evaluator
→ LKW HTTP API
→ Tier-3 application host
→ Nexus / task runner
→ Tier-2 local_search agent
→ Tier-0 rag.retrieve tool
→ runtime events
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ElasticsearchObservabilityIntegration
→ Elasticsearch index
→ Kibana Discover / CLI inspector
```

You should see:

- a real `run_id` returned by the LKW API
- runtime events indexed in Elasticsearch
- `tool_requested` and `tool_completed` for `rag.retrieve`
- Kibana Discover showing the run timeline
- CLI proof helper returning `duplicate_check=0`
- CLI proof helper returning `safety_check=passed`

---

## What you will run

The proof stack includes:

```text
local_workspace   LKW API host, port 8020
qdrant            local vector store
ollama            local LLM service used by the LKW stack
otel-collector    local OTLP proof service still present in the base stack
elasticsearch     policy-safe observability document backend, port 9200
kibana            visual UI over Elasticsearch, port 5601
```

The Elasticsearch/Kibana overlay lives here:

```text
applications/local_workspace_application/docker/docker-compose.elasticsearch.yml
```

The platform proof helper lives here:

```text
applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.bat   # Windows convenience
applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh    # Linux/macOS
```

---

## Fast path for external reviewers

This is a **local Docker-based proof path**. You do not need code changes or a hosted service.

- **Canonical start:** explicit `docker compose` from repository root (cross-platform, readable in public docs).
- **Windows helpers:** `.bat` scripts are convenience wrappers around the same Compose stack; the maintainer team validates them on Windows, but they are not the only path.
- **Linux/macOS:** use the same Compose command and `.sh` proof helpers shown in the steps below.
- **Public review assets:** screenshots or a short demo video are optional promotion assets, not requirements for the proof to pass.

---

## Prerequisites

Recommended local environment:

```text
Docker + Docker Compose
Python 3.12
uv
```

Optional on Windows:

```text
PowerShell (for .bat convenience helpers)
Docker Desktop
```

The **canonical external reviewer path** is Docker Compose from repository root. Windows `.bat` helpers are the maintainer-validated convenience path on that platform; Linux and macOS reviewers should use the explicit Compose and shell commands documented here.

---

## Step 1 — Start the local platform proof stack

Both options below start the same proof services:

```text
local_workspace
qdrant
ollama
otel-collector
elasticsearch
kibana
```

### Option A — Cross-platform Docker Compose (preferred)

From repository root:

```bash
docker compose \
  -f applications/local_workspace_application/docker/docker-compose.yml \
  -f applications/local_workspace_application/docker/docker-compose.elasticsearch.yml \
  up --build
```

This is the preferred public-docs path: explicit, readable, and identical on Linux, macOS, and Windows.

### Option B — Windows convenience helper

From repository root:

```powershell
applications\local_workspace_application\scripts\run-local-docker-all.bat
```

This wrapper runs the base `docker-compose.yml` plus every optional overlay in `applications/local_workspace_application/docker/`, including Elasticsearch/Kibana, without listing each `-f` file manually.

---

## Step 2 — Check service health

In a second terminal, from repository root:

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:9200/_cluster/health
```

Open Kibana at `http://127.0.0.1:5601` — the home page should load.

See [Expected proof outputs](#expected-proof-outputs) for health and UI pass criteria.

---

## Step 3 — Run the proof helper before the first LKW run

Windows:

```powershell
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat
```

Linux/macOS:

```bash
chmod +x applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh
applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh
```

Before any LKW run has exported observability documents, it may print:

```text
No observability index found yet.
This is expected before the first LKW run exports Elasticsearch observability documents.
```

That is a valid pre-run state. Elasticsearch is up, but the index is not created until the first export.

---

## Step 4 — Execute a real LKW run

Submit a real LKW request.

Linux/macOS / cross-platform:

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find documents about local workspace observability proof",
    "capability": "local.workspace.search",
    "metadata": { "proof": "LKW_PLATFORM_PROOF" }
  }'
```

Windows (PowerShell):

```powershell
$body = @{
  message = "Find documents about local workspace observability proof"
  capability = "local.workspace.search"
  metadata = @{
    proof = "LKW_PLATFORM_PROOF"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8020/v1/local_workspace/run" `
  -ContentType "application/json" `
  -Body $body
```

Copy the returned `run_id`. Response must include `run_id`, `state`, `answer`, `agent_id`, and `metadata` — see [Expected proof outputs](#expected-proof-outputs).

Example validated run:

```text
run_id=run_d28d5f36f5ca4240b8693ae46eaa5946
```

---

## Step 5 — Validate the run through the CLI proof helper

Replace the example with your own `run_id`.

Windows:

```powershell
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat run_d28d5f36f5ca4240b8693ae46eaa5946
```

Linux/macOS:

```bash
applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh run_d28d5f36f5ca4240b8693ae46eaa5946
```

Expected timeline and PASS criteria: [Expected proof outputs](#expected-proof-outputs).

A validated proof run produced:

```text
run_id=run_d28d5f36f5ca4240b8693ae46eaa5946
records=24
duplicate_check=0
safety_check=passed
proof_result=PASS
```

Full recorded proof artifact:

```text
applications/local_workspace_application/docs/ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md
```

---

## Step 6 — Create the Kibana data view

Open Kibana:

```text
http://127.0.0.1:5601
```

Go to:

```text
Stack Management → Kibana → Data Views → Create data view
```

Use:

```text
Name: intergrax-lkw-observability
Index pattern: intergrax-lkw-observability
Timestamp field: @timestamp
```

Save the data view.

---

## Step 7 — Inspect the run in Kibana Discover

Go to:

```text
Analytics → Discover
```

Select data view:

```text
intergrax-lkw-observability
```

Set time range to a wide enough window, for example:

```text
Last 24 hours
```

Filter by your run id:

```text
intergrax.run_id: "run_d28d5f36f5ca4240b8693ae46eaa5946"
```

Expected: Kibana shows indexed runtime documents for that run, including `tool_requested`, `tool_completed`, and `task_completed` in the timeline. See [Expected proof outputs](#expected-proof-outputs).

A validated proof run showed `Documents: 24`.

---

## Step 8 — Recommended Kibana columns

In Discover, add these columns:

```text
@timestamp
intergrax.run_id
intergrax.event_id
intergrax.event_type
intergrax.agent_id
intergrax.tool_id
intergrax.capability
intergrax.status
```

Useful KQL filters:

```text
intergrax.event_type: "tool_requested"
intergrax.event_type: "tool_completed"
intergrax.tool_id: "rag.retrieve"
intergrax.agent_id: "local_search"
```

Save the Discover view as:

```text
LKW Observability Run Timeline
```

---

## Expected proof outputs

| Check | Command / context | Expected |
|-------|-------------------|----------|
| LKW health | `curl -s http://127.0.0.1:8020/health` | `{"status":"ok"}` |
| Elasticsearch health | `curl -s http://127.0.0.1:9200/_cluster/health` | `green` or `yellow` acceptable for local single-node proof when primary shard is active |
| LKW run response | Step 4 POST | Fields: `run_id`, `state`, `answer`, `agent_id`, `metadata` |
| CLI proof helper | Step 5 with your `run_id` | `Duplicate check: duplicate groups = 0`; `Safety check: 0 forbidden keys`; `Proof result: PASS` |
| Kibana Discover | Step 7 — filter `intergrax.run_id: "<your run_id>"` | Documents visible for selected run; timeline includes `tool_requested`, `tool_completed`, `task_completed` |

---

## Public review assets

Screenshots and demo video are **optional** public-review assets. They are not required for the proof to pass.

Expected asset paths (add when captured for promotion):

```text
docs/public-adoption/assets/lkw-platform-proof/kibana-discover-run-timeline.png
docs/public-adoption/assets/lkw-platform-proof/proof-helper-pass.png
docs/public-adoption/assets/lkw-platform-proof/kibana-data-view.png
```

Suggested captions when assets are added:

```text
Kibana Discover showing indexed LKW runtime documents for one run_id.
CLI proof helper returning duplicate_check=0 and safety_check=passed.
Kibana Data View configured for intergrax-lkw-observability with @timestamp.
```

If a short public demo video is added later, link it from this section. Do not treat video as a proof requirement.

Before committing screenshots or publishing a demo, verify they do not include secrets, private documents, raw prompts, raw chunks, full local paths, tokens, or personal data.

---

## What this proves architecturally

This proof is intentionally small, but it exercises a real platform path:

| Platform concern | What the proof demonstrates |
|------------------|-----------------------------|
| Tier-3 application host | LKW exposes a real HTTP product boundary |
| Task intake | `POST /v1/local_workspace/run` creates a real task/run |
| Nexus/task runner | The task flows through the platform execution path |
| Agent routing | `local.workspace.search` reaches the `local_search` agent path |
| Tool execution | `rag.retrieve` emits `tool_requested` and `tool_completed` events |
| Observability spine | Runtime events become exportable observability envelopes |
| Export policy | Raw content/prompt/chunks/tool args/secrets/full paths are not exported by default |
| Vendor integration | Elasticsearch receives policy-safe documents via the vendor integration path |
| Visual inspection | Kibana can inspect real run timelines by `run_id` |
| Repeatable proof | CLI helper verifies duplicate and safety conditions |

---

## What this does not prove yet

This proof deliberately does not claim:

```text
production security posture
production auth/TLS
production Elasticsearch hardening
index lifecycle management
retry/backoff hardening
dead-letter export handling
throughput or load testing
dashboards as code
multi-vendor semantic mapping
commercial-use permission
```

Those are separate hardening and adoption tracks.

Token optimization proof claims must follow the claim guardrails in [`TOKEN_OPTIMIZATION_CLAIMS.md`](TOKEN_OPTIMIZATION_CLAIMS.md).

---

## Sentry controlled problem proof (local Docker stack)

The LKW platform proof can show two complementary observability views:

- **Elasticsearch/Kibana:** structured run timeline and event inspection
- **Sentry:** error issue triage for controlled platform problem signals

Architecture:

```text
LKW controlled failure
→ platform ProblemReporter
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ObservabilityVendorIntegrationContract
→ ObservabilityVendorPayload (PROBLEMS)
→ Sentry provider
→ local Sentry issue
```

### Start local Sentry overlay

```bash
docker compose \
  -f applications/local_workspace_application/docker/docker-compose.yml \
  -f applications/local_workspace_application/docker/docker-compose.sentry.yml \
  up --build
```

Local Sentry UI:

```text
http://127.0.0.1:9000
```

Bootstrap creates local proof login (`admin@intergrax.local` / `proof-local-only`) and writes the local DSN into `docker/sentry-proof/generated.env` for the LKW container. No external SaaS DSN is required.

### Run controlled LKW problem proof

```bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat
```

Expected:

```text
proof_result=PASS
backend=sentry
sentry_mode=local_docker
sentry_ui=http://127.0.0.1:9000
problem_kind=lkw.proof_controlled_failure
problem_error_code=LKW_PROOF_CONTROLLED_FAILURE
safety_check=passed
```

### Search issue by tags

```text
tag:intergrax.problem_kind:lkw.proof_controlled_failure
tag:intergrax.problem_error_code:LKW_PROOF_CONTROLLED_FAILURE
tag:intergrax.run_id:<run_id from helper output>
```

Expected issue title:

```text
Intergrax problem: lkw.proof_controlled_failure
```

Optional screenshot placeholders (not required for proof):

- `sentry-issue-lkw-controlled-failure.png`
- `sentry-tags-lkw-proof.png`

This does not claim production Sentry readiness. See [`applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md).

---

## Troubleshooting

### Kibana shows no results

Usually this is a time range issue.

Set the Discover time picker to:

```text
Last 24 hours
```

or click:

```text
Search entire time range
```

### Kibana cannot see the index

Check Elasticsearch indices:

```powershell
curl -s "http://127.0.0.1:9200/_cat/indices?v"
```

Expected:

```text
intergrax-lkw-observability
```

If the index does not exist, execute a real LKW run first.

### Proof helper says no index exists

That is expected before the first LKW run. Run Step 4, then run the helper again with the returned `run_id`.

### Proof helper fails for a selected run_id

Check:

```powershell
applications\local_workspace_application\scripts\inspect-elasticsearch-observability.bat --list-runs
```

Then use one of the listed run ids.

---

## Related docs

- LKW build/deploy runbook: [`applications/local_workspace_application/docs/BUILD_AND_DEPLOY.md`](../../applications/local_workspace_application/docs/BUILD_AND_DEPLOY.md)
- Kibana guide: [`applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md)
- Sentry guide: [`applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md)
- Recorded Elasticsearch proof: [`applications/local_workspace_application/docs/ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md`](../../applications/local_workspace_application/docs/ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md)
- LKW product-validation narrative: [`docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md`](../product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md)
- Observability architecture: [`docs/architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md)
- Observability plan: [`docs/plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md)
